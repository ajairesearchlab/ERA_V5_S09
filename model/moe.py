"""
Session 14: a mixture-of-experts feed-forward layer, and the conversion from a dense model.

    MoEFFN        drop-in replacement for SwiGLU inside a Block:
                    router (fp32, softmax over E experts)
                    -> choose top-k using score + bias      (bias only steers the CHOICE)
                    -> k small SwiGLU experts
                    -> weighted sum, weights = chosen scores rescaled to sum 1, times `scale`
                  Dropless: every token reaches all k of its experts however uneven the load.
    upcycle       dense MiniLLM -> MoE MiniLLM by partition + drop-upcycling (notes Section 15)
    param_counts  total and active parameters

Balancing (notes Sections 12-14):
    bias          one number per expert, added to the scores only when choosing the top k.
                  Updated after every optimiser step from the WHOLE batch's load
                  (train/moe_hook.py, MoEBalancer). It gets no gradient.
    sequence aux  tiny per-sequence balance loss (alpha ~1e-4). It is injected as a gradient
                  inside forward (_AddAuxGrad), so train/loop.py does not need to change.

Self-test (random weights, no data needed):
    python -m model.moe
"""
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _AddAuxGrad(torch.autograd.Function):
    """Returns y unchanged; in backward it also sends gradient 1 into `aux`.
    Same effect as adding `aux` to the loss, without touching the loss code."""

    @staticmethod
    def forward(ctx, y, aux):
        return y.view_as(y)

    @staticmethod
    def backward(ctx, grad_y):
        return grad_y, torch.ones((), device=grad_y.device, dtype=torch.float32)


class MoEFFN(nn.Module):
    def __init__(self, d_model, n_experts, top_k, hidden, scale=1.0, router_init_std=0.002,
                 n_layers=8):
        super().__init__()
        assert 1 <= top_k <= n_experts
        self.d_model, self.n_experts, self.top_k, self.hidden = d_model, n_experts, top_k, hidden
        self.scale = float(scale)
        E, H, D = n_experts, hidden, d_model
        self.router = nn.Parameter(torch.randn(E, D) * router_init_std)
        self.w_gate = nn.Parameter(torch.randn(E, H, D) * 0.02)
        self.w_up = nn.Parameter(torch.randn(E, H, D) * 0.02)
        self.w_down = nn.Parameter(torch.randn(E, D, H) * 0.02 / math.sqrt(2 * n_layers))
        self.register_buffer("bias", torch.zeros(E))                     # saved with the model
        self.register_buffer("counts", torch.zeros(E), persistent=False)  # load since last reset
        self.aux_scale = 0.0          # set by the experiment: alpha / accumulation steps
        self.last_aux = None          # the sequence balance loss of the last training forward
        self.record = False           # if True, keep the last top-k choice (analysis only)
        self.last_idx = None

    # ------------------------------------------------------------------ pieces
    def route(self, xf):
        """xf [N, D] -> scores [N, E], chosen experts [N, k], their weights [N, k]. fp32."""
        with torch.autocast(device_type=xf.device.type, enabled=False):
            scores = F.linear(xf.float(), self.router.float()).softmax(dim=-1)
            top_idx = (scores + self.bias).topk(self.top_k, dim=-1).indices
            top_w = scores.gather(-1, top_idx)
            top_w = top_w / top_w.sum(-1, keepdim=True) * self.scale
        return scores, top_idx, top_w

    def expert(self, e, x):
        """One SwiGLU expert: down( silu(gate x) * up x )."""
        return (F.silu(x @ self.w_gate[e].t()) * (x @ self.w_up[e].t())) @ self.w_down[e].t()

    def expert_sum(self, x, ids):
        """Unweighted sum of the given experts' outputs (used to check the conversion)."""
        return sum(self.expert(e, x) for e in ids)

    def _seq_aux(self, scores, top_idx, shape):
        """Per-sequence  sum_i f_i P_i  (f_i scaled so that 1.0 = perfectly even), mean over sequences."""
        E, k = self.n_experts, self.top_k
        B, T = (shape[0], shape[1]) if len(shape) == 3 else (1, scores.shape[0])
        chosen = torch.zeros_like(scores).scatter_(1, top_idx, 1.0)
        f = chosen.view(B, T, E).mean(1) * (E / k)
        P = scores.view(B, T, E).mean(1)
        return (f * P).sum(-1).mean()

    # ------------------------------------------------------------------ forward
    def forward(self, x):
        shape = x.shape
        xf = x.reshape(-1, shape[-1])
        N = xf.shape[0]
        scores, top_idx, top_w = self.route(xf)

        # dispatch: sort the N*k (token, expert) pairs by expert, run each expert once
        flat = top_idx.reshape(-1)
        order = flat.argsort()
        tok = order // self.top_k
        counts = torch.bincount(flat, minlength=self.n_experts)
        xs = xf.index_select(0, tok)
        ys = [self.expert(e, xe) for e, xe in enumerate(xs.split(counts.tolist())) if xe.shape[0]]
        y = torch.cat(ys)
        w = top_w.reshape(-1).index_select(0, order).unsqueeze(-1)

        # combine: weighted outputs added back at each token's position
        out = torch.zeros(N, shape[-1], device=x.device, dtype=torch.float32)
        out.index_add_(0, tok, y.float() * w)
        out = out.to(x.dtype)

        if self.training:
            self.counts += counts.to(self.counts.dtype)
            inject = self.aux_scale > 0 and torch.is_grad_enabled()
            with torch.set_grad_enabled(inject):
                aux = self._seq_aux(scores, top_idx, shape)
            self.last_aux = aux.detach()
            if inject:
                out = _AddAuxGrad.apply(out, aux * self.aux_scale)
        if self.record:
            self.last_idx = top_idx.detach()
        return out.view(shape)


# --------------------------------------------------------------------------- conversion
def upcycle(dense_model, mcfg, drop_r=0.0, seed=0):
    """
    Dense MiniLLM -> MoE MiniLLM. Attention, norms, embeddings and head are copied unchanged.
    Each SwiGLU (width Hd) becomes E experts of width H by partition (Qwen recipe, notes S15):
        n_copies = E*H / Hd copies of the FFN; each copy's neurons shuffled, then cut into Hd/H
        slices; expert (copy c, slice s) = those H neurons' gate rows, up rows, down columns.
        drop_r > 0: in every expert, that share of neurons is redrawn at random (drop-upcycling),
        with the std of the original matrix, so experts start further apart.
    Why the output scale is right: each expert is a random H-of-Hd subset of the neurons; with
    near-uniform routing each weight is ~scale/k. With scale = Hd/H, k experts together cover
    each neuron once on average, so the expected MoE output equals the dense output.
    """
    E, H, k = mcfg.n_experts, mcfg.expert_hidden, mcfg.top_k
    moe = copy.deepcopy(dense_model)
    g = torch.Generator().manual_seed(seed)
    for block in moe.blocks:
        ffn = block.ffn
        Wg = ffn.gate.weight.detach().float().cpu()          # [Hd, D]
        Wu = ffn.up.weight.detach().float().cpu()            # [Hd, D]
        Wd = ffn.down.weight.detach().float().cpu()          # [D, Hd]
        Hd, D = Wg.shape
        assert Hd % H == 0 and (E * H) % Hd == 0, f"need H | {Hd} and {Hd} | E*H"
        n_slices, n_copies = Hd // H, (E * H) // Hd
        sg, su, sd = Wg.std().item(), Wu.std().item(), Wd.std().item()
        gs, us, ds = [], [], []
        for _ in range(n_copies):
            perm = torch.randperm(Hd, generator=g)
            for s in range(n_slices):
                idx = perm[s * H:(s + 1) * H]
                ge, ue, de = Wg[idx].clone(), Wu[idx].clone(), Wd[:, idx].clone()
                m = int(round(drop_r * H))
                if m:
                    j = torch.randperm(H, generator=g)[:m]
                    ge[j] = torch.randn(m, D, generator=g) * sg
                    ue[j] = torch.randn(m, D, generator=g) * su
                    de[:, j] = torch.randn(D, m, generator=g) * sd
                gs.append(ge)
                us.append(ue)
                ds.append(de)
        scale = mcfg.scale if mcfg.scale is not None else Hd / H
        new = MoEFFN(D, E, k, H, scale=scale, router_init_std=mcfg.router_init_std,
                     n_layers=len(moe.blocks))
        with torch.no_grad():
            new.w_gate.copy_(torch.stack(gs))
            new.w_up.copy_(torch.stack(us))
            new.w_down.copy_(torch.stack(ds))
            new.router.copy_(torch.randn(E, D, generator=g) * mcfg.router_init_std)
        dev = ffn.gate.weight.device
        block.ffn = new.to(dev)
    moe.moe_cfg = mcfg
    moe.moe_slices = n_slices        # experts 0..n_slices-1 together are copy 0 = the whole FFN
    return moe


def moe_layers(model):
    return [b.ffn for b in model.blocks if isinstance(b.ffn, MoEFFN)]


def param_counts(model):
    """Total stored parameters, and the parameters one token actually uses."""
    total = sum(p.numel() for p in model.parameters())
    inactive = 0
    for m in moe_layers(model):
        per_expert = m.w_gate[0].numel() + m.w_up[0].numel() + m.w_down[0].numel()
        inactive += (m.n_experts - m.top_k) * per_expert
    return {"total": total, "active": total - inactive}


if __name__ == "__main__":
    from config import ModelConfig, MoEConfig
    from model.model import MiniLLM

    torch.manual_seed(0)
    cfg, mc = ModelConfig(), MoEConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dense = MiniLLM(cfg).to(device)
    moe = upcycle(dense, mc, drop_r=0.0, seed=0)
    for name, m in (("dense", dense), ("moe", moe)):
        c = param_counts(m)
        print(f"{name:<6} total {c['total']:>12,}  active {c['active']:>12,}")
    x = torch.randn(64, cfg.d_model, device=device)
    with torch.no_grad():
        yd = dense.blocks[0].ffn(x)
        ym = moe.blocks[0].ffn.expert_sum(x, range(moe.moe_slices))
    print(f"copy-0 experts summed vs dense FFN, largest difference: {(yd - ym).abs().max().item():.2e}")
    tokens = torch.randint(0, cfg.vocab_size, (2, cfg.context_length), device=device)
    moe.train()
    logits = moe(tokens)
    logits.float().mean().backward()
    print(f"logits {tuple(logits.shape)}, router grad norm {moe.blocks[0].ffn.router.grad.norm().item():.2e}")
    print("self-test OK")
