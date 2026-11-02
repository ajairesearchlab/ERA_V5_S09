"""
Session 11: the optimiser pieces, written out so each one can be checked and switched.

    AdamW            Adam + decoupled weight decay, with a bias_correction switch
                     (torch.optim.AdamW has no such switch; Task 1 checks this class against it)
    param_groups     matrices decay, 1-D weights (RMSNorm gains) don't (notes Section 7)
    cosine / wsd / constant
                     learning-rate schedules: each returns a function step -> lr (Section 10)
    UpdateRatio      |weight change| / |weight| for every parameter tensor, every step (Section 9)
"""
import csv
import math

import torch

from train.loop import lr_at


# --------------------------------------------------------------------------- AdamW
class AdamW(torch.optim.Optimizer):
    """
    For every weight, every step:
        m = b1 m + (1 - b1) g                       average gradient       (direction)
        v = b2 v + (1 - b2) g^2                     average squared grad   (scale)
        w = w - lr * wd * w                         decoupled decay
        w = w - lr * m_hat / (sqrt(v_hat) + eps)    m_hat, v_hat = m, v divided by (1 - b^t)
    With bias_correction=False, m and v are used as they are (still pulled towards 0 early).
    The arithmetic follows torch.optim.AdamW's order so the two agree to rounding.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0,
                 bias_correction=True):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        bias_correction=bias_correction)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            b1, b2 = group["betas"]
            lr, wd, eps = group["lr"], group["weight_decay"], group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state["step"] += 1
                t = state["step"]
                m, v, g = state["exp_avg"], state["exp_avg_sq"], p.grad
                m.lerp_(g, 1 - b1)                                   # b1 m + (1 - b1) g
                v.mul_(b2).addcmul_(g, g, value=1 - b2)
                if wd:
                    p.mul_(1 - lr * wd)
                if group["bias_correction"]:
                    bc1, bc2 = 1 - b1 ** t, 1 - b2 ** t
                else:
                    bc1 = bc2 = 1.0
                denom = (v.sqrt() / math.sqrt(bc2)).add_(eps)
                p.addcdiv_(m, denom, value=-lr / bc1)
        return None


def param_groups(model, weight_decay):
    """Decay the matrices; leave 1-D weights (norm gains, any biases) alone."""
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]


def build_adamw(model, tc, bias_correction=True):
    """Our AdamW with the same settings as train.loop.build_optimizer (torch's)."""
    return AdamW(param_groups(model, tc.weight_decay), lr=tc.lr,
                 betas=(tc.beta1, tc.beta2), bias_correction=bias_correction)


# --------------------------------------------------------------------------- schedules
def cosine(total, peak, warmup, min_frac=0.1):
    """Linear warmup, cosine to min_frac * peak at `total`. Needs `total` in advance."""
    return lambda s: lr_at(s, total, peak, warmup, min_frac)


def wsd(total, peak, warmup, decay_frac=0.1, min_frac=0.1):
    """Warmup, Stable (flat at peak), Decay (linear to min_frac * peak over the last decay_frac)."""
    decay_start = int(round(total * (1 - decay_frac)))

    def f(s):
        if s < warmup:
            return peak * (s + 1) / warmup
        if s < decay_start:
            return peak
        progress = (s - decay_start + 1) / max(1, total - decay_start)
        return peak * (1 - (1 - min_frac) * min(progress, 1.0))
    return f


def constant(peak, warmup=0):
    return lambda s: peak * (s + 1) / warmup if s < warmup else peak


# --------------------------------------------------------------------------- update ratio
def param_kind(name):
    """'blocks.3.attn.q_proj.weight' -> (3, 'q_proj'); 'tok_emb.weight' -> (None, 'tok_emb')."""
    parts = name.split(".")
    if parts[0] == "blocks":
        return int(parts[1]), parts[-2]
    return None, parts[0]


class UpdateRatio:
    """
    Step hook for train.loop.run_training. Before the step it copies every weight; after
    the step it records ||w_after - w_before|| / ||w_before|| for every parameter tensor.
    The change includes weight decay, because that is part of what the optimiser did.
    One .tolist() per step, so the cost is a copy of the weights and a few norms.
    """

    def __init__(self, model):
        self.names = [n for n, _ in model.named_parameters()]
        self.is_matrix = [p.dim() >= 2 for _, p in model.named_parameters()]
        self.rows = []                                     # one dict per step: name -> ratio
        self._prev = None

    @torch.no_grad()
    def before(self, model, step):
        self._prev = [p.detach().clone() for p in model.parameters()]

    @torch.no_grad()
    def after(self, model, step):
        ratios = torch.stack([(p.detach() - q).norm() / q.norm().clamp(min=1e-12)
                              for p, q in zip(model.parameters(), self._prev)])
        self._prev = None
        row = {"step": step}
        row.update(zip(self.names, ratios.float().tolist()))
        self.rows.append(row)

    def series(self, name):
        return [r[name] for r in self.rows]

    def median_matrix(self):
        """Per step: the median ratio over all 2-D weights (one summary number)."""
        mats = [n for n, m in zip(self.names, self.is_matrix) if m]
        out = []
        for r in self.rows:
            vals = sorted(r[n] for n in mats)
            out.append(vals[len(vals) // 2])
        return out

    def by_kind(self):
        """kind -> per-step median over layers, e.g. 'q_proj' -> [..]."""
        kinds = {}
        for n in self.names:
            kinds.setdefault(param_kind(n)[1], []).append(n)
        out = {}
        for k, names in kinds.items():
            series = []
            for r in self.rows:
                vals = sorted(r[n] for n in names)
                series.append(vals[len(vals) // 2])
            out[k] = series
        return out

    def save(self, path):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["step"] + self.names)
            w.writeheader()
            w.writerows(self.rows)
