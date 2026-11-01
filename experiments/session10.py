"""
Session 10 assignment: make a small model and a real training loop tell the truth about itself.

Run from the mini_llm folder:
    uv run python -m experiments.session10                      # everything (~12-15 min)
    uv run python -m experiments.session10 --section checks     # Tasks 1, 2, 6   (~1 min, no training)
    uv run python -m experiments.session10 --section train      # Tasks 3, 4      (~8-10 min)
    uv run python -m experiments.session10 --section mfu        # Task 5          (~2-3 min)
    uv run python -m experiments.session10 --quick              # smoke test, short runs

Sections
    1. shapes      every tensor in one optimiser step, with what each dimension means   (Task 1)
    2. gradcheck   nudge one weight, compare with backward(), in fp64 / fp32 / bf16     (Task 2)
    3. formats     0.1 in fp32, bf16 and fp8 E4M3, bit by bit                           (Task 6)
    4. accum       correct vs average-of-averages accumulation, gradients + training     (Task 3)
    5. gradnorm    grad norm every step; does it move before the loss?                   (Task 4)
    6. mfu         Model FLOPs Utilisation and where the time goes                       (Task 5)

Outputs go to runs/session10/: train_log.csv, range_test.csv, accumulation.png,
gradnorm.png, mfu.json.
"""
import argparse
import csv
import json
import math
import struct
import subprocess
import time

import numpy as np
import torch
import torch.nn.functional as F

import model.attention as attention_module
from config import PROJECT_ROOT, ModelConfig, TrainConfig
from data.dataset import TokenData, load_meta
from model.layers import RMSNorm
from model.model import MiniLLM
from train.loop import (accumulate, build_optimizer, make_micro_batches,
                        micro_targets, open_log, print_log_header, run_training, set_lr,
                        train_step)
from train.loss import IGNORE, lm_loss, shift_targets

RUN_DIR = PROJECT_ROOT / "runs" / "session10"
LENGTHS = (256, 128, 64, 32)          # real tokens per row in each of the 4 micro-batches (Task 3)


def banner(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def shp(t):
    return str(list(t.shape)) if t.dim() else "scalar"


def setup():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    meta = load_meta()
    cfg = ModelConfig(vocab_size=meta["vocab_size"])
    return device, cfg, meta["pad_id"], meta["eos_id"]


def fresh_model(cfg, device, seed=0):
    torch.manual_seed(seed)
    return MiniLLM(cfg).to(device)


def val_set(cfg, device, n):
    g = torch.Generator().manual_seed(1234)                          # same val batches everywhere
    val = TokenData("val")
    return [val.get_batch(16, cfg.context_length, device=device, generator=g) for _ in range(n)]


# =========================================================================== 1. shapes
def section_shapes(device, cfg, pad, eos):
    banner("1. SHAPES: every tensor in one optimiser step (Task 1)")
    tc = TrainConfig()
    B, K, T, D, H = tc.micro_batch, tc.accum_steps, cfg.context_length, cfg.d_model, cfg.n_heads
    print(f"  one step = {K} micro-batches x {B} sequences x {T} tokens = {K * B} sequences seen "
          f"by the optimiser\n")
    model = fresh_model(cfg, device)
    opt = build_optimizer(model, tc)
    gen = torch.Generator().manual_seed(0)
    micros = make_micro_batches(TokenData("train"), K, B, T, device, gen)

    rec = []                                                        # (name, tensor, meaning)

    def keep(name, meaning):
        def hook(_m, _inp, out):
            if not done["forward"]:
                rec.append((name, out, meaning))
        return hook

    done = {"forward": False}
    b0 = model.blocks[0]
    hooks = [
        (model.tok_emb, "tok_emb(tokens)", "one D-vector per token: B x T x D"),
        (b0.attn_norm, "block 1 attn_norm", "RMSNorm keeps the shape, rescales each token vector"),
        (b0.attn.q_proj, "q_proj output", "queries before splitting: B x T x D"),
        (b0.attn.k_proj, "k_proj output", "keys before splitting"),
        (b0.attn.v_proj, "v_proj output", "values before splitting"),
        (b0.attn.o_proj, "o_proj output", "heads merged back and mixed: B x T x D"),
        (b0.ffn_norm, "block 1 ffn_norm", "RMSNorm again before the FFN"),
        (b0.ffn.gate, "ffn gate(x)", f"B x T x {cfg.ffn_hidden}: widened, 'how much'"),
        (b0.ffn.up, "ffn up(x)", f"B x T x {cfg.ffn_hidden}: widened, 'what'"),
        (b0.ffn.down, "ffn down(...)", "back to B x T x D, added to the residual stream"),
        (b0, "block 1 output", "residual stream: same shape after every block"),
        (model.final_norm, "final_norm (hidden h)", "what the head reads"),
        (model.lm_head, "logits", f"B x T x V: a score for each of {cfg.vocab_size} tokens"),
    ]
    handles = [m.register_forward_hook(keep(n, s)) for m, n, s in hooks]

    sdpa = attention_module.F.scaled_dot_product_attention           # torch.nn.functional's own

    def sdpa_spy(q, k, v, *a, **kw):
        out = sdpa(q, k, v, *a, **kw)
        if not done["forward"] and not any(n == "q (heads split, RoPE)" for n, _, _ in rec):
            rec.append(("q (heads split, RoPE)", q, f"B x H x T x hd: {H} heads of {cfg.head_dim}"))
            rec.append(("k (heads split, RoPE)", k, "same: one key per token per head"))
            rec.append(("v (heads split)", v, "same: one value per token per head"))
            rec.append(("attention output", out, "B x H x T x hd, before merging heads"))
        return out

    attention_module.F.scaled_dot_product_attention = sdpa_spy
    try:
        targets = micro_targets(micros, pad, eos)
        counts = [int((t != IGNORE).sum()) for t in targets]
        # first micro-batch alone, so we can show the loss tensors too
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
            logits = model(micros[0])
        done["forward"] = True
        V = logits.size(-1)
        flat_logits = logits[:, :-1].reshape(-1, V)
        flat_targets = targets[0].reshape(-1)
        loss_sum = F.cross_entropy(flat_logits.float(), flat_targets, ignore_index=IGNORE, reduction="sum")
    finally:
        attention_module.F.scaled_dot_product_attention = sdpa
        for h in handles:
            h.remove()

    order = ["tok_emb(tokens)", "block 1 attn_norm", "q_proj output", "k_proj output", "v_proj output",
             "q (heads split, RoPE)", "k (heads split, RoPE)", "v (heads split)", "attention output",
             "o_proj output", "block 1 ffn_norm", "ffn gate(x)", "ffn up(x)", "ffn down(...)",
             "block 1 output", "final_norm (hidden h)", "logits"]
    by_name = {n: (t, s) for n, t, s in rec}

    print(f"  {'tensor':<26}{'shape':<22}{'dtype':<10}meaning")
    print(f"  {'-' * 26}{'-' * 22}{'-' * 10}{'-' * 40}")

    def row(name, t, meaning):
        print(f"  {name:<26}{shp(t):<22}{str(t.dtype).replace('torch.', ''):<10}{meaning}")

    row("tokens (micro-batch 1)", micros[0], "B sequences x T token ids")
    row("targets (shifted, masked)", targets[0], "B x (T-1): next token, -100 where masked")
    row("rope cos / sin", model.rope_cos[:T], "T x hd: one rotation angle per position, per pair")
    for n in order:
        if n in by_name:
            t, s = by_name[n]
            row(n, t, s)
            if n == "block 1 output":
                print(f"  {'... blocks 2-' + str(cfg.n_layers):<26}{shp(t):<22}{'':<10}same shape: blocks refine, never resize")
    row("logits[:, :-1] flattened", flat_logits, "B*(T-1) rows: one classification each")
    row("targets flattened", flat_targets, "one correct id per row")
    row("loss (sum, micro-batch 1)", loss_sum, f"summed over {counts[0]:,} real targets")

    # the full step: accumulate, then look at gradients and optimiser state
    out = accumulate(model, micros, pad, eos, "token")
    print(f"\n  real targets per micro-batch: {counts}  -> total {sum(counts):,}")
    print(f"  step loss = (sum of all token losses) / {sum(counts):,} = {out['reported'].item():.4f}")

    print(f"\n  after backward() on all {K} micro-batches, before clipping:")
    print(f"  {'parameter':<30}{'weight shape':<16}{'grad shape':<16}{'grad dtype':<11}meaning")
    shown = ["tok_emb.weight", "blocks.0.attn_norm.weight", "blocks.0.attn.q_proj.weight",
             "blocks.0.attn.o_proj.weight", "blocks.0.ffn.gate.weight", "blocks.0.ffn.down.weight",
             "final_norm.weight", "lm_head.weight"]
    params = dict(model.named_parameters())
    for n in shown:
        p = params[n]
        print(f"  {n:<30}{shp(p):<16}{shp(p.grad):<16}{str(p.grad.dtype).replace('torch.', ''):<11}"
              f"one number per weight")
    print(f"  ... {len(params)} parameter tensors in all; every .grad has its weight's shape")

    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
    print(f"\n  {'grad norm (before clip)':<30}{'scalar':<16}{gn.item():.4f}  one length for ALL "
          f"{sum(p.numel() for p in model.parameters()):,} gradients")
    opt.step()

    st = opt.state[params["blocks.0.attn.q_proj.weight"]]
    print("\n  AdamW state after opt.step(), for blocks.0.attn.q_proj.weight:")
    for k, v in st.items():
        meaning = {"exp_avg": "running mean of gradients (momentum)",
                   "exp_avg_sq": "running mean of squared gradients",
                   "step": "how many updates so far"}.get(k, "")
        print(f"    {k:<14}{shp(v):<16}{str(v.dtype).replace('torch.', ''):<10}{meaning}")

    n_w = sum(p.numel() for p in model.parameters())
    b_w = sum(p.numel() * p.element_size() for p in model.parameters())
    b_g = sum(p.grad.numel() * p.grad.element_size() for p in model.parameters())
    b_o = sum(v.numel() * v.element_size() for s in opt.state.values() for v in s.values() if v.dim())
    opt.zero_grad(set_to_none=True)
    print("\n  bytes held for every weight in this run (Section 13):")
    print(f"    weights fp32 {b_w / n_w:.0f} + grads {b_g / n_w:.0f} + AdamW m and v {b_o / n_w:.0f} "
          f"= {(b_w + b_g + b_o) / n_w:.0f} bytes/weight -> {(b_w + b_g + b_o) / 2 ** 20:,.0f} MiB "
          f"for {n_w / 1e6:.2f}M weights, before activations")


# =========================================================================== 2. gradcheck
def section_gradcheck(device, cfg, pad, eos):
    banner("2. GRADIENT CHECK: nudge one weight, compare with backward() (Task 2)")
    print(f"  dropout = {cfg.dropout}: the forward pass is repeatable, so a nudge measures only the weight")
    g = torch.Generator().manual_seed(7)
    tokens = TokenData("val").get_batch(2, 64, device=device, generator=g)

    targets = shift_targets(tokens, pad_id=pad, eos_id=eos, mask_boundaries=True).reshape(-1)

    def loss_fn(model, mode):
        # Not lm_loss: it casts logits to fp32, which would cap an fp64 check at fp32 precision.
        if mode == "bf16":
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits = model(tokens).float()
        else:
            logits = model(tokens)
        return F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), targets, ignore_index=IGNORE)

    # RMSNorm normalises in fp32 (x.float()) for stability. In an fp64 check that silently
    # rounds every activation to fp32 and the nudge drowns in it, so for this section only
    # RMSNorm computes in whatever dtype it is given.
    def rmsnorm_same_dtype(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

    original_forward = RMSNorm.forward
    RMSNorm.forward = rmsnorm_same_dtype
    try:
        _gradcheck_body(device, cfg, loss_fn)
    finally:
        RMSNorm.forward = original_forward


def _gradcheck_body(device, cfg, loss_fn):
    model = fresh_model(cfg, device).double()
    L1, L2 = loss_fn(model, "fp64").item(), loss_fn(model, "fp64").item()
    print(f"  two identical forward passes: {L1:.12f} vs {L2:.12f} (difference {abs(L1 - L2):.1e})")

    names = ["blocks.0.attn.q_proj.weight", "blocks.7.ffn.down.weight", "lm_head.weight",
             "final_norm.weight", "tok_emb.weight"]
    params = dict(model.named_parameters())
    model.zero_grad(set_to_none=True)
    loss_fn(model, "fp64").backward()
    picks = []
    for n in names:
        idx = tuple(int(i) for i in np.unravel_index(int(params[n].grad.abs().argmax()), params[n].shape))
        picks.append((n, idx, params[n].grad[idx].item()))

    def slope(model, name, idx, eps, mode, central=True):
        p = dict(model.named_parameters())[name]
        with torch.no_grad():
            orig = p[idx].item()
            p[idx] = orig + eps
            up = loss_fn(model, mode).item()
            if central:
                p[idx] = orig - eps
                down = loss_fn(model, mode).item()
            else:
                p[idx] = orig
                down = loss_fn(model, mode).item()
            p[idx] = orig
        return (up - down) / (2 * eps if central else eps)

    name, idx, auto = picks[0]
    print(f"\n  weight {name}{list(idx)} (largest gradient in that matrix)")
    print(f"  backward() says dLoss/dw = {auto:+.10f}  (fp64)\n")
    print(f"  {'nudge':>8}{'one-sided slope':>20}{'rel. error':>12}{'central slope':>20}{'rel. error':>12}")
    for eps in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6):
        a = slope(model, name, idx, eps, "fp64", central=False)
        c = slope(model, name, idx, eps, "fp64", central=True)
        print(f"  {eps:>8.0e}{a:>+20.10f}{abs(a - auto) / abs(auto):>12.1e}"
              f"{c:>+20.10f}{abs(c - auto) / abs(auto):>12.1e}")
    print("  One-sided: error shrinks with the nudge (curvature). Central cancels most of it.")
    print("  Too small a nudge and rounding in the loss itself takes over.")

    print(f"\n  five weights, central nudge 1e-5, fp64:")
    print(f"  {'weight':<30}{'index':<14}{'backward()':>16}{'nudge':>16}{'rel. error':>12}")
    for n, ix, a in picks:
        c = slope(model, n, ix, 1e-5, "fp64")
        print(f"  {n:<30}{str(list(ix)):<14}{a:>+16.10f}{c:>+16.10f}{abs(c - a) / abs(a):>12.1e}")

    print("\n  the same check at lower precision (first weight, compared with the fp64 answer):")
    print(f"  {'precision':<12}{'nudge':>8}{'measured slope':>18}{'rel. error':>12}")
    m32 = fresh_model(cfg, device).float()
    for mode, model_p, epss in (("fp64", model, (1e-5,)), ("fp32", m32, (1e-2, 1e-3, 1e-4)),
                                ("bf16", m32, (1e-1, 1e-2, 1e-3))):
        for eps in epss:
            c = slope(model_p, name, idx, eps, mode)
            print(f"  {mode:<12}{eps:>8.0e}{c:>+18.8f}{abs(c - auto) / abs(auto):>12.1e}")
    print("  In bf16 the loss itself has only ~3 significant digits, so a small nudge is lost in")
    print("  rounding: gradient checks must run in fp32 or fp64.")


# =========================================================================== 3. formats
def encode(x, e_bits, m_bits):
    """Hand method: sign, power of two, mantissa bits with round-to-nearest-even (normal numbers)."""
    bias = 2 ** (e_bits - 1) - 1
    s = 0 if x >= 0 else 1
    x = abs(x)
    e = math.floor(math.log2(x))
    frac = x / 2 ** e - 1                                         # in [0, 1)
    scaled = frac * 2 ** m_bits
    m = math.floor(scaled)
    rest = scaled - m
    if rest > 0.5 or (rest == 0.5 and m % 2 == 1):
        m += 1
    if m == 2 ** m_bits:
        m, e = 0, e + 1
    return s, e + bias, m, e, bias


def section_formats():
    banner("3. THE NUMBER 0.1 IN fp32, bf16 AND fp8 E4M3 (Task 6)")
    x = 0.1
    print("  Step 1. Write 0.1 as (1 + fraction) x 2^power:  0.1 = 1.6 x 2^-4")
    print("  Step 2. The fraction 0.6 in binary, by repeated doubling:")
    f, digits = 0.6, ""
    for i in range(12):
        f2 = f * 2
        bit = int(f2 >= 1)
        digits += str(bit)
        print(f"            {f:.1f} x 2 = {f2:.1f}  -> bit {bit}")
        f = f2 - bit
    print(f"          0.6 = 0.{digits}...  (1001 repeats forever, so every format must round)")

    exact = {}
    b32 = struct.unpack(">I", struct.pack(">f", x))[0]
    exact["fp32"] = f"{b32:032b}"
    exact["bf16"] = f"{torch.tensor(x, dtype=torch.float32).to(torch.bfloat16).view(torch.int16).item() & 0xFFFF:016b}"
    try:
        exact["fp8 E4M3"] = f"{torch.tensor(x).to(torch.float8_e4m3fn).view(torch.uint8).item():08b}"
    except (AttributeError, RuntimeError):
        exact["fp8 E4M3"] = None

    formats = [("fp32", 8, 23), ("bf16", 8, 7), ("fp8 E4M3", 4, 3)]
    print("\n  Step 3. Exponent field = power + bias; mantissa = first bits of the fraction, rounded:\n")
    print(f"  {'format':<10}{'s':<3}{'exponent':<11}{'mantissa':<26}{'stored value':<22}{'rel. error':<12}{'PyTorch agrees'}")
    for name, eb, mb in formats:
        s, ef, m, e, bias = encode(x, eb, mb)
        bits = f"{s}{ef:0{eb}b}{m:0{mb}b}"
        value = (1 + m / 2 ** mb) * 2.0 ** e
        ok = "yes" if exact[name] == bits else ("n/a" if exact[name] is None else f"NO ({exact[name]})")
        print(f"  {name:<10}{s:<3}{ef:0{eb}b}{'':<{11 - eb}}{m:0{mb}b}{'':<{26 - mb}}"
              f"{value:<22.15g}{abs(value - x) / x:<12.2e}{ok}")
        print(f"  {'':<10}   = {ef} - bias {bias} = 2^{e};  mantissa {m}/{2 ** mb}"
              f" -> (1 + {m / 2 ** mb:.10g}) x 2^{e}")

    print("\n  What each format can hold (range is set by exponent bits, detail by mantissa bits):")
    print(f"  {'format':<10}{'exp':>4}{'mant':>6}{'decimal digits':>16}{'largest':>12}{'smallest normal':>17}{'smallest subnormal':>20}")
    rows = [("fp32", 8, 23, (2 - 2 ** -23) * 2.0 ** 127, 2.0 ** -126, 2.0 ** -149),
            ("bf16", 8, 7, (2 - 2 ** -7) * 2.0 ** 127, 2.0 ** -126, 2.0 ** -133),
            ("fp16", 5, 10, 65504.0, 2.0 ** -14, 2.0 ** -24),
            ("fp8 E4M3", 4, 3, 448.0, 2.0 ** -6, 2.0 ** -9)]
    for name, eb, mb, big, small, sub in rows:
        print(f"  {name:<10}{eb:>4}{mb:>6}{(mb + 1) * math.log10(2):>16.1f}{big:>12.3g}{small:>17.3g}{sub:>20.3g}")
    print("  (fp8 E4M3 has no infinity and only 448 as its top: real fp8 training shares a scale")
    print("   per block or tensor, exactly the idea in Section 11.)")


# =========================================================================== 4. accumulation
def flat_grad(model):
    return torch.cat([p.grad.detach().float().flatten() for p in model.parameters()])


def section_accum(device, cfg, pad, eos, args):
    banner("4. GRADIENT ACCUMULATION: correct vs average of averages (Task 3)")
    tc = TrainConfig()
    train = TokenData("train")
    print(f"  {tc.accum_steps} micro-batches of {tc.micro_batch} rows; rows in micro-batch i keep "
          f"{list(LENGTHS)} real tokens, the rest is <pad>\n")

    # ---- 4a. one step, three ways, in fp32 (no rounding noise)
    print("  4a. One step's gradient, compared with ONE big batch holding the same rows (fp32):")
    model = fresh_model(cfg, device)
    for case, lengths in (("different lengths", LENGTHS), ("equal lengths", None)):
        gen = torch.Generator().manual_seed(11)
        micros = make_micro_batches(train, tc.accum_steps, tc.micro_batch, cfg.context_length,
                                    device, gen, lengths, pad)
        counts = [int((t != IGNORE).sum()) for t in micro_targets(micros, pad, eos)]
        model.zero_grad(set_to_none=True)
        big = torch.cat(micros)
        ref_loss = lm_loss(model(big), big, pad_id=pad, eos_id=eos, mask_boundaries=True)[0]
        ref_loss.backward()
        ref = flat_grad(model)
        print(f"\n    {case}: real targets per micro-batch {counts}")
        if case == "different lengths":
            tot = sum(counts)
            print(f"    share of tokens  {[f'{c / tot:.1%}' for c in counts]}   vs the bug's vote "
                  f"{[f'{1 / len(counts):.1%}'] * len(counts)}")
        print(f"    {'method':<26}{'loss':>9}{'grad rel. error':>17}{'cosine':>10}")
        print(f"    {'one big batch (reference)':<26}{ref_loss.item():>9.4f}{'-':>17}{'-':>10}")
        for red, label in (("token", "token-weighted (correct)"), ("mean_of_means", "average of averages")):
            model.zero_grad(set_to_none=True)
            out = accumulate(model, micros, pad, eos, red, use_autocast=False)
            g = flat_grad(model)
            err = ((g - ref).norm() / ref.norm()).item()
            cos = F.cosine_similarity(g, ref, dim=0).item()
            print(f"    {label:<26}{out['reported'].item():>9.4f}{err:>17.2e}{cos:>10.6f}")
    model.zero_grad(set_to_none=True)
    del model

    # ---- 4b. two training runs, identical start and batches
    steps = args.steps
    print(f"\n  4b. Two training runs, {steps} steps each, same initial weights, same batches:")
    tc = TrainConfig(steps=steps, eval_every=max(10, steps // 12))
    vb = val_set(cfg, device, tc.eval_batches)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    f, w = open_log(RUN_DIR / "train_log.csv")
    runs = {}
    with f:
        for name, red in (("correct", "token"), ("bug", "mean_of_means")):
            print(f"\n  RUN {name}: reduction = {red}  (loss = what this run optimises; "
                  f"true = token-weighted, same yardstick for both)")
            print_log_header()
            model = fresh_model(cfg, device)
            runs[name] = run_training(name, model, tc, train, vb, pad, eos, device, w,
                                      reduction=red, lengths=LENGTHS, log_every=max(10, steps // 12))
            f.flush()
            del model
            if device == "cuda":
                torch.cuda.empty_cache()

    c = {r["step"]: r for r in runs["correct"]}
    b = {r["step"]: r for r in runs["bug"]}
    print(f"\n  {'step':>6}{'correct':>10}{'bug says':>10}{'bug true':>10}{'bug error':>11}"
          f"{'val correct':>13}{'val bug':>9}")
    for s in sorted(c):
        if s not in b or "loss" not in c[s] or (s % tc.eval_every and s != steps - 1):
            continue
        cv, bv = c[s].get("val_loss", float("nan")), b[s].get("val_loss", float("nan"))
        err = (b[s]["loss"] - b[s]["true_loss"]) / b[s]["true_loss"]
        print(f"  {s:>6}{c[s]['loss']:>10.4f}{b[s]['loss']:>10.4f}{b[s]['true_loss']:>10.4f}"
              f"{err:>+11.1%}{cv:>13.4f}{bv:>9.4f}")
    fc, fb = runs["correct"][-1].get("val_loss"), runs["bug"][-1].get("val_loss")
    print(f"\n  final validation loss: correct {fc:.4f}, bug {fb:.4f} ({fb - fc:+.4f})")
    print("  'bug says' is what the buggy run would put on its dashboard. 'bug true' is the same")
    print("  batch scored per token. The short micro-batches hold the early, poorly-predicted")
    print("  positions, and the bug gives them a full quarter of the vote.")
    plot_accum(runs)
    return runs


def ema(x, a=0.05):
    out, m = [], None
    for v in x:
        m = v if m is None else (1 - a) * m + a * v
        out.append(m)
    return out


def get_plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        print("  (matplotlib not installed: uv add matplotlib)")
        return None


def plot_accum(runs):
    plt = get_plt()
    if not plt:
        return
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.5))
    for name, colour in (("correct", "tab:blue"), ("bug", "tab:red")):
        rows = [r for r in runs[name] if "loss" in r]
        st = [r["step"] for r in rows]
        a.plot(st, ema([r["loss"] for r in rows]), color=colour,
               label=f"{name}: loss it reports")
        if name == "bug":
            a.plot(st, ema([r["true_loss"] for r in rows]), "--", color=colour,
                   label="bug: same batches, token-weighted")
        v = [r for r in runs[name] if r.get("val_loss") == r.get("val_loss") and r.get("val_loss") is not None]
        b.plot([r["step"] for r in v], [r["val_loss"] for r in v], "o-", color=colour, label=name)
    a.set(xlabel="step", ylabel="training loss (EMA)", title="Training loss: the gap is the bug")
    b.set(xlabel="step", ylabel="validation loss", title="Validation loss (same yardstick)")
    for ax in (a, b):
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(RUN_DIR / "accumulation.png", dpi=120)
    plt.close(fig)
    print(f"\n  plot saved: {RUN_DIR / 'accumulation.png'}")


# =========================================================================== 5. grad norm
def onsets(rows, start=20):
    """First step the grad norm leaves its band, and first step the loss turns up."""
    gn = [r["grad_norm"] for r in rows if "loss" in r]
    lo = ema([r["loss"] for r in rows if "loss" in r], 0.1)
    g_on = l_on = None
    for s in range(start, len(gn) - 2):
        base = float(np.median(gn[max(0, s - 50):s]))
        if all(x > 2.0 * base for x in gn[s:s + 3]) and all(math.isfinite(x) for x in gn[s:s + 3]):
            g_on = s
            break
    for s in range(start, len(lo)):
        if not math.isfinite(lo[s]) or lo[s] > min(lo[:s]) + 0.3:
            l_on = s
            break
    return g_on, l_on


def section_gradnorm(device, cfg, pad, eos, args, runs=None):
    banner("5. GRAD NORM AT EVERY STEP: does it move before the loss? (Task 4)")
    if runs is None:
        print("  (run with --section train to include the normal run)")
    else:
        rows = [r for r in runs["correct"] if "loss" in r]
        gn = np.array([r["grad_norm"] for r in rows])
        print("  5a. The normal (correct) run. Largest grad-norm jumps after warmup, against the")
        print("      loss in the 10 steps before and after:")
        print(f"  {'step':>6}{'grad norm':>11}{'x median':>10}{'loss before':>13}{'loss after':>12}")
        cands = []
        for s in range(100, len(rows) - 10):
            med = float(np.median(gn[s - 50:s]))
            cands.append((gn[s] / med, s))
        for ratio, s in sorted(cands, reverse=True)[:5]:
            before = np.mean([r["loss"] for r in rows[s - 10:s]])
            after = np.mean([r["loss"] for r in rows[s + 1:s + 11]])
            print(f"  {s:>6}{gn[s]:>11.3f}{ratio:>10.2f}{before:>13.4f}{after:>12.4f}")
        print("  A healthy run with clipping on rarely gives a clean spike; that is the point of")
        print("  watching it. 5b makes the run go wrong on purpose.")

    steps = args.range_steps
    lo, hi = 1e-4, 1e-1
    print(f"\n  5b. Learning-rate range test: lr grows from {lo:g} to {hi:g} over {steps} steps,")
    print("      clipping OFF (the norm is still measured), until training breaks.")
    tc = TrainConfig(steps=steps, grad_clip=0.0, eval_every=10 ** 9, seed=1)
    print_log_header()
    model = fresh_model(cfg, device)
    rows = run_training("range_test", model, tc, TokenData("train"), [], pad, eos, device,
                        lr_fn=lambda s: lo * (hi / lo) ** (s / steps), log_every=max(10, steps // 15))
    del model
    rows = [r for r in rows if "loss" in r]
    with open(RUN_DIR / "range_test.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    g_on, l_on = onsets(rows)
    print(f"\n  grad norm leaves its band (> 2x its recent median for 3 steps): "
          f"step {g_on if g_on is not None else 'not found'}")
    print(f"  loss turns up (EMA rises 0.3 above its best so far):            "
          f"step {l_on if l_on is not None else 'not found'}")
    if g_on is not None and l_on is not None:
        lead = l_on - g_on
        print(f"  -> the grad norm moved {lead} steps {'BEFORE' if lead > 0 else 'after'} the loss "
              f"(lr {rows[g_on]['lr']:.2e} vs {rows[l_on]['lr']:.2e})")
    plot_gradnorm(runs, rows, g_on, l_on)


def plot_gradnorm(runs, rt, g_on, l_on):
    plt = get_plt()
    if not plt:
        return
    panels = ([("normal run (correct accumulation)", [r for r in runs["correct"] if "loss" in r], None, None)]
              if runs else []) + [("learning-rate range test, no clipping", rt, g_on, l_on)]
    fig, axes = plt.subplots(len(panels), 1, figsize=(12, 4.2 * len(panels)), squeeze=False)
    for ax, (title, rows, g, l) in zip(axes[:, 0], panels):
        st = [r["step"] for r in rows]
        ax.plot(st, [r["grad_norm"] for r in rows], color="tab:orange", lw=0.8, label="grad norm")
        ax.set_yscale("log")
        ax.set_ylabel("grad norm (log)", color="tab:orange")
        ax2 = ax.twinx()
        ax2.plot(st, [r["loss"] for r in rows], color="tab:blue", lw=0.8, label="loss")
        ax2.set_ylabel("loss", color="tab:blue")
        if g is not None:
            ax.axvline(g, color="tab:orange", ls="--", label=f"norm moves: {g}")
        if l is not None:
            ax.axvline(l, color="tab:blue", ls="--", label=f"loss moves: {l}")
        ax.set(xlabel="step", title=title)
        ax.legend(loc="upper left")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RUN_DIR / "gradnorm.png", dpi=120)
    plt.close(fig)
    print(f"\n  plot saved: {RUN_DIR / 'gradnorm.png'}")


# =========================================================================== 6. MFU
def matmul_tflops(n, iters=30):
    a = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    for _ in range(3):
        a @ b
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        a @ b
    torch.cuda.synchronize()
    return 2 * n ** 3 * iters / (time.perf_counter() - t) / 1e12


def time_config(cfg, pad, eos, micro, accum, compile_=False, steps=20, warm=5):
    model = fresh_model(cfg, "cuda")
    if compile_:
        model = torch.compile(model)
    tc = TrainConfig(micro_batch=micro, accum_steps=accum)
    opt = build_optimizer(model, tc)
    set_lr(opt, 1e-4)
    train = TokenData("train")
    gen = torch.Generator().manual_seed(0)
    torch.cuda.reset_peak_memory_stats()
    for i in range(warm + steps):
        if i == warm:
            torch.cuda.synchronize()
            t = time.perf_counter()
        micros = make_micro_batches(train, accum, micro, cfg.context_length, "cuda", gen)
        train_step(model, opt, micros, pad, eos, "token", 1.0)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t
    tok_s = steps * micro * accum * cfg.context_length / dt
    return tok_s, torch.cuda.max_memory_allocated() / 2 ** 20


def phase_breakdown(cfg, pad, eos, micro=32, steps=20):
    """Wall time per phase of a step, with a sync after each so phases don't overlap."""
    model = fresh_model(cfg, "cuda")
    tc = TrainConfig(micro_batch=micro, accum_steps=1)
    opt = build_optimizer(model, tc)
    set_lr(opt, 1e-4)
    train = TokenData("train")
    gen = torch.Generator().manual_seed(0)
    phases = {k: 0.0 for k in ("data (get_batch + copy)", "forward + loss", "backward",
                               "grad norm + clip", "optimizer step", "zero_grad", "logging (.item())")}
    keys = list(phases)
    for i in range(steps + 3):
        ts = [time.perf_counter()]

        def mark():
            torch.cuda.synchronize()
            ts.append(time.perf_counter())
        tokens = train.get_batch(micro, cfg.context_length, device="cuda", generator=gen)
        mark()
        tg = micro_targets([tokens], pad, eos)[0]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(tokens)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)).float(), tg.reshape(-1),
                               ignore_index=IGNORE)
        mark()
        loss.backward()
        mark()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        mark()
        opt.step()
        mark()
        opt.zero_grad(set_to_none=True)
        mark()
        _ = loss.item(), gn.item()
        mark()
        if i >= 3:
            for k, a, b in zip(keys, ts[:-1], ts[1:]):
                phases[k] += (b - a) * 1000 / steps
    return phases


def section_mfu(device, cfg, pad, eos, args):
    banner("6. MFU: how much of the GPU are we using? (Task 5)")
    if device != "cuda":
        print("  needs a CUDA GPU: skipped")
        return
    props = torch.cuda.get_device_properties(0)
    print(f"  GPU: {props.name}, {props.multi_processor_count} SMs, {props.total_memory / 2 ** 30:.1f} GiB")
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=power.limit,power.max_limit,clocks.sm,clocks.max.sm",
                            "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
        print(f"  nvidia-smi (power limit, max power limit, SM clock now, SM clock max): {q.stdout.strip()}")
    except Exception as e:                                          # noqa: BLE001
        print(f"  nvidia-smi not available ({e.__class__.__name__})")

    print("\n  6a. What the GPU can actually do: bf16 matrix multiply, measured")
    peaks = {}
    for n in (1024, 2048, 4096, 8192):
        try:
            peaks[n] = matmul_tflops(n)
            print(f"    {n:>5} x {n:<5} {peaks[n]:>7.1f} TFLOP/s")
        except torch.cuda.OutOfMemoryError:
            print(f"    {n:>5} x {n:<5} out of memory")
    peak = max(peaks.values())
    spec = args.spec_tflops
    print(f"    measured peak: {peak:.1f} TFLOP/s" +
          (f"   spec sheet (you supplied): {spec:.1f} TFLOP/s" if spec else
           "   (pass --spec-tflops X to also report against the spec sheet)"))

    m = MiniLLM(cfg)
    n_all = sum(p.numel() for p in m.parameters())
    n_emb = m.tok_emb.weight.numel()
    del m
    L, T, D = cfg.n_layers, cfg.context_length, cfg.d_model
    f_notes = 6 * n_all
    f_refined = 6 * (n_all - n_emb) + 12 * L * T * D
    print(f"\n  6b. FLOPs per token (forward + backward):")
    print(f"    notes' rule 6N, N = all {n_all:,} weights             {f_notes / 1e6:>8.1f} MFLOP")
    print(f"    refined: 6 x {n_all - n_emb:,} matmul weights + attention    {f_refined / 1e6:>8.1f} MFLOP")
    print(f"    (the embedding is a lookup, not a multiply; attention scores add 12 x L x T x D)")

    print("\n  6c. Training throughput, 20 timed steps after 5 warm-up steps:")
    configs = [("micro 8 x accum 4 (our loop)", 8, 4, False),
               ("micro 32 x accum 1", 32, 1, False),
               ("micro 64 x accum 1", 64, 1, False),
               ("micro 32 + torch.compile", 32, 1, True)]
    if args.quick:
        configs = configs[:2]
    print(f"    {'configuration':<30}{'tokens/s':>10}{'MFU 6N':>9}{'MFU refined':>13}"
          + (f"{'MFU vs spec':>13}" if spec else "") + f"{'peak MiB':>10}")
    results = []
    for label, micro, accum, comp in configs:
        try:
            tok_s, mem = time_config(cfg, pad, eos, micro, accum, comp)
        except Exception as e:                                      # noqa: BLE001
            msg = str(e).strip().splitlines()[0][:70] if str(e).strip() else e.__class__.__name__
            print(f"    {label:<30} failed: {msg}")
            torch.cuda.empty_cache()
            continue
        mfu, mfu_r = f_notes * tok_s / (peak * 1e12), f_refined * tok_s / (peak * 1e12)
        line = f"    {label:<30}{tok_s:>10,.0f}{mfu:>9.1%}{mfu_r:>13.1%}"
        if spec:
            line += f"{f_notes * tok_s / (spec * 1e12):>13.1%}"
        print(line + f"{mem:>10,.0f}")
        results.append({"config": label, "tokens_per_s": tok_s, "mfu_6N": mfu, "mfu_refined": mfu_r,
                        "peak_mib": mem})
        torch.cuda.empty_cache()

    print("\n  6d. Where one step's time goes (micro 32, sync after each phase):")
    ph = phase_breakdown(cfg, pad, eos)
    tot = sum(ph.values())
    for k, v in ph.items():
        print(f"    {k:<26}{v:>8.1f} ms {v / tot:>7.1%}")
    print(f"    {'total':<26}{tot:>8.1f} ms")
    print("    (syncing after every phase adds a little time; the shares are what matter)")

    best = max(results, key=lambda r: r["mfu_6N"]) if results else None
    if best:
        print(f"\n  best: {best['config']} at {best['mfu_6N']:.1%} MFU (6N, measured peak). "
              f"Distance to 40%: {0.40 - best['mfu_6N']:+.1%}")
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    (RUN_DIR / "mfu.json").write_text(json.dumps(
        {"gpu": props.name, "measured_peak_tflops": peak, "spec_tflops": spec,
         "flops_per_token_6N": f_notes, "flops_per_token_refined": f_refined,
         "configs": results, "phases_ms": ph}, indent=2))
    print(f"  saved {RUN_DIR / 'mfu.json'}")


# =========================================================================== main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--section", choices=["all", "checks", "train", "mfu"], default="all")
    p.add_argument("--steps", type=int, default=600, help="steps per accumulation run")
    p.add_argument("--range-steps", type=int, default=300, help="steps in the lr range test")
    p.add_argument("--spec-tflops", type=float, default=None, help="your GPU's spec-sheet dense bf16 TFLOP/s")
    p.add_argument("--quick", action="store_true", help="short smoke test")
    args = p.parse_args()
    if args.quick:
        args.steps, args.range_steps = 60, 60

    device, cfg, pad, eos = setup()
    print(f"device {device}, model D={cfg.d_model} L={cfg.n_layers} H={cfg.n_heads} T={cfg.context_length} "
          f"V={cfg.vocab_size}, <pad>={pad} <eos>={eos}")
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if args.section in ("all", "checks"):
        section_shapes(device, cfg, pad, eos)
        section_gradcheck(device, cfg, pad, eos)
        section_formats()
    if args.section in ("all", "train"):
        runs = section_accum(device, cfg, pad, eos, args)
        section_gradnorm(device, cfg, pad, eos, args, runs)
    if args.section in ("all", "mfu"):
        section_mfu(device, cfg, pad, eos, args)
    print(f"\nSession 10 done in {(time.time() - t0) / 60:.1f} min.")


if __name__ == "__main__":
    main()
