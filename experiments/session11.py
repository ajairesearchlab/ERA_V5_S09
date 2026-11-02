"""
Session 11 assignment: optimizers and learning-rate schedules, measured on our own model.

Run from the mini_llm folder:
    uv run python -m experiments.session11 --quick              # smoke test, ~3-4 min
    uv run python -m experiments.session11                      # everything (~2-2.5 h, mostly the sweep)
    uv run python -m experiments.session11 --section adam       # Task 1  (~1 min)
    uv run python -m experiments.session11 --section bias       # Task 2  (~5 min)
    uv run python -m experiments.session11 --section ratio      # Task 3  (~6 min)
    uv run python -m experiments.session11 --section schedule   # Task 4  (~5 min)
    uv run python -m experiments.session11 --section sweep      # Task 5  (~1.5-2 h, resumable)

Sections
    1. adam       Adam by hand for one weight, checked against PyTorch; our AdamW checked on the model
    2. bias       bias correction on/off: 20 steps by hand, then the real model, beta2 0.95 and 0.999
    3. ratio      update-to-weight ratio for every layer, warmup vs no warmup
    4. schedule   cosine vs WSD (both planned for 300, compared at 200) + cosine planned for 200
    5. sweep      learning rate at widths 256, 384, 512, 1024; minima; estimate for 4096

Outputs go to runs/session11/.
"""
import argparse
import csv
import dataclasses
import json
import math
import random
import time

import torch

from config import PROJECT_ROOT, TrainConfig
from data.dataset import TokenData
from experiments.session10 import banner, ema, fresh_model, get_plt, setup, val_set
from model.model import MiniLLM
from train.adam_by_hand import (bias_factor, first_sustained, hand_adam, loglog_fit,
                                parabola_vertex, rolling_median, steps_until_small)
from train.loop import (accumulate, build_optimizer, make_micro_batches, print_log_header,
                        run_training)
from train.optim import AdamW, UpdateRatio, constant, cosine, param_groups, param_kind, wsd

RUN_DIR = PROJECT_ROOT / "runs" / "session11"
GRADS = (0.50, 0.40, 0.60, 0.45, 0.55)              # the notes' five gradients (Section 6)
LR_TOY = 1e-3


def width_cfg(base, D):
    """Same depth and head size (64), FFN ~ 8/3 D rounded up to a multiple of 64."""
    return dataclasses.replace(base, d_model=D, n_heads=D // 64,
                               ffn_hidden=64 * math.ceil(8 * D / 3 / 64))


def n_params(cfg):
    return sum(p.numel() for p in MiniLLM(cfg).parameters())


def free(device):
    if device == "cuda":
        torch.cuda.empty_cache()


def fmt_step(i):
    """0-based index -> 1-based step label, or 'never'."""
    return str(i + 1) if i is not None else "never"


# =========================================================================== 1. Adam by hand
def torch_adam_single(grads, w0, lr, b1, b2, eps, wd, dtype, opt_cls=torch.optim.AdamW, **kw):
    """One weight through PyTorch's optimiser. m_hat, v_hat are computed from PyTorch's
    stored m and v (PyTorch never stores the corrected values)."""
    w = torch.tensor([w0], dtype=dtype, requires_grad=True)
    opt = opt_cls([w], lr=lr, betas=(b1, b2), eps=eps, weight_decay=wd, **kw)
    rows = []
    for t, g in enumerate(grads, start=1):
        before = w.detach().clone()
        w.grad = torch.tensor([g], dtype=dtype)
        opt.step()
        st = opt.state[w]
        m, v = st["exp_avg"].item(), st["exp_avg_sq"].item()
        rows.append({"t": t, "g": g, "m": m, "v": v, "m_hat": m / (1 - b1 ** t),
                     "v_hat": v / (1 - b2 ** t), "step": (w.detach() - before).item(),
                     "w": w.item()})
    return rows


COLS = ("m", "v", "m_hat", "v_hat", "step", "w")


def compare(hand, other):
    """Largest absolute and relative difference per column."""
    out = {}
    for c in COLS:
        a = max(abs(h[c] - o[c]) for h, o in zip(hand, other))
        r = max(abs(h[c] - o[c]) / max(abs(h[c]), 1e-30) for h, o in zip(hand, other))
        out[c] = (a, r)
    return out


def print_compare(label, diffs):
    print(f"  {label}")
    print(f"    {'':<8}" + "".join(f"{c:>12}" for c in COLS))
    print(f"    {'max abs':<8}" + "".join(f"{diffs[c][0]:>12.1e}" for c in COLS))
    print(f"    {'max rel':<8}" + "".join(f"{diffs[c][1]:>12.1e}" for c in COLS))
    worst = max(d[1] for d in diffs.values())
    digits = -math.log10(worst) if worst > 0 else float("inf")
    print(f"    -> agree to about {digits:.0f} significant digits" if math.isfinite(digits)
          else "    -> identical")


def section_adam(device, cfg, pad, eos, args):
    banner("1. ADAM BY HAND, CHECKED AGAINST PYTORCH (Task 1)")
    b1, b2, eps = 0.9, 0.999, 1e-8
    print(f"  one weight, w0 = 1.0, gradients {list(GRADS)}, lr {LR_TOY}, betas ({b1}, {b2}), eps {eps}")
    print("  by hand: plain Python floats (fp64), train/adam_by_hand.py\n")
    hand = hand_adam(GRADS, 1.0, LR_TOY, b1, b2, eps)
    print(f"  {'t':>3}{'g':>7}{'m':>11}{'v':>12}{'m_hat':>10}{'v_hat':>10}{'step':>12}{'step/lr':>9}{'w':>11}")
    for r in hand:
        print(f"  {r['t']:>3}{r['g']:>7.2f}{r['m']:>11.6f}{r['v']:>12.8f}{r['m_hat']:>10.6f}"
              f"{r['v_hat']:>10.6f}{r['step']:>12.8f}{r['step'] / LR_TOY:>9.4f}{r['w']:>11.6f}")
    print("  m_hat / sqrt(v_hat) stays near 1, so every step stays near lr while g ranges 0.40-0.60.")

    print("\n  The same five steps through torch.optim.AdamW (weight_decay=0, so it is plain Adam):")
    t64 = torch_adam_single(GRADS, 1.0, LR_TOY, b1, b2, eps, 0.0, torch.float64)
    t32 = torch_adam_single(GRADS, 1.0, LR_TOY, b1, b2, eps, 0.0, torch.float32)
    print_compare("PyTorch fp64 vs hand", compare(hand, t64))
    print_compare("PyTorch fp32 vs hand", compare(hand, t32))
    print("    (fp32 keeps ~7 significant digits; 'step' is a difference of two numbers near 1.0,")
    print("     so in fp32 it keeps fewer digits than the weight itself)")

    print("\n  With decoupled weight decay 0.1 (AdamW proper):")
    hand_wd = hand_adam(GRADS, 1.0, LR_TOY, b1, b2, eps, wd=0.1)
    print_compare("PyTorch AdamW fp64 vs hand", compare(hand_wd, torch_adam_single(
        GRADS, 1.0, LR_TOY, b1, b2, eps, 0.1, torch.float64)))
    extra = hand_wd[0]["step"] - hand[0]["step"]
    print(f"    decay adds {extra:+.2e} to step 1 = -lr x wd x w = {-LR_TOY * 0.1 * 1.0:+.2e}")

    print("\n  Our AdamW class (train/optim.py), which Task 2 needs for its bias-correction switch:")
    ours = torch_adam_single(GRADS, 1.0, LR_TOY, b1, b2, eps, 0.1, torch.float64, opt_cls=AdamW)
    print_compare("ours fp64 vs hand (decay 0.1)", compare(hand_wd, ours))
    hand_off = hand_adam(GRADS, 1.0, LR_TOY, b1, b2, eps, bias_correction=False)
    ours_off = torch_adam_single(GRADS, 1.0, LR_TOY, b1, b2, eps, 0.0, torch.float64, opt_cls=AdamW,
                                 bias_correction=False)
    worst = max(abs(h["w"] - o["w"]) for h, o in zip(hand_off, ours_off))
    print(f"    ours vs hand with bias correction OFF: largest weight difference {worst:.1e}")

    model_check(device, cfg, pad, eos)


def model_check(device, cfg, pad, eos, steps=5):
    """Our AdamW vs torch's on the real model: same weights, same gradients each step."""
    print(f"\n  On the real model ({sum(p.numel() for p in MiniLLM(cfg).parameters()):,} weights), "
          f"{steps} steps: two identical copies, the gradient")
    print("  computed once and copied, so any difference comes from the optimisers alone.")
    tc = TrainConfig()
    a, b = fresh_model(cfg, device), fresh_model(cfg, device)
    opt_t = build_optimizer(a, tc)                      # torch.optim.AdamW, as in Session 10
    opt_o = AdamW(param_groups(b, tc.weight_decay), lr=tc.lr, betas=(tc.beta1, tc.beta2))
    gen = torch.Generator().manual_seed(0)
    train = TokenData("train")
    print(f"    {'step':>5}{'largest |update|':>18}{'largest |w_torch - w_ours|':>28}{'as share of update':>20}")
    for s in range(steps):
        micros = make_micro_batches(train, 1, tc.micro_batch, cfg.context_length, device, gen)
        accumulate(a, micros, pad, eos)
        for pa, pb in zip(a.parameters(), b.parameters()):
            pb.grad = pa.grad.detach().clone()
        before = [p.detach().clone() for p in a.parameters()]
        opt_t.step()
        opt_o.step()
        with torch.no_grad():
            upd = max((pa - q).abs().max().item() for pa, q in zip(a.parameters(), before))
            diff = max((pa - pb).abs().max().item() for pa, pb in zip(a.parameters(), b.parameters()))
        print(f"    {s + 1:>5}{upd:>18.3e}{diff:>28.3e}{diff / upd:>20.1e}")
        opt_t.zero_grad(set_to_none=True)
        opt_o.zero_grad(set_to_none=True)
    print("  Differences at fp32 rounding level: our AdamW is the same optimiser.")
    del a, b, opt_t, opt_o
    free(device)


# =========================================================================== 2. bias correction
def toy_gradients(n=20, seed=0):
    """The notes' five gradients, then more of the same kind: about 0.5, a little noise."""
    rng = random.Random(seed)
    return list(GRADS) + [round(rng.gauss(0.5, 0.05), 4) for _ in range(n - len(GRADS))]


def section_bias(device, cfg, pad, eos, args):
    banner("2. BIAS CORRECTION ON vs OFF (Task 2)")
    grads = toy_gradients(20)
    b1 = 0.9
    print("  2a. One weight, 20 gradients (the notes' five, then ~0.5 +/- 0.05), lr 1e-3.")
    print("      Step size in units of lr. The ratio OFF/ON at the same step is (1 - b1^t) / sqrt(1 - b2^t).\n")
    toy = {}
    for b2 in (0.999, 0.95):
        toy[b2] = (hand_adam(grads, 1.0, LR_TOY, b1, b2), hand_adam(grads, 1.0, LR_TOY, b1, b2,
                                                                     bias_correction=False))
    print(f"  {'':>4}{'':>8}{'---- beta2 = 0.999 ----':>33}{'---- beta2 = 0.95 ----':>33}")
    print(f"  {'t':>4}{'g':>8}" + f"{'ON':>10}{'OFF':>10}{'factor':>10}   " * 2)
    for i, g in enumerate(grads):
        line = f"  {i + 1:>4}{g:>8.4f}"
        for b2 in (0.999, 0.95):
            on, off = toy[b2]
            line += (f"{-on[i]['step'] / LR_TOY:>10.3f}{-off[i]['step'] / LR_TOY:>10.3f}"
                     f"{bias_factor(i + 1, b1, b2):>10.3f}   ")
        print(line)
    for b2 in (0.999, 0.95):
        on, off = toy[b2]
        print(f"  beta2 {b2}: after 20 steps w = {on[-1]['w']:.6f} (ON) vs {off[-1]['w']:.6f} (OFF); "
              f"OFF has moved {(1 - off[-1]['w']) / (1 - on[-1]['w']):.2f}x as far")

    print("\n  When does it stop mattering? The factor depends only on t, b1 and b2:")
    print(f"  {'beta2':>8}{'factor t=1':>12}{'peak':>8}{'at t':>6}{'within 10%':>12}{'within 1%':>11}")
    for b2 in (0.999, 0.95):
        fs = [bias_factor(t, b1, b2) for t in range(1, 2001)]
        pk = max(range(len(fs)), key=lambda i: fs[i])
        print(f"  {b2:>8}{fs[0]:>12.2f}{fs[pk]:>8.2f}{pk + 1:>6}"
              f"{steps_until_small(b1, b2, 0.10):>12,}{steps_until_small(b1, b2, 0.01):>11,}")
    print("  With beta2 0.999 the uncorrected v is pulled towards 0 for ~1/(1-0.999) = 1,000 steps,")
    print("  so the OFF step is too BIG (up to 6.6x) and the gap widens before it closes.")
    print("  With beta2 0.95, v catches up within ~20 steps and m dominates: the OFF step is first")
    print("  too SMALL (0.45x), overshoots slightly (1.10x near t=20), then fades by ~76 steps.")

    # ---- 2b. the real model
    steps = args.bias_steps
    print(f"\n  2b. The real model, {steps} steps, constant lr {TrainConfig().lr:g} and NO warmup, so the")
    print("      correction is not hidden by a tiny early learning rate. Same weights, same batches.")
    tc = TrainConfig(steps=steps, eval_every=steps)
    vb = val_set(cfg, device, tc.eval_batches)
    train = TokenData("train")
    runs = {}
    for b2 in (0.95, 0.999):
        for bc in (True, False):
            key = (b2, bc)
            print(f"\n  RUN beta2 {b2}, bias correction {'ON' if bc else 'OFF'}")
            print_log_header()
            model = fresh_model(cfg, device)
            hook = UpdateRatio(model)
            opt = AdamW(param_groups(model, tc.weight_decay), lr=tc.lr, betas=(tc.beta1, b2),
                        bias_correction=bc)
            rows = run_training(f"b2={b2},bc={bc}", model, dataclasses.replace(tc, beta2=b2), train, vb,
                                pad, eos, device, lr_fn=constant(tc.lr), log_every=max(10, steps // 6),
                                opt=opt, step_hook=hook)
            runs[key] = {"rows": [r for r in rows if "loss" in r], "final_val": rows[-1].get("val_loss")
                         if rows[-1]["step"] == steps else float("nan"),
                         "ratio": hook.median_matrix()}
            del model, opt, hook
            free(device)

    show = [s for s in (0, 1, 2, 4, 9, 11, 19, 29, 49, 74, 99, 149, 199, 299, 499) if s < steps]
    summary = {}
    for b2 in (0.95, 0.999):
        on, off = runs[(b2, True)], runs[(b2, False)]
        n = min(len(on["rows"]), len(off["rows"]))
        lo_on = ema([r["loss"] for r in on["rows"][:n]], 0.1)
        lo_off = ema([r["loss"] for r in off["rows"][:n]], 0.1)
        gap = [b - a for a, b in zip(lo_on, lo_off)]
        settle_loss = first_sustained(gap, lambda x: abs(x) < 0.02)
        # step sizes: smoothed median update ratio of OFF vs ON, within 10%
        r_on, r_off = rolling_median(on["ratio"][:n], 5), rolling_median(off["ratio"][:n], 5)
        rel = [b / a - 1 for a, b in zip(r_on, r_off)]
        settle_step = first_sustained(rel, lambda x: abs(x) < 0.10)
        summary[b2] = {"step_size": fmt_step(settle_step), "loss": fmt_step(settle_loss),
                       "final_gap": gap[-1] if gap else float("nan")}
        print(f"\n  beta2 {b2}: loss (raw, same batch for both runs) and the median update-to-weight")
        print("  ratio over all weight matrices")
        print(f"  {'step':>6}{'loss ON':>10}{'loss OFF':>10}{'OFF-ON':>9}{'ratio ON':>11}{'ratio OFF':>11}"
              f"{'OFF/ON':>8}{'factor':>8}")
        for s in show:
            if s >= n:
                break
            a, b = on["rows"][s]["loss"], off["rows"][s]["loss"]
            ra, rb = on["ratio"][s], off["ratio"][s]
            print(f"  {s + 1:>6}{a:>10.4f}{b:>10.4f}{b - a:>+9.4f}{ra:>11.2e}{rb:>11.2e}"
                  f"{rb / ra:>8.2f}{bias_factor(s + 1, b1, b2):>8.2f}")
        print(f"  final validation loss: ON {on['final_val']:.4f}, OFF {off['final_val']:.4f} "
              f"({off['final_val'] - on['final_val']:+.4f})")
        print(f"  step sizes: OFF/ON update ratio stays within 10% from step {fmt_step(settle_step)}")
        print(f"  loss: smoothed training-loss gap stays under 0.02 from step {fmt_step(settle_loss)}"
              f" (gap at the end {summary[b2]['final_gap']:+.4f})")
    print("\n  'factor' is the size difference the formula predicts for the same m and v. OFF/ON")
    print("  also includes the two runs having drifted apart, so it matches only early on.")
    print("  Two different answers to 'when does it stop mattering': the step SIZES converge when")
    print("  the factor reaches 1, but any distance gained or lost before then stays in the weights,")
    print("  so the LOSS gap can persist long after the steps themselves agree.")

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    with open(RUN_DIR / "bias_model.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["beta2", "bias_correction", "step", "loss", "grad_norm", "median_update_ratio"])
        for (b2, bc), r in runs.items():
            for row, ratio in zip(r["rows"], r["ratio"]):
                w.writerow([b2, bc, row["step"], row["loss"], row["grad_norm"], ratio])
    plot_bias(toy, runs, b1)
    return summary


def plot_bias(toy, runs, b1):
    plt = get_plt()
    if not plt:
        return
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    colours = {0.999: "tab:red", 0.95: "tab:blue"}
    ax = axes[0]
    for b2, (on, off) in toy.items():
        t = [r["t"] for r in on]
        ax.plot(t, [-r["step"] / LR_TOY for r in on], "o-", color=colours[b2], ms=3,
                label=f"beta2 {b2}, ON")
        ax.plot(t, [-r["step"] / LR_TOY for r in off], "s--", color=colours[b2], ms=3,
                label=f"beta2 {b2}, OFF")
    ax.set(xlabel="step", ylabel="step size / lr", title="One weight, first 20 steps",
           xticks=range(0, 21, 2))
    ax = axes[1]
    ts = list(range(1, 10001))
    for b2 in (0.999, 0.95):
        ax.plot(ts, [bias_factor(t, b1, b2) for t in ts], color=colours[b2], label=f"beta2 {b2}")
    ax.axhspan(0.99, 1.01, color="grey", alpha=0.3, label="within 1%")
    ax.set_xscale("log")
    ax.set(xlabel="step (log)", ylabel="OFF step / ON step", title="When the difference fades")
    ax = axes[2]
    for (b2, bc), r in runs.items():
        st = [x["step"] for x in r["rows"]]
        ax.plot(st, ema([x["loss"] for x in r["rows"]], 0.1), "-" if bc else "--",
                color=colours[b2], label=f"beta2 {b2}, {'ON' if bc else 'OFF'}")
    ax.set(xlabel="step", ylabel="training loss (EMA)", title="Real model, constant lr, no warmup")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RUN_DIR / "bias_correction.png", dpi=120)
    plt.close(fig)
    print(f"\n  plot saved: {RUN_DIR / 'bias_correction.png'}")


# =========================================================================== 3. update ratio
KINDS = ("tok_emb", "q_proj", "k_proj", "v_proj", "o_proj", "gate", "up", "down",
         "attn_norm", "ffn_norm", "final_norm", "lm_head")
MATRIX_KINDS = ("tok_emb", "q_proj", "k_proj", "v_proj", "o_proj", "gate", "up", "down", "lm_head")
BLOCK_KINDS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate", "up", "down", "attn_norm", "ffn_norm")


def converge_step(r_w, r_nw, tol=0.10, k=11):
    """First step from which the warmup run's ratio stays within tol of the no-warmup run's
    (both smoothed with a k-step rolling median). 0-based index or None."""
    a, b = rolling_median(r_w, k), rolling_median(r_nw, k)
    return first_sustained([x / y - 1 for x, y in zip(a, b)], lambda d: abs(d) < tol)


def section_ratio(device, cfg, pad, eos, args):
    banner("3. UPDATE-TO-WEIGHT RATIO FOR EVERY LAYER: when does warmup stop mattering? (Task 3)")
    steps = args.ratio_steps
    base = TrainConfig(steps=steps, eval_every=max(50, steps // 6))
    W = base.warmup if not args.quick else 20
    sched_w = cosine(steps, base.lr, W, base.min_lr_frac)
    # the same schedule, except full lr from step 0: the two runs differ ONLY in the warmup
    sched_nw = lambda s: base.lr if s < W else sched_w(s)              # noqa: E731
    print(f"  {steps} steps each, lr peak {base.lr:g}. Run A warms up over {W} steps; run B starts at the")
    print(f"  peak. After step {W} both follow the identical cosine. Same weights, same batches.")
    print("  ratio = ||w_after - w_before|| / ||w_before|| per parameter tensor, every step\n")
    vb = val_set(cfg, device, base.eval_batches)
    train = TokenData("train")
    hooks, finals = {}, {}
    for name, fn in (("warmup", sched_w), ("no_warmup", sched_nw)):
        print(f"  RUN {name}")
        print_log_header()
        model = fresh_model(cfg, device)
        hook = UpdateRatio(model)
        rows = run_training(name, model, base, train, vb, pad, eos, device, lr_fn=fn,
                            log_every=max(25, steps // 12), step_hook=hook)
        hooks[name], finals[name] = hook, rows[-1].get("val_loss", float("nan"))
        RUN_DIR.mkdir(parents=True, exist_ok=True)
        hook.save(RUN_DIR / f"ratio_{name}.csv")
        del model
        free(device)

    hw, hn = hooks["warmup"], hooks["no_warmup"]
    n = min(len(hw.rows), len(hn.rows))
    kw, kn = hw.by_kind(), hn.by_kind()
    pts = sorted({s for s in (0, 9, W - 1, 2 * W - 1, n // 2, n - 1) if 0 <= s < n})

    print("  3a. Median ratio over the 8 layers, by kind of weight (warmup | no warmup):")
    print(f"  {'kind':<11}" + "".join(f"{'step ' + str(s + 1):>17}" for s in pts))
    for k in KINDS:
        a, b = kw[k][:n], kn[k][:n]
        print(f"  {k:<11}" + "".join(f"{a[s]:>8.1e}|{b[s]:<8.1e}" for s in pts))
    print("  The notes' target for a healthy run is about 1e-3.")

    print(f"\n  3b. By kind of weight (median over layers, 11-step rolling median): the step from which")
    print("      the warmup run's ratio stays within 10% of the no-warmup run's, and where it peaks.")
    print(f"  {'kind':<11}{'converges':>11}{'warmup peak':>13}{'peak ratio':>12}{'no-warmup step 1':>18}")
    kind_conv, peak_at = {}, {}
    for k in KINDS:
        a, b = kw[k][:n], kn[k][:n]
        kind_conv[k] = converge_step(a, b)
        sm = rolling_median(a, 11)
        peak_at[k] = max(range(n), key=lambda i: sm[i])
        print(f"  {k:<11}{fmt_step(kind_conv[k]):>11}{'step ' + str(peak_at[k] + 1):>13}"
              f"{sm[peak_at[k]]:>12.1e}{b[0]:>18.1e}")
    mats = sorted(v for k, v in kind_conv.items() if k in MATRIX_KINDS and v is not None)
    if mats:
        print(f"\n  -> weight matrices: warmup stops changing the ratio at step {mats[len(mats) // 2] + 1} "
              f"(median over kinds), latest {mats[-1] + 1}; warmup itself ends at step {W}")
    never = [k for k in MATRIX_KINDS if kind_conv[k] is None]
    if never:
        print(f"     never within 10% in this run: {', '.join(never)}")

    print("\n  3c. The same test for every individual tensor (noisier than 3b):")
    grid = {}
    for nm in hw.names:
        grid[param_kind(nm)] = converge_step(hw.series(nm)[:n], hn.series(nm)[:n])
    print(f"  {'layer':<7}" + "".join(f"{k:>11}" for k in BLOCK_KINDS))
    for l in sorted({l for l, _ in grid if l is not None}):
        print(f"  {l:<7}" + "".join(f"{fmt_step(grid[(l, k)]):>11}" for k in BLOCK_KINDS))
    for k in ("tok_emb", "lm_head", "final_norm"):
        print(f"  {k:<11}{fmt_step(grid[(None, k)]):>7}")
    nc = sum(v is None for v in grid.values())
    print(f"  {len(grid) - nc} of {len(grid)} tensors converge within the run")
    print(f"\n  final validation loss: warmup {finals['warmup']:.4f}, no warmup {finals['no_warmup']:.4f}")

    summary = {"warmup_steps": W,
               "converge_step_by_kind": {k: (v + 1 if v is not None else None) for k, v in kind_conv.items()},
               "converge_step_by_tensor": {f"{l}.{k}": (v + 1 if v is not None else None)
                                           for (l, k), v in grid.items()},
               "peak_step_by_kind": {k: v + 1 for k, v in peak_at.items()},
               "final_val": finals}
    (RUN_DIR / "ratio_summary.json").write_text(json.dumps(summary, indent=2))
    plot_ratio(kw, kn, n, W, kind_conv)
    return summary


def plot_ratio(kw, kn, n, W, kind_conv):
    plt = get_plt()
    if not plt:
        return
    show = ("q_proj", "o_proj", "up", "down", "tok_emb", "lm_head")
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
    for ax, k in zip(axes.flat, show):
        st = list(range(1, n + 1))
        ax.plot(st, kw[k][:n], color="tab:blue", lw=0.9, label="warmup")
        ax.plot(st, kn[k][:n], color="tab:red", lw=0.9, alpha=0.8, label="no warmup")
        ax.axhline(1e-3, color="grey", ls=":", label="1e-3 target")
        ax.axvline(W, color="tab:blue", ls=":", alpha=0.6, label=f"warmup ends ({W})")
        if kind_conv.get(k) is not None:
            c = kind_conv[k] + 1
            ax.axvline(c, color="black", ls="--", label=f"within 10% from {c}")
        ax.set_yscale("log")
        ax.set(title=f"{k} (median over layers)" if k not in ("tok_emb", "lm_head") else k)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    for ax in axes[1]:
        ax.set_xlabel("step")
    for ax in axes[:, 0]:
        ax.set_ylabel("update / weight")
    fig.tight_layout()
    fig.savefig(RUN_DIR / "update_ratio.png", dpi=120)
    plt.close(fig)
    print(f"\n  plot saved: {RUN_DIR / 'update_ratio.png'}")


# =========================================================================== 4. schedules
def section_schedule(device, cfg, pad, eos, args):
    total, stop = args.sched_total, args.sched_stop
    banner(f"4. COSINE vs WSD, BOTH PLANNED FOR {total} STEPS, COMPARED AT {stop} (Task 4)")
    peak = TrainConfig().lr
    warm = max(1, round(0.1 * total))
    tc = TrainConfig(steps=total, eval_every=max(5, total // 12), eval_batches=20)
    cn, wn = f"cosine_{total}", f"wsd_{total}"
    plans = [
        (cn, total, cosine(total, peak, warm)),
        (wn, total, wsd(total, peak, warm, decay_frac=0.1)),
        (f"cosine_{stop}", stop, cosine(stop, peak, warm)),     # planned for the stopping point
    ]
    print(f"  peak lr {peak:g}, warmup {warm} steps, both decay to 10% of peak.")
    print(f"  {cn}: cosine over the whole run.  {wn}: flat at peak, linear decay over the")
    print(f"  last {total - round(total * 0.9)} steps.  cosine_{stop}: a run that KNEW it would stop at {stop}.")
    print(f"  Validation on {tc.eval_batches} fixed batches every {tc.eval_every} steps; step {stop} is one of them.")
    vb = val_set(cfg, device, tc.eval_batches)
    train = TokenData("train")
    res = {}
    for name, steps, fn in plans:
        print(f"\n  RUN {name}  (lr at step {stop}: {fn(stop - 1):.2e})")
        print_log_header()
        model = fresh_model(cfg, device)
        rows = run_training(name, model, dataclasses.replace(tc, steps=steps), train, vb, pad, eos,
                            device, lr_fn=fn, log_every=max(25, total // 12))
        res[name] = {"rows": rows, "lr": [fn(s) for s in range(steps)]}
        del model
        free(device)

    def val_at(name, s):
        for r in res[name]["rows"]:
            if r["step"] == s and r.get("val_loss") == r.get("val_loss"):
                return r["val_loss"]
        return float("nan")

    def train_ema_at(name, s):
        lo = [r["loss"] for r in res[name]["rows"] if "loss" in r][:s]
        return ema(lo, 0.1)[-1] if lo else float("nan")

    print(f"\n  {'run':<14}{'lr at ' + str(stop):>12}{'val @' + str(stop):>11}{'train EMA @' + str(stop):>16}"
          f"{'val @' + str(total):>11}")
    for name, steps, fn in plans:
        v_end = f"{val_at(name, total):>11.4f}" if steps == total else f"{'-':>11}"
        print(f"  {name:<14}{fn(stop - 1):>12.2e}{val_at(name, stop):>11.4f}{train_ema_at(name, stop):>16.4f}{v_end}")
    c, w_, c2 = val_at(cn, stop), val_at(wn, stop), val_at(f"cosine_{stop}", stop)
    print(f"\n  at step {stop}: WSD - cosine = {w_ - c:+.4f}   cosine planned for {stop} - cosine planned "
          f"for {total} = {c2 - c:+.4f}")
    print(f"  at step {total}: WSD - cosine = {val_at(wn, total) - val_at(cn, total):+.4f}")
    print(f"  Step-{stop} numbers compare models that are still mid-schedule: cosine has already decayed")
    print(f"  to ~{plans[0][2](stop - 1) / peak:.0%} of peak, WSD has not started its decay. "
          "Which one to keep is in the write-up.")

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    with open(RUN_DIR / "schedules.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "step", "lr", "loss", "val_loss"])
        for name in res:
            for r in res[name]["rows"]:
                w.writerow([name, r["step"], r.get("lr", ""), r.get("loss", ""), r.get("val_loss", "")])
    plot_schedules(res, stop, total, cn, wn)
    return {"val_at_stop": {n: val_at(n, stop) for n in res},
            "val_at_total": {n: val_at(n, total) for n in res}}


def plot_schedules(res, stop, total, cn, wn):
    plt = get_plt()
    if not plt:
        return
    colours = {cn: "tab:blue", wn: "tab:orange"}
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(17, 4.6))
    for name, r in res.items():
        col = colours.get(name, "tab:green")
        a.plot(range(len(r["lr"])), r["lr"], color=col, label=name)
        tr = [x for x in r["rows"] if "loss" in x]
        b.plot([x["step"] for x in tr], ema([x["loss"] for x in tr], 0.1), color=col, label=name)
        v = [x for x in r["rows"] if x.get("val_loss") == x.get("val_loss") and x["step"] > 0]
        c.plot([x["step"] for x in v], [x["val_loss"] for x in v], "o-", color=col, ms=3, label=name)
    a.set(xlabel="step", ylabel="learning rate", title="Schedules")
    b.set(xlabel="step", ylabel="training loss (EMA)", title="Training loss")
    c.set(xlabel="step", ylabel="validation loss", title="Validation loss")
    vals = [x["val_loss"] for r in res.values() for x in r["rows"]
            if x.get("val_loss") == x.get("val_loss") and x["step"] >= total // 3]
    if vals:
        c.set_ylim(min(vals) - 0.05, max(vals) + 0.05)
    for ax in (a, b, c):
        ax.axvline(stop, color="black", ls="--", alpha=0.6, label=f"stop at {stop}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RUN_DIR / "schedules.png", dpi=120)
    plt.close(fig)
    print(f"\n  plot saved: {RUN_DIR / 'schedules.png'}")


# =========================================================================== 5. lr sweep
SWEEP_FIELDS = ["width", "lr", "params", "steps", "final_val", "final_train_ema", "diverged", "minutes"]
NOTES_TABLE = {256: 3.0e-3, 512: 1.5e-3, 1024: 7.5e-4, 2048: 3.8e-4, 4096: 1.9e-4}   # notes, Section 12


def load_sweep(path):
    done = {}
    if path.exists():
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                done[(int(r["width"]), float(r["lr"]))] = {
                    "width": int(r["width"]), "lr": float(r["lr"]), "params": int(r["params"]),
                    "steps": int(r["steps"]), "final_val": float(r["final_val"]),
                    "final_train_ema": float(r["final_train_ema"]),
                    "diverged": r["diverged"] == "True", "minutes": float(r["minutes"])}
    return done


def fit_minimum(points):
    """points: [(lr, val)] sorted by lr, equally spaced in log2. Parabola through the best
    grid point and its neighbours. Returns (lr*, best grid lr, at_edge)."""
    ok = [(lr, v) for lr, v in points if math.isfinite(v)]
    i = min(range(len(points)), key=lambda j: points[j][1] if math.isfinite(points[j][1]) else 1e9)
    best_lr = points[i][0]
    if i == 0 or i == len(points) - 1 or len(ok) < 3:
        return best_lr, best_lr, True
    nb = points[i - 1:i + 2]
    if not all(math.isfinite(v) for _, v in nb):
        return best_lr, best_lr, False
    x = parabola_vertex([math.log2(lr) for lr, _ in nb], [v for _, v in nb])
    return 2 ** x, best_lr, False


def section_sweep(device, base_cfg, pad, eos, args):
    banner("5. LEARNING-RATE SWEEP ACROSS WIDTHS, AND AN ESTIMATE FOR 4,096 (Task 5)")
    widths, lrs, steps = args.widths, args.lrs, args.sweep_steps
    tc = TrainConfig(steps=steps, eval_every=steps, eval_batches=20,
                     micro_batch=16, accum_steps=2,                 # same 32 sequences, faster
                     warmup=max(1, steps // 6))
    print(f"  widths {widths}, 8 layers, head_dim 64, FFN ~8/3 D. Learning rates {[f'{x:g}' for x in lrs]}")
    print(f"  {steps} steps each = {steps * 32 * base_cfg.context_length / 1e6:.1f}M tokens, warmup {tc.warmup}, "
          f"cosine to 10%, AdamW (0.9, 0.95), wd 0.1, clip 1.0, same seed and batches for every run.")
    print("  Score = validation loss on 20 fixed batches at the end. Init is std 0.02 at every width")
    print("  (standard parameterisation, not muP), so the best lr is expected to move with width.\n")
    for D in widths:
        c = width_cfg(base_cfg, D)
        print(f"    width {D:>5}: heads {c.n_heads:>2}, FFN {c.ffn_hidden:>5}, {n_params(c) / 1e6:>6.1f}M parameters")

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    path = RUN_DIR / ("sweep_quick.csv" if args.quick else "sweep.csv")
    if args.fresh_sweep and path.exists():
        path.unlink()
    done = load_sweep(path)
    if done:
        print(f"\n  resuming: {len(done)} runs already in {path.name} are reused (--fresh-sweep to redo)")
    vb = val_set(base_cfg, device, tc.eval_batches)
    train = TokenData("train")
    new_file = not path.exists()
    f = open(path, "a", newline="")
    w = csv.DictWriter(f, fieldnames=SWEEP_FIELDS)
    if new_file:
        w.writeheader()
    t_all = time.time()
    with f:
        for D in widths:
            cfg = width_cfg(base_cfg, D)
            todo = [lr for lr in lrs if (D, lr) not in done]
            for j, lr in enumerate(todo):
                t0 = time.time()
                model = fresh_model(cfg, device)
                rows = run_training(f"w{D}_lr{lr:g}", model, dataclasses.replace(tc, lr=lr), train, vb,
                                    pad, eos, device, log_every=10 ** 9)
                tr = [r["loss"] for r in rows if "loss" in r]
                finished = rows[-1]["step"] == steps and math.isfinite(rows[-1].get("val_loss", float("nan")))
                rec = {"width": D, "lr": lr, "params": sum(p.numel() for p in model.parameters()),
                       "steps": steps, "final_val": rows[-1]["val_loss"] if finished else float("nan"),
                       "final_train_ema": ema(tr, 0.1)[-1] if tr else float("nan"),
                       "diverged": not finished, "minutes": (time.time() - t0) / 60}
                done[(D, lr)] = rec
                w.writerow(rec)
                f.flush()
                left = len(todo) - j - 1
                print(f"    width {D:>5}  lr {lr:>9.2e}  val {rec['final_val']:>8.4f}"
                      f"{'  DIVERGED' if rec['diverged'] else ''}   {rec['minutes']:.1f} min "
                      f"(total {(time.time() - t_all) / 60:.0f} min"
                      + (f", ~{left * rec['minutes']:.0f} min left at this width)" if left else ")"))
                del model
                free(device)

    # ---- table and minima
    print(f"\n  validation loss after {steps} steps (* = best on the grid)")
    print(f"  {'lr':>10}" + "".join(f"{'w=' + str(D):>11}" for D in widths))
    best = {D: min((done[(D, lr)]["final_val"] for lr in lrs if math.isfinite(done[(D, lr)]["final_val"])),
                   default=float("nan")) for D in widths}
    for lr in lrs:
        line = f"  {lr:>10.2e}"
        for D in widths:
            v = done[(D, lr)]["final_val"]
            line += f"{'diverged':>11}" if not math.isfinite(v) else f"{v:>10.4f}{'*' if v == best[D] else ' '}"
        print(line)

    minima = {}
    print(f"\n  {'width':>7}{'best grid lr':>14}{'fitted lr*':>12}{'loss at best':>14}{'notes (SP)':>12}")
    for D in widths:
        pts = [(lr, done[(D, lr)]["final_val"]) for lr in lrs]
        lr_star, grid_lr, edge = fit_minimum(pts)
        minima[D] = (lr_star, edge)
        nt = f"{NOTES_TABLE[D]:.1e}" if D in NOTES_TABLE else "-"
        print(f"  {D:>7}{grid_lr:>14.2e}{lr_star:>12.2e}{best[D]:>14.4f}{nt:>12}"
              f"{'   AT GRID EDGE: widen the range' if edge else ''}")

    # ---- extrapolation to 4096
    target = 4096
    ws = [D for D in widths if math.isfinite(minima[D][0])]
    est = {}
    if len(ws) >= 2:
        a, b, r2 = loglog_fit(ws, [minima[D][0] for D in ws])
        est["log-log fit through all widths"] = math.exp(a) * target ** b
        big = max(ws)
        est[f"1/width from width {big}"] = minima[big][0] * big / target
        if len(ws) >= 3:
            top2 = sorted(ws)[-2:]
            _, b2_, _ = loglog_fit(top2, [minima[D][0] for D in top2])
            est[f"slope of the two widest ({top2[0]}, {top2[1]})"] = minima[top2[1]][0] * (target / top2[1]) ** b2_
        r2_txt = f"R^2 {r2:.3f}" if math.isfinite(r2) else "R^2 n/a: every minimum is the same value"
        print(f"\n  fit: lr* ~ width^{b:.2f}  ({r2_txt}; the notes' standard-parameterisation rule is width^-1)")
        print(f"\n  estimates for width {target}:")
        for k, v in est.items():
            print(f"    {k:<42}{v:.2e}")
        lo, hi = min(est.values()), max(est.values())
        geo = math.exp(sum(math.log(v) for v in est.values()) / len(est))
        print(f"    {'notes, Section 12':<42}{NOTES_TABLE[target]:.2e}")
        print(f"\n  -> use about {geo:.1e} at width {target} (estimates span {lo:.1e} to {hi:.1e}, "
              f"a factor of {hi / lo:.1f})")
        edges = [D for D in widths if minima[D][1]]
        issues = []
        if edges:
            issues.append(f"minimum at the grid edge for width(s) {edges}")
        if not math.isfinite(r2):
            issues.append("all minima are equal, so the fit has no slope information")
        elif r2 < 0.9:
            issues.append(f"the points are not on a clean line (R^2 {r2:.2f})")
        if hi / lo > 2:
            issues.append("the estimates disagree by more than 2x")
        level = "low" if issues else "moderate"
        print(f"  confidence: {level}. " + ("; ".join(issues) + ". " if issues else "")
              + f"Always: a {target / max(ws):.0f}x extrapolation beyond the widest point, one seed, "
              f"{steps}-step runs (short runs favour higher lr), and loss curves that are flat near the minimum.")
        (RUN_DIR / ("sweep_summary_quick.json" if args.quick else "sweep_summary.json")).write_text(json.dumps(
            {"minima": {D: {"lr_star": minima[D][0], "at_edge": minima[D][1]} for D in widths},
             "slope": b, "r2": r2 if math.isfinite(r2) else None,
             "estimates_4096": est, "recommended_4096": geo}, indent=2))
    plot_sweep(done, widths, lrs, minima, est, target, args.quick)


def plot_sweep(done, widths, lrs, minima, est, target, quick=False):
    plt = get_plt()
    if not plt:
        return
    fig, (a, b) = plt.subplots(1, 2, figsize=(15, 5))
    cmap = plt.get_cmap("viridis")
    for i, D in enumerate(widths):
        col = cmap(i / max(1, len(widths) - 1))
        xs = [lr for lr in lrs if math.isfinite(done[(D, lr)]["final_val"])]
        ys = [done[(D, lr)]["final_val"] for lr in xs]
        a.plot(xs, ys, "o-", color=col, label=f"width {D}")
        lr_star = minima[D][0]
        pts = [(lr, done[(D, lr)]["final_val"]) for lr in lrs]
        near = min(pts, key=lambda p: abs(math.log(p[0] / lr_star)))
        a.plot([lr_star], [near[1]], marker="v", ms=12, color=col, markeredgecolor="black")
        for lr in lrs:
            if not math.isfinite(done[(D, lr)]["final_val"]):
                a.plot([lr], [max(ys) if ys else 0], "x", color=col, ms=10)
    a.set_xscale("log")
    a.set(xlabel="learning rate (log)", ylabel="validation loss", title="Loss against learning rate (v = fitted minimum)")
    ws = [D for D in widths if math.isfinite(minima[D][0])]
    b.plot(ws, [minima[D][0] for D in ws], "o", ms=9, color="tab:blue", label="our fitted minima")
    if len(ws) >= 2:
        A, B, _ = loglog_fit(ws, [minima[D][0] for D in ws])
        xs = [min(ws), target]
        b.plot(xs, [math.exp(A) * x ** B for x in xs], "--", color="tab:blue", label=f"fit: width^{B:.2f}")
    nt = sorted(NOTES_TABLE)
    b.plot(nt, [NOTES_TABLE[x] for x in nt], "s:", color="grey", label="notes (SP, ~1/width)")
    for k, v in est.items():
        b.plot([target], [v], "*", ms=12, label=f"4096: {k}")
    b.set_xscale("log", base=2)
    b.set_yscale("log")
    b.set(xlabel="width (log2)", ylabel="best learning rate (log)", title="Best lr against width")
    for ax in (a, b):
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(RUN_DIR / ("lr_sweep_quick.png" if quick else "lr_sweep.png"), dpi=120)
    plt.close(fig)
    print(f"\n  plot saved: {RUN_DIR / ('lr_sweep_quick.png' if quick else 'lr_sweep.png')}")


# =========================================================================== main
SECTIONS = ("adam", "bias", "ratio", "schedule", "sweep")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--section", choices=("all",) + SECTIONS, default="all")
    p.add_argument("--bias-steps", type=int, default=300)
    p.add_argument("--ratio-steps", type=int, default=600)
    p.add_argument("--sched-total", type=int, default=300)
    p.add_argument("--sched-stop", type=int, default=200)
    p.add_argument("--widths", type=int, nargs="+", default=[256, 384, 512, 1024])
    p.add_argument("--lrs", type=float, nargs="+",
                   default=[2.5e-4, 5e-4, 1e-3, 2e-3, 4e-3, 8e-3, 1.6e-2])
    p.add_argument("--sweep-steps", type=int, default=600)
    p.add_argument("--fresh-sweep", action="store_true", help="ignore saved sweep results")
    p.add_argument("--quick", action="store_true", help="short smoke test")
    args = p.parse_args()
    if args.quick:
        args.bias_steps, args.ratio_steps = 40, 80
        args.sched_total, args.sched_stop = 60, 40
        args.widths, args.lrs, args.sweep_steps = [256, 384], [1e-3, 2e-3, 4e-3], 30

    device, cfg, pad, eos = setup()
    print(f"device {device}, model D={cfg.d_model} L={cfg.n_layers} H={cfg.n_heads} T={cfg.context_length} "
          f"V={cfg.vocab_size}, <pad>={pad} <eos>={eos}")
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    run = SECTIONS if args.section == "all" else (args.section,)
    fns = {"adam": section_adam, "bias": section_bias, "ratio": section_ratio,
           "schedule": section_schedule, "sweep": section_sweep}
    for s in run:
        t = time.time()
        fns[s](device, cfg, pad, eos, args)
        print(f"\n  [{s} took {(time.time() - t) / 60:.1f} min]")
    print(f"\nSession 11 done in {(time.time() - t0) / 60:.1f} min.")


if __name__ == "__main__":
    main()
