"""
Session 14 assignment: train a dense model, convert it into a mixture-of-experts, keep training.

Run from the mini_llm folder (once, first): uv run python -m data.extend_data      # 20M -> 60M tokens
    uv run python -m experiments.session14 --quick              # smoke test, a few minutes
    uv run python -m experiments.session14                      # everything (~1-1.5 h)
    uv run python -m experiments.session14 --section dense      # Task 1  dense, 2,000 steps -> checkpoint
    uv run python -m experiments.session14 --section convert    # Task 2  conversion checks, loss at step 0
    uv run python -m experiments.session14 --section continue   # Task 3  dense vs MoE r=0 vs MoE r=0.5
    uv run python -m experiments.session14 --section routing    # Task 4  load balance over training
    uv run python -m experiments.session14 --section nobalance  # Task 5a the same run with balancing off
    uv run python -m experiments.session14 --section special    # Task 5b what the experts learned

Every section that trains saves its result and is skipped next time (--fresh to redo).
Outputs: runs/session14/, checkpoints/s14_*.pt
"""
import argparse
import copy
import csv
import json
import math
import string
import sys
import time
from types import SimpleNamespace

import torch

from config import CHECKPOINT_DIR, PROJECT_ROOT, MoEConfig, TrainConfig
from data.dataset import TokenData, load_tokenizer
from experiments.session10 import banner, ema, fresh_model, get_plt, setup, val_set
from model.model import MiniLLM
from model.moe import moe_layers, param_counts, upcycle
from train.loop import autocast, evaluate, open_log, print_log_header, run_training
from train.moe_hook import MoEBalancer, read_stats
from train.optim import constant, wsd

RUN_DIR = PROJECT_ROOT / "runs" / "session14"
NAN = float("nan")


# =========================================================================== helpers
class Tee:
    """Print to the console and to a file at the same time (--log)."""
    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8")
        self.out = sys.stdout

    def write(self, s):
        self.out.write(s)
        self.f.write(s)

    def flush(self):
        self.out.flush()
        self.f.flush()


def free(device):
    if device == "cuda":
        torch.cuda.empty_cache()


def read_log(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            row = {k: (v if k == "run" else (float(v) if v not in ("", None) else NAN)) for k, v in r.items()}
            row["step"] = int(row["step"])
            rows.append(row)
    return rows


def train_rows(rows):
    return [r for r in rows if math.isfinite(r.get("loss", NAN))]


def val_points(rows):
    return [(r["step"], r["val_loss"]) for r in rows if math.isfinite(r.get("val_loss", NAN))]


def val_at(rows, step):
    for s, v in val_points(rows):
        if s == step:
            return v
    return NAN


def median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else NAN


def tc_for(a, steps, seed):
    """32 sequences x 256 tokens = 8,192 tokens per optimiser step, as in Sessions 10-11."""
    return TrainConfig(steps=steps, lr=a.lr, warmup=a.warmup, micro_batch=16, accum_steps=2,
                       eval_every=a.eval_every, eval_batches=a.eval_batches, seed=seed)


def ckpt(ctx, name):
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    return CHECKPOINT_DIR / f"s14_{name}{ctx.suf}.pt"


def out(ctx, name):
    stem, dot, ext = name.rpartition(".")
    return RUN_DIR / (f"{stem}{ctx.suf}.{ext}" if dot else f"{name}{ctx.suf}")
    
    #return RUN_DIR / f"{name}{ctx.suf}"


def load_dense(ctx):
    path = ckpt(ctx, "dense_stable")
    if not path.exists():
        raise SystemExit(f"{path.name} not found: run --section dense first")
    m = MiniLLM(ctx.cfg).to(ctx.device)
    m.load_state_dict(torch.load(path, map_location=ctx.device))
    return m


def load_moe(ctx, name, drop_r):
    """Rebuild the MoE structure from the dense checkpoint, then load the trained MoE weights."""
    path = ckpt(ctx, name)
    if not path.exists():
        raise SystemExit(f"{path.name} not found: run --section continue first")
    m = upcycle(load_dense(ctx), ctx.mcfg, drop_r=drop_r, seed=ctx.args.seed)
    m.load_state_dict(torch.load(path, map_location=ctx.device))
    return m


def variants(ctx):
    """(name, drop_r) of the two MoE runs: plain partition, and drop-upcycling."""
    return [("moe_r0", 0.0), (f"moe_r{ctx.mcfg.drop_r:g}", ctx.mcfg.drop_r)]


@torch.no_grad()
def route_counts(model, batches, device):
    """Eval-mode routing: per layer, how many routing slots each expert received. [L, E]"""
    mods = moe_layers(model)
    for m in mods:
        m.record = True
    model.eval()
    tot = torch.zeros(len(mods), mods[0].n_experts)
    for tokens in batches:
        with autocast(device):
            model(tokens)
        for l, m in enumerate(mods):
            tot[l] += torch.bincount(m.last_idx.reshape(-1), minlength=m.n_experts).float().cpu()
    for m in mods:
        m.record, m.last_idx = False, None
    model.train()
    return tot


def maxvio(counts):
    load = counts / counts.sum(-1, keepdim=True)
    even = 1.0 / counts.shape[-1]
    return (load.max(-1).values - even) / even


def save_summary(ctx):
    path = out(ctx, "summary.json")
    old = json.loads(path.read_text()) if path.exists() else {}
    old.update(ctx.summary)
    path.write_text(json.dumps(old, indent=2, default=float))


# =========================================================================== 1. dense
def section_dense(ctx):
    a = ctx.args
    banner("1. TRAIN THE DENSE MODEL (Task 1)")
    path, log = ckpt(ctx, "dense_stable"), out(ctx, "dense_phase1.csv")
    tc = tc_for(a, a.dense_steps, seed=0)
    print(f"  Session 11 model: {ctx.cfg.n_layers} layers, D={ctx.cfg.d_model}, SwiGLU {ctx.cfg.ffn_hidden}, "
          f"{sum(p.numel() for p in MiniLLM(ctx.cfg).parameters()):,} parameters")
    print(f"  {a.dense_steps} steps x 8,192 tokens = {a.dense_steps * 8192 / 1e6:.1f}M tokens, AdamW (0.9, 0.95), "
          f"warmup {a.warmup}, then lr flat at {a.lr:g}.")
    print("  No decay: this is the 'stable' part of WSD, so the checkpoint can keep training (Session 11, Task 4).")
    if path.exists() and log.exists() and not a.fresh:
        print(f"\n  reusing {path.name} and {log.name} (--fresh to retrain)")
        rows = read_log(log)
    else:
        model = fresh_model(ctx.cfg, ctx.device)
        f, w = open_log(log)
        print_log_header()
        rows = run_training("dense_phase1", model, tc, ctx.train, ctx.vb, ctx.pad, ctx.eos, ctx.device,
                            writer=w, lr_fn=constant(a.lr, a.warmup), log_every=max(10, a.dense_steps // 20))
        f.close()
        torch.save(model.state_dict(), path)
        print(f"  saved {path}")
        del model
        free(ctx.device)
    vp = val_points(rows)
    print("\n  validation loss:  " + "  ".join(f"{s}: {v:.3f}" for s, v in vp[::max(1, len(vp) // 8)]))
    tr = train_rows(rows)
    final = vp[-1][1]
    print(f"  checkpoint (step {vp[-1][0]}): val {final:.4f}, training loss (EMA) {ema([r['loss'] for r in tr], 0.1)[-1]:.4f}")
    print(f"  median step time {median([r['step_ms'] for r in tr[5:]]):.0f} ms")
    ctx.summary["dense"] = {"steps": a.dense_steps, "final_val": final}


# =========================================================================== 2. convert
def section_convert(ctx):
    a, mc = ctx.args, ctx.mcfg
    banner("2. CONVERT THE DENSE MODEL INTO A MIXTURE-OF-EXPERTS (Task 2)")
    dense = load_dense(ctx)
    Hd, H, E, k = ctx.cfg.ffn_hidden, mc.expert_hidden, mc.n_experts, mc.top_k
    print(f"  every block's SwiGLU (width {Hd}) -> {E} experts of width {H}, top-{k}: {k} x {H} = {k * H} "
          f"= the dense width")
    print(f"  partition: {E * H // Hd} shuffled copies of the FFN, each cut into {Hd // H} slices of {H} neurons")
    print(f"  router: softmax over {E}, fp32, init std {mc.router_init_std:g}; routed scale = {Hd}/{H} = {Hd / H:g}")
    moes = {name: upcycle(dense, mc, drop_r=r, seed=a.seed) for name, r in variants(ctx)}
    m0 = moes["moe_r0"]

    pd, pm = param_counts(dense), param_counts(m0)
    print(f"\n  {'':<26}{'dense':>14}{'MoE':>14}")
    print(f"  {'total parameters':<26}{pd['total']:>14,}{pm['total']:>14,}   ({pm['total'] / pd['total']:.2f}x)")
    print(f"  {'active per token':<26}{pd['active']:>14,}{pm['active']:>14,}   "
          f"(+{pm['active'] - pd['active']:,}: the routers)")
    print(f"  {'training state, 16 B/w':<26}{pd['total'] * 16 / 2**30:>13.2f}G{pm['total'] * 16 / 2**30:>13.2f}G")

    # ---- check 1: the conversion moved the weights correctly
    print("\n  Check: the experts from copy 0 (slices 0..3), summed with weight 1, must equal the dense FFN.")
    x = torch.randn(512, ctx.cfg.d_model, generator=torch.Generator().manual_seed(1)).to(ctx.device)
    with torch.no_grad():
        for name, m in moes.items():
            worst = 0.0
            for l, block in enumerate(dense.blocks):
                yd = block.ffn(x)
                ym = m.blocks[l].ffn.expert_sum(x, range(m.moe_slices))
                worst = max(worst, ((yd - ym).abs().max() / yd.abs().max()).item())
            print(f"    {name:<10} largest difference over 8 layers, relative to the output size: {worst:.1e}")
    print("    r=0 matches to fp32 rounding; r=0.5 does not, because half of every expert was redrawn.")

    # ---- check 2: loss right after conversion
    print("\n  Validation loss at step 0 (no training after conversion):")
    vd = evaluate(dense, ctx.vb, ctx.eos)
    res = {"dense": vd}
    print(f"    {'dense checkpoint':<22}{vd:.4f}")
    for name, m in moes.items():
        v = evaluate(m, ctx.vb, ctx.eos)
        res[name] = v
        print(f"    {name:<22}{v:.4f}   ({v - vd:+.4f})")

    # ---- check 3: routing before any router training
    print("\n  Routing at step 0 (validation batches). MaxVio = (busiest - even) / even, 0 = perfectly even:")
    for name, m in moes.items():
        mv = maxvio(route_counts(m, ctx.vb, ctx.device))
        print(f"    {name:<10} MaxVio by layer: " + " ".join(f"{v:.2f}" for v in mv.tolist()))
    ctx.summary["convert"] = {"params_dense": pd, "params_moe": pm, "val_step0": res}
    del moes, dense
    free(ctx.device)


# =========================================================================== 3. continue
def section_continue(ctx):
    a, mc = ctx.args, ctx.mcfg
    banner("3. KEEP TRAINING: DENSE vs MoE FROM THE SAME CHECKPOINT (Task 3)")
    dense = load_dense(ctx)
    S = a.cont_steps
    tc = tc_for(a, S, seed=1)                      # new batches, the same for every branch
    lr_fn = wsd(S, a.lr, a.warmup, decay_frac=0.1, min_frac=0.1)
    print(f"  three branches from the step-{a.dense_steps} checkpoint, {S} steps each "
          f"({S * 8192 / 1e6:.1f}M new tokens, the same batches for all three)")
    print(f"  WSD: fresh AdamW, warmup {a.warmup}, flat {a.lr:g}, linear decay to 10% over the last {S - round(S * 0.9)} steps")
    print(f"  MoE balancing: bias gamma {mc.bias_update:g} on whole-batch load, sequence aux {mc.seq_aux_coef:g}, dropless")
    branches = [("dense", None)] + variants(ctx)
    res = {}
    for name, r in branches:
        log, stats, path = out(ctx, f"cont_{name}.csv"), out(ctx, f"moe_stats_{name}.csv"), ckpt(ctx, name)
        done = log.exists() and path.exists() and (r is None or stats.exists())
        if done and not a.fresh:
            print(f"\n  {name}: reusing {log.name} (--fresh to retrain)")
            res[name] = read_log(log)
            continue
        print(f"\n  RUN {name}")
        model = copy.deepcopy(dense) if r is None else upcycle(dense, mc, drop_r=r, seed=a.seed)
        hook = None
        if r is not None:
            for m in moe_layers(model):
                m.aux_scale = mc.seq_aux_coef / tc.accum_steps      # the loop sums micro-batches
            hook = MoEBalancer(model, gamma=mc.bias_update)
        f, w = open_log(log)
        print_log_header()
        rows = run_training(name, model, tc, ctx.train, ctx.vb, ctx.pad, ctx.eos, ctx.device, writer=w,
                            lr_fn=lr_fn, log_every=max(10, S // 20), step_hook=hook)
        f.close()
        torch.save(model.state_dict(), path)
        if hook:
            hook.save(stats)
        res[name] = rows
        del model, hook
        free(ctx.device)

    # ---- tables
    ck = val_points(read_log(out(ctx, "dense_phase1.csv")))[-1][1]
    pts = sorted({s for s, _ in val_points(res["dense"])})
    show = [s for s in pts if s in {0, a.eval_every, 2 * a.eval_every} or s % max(1, S // 8) == 0 or s == pts[-1]]
    print(f"\n  validation loss (dense checkpoint before branching: {ck:.4f})")
    print(f"  {'step':>6}" + "".join(f"{n:>12}" for n, _ in branches) + "".join(f"{n + '-dense':>16}" for n, _ in branches[1:]))
    for s in show:
        vals = [val_at(res[n], s) for n, _ in branches]
        print(f"  {s:>6}" + "".join(f"{v:>12.4f}" for v in vals) + "".join(f"{v - vals[0]:>+16.4f}" for v in vals[1:]))

    print(f"\n  {'branch':<12}{'step 0':>9}{'lowest':>9}{'final':>9}{'vs ckpt':>9}{'vs step0':>10}"
          f"{'ms/step':>9}{'tok/s':>9}{'recovers ckpt at':>18}")
    summ = {}
    for n, _ in branches:
        vp = val_points(res[n])
        v0, vmin, vend = vp[0][1], min(v for _, v in vp), vp[-1][1]
        rec = next((s for s, v in vp if v <= ck), None)
        ms = median([r["step_ms"] for r in train_rows(res[n])[5:]])
        print(f"  {n:<12}{v0:>9.4f}{vmin:>9.4f}{vend:>9.4f}{vend - ck:>+9.4f}{vend - v0:>+10.4f}"
              f"{ms:>9.0f}{8192 / ms * 1000:>9,.0f}{('step ' + str(rec)) if rec is not None else 'never':>18}")
        summ[n] = {"val_step0": v0, "val_min": vmin, "val_final": vend, "ms_per_step": ms,
                   "recovers_ckpt_step": rec, "val": vp}
    print("  'vs ckpt' < 0: the branch ends below the dense checkpoint, i.e. it kept training and kept improving.")
    ctx.summary["continue"] = {"ckpt_val": ck, "branches": summ}
    plot_continue(ctx, read_log(out(ctx, "dense_phase1.csv")), res, a.dense_steps)
    del dense
    free(ctx.device)


def plot_continue(ctx, phase1, res, offset):
    plt = get_plt()
    if not plt:
        return
    colours = {"dense": "black", "moe_r0": "tab:blue"}
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(15, 5))
    vp = val_points(phase1)
    ax.plot([s for s, _ in vp], [v for _, v in vp], "-", color="grey", label="dense, phase 1")
    for n, rows in res.items():
        c = colours.get(n, "tab:orange")
        vp = val_points(rows)
        ax.plot([s + offset for s, _ in vp], [v for _, v in vp], "o-", ms=3, color=c, label=n)
        tr = train_rows(rows)
        bx.plot([r["step"] for r in tr], ema([r["loss"] for r in tr], 0.05), color=c, label=n)
    ax.axvline(offset, color="grey", ls=":", label="conversion")
    tail = [v for rows in res.values() for _, v in val_points(rows)]
    ax.set_ylim(min(tail) - 0.1, max(tail) + 0.3)
    ax.set(xlabel="optimiser step (phase 1 + phase 2)", ylabel="validation loss", title="Dense, then three branches")
    bx.set(xlabel="phase-2 step", ylabel="training loss (EMA)", title="Phase 2 training loss")
    for x in (ax, bx):
        x.grid(alpha=0.3)
        x.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out(ctx, "continue.png"), dpi=120)
    plt.close(fig)
    print(f"\n  plot saved: {out(ctx, 'continue.png')}")


# =========================================================================== 4. routing
def section_routing(ctx):
    a = ctx.args
    E = ctx.mcfg.n_experts
    banner("4. LOAD BALANCE DURING TRAINING (Task 4)")
    stats = {}
    for name, _ in variants(ctx):
        p = out(ctx, f"moe_stats_{name}.csv")
        if not p.exists():
            raise SystemExit(f"{p.name} not found: run --section continue first")
        stats[name] = read_stats(p)
    print(f"  per optimiser step, per layer. Even load = 1/{E} = {100 / E:.2f}% of routing slots.")
    print("  MaxVio: (busiest - even) / even. starved: experts under 10% of the even share. zero: no tokens at all.")
    summ = {}
    for name, st in stats.items():
        steps = st["steps"].tolist()
        n = len(steps)
        show = sorted({i for i in (0, 1, 4, 9, 24, 49, 99, 249, 499, 999, n - 1) if i < n})
        print(f"\n  {name}")
        print(f"  {'step':>6}{'MaxVio med':>12}{'MaxVio max':>12}{'starved':>9}{'zero':>6}{'bias range':>12}{'seq aux':>9}")
        for i in show:
            mv = st["maxvio"][i]
            print(f"  {steps[i]:>6}{mv.median().item():>12.3f}{mv.max().item():>12.3f}"
                  f"{int(st['starved'][i].sum()):>9}{int(st['zero'][i].sum()):>6}"
                  f"{(st['bias_max'][i] - st['bias_min'][i]).max().item():>12.4f}{st['aux'][i].mean().item():>9.3f}")
        last = st["load"][-min(100, n):].mean(0)                    # [L, E], last 100 steps
        print(f"  last {min(100, n)} steps, per layer: busiest and quietest expert as a multiple of even")
        print("    layer " + " ".join(f"{l:>6}" for l in range(last.shape[0])))
        print("    max   " + " ".join(f"{v * E:>6.2f}" for v in last.max(1).values.tolist()))
        print("    min   " + " ".join(f"{v * E:>6.2f}" for v in last.min(1).values.tolist()))
        med = st["maxvio"].median(1).values
        tail_max = med.flip(0).cummax(0).values.flip(0)              # worst median from step i on
        first = next((steps[i] for i in range(n) if tail_max[i] < 0.5), None)
        print(f"  median MaxVio stays below 0.5 from step: {first if first is not None else 'never'}")
        summ[name] = {"maxvio_median_final": st["maxvio"][-1].median().item(),
                      "starved_total_final": int(st["starved"][-1].sum()), "settles_below_0.5": first}
    ctx.summary["routing"] = summ
    plot_routing(ctx, stats)


def plot_routing(ctx, stats):
    plt = get_plt()
    if not plt:
        return
    E = ctx.mcfg.n_experts
    fig, axes = plt.subplots(1, 1 + len(stats), figsize=(6 + 5 * len(stats), 4.6))
    for name, st in stats.items():
        axes[0].plot(st["steps"].tolist(), st["maxvio"].median(1).values.tolist(), label=f"{name}, median layer")
        axes[0].plot(st["steps"].tolist(), st["maxvio"].max(1).values.tolist(), alpha=0.4, label=f"{name}, worst layer")
    axes[0].set_yscale("log")
    axes[0].set(xlabel="step", ylabel="MaxVio (log)", title="Load imbalance")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    for ax, (name, st) in zip(axes[1:], stats.items()):
        last = st["load"][-min(100, len(st["steps"])):].mean(0) * E
        im = ax.imshow(last.numpy(), cmap="coolwarm", vmin=0, vmax=2, aspect="auto")
        ax.set(xlabel="expert", ylabel="layer", title=f"{name}: load / even, last 100 steps")
        fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out(ctx, "routing.png"), dpi=120)
    plt.close(fig)
    print(f"\n  plot saved: {out(ctx, 'routing.png')}")


# =========================================================================== 5a. no balancing
def section_nobalance(ctx):
    a, mc = ctx.args, ctx.mcfg
    name, r = variants(ctx)[1]
    S, n = a.cont_steps, a.nob_steps
    banner(f"5a. THE SAME MoE RUN WITH BALANCING SWITCHED OFF (Task 5, {name}, {n} steps)")
    print("  bias frozen at 0 and no sequence aux loss; same conversion, schedule and batches as the balanced run")
    log, sp = out(ctx, f"nobal_{name}.csv"), out(ctx, f"moe_stats_nobal_{name}.csv")
    if log.exists() and sp.exists() and not a.fresh:
        print(f"  reusing {log.name} (--fresh to retrain)")
        rows = read_log(log)
    else:
        model = upcycle(load_dense(ctx), mc, drop_r=r, seed=a.seed)
        hook = MoEBalancer(model, gamma=0.0)           # measures only
        tc = tc_for(a, n, seed=1)
        f, w = open_log(log)
        print_log_header()
        rows = run_training(f"nobal_{name}", model, tc, ctx.train, ctx.vb, ctx.pad, ctx.eos, ctx.device,
                            writer=w, lr_fn=wsd(S, a.lr, a.warmup, decay_frac=0.1, min_frac=0.1),
                            log_every=max(10, n // 10), step_hook=hook)
        f.close()
        hook.save(sp)
        del model, hook
        free(ctx.device)
    bal_rows = read_log(out(ctx, f"cont_{name}.csv"))
    sb, su = read_stats(out(ctx, f"moe_stats_{name}.csv")), read_stats(sp)
    print(f"\n  {'step':>6}{'val bal':>10}{'val off':>10}{'MaxVio bal':>12}{'MaxVio off':>12}"
          f"{'starved bal':>13}{'starved off':>13}{'zero off':>10}")
    for s, _ in val_points(rows):
        i = min(s, n - 1, len(su["steps"]) - 1)                     # stats row i = step i
        print(f"  {s:>6}{val_at(bal_rows, s):>10.4f}{val_at(rows, s):>10.4f}"
              f"{sb['maxvio'][i].median().item():>12.3f}{su['maxvio'][i].median().item():>12.3f}"
              f"{int(sb['starved'][i].sum()):>13}{int(su['starved'][i].sum()):>13}{int(su['zero'][i].sum()):>10}")
    E = mc.n_experts
    lo = max(0, n - 50)
    lb, lu = sb["load"][lo:n].mean(0), su["load"][lo:n].mean(0)
    print(f"\n  steps {lo}-{n}, busiest expert as a multiple of even, by layer:")
    print("    balanced " + " ".join(f"{v * E:>5.2f}" for v in lb.max(1).values.tolist()))
    print("    off      " + " ".join(f"{v * E:>5.2f}" for v in lu.max(1).values.tolist()))
    print(f"  experts starved (<10% of even) at the end, all layers: balanced {int((lb < 0.1 / E).sum())}, "
          f"off {int((lu < 0.1 / E).sum())} of {lb.numel()}")
    ctx.summary["nobalance"] = {"val_bal": val_at(bal_rows, n), "val_off": val_at(rows, n),
                                "maxvio_bal": sb["maxvio"][n - 1].median().item(),
                                "maxvio_off": su["maxvio"][-1].median().item()}
    plt = get_plt()
    if plt:
        fig, (ax, bx) = plt.subplots(1, 2, figsize=(14, 4.6))
        for lab, st, m in (("balanced", sb, n), ("off", su, len(su["steps"]))):
            ax.plot(st["steps"][:m].tolist(), st["maxvio"][:m].median(1).values.tolist(), label=lab)
            bx.plot(st["steps"][:m].tolist(), st["starved"][:m].sum(1).tolist(), label=lab)
        ax.set_yscale("log")
        ax.set(xlabel="step", ylabel="MaxVio, median layer (log)", title="Imbalance")
        bx.set(xlabel="step", ylabel="starved experts, all layers", title=f"Experts under 10% of even (of {lb.numel()})")
        for x in (ax, bx):
            x.grid(alpha=0.3)
            x.legend()
        fig.tight_layout()
        fig.savefig(out(ctx, "nobalance.png"), dpi=120)
        plt.close(fig)
        print(f"\n  plot saved: {out(ctx, 'nobalance.png')}")


# =========================================================================== 5b. specialisation
FUNCTION_WORDS = {"the", "of", "and", "to", "a", "in", "is", "that", "for", "it", "as", "with", "was",
                  "on", "be", "by", "are", "or", "an", "this", "from", "at", "not", "but", "have", "which"}
CATS = ["whitespace", "punctuation", "digits", "function word", "Capitalised", "word start", "word piece", "other"]


def token_category(s):
    core = s.replace("Ġ", " ").replace("Ċ", "\n").replace("ĉ", "\t")
    t = core.strip()
    if not t:
        return "whitespace"
    if all(ch in string.punctuation for ch in t):
        return "punctuation"
    if t.isdigit():
        return "digits"
    if t.isalpha():
        if t.lower() in FUNCTION_WORDS and core.startswith(" "):
            return "function word"
        if core.startswith(" ") and t[0].isupper():
            return "Capitalised"
        return "word start" if core.startswith(" ") else "word piece"
    return "other"


def section_special(ctx):
    a = ctx.args
    name, r = variants(ctx)[1]
    banner(f"5b. WHAT DID THE EXPERTS LEARN? ({name} after training, Task 5)")
    model = load_moe(ctx, name, r)
    tok = load_tokenizer()
    cat_of = torch.tensor([CATS.index(token_category(tok.id_to_token(i) or "")) for i in range(ctx.cfg.vocab_size)])
    mods = moe_layers(model)
    E, k, L, C = mods[0].n_experts, mods[0].top_k, len(mods), len(CATS)
    counts = torch.zeros(L, C, E)
    batches = val_set(ctx.cfg, ctx.device, a.special_batches)
    for m in mods:
        m.record = True
    model.eval()
    with torch.no_grad():
        for tokens in batches:
            with autocast(ctx.device):
                model(tokens)
            c = cat_of[tokens.reshape(-1).cpu()].repeat_interleave(k)
            for l, m in enumerate(mods):
                counts[l].index_put_((c, m.last_idx.reshape(-1).cpu()), torch.ones(c.numel()), accumulate=True)
    ntok = int(counts[0].sum().item() / k)
    print(f"  {ntok:,} validation tokens, each token's category is that of the token itself.")
    print("  top share: share of the category's routing slots taken by its favourite expert (even = "
          f"{100 / E:.2f}%)")
    print("  lift: for that expert, P(category | expert) / P(category). 1.0 = no preference.")
    share_c = counts[0].sum(1) / counts[0].sum()
    summ = {}
    for l in sorted({0, L // 2, L - 1}):
        cl = counts[l]
        print(f"\n  layer {l}")
        print(f"  {'category':<15}{'tokens':>8}{'top expert':>12}{'top share':>11}{'lift':>7}")
        for ci, cname in enumerate(CATS):
            if cl[ci].sum() == 0:
                continue
            e = int(cl[ci].argmax())
            top = (cl[ci, e] / cl[ci].sum()).item()
            lift = ((cl[ci, e] / cl[:, e].sum()) / share_c[ci]).item()
            print(f"  {cname:<15}{100 * share_c[ci].item():>7.1f}%{e:>12}{100 * top:>10.1f}%{lift:>7.2f}")
            summ[f"L{l}/{cname}"] = {"top_expert": e, "top_share": top, "lift": lift}
    # layer x category heatmap of the best lift
    lifts = torch.zeros(L, C)
    for l in range(L):
        pe = counts[l] / counts[l].sum(0, keepdim=True).clamp(min=1)           # P(cat | expert)
        lifts[l] = (pe / share_c[:, None].clamp(min=1e-9)).max(1).values
    print("\n  best lift per layer (any expert), rows = layers:")
    print("  " + " " * 6 + "".join(f"{c[:11]:>12}" for c in CATS))
    for l in range(L):
        print(f"  {l:>6}" + "".join(f"{v:>12.2f}" for v in lifts[l].tolist()))
    ctx.summary["special"] = summ
    plt = get_plt()
    if plt:
        fig, ax = plt.subplots(figsize=(10, 4.8))
        im = ax.imshow(lifts.numpy(), cmap="viridis", aspect="auto")
        ax.set_xticks(range(C), CATS, rotation=30, ha="right")
        ax.set(ylabel="layer", title=f"{name}: strongest preference of any expert for each token type (lift)")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(out(ctx, "specialisation.png"), dpi=120)
        plt.close(fig)
        print(f"\n  plot saved: {out(ctx, 'specialisation.png')}")


# =========================================================================== main
SECTIONS = ("dense", "convert", "continue", "routing", "nobalance", "special")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--section", choices=("all",) + SECTIONS, default="all")
    p.add_argument("--dense-steps", type=int, default=2000)
    p.add_argument("--cont-steps", type=int, default=2000)
    p.add_argument("--nob-steps", type=int, default=600)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--eval-batches", type=int, default=20)
    p.add_argument("--special-batches", type=int, default=40)
    p.add_argument("--drop-r", type=float, default=None, help="override MoEConfig.drop_r")
    p.add_argument("--seed", type=int, default=0, help="conversion seed")
    p.add_argument("--fresh", action="store_true", help="retrain even if results exist")
    p.add_argument("--quick", action="store_true", help="short smoke test, separate files")
    p.add_argument("--log", default=None, help="also write everything printed to this file")
    a = p.parse_args(argv)
    if a.quick:
        a.dense_steps, a.cont_steps, a.nob_steps = 60, 60, 60
        a.warmup, a.eval_every, a.eval_batches, a.special_batches = 10, 20, 4, 4

    tee = None
    if a.log:
        tee = Tee(a.log)
        sys.stdout = tee
    try:
        device, cfg, pad, eos = setup()
        RUN_DIR.mkdir(parents=True, exist_ok=True)
        mcfg = MoEConfig()
        if a.drop_r is not None:
            mcfg.drop_r = a.drop_r
        train = TokenData("train")
        print(f"device {device}, model D={cfg.d_model} L={cfg.n_layers} FFN={cfg.ffn_hidden}; "
              f"MoE {mcfg.n_experts} x {mcfg.expert_hidden}, top-{mcfg.top_k}; train.bin {len(train):,} tokens")
        need = 8192 * (a.dense_steps + a.cont_steps)
        if len(train) < 0.75 * need:
            print(f"  NOTE: one run reads ~{need / 1e6:.0f}M tokens; train.bin has {len(train) / 1e6:.0f}M, so "
                  f"windows repeat. Run: uv run python -m data.extend_data")
        ctx = SimpleNamespace(device=device, cfg=cfg, pad=pad, eos=eos, train=train, args=a, mcfg=mcfg,
                              vb=val_set(cfg, device, a.eval_batches), summary={},
                              suf="_quick" if a.quick else "")
        fns = {"dense": section_dense, "convert": section_convert, "continue": section_continue,
               "routing": section_routing, "nobalance": section_nobalance, "special": section_special}
        t0 = time.time()
        for s in (SECTIONS if a.section == "all" else (a.section,)):
            t = time.time()
            fns[s](ctx)
            save_summary(ctx)
            print(f"\n  [{s} took {(time.time() - t) / 60:.1f} min]")
        print(f"\nSession 14 done in {(time.time() - t0) / 60:.1f} min.")
    finally:
        if tee:
            sys.stdout = tee.out
            tee.f.close()


if __name__ == "__main__":
    main()
