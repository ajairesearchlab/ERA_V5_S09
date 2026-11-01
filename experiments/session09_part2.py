"""
Session 9 assignment, Part 2: a second output head that predicts token t+2.

Two training runs from the same starting weights, on the same batches:
    baseline : shared trunk + head 1 (predicts t+1)              loss = L1
    mtp      : shared trunk + head 1 (t+1) + head 2 (t+2)        loss = L1 + L2

Run from the mini_llm folder:
    uv run python -m experiments.session09_part2 --steps 200      # quick smoke test (~1-2 min)
    uv run python -m experiments.session09_part2 | Tee-Object -FilePath s09_part2.txt
    uv run python -m experiments.session09_part2 --only mtp       # one run only

Optional plot (the script runs without it):
    uv add matplotlib

Outputs
    runs/session09_part2/log.csv       one row per evaluation, both runs
    runs/session09_part2/losses.png    if matplotlib is installed
    checkpoints/s09_baseline.pt, checkpoints/s09_mtp.pt   reusable in later sessions
"""
import argparse
import csv
import math
import time

import torch
import torch.nn.functional as F

from config import CHECKPOINT_DIR, PROJECT_ROOT, ModelConfig
from data.dataset import TokenData, load_meta
from model.model import MiniLLM
from model.mtp import MiniLLMMTP
from train.loss import IGNORE, future_targets, mtp_losses

RUN_DIR = PROJECT_ROOT / "runs" / "session09_part2"
NAN = float("nan")


def autocast(device):
    return torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda")


def banner(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def fmt(x, w=9, p=4):
    return f"{x:>{w}.{p}f}" if x == x else f"{'-':>{w}}"   # x != x only for NaN


def csv_num(x):
    return "" if x != x else f"{x:.5f}"


def lr_at(step, total, peak, warmup):
    """Linear warmup, then cosine decay to 10% of peak."""
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def head_weights(model):
    return model.head_weights() if isinstance(model, MiniLLMMTP) else [model.lm_head.weight]


def build_optimizer(model, lr):
    decay = [p for p in model.parameters() if p.dim() >= 2]       # matrices
    no_decay = [p for p in model.parameters() if p.dim() < 2]     # norm gains
    return torch.optim.AdamW([{"params": decay, "weight_decay": 0.1},
                              {"params": no_decay, "weight_decay": 0.0}],
                             lr=lr, betas=(0.9, 0.95))


@torch.no_grad()
def evaluate(model, val_batches, eos_id):
    """Per head: (mean val loss, top-1 accuracy) on fixed val batches, boundaries masked."""
    model.eval()
    weights = head_weights(model)
    stats = [[0.0, 0, 0] for _ in weights]                         # loss sum, correct, count
    for tokens in val_batches:
        with autocast(tokens.device.type):
            h = model.hidden_states(tokens)
        D = h.size(-1)
        for k, W in enumerate(weights, start=1):
            tg = future_targets(tokens, k, eos_id=eos_id, mask_boundaries=True).reshape(-1)
            logits = F.linear(h[:, :-k].reshape(-1, D).float(), W.float())
            valid = tg != IGNORE
            stats[k - 1][0] += F.cross_entropy(logits, tg, ignore_index=IGNORE, reduction="sum").item()
            stats[k - 1][1] += (logits.argmax(-1)[valid] == tg[valid]).sum().item()
            stats[k - 1][2] += int(valid.sum())
    model.train()
    return [(s[0] / s[2], s[1] / s[2]) for s in stats]


def train_run(name, model, args, cfg, train, val_batches, eos_id, device, writer):
    weights = head_weights(model)
    n_heads = len(weights)
    n_params = sum(p.numel() for p in model.parameters())
    banner(f"RUN: {name}  ({n_heads} head{'s' if n_heads > 1 else ''}, {n_params:,} parameters)")
    print(f"  {'step':>6}{'lr':>10}{'train L1':>9}{'train L2':>9}{'val L1':>9}{'val L2':>9}"
          f"{'acc1 %':>8}{'acc2 %':>8}{'time':>8}")

    opt = build_optimizer(model, args.lr)
    gen = torch.Generator().manual_seed(args.seed)                 # identical batches in every run
    sums, n_acc, history = [0.0] * n_heads, 0, []
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    model.train()

    for step in range(args.steps + 1):
        if step % args.eval_every == 0 or step == args.steps:
            val = evaluate(model, val_batches, eos_id)
            tr = [s / n_acc for s in sums] if n_acc else [NAN] * n_heads
            lr = lr_at(step, args.steps, args.lr, args.warmup)
            history.append((step, val))
            tr2 = tr + [NAN] * (2 - n_heads)
            vl2 = [v[0] for v in val] + [NAN] * (2 - n_heads)
            ac2 = [v[1] * 100 for v in val] + [NAN] * (2 - n_heads)
            writer.writerow([name, step, f"{lr:.3e}", *map(csv_num, tr2), *map(csv_num, vl2),
                             *map(csv_num, ac2)])
            print(f"  {step:>6}{lr:>10.2e}{fmt(tr2[0])}{fmt(tr2[1])}{fmt(vl2[0])}{fmt(vl2[1])}"
                  f"{fmt(ac2[0], 8, 1)}{fmt(ac2[1], 8, 1)}{time.time() - t0:>7.0f}s")
            sums, n_acc = [0.0] * n_heads, 0
        if step == args.steps:
            break

        lr = lr_at(step, args.steps, args.lr, args.warmup)
        for g in opt.param_groups:
            g["lr"] = lr
        tokens = train.get_batch(args.batch, cfg.context_length, device=device, generator=gen)
        with autocast(device):
            h = model.hidden_states(tokens)                        # [B, T, D], shared by all heads
            losses = mtp_losses(h, weights, tokens, args.chunk, eos_id=eos_id, mask_boundaries=True)
        loss = losses[0] + args.mtp_weight * sum(losses[1:])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        for i, l in enumerate(losses):
            sums[i] += l.item()
        n_acc += 1

    mins = (time.time() - t0) / 60
    peak = torch.cuda.max_memory_allocated() / 2 ** 20 if device == "cuda" else 0
    print(f"\n  done in {mins:.1f} min, peak GPU memory {peak:,.0f} MiB")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = CHECKPOINT_DIR / f"s09_{name}.pt"
    torch.save({"model": model.state_dict(), "cfg": cfg.to_dict(), "n_future": n_heads,
                "steps": args.steps, "tokens_seen": args.steps * args.batch * cfg.context_length}, path)
    print(f"  saved {path}")
    return history


def summarize(hist):
    banner("SUMMARY: PART 2 NUMBERS FOR THE WRITE-UP (validation, boundaries masked)")
    base, mtp = dict(hist.get("baseline", [])), dict(hist.get("mtp", []))
    steps = sorted(set(base) | set(mtp))
    print(f"  {'step':>6}{'base L1':>10}{'mtp L1':>10}{'mtp L2':>10}{'L1+L2':>10}{'L2-L1':>9}"
          f"{'acc1 %':>8}{'acc2 %':>8}")
    for s in steps:
        b = base[s][0][0] if s in base else NAN
        (l1, a1), (l2, a2) = mtp[s] if s in mtp else ((NAN, NAN), (NAN, NAN))
        print(f"  {s:>6}{fmt(b, 10)}{fmt(l1, 10)}{fmt(l2, 10)}{fmt(l1 + l2, 10)}{fmt(l2 - l1, 9, 3)}"
              f"{fmt(a1 * 100, 8, 1)}{fmt(a2 * 100, 8, 1)}")

    if not mtp:
        return
    ms = sorted(mtp)
    s0, s1, sN = ms[0], ms[1] if len(ms) > 1 else ms[0], ms[-1]
    l1_0, l2_0 = mtp[s0][0][0], mtp[s0][1][0]
    (l1_N, a1_N), (l2_N, a2_N) = mtp[sN]
    gap1, gapN = mtp[s1][1][0] - mtp[s1][0][0], l2_N - l1_N

    print("\n  What happened to each head (MTP run):")
    print(f"    head 1 (t+1): {l1_0:.3f} -> {l1_N:.3f}   fell {l1_0 - l1_N:.3f}")
    print(f"    head 2 (t+2): {l2_0:.3f} -> {l2_N:.3f}   fell {l2_0 - l2_N:.3f}")
    print(f"    gap L2 - L1 : {gap1:+.3f} at step {s1} -> {gapN:+.3f} at step {sN} "
          f"({'widened' if gapN > gap1 else 'narrowed'})")
    print("\n  Both heads start near ln(V): neither knows anything yet. Head 2 stays above head 1")
    print("  because it must predict t+2 without seeing t+1, so it has to spread its probability")
    print("  over every way the text could continue. Easy local patterns (finishing a word, like")
    print("  ' S' -> 'aint') help head 1 a lot and head 2 much less, which is why the gap usually")
    print("  widens as training finds those patterns.")
    print(f"\n  Top-1 accuracy at the end: head 1 {a1_N:.1%}, head 2 {a2_N:.1%}. Used as a draft for")
    print(f"  speculative decoding, head 2 would guess the next-next token right about {a2_N:.0%} of the")
    print("  time: a rough proxy for the acceptance rate the notes discuss.")

    if base and sN in base:
        b_N = base[sN][0][0]
        d = l1_N - b_N
        print(f"\n  Head 1 with vs without head 2: {l1_N:.4f} vs {b_N:.4f}  ({d:+.4f})")
        if abs(d) < 0.02:
            print("  Within run-to-run noise at this scale: no clear effect either way.")
        elif d < 0:
            print("  The extra t+2 signal helped head 1: the shared trunk learned more useful features.")
        else:
            print("  The extra head cost head 1 some quality: two objectives share one small trunk.")
            print("  Published MTP results report the benefit growing with model size; small models")
            print("  can lose quality, as here.")


def plot(hist, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n  (matplotlib not installed, so no plot. To add it: uv add matplotlib)")
        return
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, style in (("baseline", "--"), ("mtp", "-")):
        if name not in hist:
            continue
        steps = [s for s, _ in hist[name]]
        a.plot(steps, [v[0][0] for _, v in hist[name]], style, label=f"{name}: head 1 (t+1)")
        if name == "mtp":
            a.plot(steps, [v[1][0] for _, v in hist[name]], style, label="mtp: head 2 (t+2)")
            b.plot(steps, [v[1][0] - v[0][0] for _, v in hist[name]], label="L2 - L1")
    a.set(xlabel="step", ylabel="validation loss (nats)", title="Validation loss")
    a.legend()
    a.grid(alpha=0.3)
    b.set(xlabel="step", ylabel="nats", title="Gap: head 2 minus head 1")
    b.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"\n  plot saved: {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--eval-batches", type=int, default=20)
    p.add_argument("--chunk", type=int, default=1024)
    p.add_argument("--mtp-weight", type=float, default=1.0, help="weight on L2 (assignment: plain sum)")
    p.add_argument("--only", choices=["both", "baseline", "mtp"], default="both")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    meta = load_meta()
    cfg = ModelConfig(vocab_size=meta["vocab_size"])
    eos = meta["eos_id"]
    train, val = TokenData("train"), TokenData("val")

    g = torch.Generator().manual_seed(1234)                        # fixed val set for every eval
    val_batches = [val.get_batch(16, cfg.context_length, device=device, generator=g)
                   for _ in range(args.eval_batches)]

    tokens_per_step = args.batch * cfg.context_length
    total = tokens_per_step * args.steps
    banner("SESSION 9 PART 2: one extra head predicting t+2")
    print(f"  steps {args.steps:,} x batch {args.batch} x T {cfg.context_length} = {total / 1e6:.1f}M tokens "
          f"({total / len(train):.2f} passes over the {len(train) / 1e6:.0f}M training tokens)")
    print(f"  lr {args.lr:g} (warmup {args.warmup}, cosine to 10%), AdamW, grad clip 1.0, bf16, chunked loss")
    print(f"  validation: {args.eval_batches} fixed batches of 16 x {cfg.context_length}, every {args.eval_every} steps")
    print(f"  loss: baseline L1;  mtp L1 + {args.mtp_weight:g} x L2")

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    hist = {}
    names = ("baseline", "mtp") if args.only == "both" else (args.only,)
    with open(RUN_DIR / "log.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "step", "lr", "train_l1", "train_l2", "val_l1", "val_l2",
                    "val_acc1_pct", "val_acc2_pct"])
        for name in names:
            torch.manual_seed(args.seed)                            # same trunk init in both runs
            model = (MiniLLMMTP(cfg, n_future=2) if name == "mtp" else MiniLLM(cfg)).to(device)
            hist[name] = train_run(name, model, args, cfg, train, val_batches, eos, device, w)
            f.flush()
            del model
            if device == "cuda":
                torch.cuda.empty_cache()

    summarize(hist)
    plot(hist, RUN_DIR / "losses.png")
    print(f"\n  log saved: {RUN_DIR / 'log.csv'}")
    print("\nPart 2 done.")


if __name__ == "__main__":
    main()
