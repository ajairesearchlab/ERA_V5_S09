"""
Reusable training step (Session 10), for this session and later ones.

One optimiser step:
    for each micro-batch:   forward -> loss -> backward      (gradients pile up in .grad)
    clip                    measure the grad norm, shrink all gradients if it exceeds the cap
    update                  optimizer.step()
    wipe                    optimizer.zero_grad()

How the micro-batch losses are combined (Session 10 Section 8):
    "token"           sum of every token's loss / total real tokens       (correct)
    "mean_of_means"   average of each micro-batch's own average           (the pre-2024 bug)
They agree exactly when every micro-batch holds the same number of real tokens,
and disagree when lengths differ.

Session 11 hooks (run_training): pass your own optimiser with opt=..., and a step_hook
with before(model, step) / after(model, step) to watch every update (e.g. UpdateRatio).
"""
import csv
import math
import time

import torch
import torch.nn.functional as F

from train.loss import IGNORE, shift_targets

REDUCTIONS = ("token", "mean_of_means")


def autocast(device, enabled=True):
    return torch.autocast(device_type=device, dtype=torch.bfloat16,
                          enabled=enabled and device == "cuda")


def lr_at(step, total, peak, warmup, min_frac=0.1):
    """Linear warmup, then cosine decay to min_frac * peak (same schedule as Session 9)."""
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return peak * (min_frac + (1 - min_frac) * 0.5 * (1 + math.cos(math.pi * progress)))


def build_optimizer(model, tc):
    decay = [p for p in model.parameters() if p.dim() >= 2]       # matrices
    no_decay = [p for p in model.parameters() if p.dim() < 2]     # norm gains
    return torch.optim.AdamW([{"params": decay, "weight_decay": tc.weight_decay},
                              {"params": no_decay, "weight_decay": 0.0}],
                             lr=tc.lr, betas=(tc.beta1, tc.beta2))


def set_lr(opt, lr):
    for g in opt.param_groups:
        g["lr"] = lr


# --------------------------------------------------------------------------- batches
def make_micro_batches(data, n_micro, micro_batch, T, device, generator,
                       lengths=None, pad_id=0):
    """
    n_micro tensors of shape [micro_batch, T]. If lengths is given, micro-batch i keeps
    only its first lengths[i] tokens per row and the rest is <pad>: same tensor shape,
    different number of REAL tokens, which is what Section 8 is about.
    """
    micros = []
    for i in range(n_micro):
        tokens = data.get_batch(micro_batch, T, device=device, generator=generator)
        if lengths is not None and lengths[i] < T:
            tokens[:, lengths[i]:] = pad_id
        micros.append(tokens)
    return micros


def micro_targets(micros, pad_id, eos_id):
    return [shift_targets(m, pad_id=pad_id, eos_id=eos_id, mask_boundaries=True) for m in micros]


# --------------------------------------------------------------------------- one step
def micro_loss_sum(model, tokens, targets, device, use_autocast=True):
    """Summed (not averaged) loss over the real targets of one micro-batch."""
    with autocast(device, use_autocast):
        logits = model(tokens)                                       # [b, T, V]
    V = logits.size(-1)
    return F.cross_entropy(logits[:, :-1].reshape(-1, V).float(), targets.reshape(-1),
                           ignore_index=IGNORE, reduction="sum")


def accumulate(model, micros, pad_id, eos_id, reduction="token", use_autocast=True):
    """
    Forward + backward over every micro-batch, leaving the combined gradient in .grad.
    Does NOT clip, step or wipe. Returns detached tensors (no GPU sync):
        reported  the loss this reduction optimises
        true      sum of all token losses / all real tokens (same yardstick for both)
        n_tokens  real target tokens in the whole step
    """
    assert reduction in REDUCTIONS
    device = micros[0].device.type
    targets = micro_targets(micros, pad_id, eos_id)
    counts = [(t != IGNORE).sum() for t in targets]                  # tensors: no sync
    total = torch.stack(counts).sum().clamp(min=1)
    K = len(micros)

    reported = torch.zeros((), device=micros[0].device)
    true = torch.zeros((), device=micros[0].device)
    for tokens, tg, n in zip(micros, targets, counts):
        s = micro_loss_sum(model, tokens, tg, device, use_autocast)
        if reduction == "token":
            part = s / total                       # every token weighs the same
        else:
            part = s / n.clamp(min=1) / K          # every micro-batch weighs the same
        part.backward()
        reported += part.detach()
        true += s.detach() / total
    return {"reported": reported, "true": true, "n_tokens": total}


def train_step(model, opt, micros, pad_id, eos_id, reduction="token", clip=1.0):
    """Forward, loss, backward (accumulated), clip, update, wipe."""
    out = accumulate(model, micros, pad_id, eos_id, reduction)
    cap = clip if clip and clip > 0 else float("inf")
    out["grad_norm"] = torch.nn.utils.clip_grad_norm_(model.parameters(), cap)   # norm BEFORE clipping
    opt.step()
    opt.zero_grad(set_to_none=True)                # the wipe is not optional
    return out


@torch.no_grad()
def evaluate(model, val_batches, eos_id, pad_id=None):
    """Token-weighted validation loss, boundaries masked."""
    model.eval()
    total, n = 0.0, 0
    for tokens in val_batches:
        tg = shift_targets(tokens, pad_id=pad_id, eos_id=eos_id, mask_boundaries=True)
        total += micro_loss_sum(model, tokens, tg, tokens.device.type).item()
        n += int((tg != IGNORE).sum())
    model.train()
    return total / max(n, 1)


# --------------------------------------------------------------------------- a full run
def run_training(name, model, tc, train_data, val_batches, pad_id, eos_id, device,
                 writer=None, reduction="token", lengths=None, lr_fn=None, log_every=50,
                 opt=None, step_hook=None):
    """
    Train for tc.steps optimiser steps and log EVERY step:
        loss (reported and true), grad norm before clipping, lr, real tokens, step time.
    Batches come from a generator seeded with tc.seed, so two runs with the same tc
    see identical data. lr_fn(step) overrides the warmup + cosine schedule.
    opt (Session 11): use this optimiser instead of torch AdamW from build_optimizer.
    step_hook (Session 11): object with before(model, step) and after(model, step),
    called around every optimiser step.
    Returns a list of per-step dicts.
    """
    opt = opt if opt is not None else build_optimizer(model, tc)
    gen = torch.Generator().manual_seed(tc.seed)
    T = model.cfg.context_length
    rows = []
    model.train()
    t_start = time.perf_counter()

    for step in range(tc.steps + 1):
        val = float("nan")
        if step % tc.eval_every == 0 or step == tc.steps:
            val = evaluate(model, val_batches, eos_id)
        if step == tc.steps:
            rows.append({"run": name, "step": step, "val_loss": val})
            if writer:
                writer.writerow(rows[-1])
            break

        lr = lr_fn(step) if lr_fn else lr_at(step, tc.steps, tc.lr, tc.warmup, tc.min_lr_frac)
        set_lr(opt, lr)
        t0 = time.perf_counter()
        micros = make_micro_batches(train_data, tc.accum_steps, tc.micro_batch, T, device,
                                    gen, lengths, pad_id)
        if step_hook is not None:
            step_hook.before(model, step)
        out = train_step(model, opt, micros, pad_id, eos_id, reduction, tc.grad_clip)
        if step_hook is not None:
            step_hook.after(model, step)
        row = {"run": name, "step": step, "lr": lr,
               "loss": out["reported"].item(),        # .item() waits for the GPU: step time is real
               "true_loss": out["true"].item(),
               "grad_norm": out["grad_norm"].item(),
               "tokens": int(out["n_tokens"]),
               "step_ms": (time.perf_counter() - t0) * 1000,
               "val_loss": val}
        rows.append(row)
        if writer:
            writer.writerow(row)
        if step % log_every == 0:
            v = f"{val:8.4f}" if val == val else f"{'-':>8}"
            print(f"  {step:>6}{lr:>10.2e}{row['loss']:>9.4f}{row['true_loss']:>9.4f}"
                  f"{row['grad_norm']:>9.3f}{v}{row['step_ms']:>8.0f}"
                  f"{time.perf_counter() - t_start:>7.0f}s")
        if not math.isfinite(row["loss"]):
            print(f"  loss is {row['loss']} at step {step}: stopping this run")
            break
    return rows


LOG_FIELDS = ["run", "step", "lr", "loss", "true_loss", "grad_norm", "tokens", "step_ms", "val_loss"]


def open_log(path):
    f = open(path, "w", newline="")
    w = csv.DictWriter(f, fieldnames=LOG_FIELDS, restval="")
    w.writeheader()
    return f, w


def print_log_header():
    print(f"  {'step':>6}{'lr':>10}{'loss':>9}{'true':>9}{'gnorm':>9}{'val':>8}{'ms':>8}{'time':>8}")
