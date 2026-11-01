"""
Session 9 assignment, Part 1: the loss harness (Stages 2a + 2b + 2c).

Run from the mini_llm folder:
    uv run python -m experiments.session09 | Tee-Object -FilePath s09_stage2c.txt
    uv run python -m experiments.session09 --skip-train      # skip run B in section 4
    uv run python -m experiments.session09 --mem-batch 32    # smaller memory test if VRAM is tight

Sections
    1. Every tensor shape between tokens and the scalar loss
    2. The shift, printed as STRINGS (incl. one document boundary)
    3. Untrained loss and perplexity vs ln(V)
    4. Why the shift bug is dangerous: 150 steps, correct vs. no shift
    5. Padding mask: contributing-token counts and three ways to average
    6. Document-boundary mask: loss before and after, and why it differs
    7. Tied vs untied output head: parameter counts
    8. Ordinary vs chunked cross-entropy: same loss, peak GPU memory
    9. Summary: the Part 1 numbers for the write-up

Sections 5, 6 and 8 use the model trained for 150 steps in section 4 (run A).
"""
import argparse
import dataclasses
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from config import ModelConfig
from data.dataset import TokenData, load_meta, load_tokenizer
from model.model import MiniLLM
from train.loss import IGNORE, chunked_cross_entropy, lm_loss, per_token_loss, shift_targets

B = 4
MiB = 2 ** 20


def banner(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def show(name, t, meaning):
    shape = "scalar" if t.dim() == 0 else str(tuple(t.shape))
    print(f"  {name:<26}{shape:<20}{meaning}")


def tok_str(tok, i, meta):
    """Readable single token: special tokens by name, everything else as repr(text)."""
    if i == meta["eos_id"]:
        return "<eos>"
    if i == meta["pad_id"]:
        return "<pad>"
    return repr(tok.decode([i]))


def autocast(device):
    return torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda")


def boundary_note(i, t, eos):
    if t == eos:
        return "  <- fine: learns to END a document"
    if i == eos:
        return "  <- BAD: predicts doc 2 from doc 1 (masked in section 6)"
    return ""


# --------------------------------------------------------------------------- 1
def section_shapes(model, tokens, V, device):
    banner("1. SHAPES: tokens -> logits -> shift -> flatten -> scalar")
    with torch.no_grad(), autocast(device):
        logits = model(tokens)                       # [B, T, V]

    inputs = tokens[:, :-1]
    targets = tokens[:, 1:]
    logits_used = logits[:, :-1]
    flat_logits = logits_used.reshape(-1, V).float()
    flat_targets = targets.reshape(-1)
    loss = F.cross_entropy(flat_logits, flat_targets)

    show("tokens", tokens, "B sequences x T token ids")
    show("logits", logits, "B x T positions x V scores")
    show("inputs = tokens[:, :-1]", inputs, "T-1: the last token predicts nothing we have")
    show("targets = tokens[:, 1:]", targets, "T-1: the first token is predicted by nothing")
    show("logits[:, :-1]", logits_used, "scores aligned with targets")
    show("flat logits", flat_logits, "B*(T-1) rows, one classification each")
    show("flat targets", flat_targets, "one correct class id per row")
    show("loss", loss, "mean over all B*(T-1) rows")

    assert torch.equal(inputs[:, 1:], targets[:, :-1]), "shift identity violated"
    print("\n  check: targets[:, t] == inputs[:, t+1] for every t  -> OK")
    return logits


# --------------------------------------------------------------------------- 2
def section_shift_strings(tokens, tok, meta, val_ids, n_show=12):
    banner("2. THE SHIFT, AS STRINGS (each row: this input predicts that target)")
    inputs, targets = tokens[:, :-1], tokens[:, 1:]
    eos = meta["eos_id"]

    print("\nRow 0, first positions:")
    print(f"  {'pos':>4}  {'input':<20}->  {'target':<20}")
    for p in range(n_show):
        i, t = inputs[0, p].item(), targets[0, p].item()
        print(f"  {p:>4}  {tok_str(tok, i, meta):<20}->  {tok_str(tok, t, meta):<20}")

    hits = (inputs == eos).nonzero()
    if len(hits):
        row, p = hits[0].tolist()
        ins = inputs[row, max(0, p - 5):p + 6].tolist()
        tgs = targets[row, max(0, p - 5):p + 6].tolist()
        print(f"\nRow {row}, around the document boundary at position {p}:")
    else:
        # The batch is seeded, so rerunning gives the same windows. Instead, cut a
        # window from the val data that is guaranteed to contain a boundary.
        p = int(np.flatnonzero(np.asarray(val_ids[:200_000]) == eos)[0])
        window = np.asarray(val_ids[p - 6:p + 7]).astype(np.int64).tolist()
        ins, tgs = window[:-1], window[1:]
        print("\nNo <eos> in this batch, so here is the first boundary in the val data:")

    print(f"  {'input':<20}->  {'target':<20}")
    for i, t in zip(ins, tgs):
        print(f"  {tok_str(tok, i, meta):<20}->  {tok_str(tok, t, meta):<20}{boundary_note(i, t, eos)}")


# --------------------------------------------------------------------------- 3
def section_untrained_loss(logits, tokens, V):
    banner("3. UNTRAINED LOSS AND PERPLEXITY")
    flat_logits = logits[:, :-1].reshape(-1, V).float()
    flat_targets = tokens[:, 1:].reshape(-1)

    loss = F.cross_entropy(flat_logits, flat_targets)
    manual = -(flat_logits.log_softmax(-1).gather(1, flat_targets[:, None])).mean()

    sigma = flat_logits.std().item()
    uniform = math.log(V)
    predicted = uniform + sigma ** 2 / 2

    print(f"  positions counted         {flat_targets.numel():,}")
    print(f"  loss (F.cross_entropy)    {loss.item():.4f}")
    print(f"  loss (by hand)            {manual.item():.4f}")
    print(f"  perplexity = exp(loss)    {math.exp(loss.item()):,.0f}")
    print(f"  ln(V), uniform guess      {uniform:.4f}   (perplexity {V:,})")
    print(f"  ln(V) + sigma^2/2         {predicted:.4f}   (sigma = logit std {sigma:.3f})")

    gap = loss.item() - uniform
    verdict = "OK" if abs(gap) < 0.5 else "SUSPICIOUS - check the shift before going further"
    print(f"\n  loss - ln(V) = {gap:+.4f}  -> {verdict}")
    return loss.item()


# --------------------------------------------------------------------------- 4
def short_train(cfg, train, device, shift, steps=150, batch=16, lr=6e-4):
    torch.manual_seed(0)
    model = MiniLLM(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    gen = torch.Generator().manual_seed(0)   # identical batches for both runs
    V, history = cfg.vocab_size, []

    for step in range(steps + 1):
        tokens = train.get_batch(batch, cfg.context_length, device=device, generator=gen)
        with autocast(device):
            logits = model(tokens)
        if shift:
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, V).float(), tokens[:, 1:].reshape(-1))
        else:  # THE BUG: position t is asked to predict its own token
            loss = F.cross_entropy(logits.reshape(-1, V).float(), tokens.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 25 == 0:
            history.append((step, loss.item()))
    model.zero_grad(set_to_none=True)
    return history, model.eval()


def section_shift_bug(cfg, device, train, skip_b):
    banner("4. WHY THE SHIFT MATTERS: 150 training steps, correct vs. no shift")
    print("  training run A (correct shift) ...")
    good, model_a = short_train(cfg, train, device, shift=True)
    if skip_b:
        print(f"  run A final loss {good[-1][1]:.4f}  (run B skipped: --skip-train)")
        return model_a

    print("  training run B (no shift: predict your own token) ...")
    bad, _ = short_train(cfg, train, device, shift=False)
    print(f"\n  {'step':>6}  {'A correct':>12}  {'B no shift':>12}")
    for (s, a), (_, b) in zip(good, bad):
        print(f"  {s:>6}  {a:>12.4f}  {b:>12.4f}")
    print("\n  Both start near ln(V): the bug is invisible at initialisation.")
    print("  B falls far faster because copying the input is trivial, not because it learned language.")
    return model_a


# --------------------------------------------------------------------------- 5
def split_docs(val_ids, eos, limit=300_000):
    """val starts at a document start, so cutting at every <eos> gives whole documents."""
    arr = np.asarray(val_ids[:limit])
    docs, start = [], 0
    for e in np.flatnonzero(arr == eos):
        docs.append(arr[start:e + 1])
        start = e + 1
    return docs


def section_padding(model, val_ids, tok, meta, cfg, device):
    banner("5. PADDING MASK")
    pad, eos, T = meta["pad_id"], meta["eos_id"], cfg.context_length
    lengths = [T, 192, 128, 64]

    docs = [d for d in split_docs(val_ids, eos) if len(d) > T][:len(lengths)]
    tokens = torch.full((len(lengths), T), pad, dtype=torch.long)
    for r, (d, L) in enumerate(zip(docs, lengths)):
        tokens[r, :L] = torch.from_numpy(d[:L].astype(np.int64))
    tokens = tokens.to(device)

    with torch.no_grad(), autocast(device):
        logits = model(tokens)

    per_tok = per_token_loss(logits, tokens)
    targets = tokens[:, 1:]
    valid = targets != pad
    n_all, n_valid = targets.numel(), int(valid.sum())

    loss_all = per_tok.mean().item()
    loss_masked = per_tok[valid].mean().item()
    loss_wrong_den = (per_tok[valid].sum() / n_all).item()
    loss_pad_only = per_tok[~valid].mean().item()

    lib_loss, lib_n = lm_loss(logits, tokens, pad_id=pad)
    assert int(lib_n) == n_valid and abs(lib_loss.item() - loss_masked) < 1e-3, "lm_loss disagrees"

    print("  Batch: 4 real documents cut to different lengths, right-padded with <pad> to T=256")
    for r, L in enumerate(lengths):
        print(f"    row {r}: {L:>3} real tokens + {T - L:>3} pads -> {L - 1:>3} real targets")

    print("\n  Row 3 at the edge of its real text:")
    print(f"  {'pos':>4}  {'input':<20}->  {'target':<20}{'counted?'}")
    for p in range(lengths[3] - 4, lengths[3] + 2):
        i, t = tokens[3, p].item(), tokens[3, p + 1].item()
        counted = "yes" if t != pad else "NO (target is <pad>)"
        print(f"  {p:>4}  {tok_str(tok, i, meta):<20}->  {tok_str(tok, t, meta):<20}{counted}")

    print(f"\n  {'how the loss is averaged':<44}{'positions':>10}{'loss':>10}")
    print(f"  {'A. everything, pads included':<44}{n_all:>10,}{loss_all:>10.4f}")
    print(f"  {'B. pads masked (correct)':<44}{n_valid:>10,}{loss_masked:>10.4f}")
    print(f"  {'C. pads masked, but divided by B*(T-1)':<44}{n_all:>10,}{loss_wrong_den:>10.4f}")
    print(f"  {'   (loss on the pad targets alone)':<44}{n_all - n_valid:>10,}{loss_pad_only:>10.4f}")
    print(f"\n  C = B x {n_valid}/{n_all} = B x {n_valid / n_all:.3f}: the loss shrinks with the share of real tokens,")
    print("  which changes from batch to batch, so C is not comparable across batches.")
    print("  A is high here because this model never saw <pad> as a target. If you TRAINED with")
    print("  pads included, it would learn '<pad> follows <pad>' and A would look better than it is.")

    with torch.no_grad():
        alone = model(tokens[3:4, :lengths[3]])
        padded = model(tokens[3:4])[:, :lengths[3]]
    diff = (alone - padded).abs().max().item()
    print(f"\n  causal check: row 3's logits alone vs. with pads after it, max |diff| = {diff:.2e}")
    print("  (~0: pads sit to the RIGHT, and causal attention only looks LEFT)")
    return {"n_all": n_all, "n_valid": n_valid, "loss_all": loss_all, "loss_masked": loss_masked}


# --------------------------------------------------------------------------- 6
def section_boundaries(model, val_ids, tok, meta, cfg, device, n_windows=16):
    banner("6. DOCUMENT-BOUNDARY MASK")
    eos, T = meta["eos_id"], cfg.context_length
    arr = np.asarray(val_ids)

    eos_pos = np.flatnonzero(arr == eos)
    eos_pos = eos_pos[(eos_pos > T) & (eos_pos < len(arr) - T)]
    rng = np.random.default_rng(0)
    picks = rng.choice(eos_pos, n_windows, replace=False)
    offsets = rng.integers(16, T - 16, n_windows)
    rows = [arr[p - o:p - o + T] for p, o in zip(picks, offsets)]
    tokens = torch.from_numpy(np.stack(rows).astype(np.int64)).to(device)

    with torch.no_grad(), autocast(device):
        logits = model(tokens)

    per_tok = per_token_loss(logits, tokens)
    inputs, targets = tokens[:, :-1], tokens[:, 1:]
    cross = inputs == eos                              # <eos> -> first token of next doc
    into = targets == eos                              # last token -> <eos>
    rest = ~(cross | into)

    loss_unmasked = per_tok.mean().item()
    loss_masked = per_tok[~cross].mean().item()
    lib_loss, lib_n = lm_loss(logits, tokens, eos_id=eos, mask_boundaries=True)
    assert int(lib_n) == int((~cross).sum()) and abs(lib_loss.item() - loss_masked) < 1e-3

    cross_mean, rest_mean = per_tok[cross].mean().item(), per_tok[rest].mean().item()

    print(f"  Batch: {n_windows} windows of T={T}, each containing a document boundary")
    print(f"\n  {'':<36}{'positions':>10}{'loss':>10}")
    print(f"  {'without boundary mask':<36}{per_tok.numel():>10,}{loss_unmasked:>10.4f}")
    print(f"  {'with boundary mask':<36}{int((~cross).sum()):>10,}{loss_masked:>10.4f}")
    print(f"\n  Mean loss by kind of position:")
    print(f"  {'<eos> -> next doc (masked)':<36}{int(cross.sum()):>10,}{cross_mean:>10.4f}")
    print(f"  {'last token -> <eos> (kept)':<36}{int(into.sum()):>10,}{per_tok[into].mean().item():>10.4f}")
    print(f"  {'everything else':<36}{int(rest.sum()):>10,}{rest_mean:>10.4f}")

    print("\n  First 4 boundaries, with the loss on each pair:")
    print(f"  {'input':<20}->  {'target':<20}{'loss':>8}")
    for r in range(4):
        p = int(cross[r].nonzero()[0])
        for q in (p - 1, p):
            i, t = inputs[r, q].item(), targets[r, q].item()
            note = "  kept" if t == eos else "  MASKED"
            print(f"  {tok_str(tok, i, meta):<20}->  {tok_str(tok, t, meta):<20}{per_tok[r, q].item():>8.3f}{note}")

    # Explanation driven by the measured numbers, not assumed in advance.
    change = loss_masked - loss_unmasked
    if cross_mean < rest_mean:
        print(f"\n  Here the masked pairs are EASIER than average ({cross_mean:.2f} vs {rest_mean:.2f}).")
        print("  Documents often start the same way ('The', a title word), so even early in training")
        print(f"  the model has a generic start-of-document guess. Masking them RAISES the loss by {change:+.4f}.")
    else:
        print(f"\n  Here the masked pairs are HARDER than average ({cross_mean:.2f} vs {rest_mean:.2f}).")
        print(f"  Masking them LOWERS the loss by {change:+.4f}.")
    share = 1 / meta["avg_tokens_per_doc"]
    print(f"  In a normal random batch only ~{share:.2%} of positions are boundaries, so the average barely")
    print("  moves either way. The size of the change isn't the point. The reason to mask is that the")
    print("  target asks the model to predict document 2 from document 1, a relationship that doesn't exist.")
    print("  Note: the loss mask doesn't stop ATTENTION from crossing the boundary. Tokens after")
    print("  <eos> can still attend to document 1; a document-aware attention mask would fix that.")
    return {"before": loss_unmasked, "after": loss_masked}


# --------------------------------------------------------------------------- 7
def section_tying(cfg):
    banner("7. TIED VS UNTIED OUTPUT HEAD")
    untied = MiniLLM(cfg)
    tied = MiniLLM(dataclasses.replace(cfg, tie_weights=True))
    assert tied.lm_head.weight is tied.tok_emb.weight, "tying failed"

    a, b = untied.param_breakdown(), tied.param_breakdown()
    print(f"  {'component':<28}{'untied':>14}{'tied':>14}")
    for k in a:
        print(f"  {k:<28}{a[k]:>14,}{b[k]:>14,}")

    V, D = cfg.vocab_size, cfg.d_model
    saved = a["TOTAL"] - b["TOTAL"]
    print(f"\n  saved by tying: {saved:,} = V x D = {V:,} x {D}  ({saved / a['TOTAL']:.1%} of the untied model)")
    print("  (tied: the head reuses the embedding matrix, so it adds 0 new parameters)")
    print(f"\n  Same arithmetic at the course's V5 size: 131,072 x 4,096 = {131_072 * 4_096:,} per matrix.")
    print("  In a small model like ours the head is a big share of all parameters; that share")
    print("  is why tying is popular at small scale.")
    del untied, tied
    return {"untied": a["TOTAL"], "tied": b["TOTAL"]}


# --------------------------------------------------------------------------- 8
def section_chunked(model, meta, cfg, device, mem_batch, chunk_sizes=(4096, 1024, 256)):
    banner("8. ORDINARY VS CHUNKED CROSS-ENTROPY (output head + loss, forward + backward)")
    if device != "cuda":
        print("  skipped: needs a CUDA GPU to measure memory")
        return None

    T, D, V = cfg.context_length, cfg.d_model, cfg.vocab_size
    tokens = TokenData("val").get_batch(mem_batch, T, device=device,
                                        generator=torch.Generator().manual_seed(1))
    with torch.no_grad(), autocast(device):
        h_full = model.hidden_states(tokens)
    # Isolate the part we're comparing: hidden states -> head -> loss.
    h = h_full[:, :-1].reshape(-1, D).float().detach().requires_grad_(True)    # [N, D]
    targets = shift_targets(tokens, meta["pad_id"], meta["eos_id"], mask_boundaries=True).reshape(-1)
    W = model.lm_head.weight                                                   # [V, D]
    N = h.size(0)
    del h_full

    def ordinary():
        with autocast(device):
            logits = F.linear(h, W)                                            # [N, V] all at once
            return F.cross_entropy(logits.float(), targets, ignore_index=IGNORE)

    def chunked(c):
        def fn():
            with autocast(device):
                return chunked_cross_entropy(h, W, targets, chunk_size=c)
        return fn

    def run(fn):
        h.grad, W.grad = None, None
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        t0 = time.perf_counter()
        loss = fn()
        loss.backward()
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000
        peak = torch.cuda.max_memory_allocated() - base
        return loss.item(), peak, ms, h.grad.clone(), W.grad.clone()

    # Warm-up so one-time CUDA/cuBLAS allocations don't land in the first measurement.
    run(ordinary)
    run(chunked(1024))

    try:
        ref_loss, ref_peak, ref_ms, ref_gh, ref_gw = run(ordinary)
    except torch.cuda.OutOfMemoryError:
        print(f"  ordinary CE ran out of memory at --mem-batch {mem_batch}; rerun with a smaller value")
        return None

    print(f"  hidden states h: [N, D] = [{N:,}, {D}]   (batch {mem_batch} x {T - 1} positions)")
    print(f"  full logits    : [N, V] = [{N:,}, {V:,}] = {N * V * 4 / MiB:,.0f} MiB in fp32")
    print(f"\n  {'method':<20}{'logits held':>14}{'loss':>10}{'peak MiB':>11}{'vs ordinary':>13}"
          f"{'time ms':>10}{'max grad diff':>15}")
    print(f"  {'ordinary':<20}{N:>14,}{ref_loss:>10.4f}{ref_peak / MiB:>11.1f}{'1.0x':>13}{ref_ms:>10.1f}{'-':>15}")

    rows = {}
    for c in chunk_sizes:
        loss, peak, ms, gh, gw = run(chunked(c))
        gdiff = max((gh - ref_gh).abs().max().item() / ref_gh.abs().max().item(),
                    (gw - ref_gw).abs().max().item() / ref_gw.abs().max().item())
        rows[c] = (loss, peak)
        print(f"  {f'chunked {c}':<20}{c:>14,}{loss:>10.4f}{peak / MiB:>11.1f}"
              f"{f'{ref_peak / peak:.1f}x less':>13}{ms:>10.1f}{gdiff:>15.1e}")

    l1024, p1024 = rows[1024]
    print("\n  'logits held' = rows of logits in memory at once. 'max grad diff' is relative to the")
    print("  largest gradient; values around 1e-3 or below are bf16 rounding, not a different result.")
    print(f"\n  Loss: ordinary {ref_loss:.4f} vs chunked {l1024:.4f}  (difference {abs(ref_loss - l1024):.1e})")
    print(f"  Peak memory: {ref_peak / MiB:.1f} MiB -> {p1024 / MiB:.1f} MiB at chunk 1024 = {ref_peak / p1024:.1f}x less")
    print("  Time: chunked is slower because backward recomputes each chunk's logits (one extra matmul).")
    print("  The objective is unchanged; only what exists in memory at once is different.")

    val_ppl = math.exp(ref_loss)
    print(f"\n  Bonus: this loss is run A's validation loss on {int((targets != IGNORE).sum()):,} positions,")
    print(f"  perplexity exp({ref_loss:.4f}) = {val_ppl:,.0f}  (untrained was ~{cfg.vocab_size:,})")
    return {"ordinary": ref_peak, "chunk1024": p1024, "loss_ord": ref_loss, "loss_chunk": l1024,
            "val_ppl": val_ppl, "mem_batch": mem_batch}


# --------------------------------------------------------------------------- 9
def section_summary(r, V):
    banner("9. SUMMARY: PART 1 NUMBERS FOR THE WRITE-UP")
    pad, bd, ty, mem = r["padding"], r["boundary"], r["tying"], r["memory"]
    lines = [
        ("1. shapes", "tokens [4,256] -> hidden [4,256,384] -> logits [4,256,8192] -> scalar"),
        ("2. shift check", "targets[t] == inputs[t+1]: OK (strings printed in section 2)"),
        ("3. padding mask", f"contributing tokens {pad['n_all']:,} -> {pad['n_valid']:,}; "
                            f"loss {pad['loss_all']:.4f} -> {pad['loss_masked']:.4f}"),
        ("4. boundary mask", f"loss {bd['before']:.4f} -> {bd['after']:.4f} (16 boundaries masked)"),
        ("5. perplexity", f"untrained {math.exp(r['untrained']):,.0f} vs V = {V:,}"
                          + (f"; after 150 steps {mem['val_ppl']:,.0f}" if mem else "")),
        ("6. head params", f"untied {ty['untied']:,} vs tied {ty['tied']:,}"),
    ]
    if mem:
        lines.append(("7. peak memory", f"ordinary {mem['ordinary'] / MiB:.1f} MiB vs chunked(1024) "
                                        f"{mem['chunk1024'] / MiB:.1f} MiB = {mem['ordinary'] / mem['chunk1024']:.1f}x "
                                        f"(batch {mem['mem_batch']})"))
    else:
        lines.append(("7. peak memory", "not measured (no GPU or out of memory)"))
    for k, v in lines:
        print(f"  {k:<18}{v}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-train", action="store_true", help="skip run B in section 4")
    parser.add_argument("--mem-batch", type=int, default=64, help="batch size for the memory test")
    args = parser.parse_args()

    torch.manual_seed(1337)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    meta, tok = load_meta(), load_tokenizer()
    cfg = ModelConfig(vocab_size=meta["vocab_size"])
    V = cfg.vocab_size
    val_ids = TokenData("val").ids

    model = MiniLLM(cfg).to(device).eval()
    tokens = TokenData("val").get_batch(B, cfg.context_length, device=device)

    results = {}
    logits = section_shapes(model, tokens, V, device)
    section_shift_strings(tokens, tok, meta, val_ids)
    results["untrained"] = section_untrained_loss(logits, tokens, V)
    del model, logits

    model_a = section_shift_bug(cfg, device, TokenData("train"), args.skip_train)
    results["padding"] = section_padding(model_a, val_ids, tok, meta, cfg, device)
    results["boundary"] = section_boundaries(model_a, val_ids, tok, meta, cfg, device)
    results["tying"] = section_tying(cfg)
    results["memory"] = section_chunked(model_a, meta, cfg, device, args.mem_batch)
    section_summary(results, V)

    print("\nPart 1 done.")


if __name__ == "__main__":
    main()
