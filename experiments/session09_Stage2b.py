"""
Session 9 assignment: Stage 2a (shapes, shift, untrained loss) + Stage 2b (padding, boundaries).

Run from the mini_llm folder:
    uv run python -m experiments.session09               # all six sections (~1-2 min)
    uv run python -m experiments.session09 --skip-train  # skip the shift-bug comparison (section 4)

Sections
    1. Every tensor shape between tokens and the scalar loss
    2. The shift, printed as STRINGS (incl. one document boundary)
    3. Untrained loss and perplexity vs ln(V)
    4. Why the shift bug is dangerous: 150 steps, correct vs. no shift
    5. Padding mask: contributing-token counts and three ways to average
    6. Document-boundary mask: loss before and after, and why it differs

Sections 5 and 6 use a model trained for 150 steps (run A), so the losses mean
something. With --skip-train, run A is still trained, only run B is skipped.
"""
import argparse
import math

import numpy as np
import torch
import torch.nn.functional as F

from config import ModelConfig
from data.dataset import TokenData, load_meta, load_tokenizer
from model.model import MiniLLM
from train.loss import lm_loss, per_token_loss

B = 4


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

    inputs = tokens[:, :-1]                          # positions that make a prediction
    targets = tokens[:, 1:]                          # the token that actually came next
    logits_used = logits[:, :-1]                     # drop the last position: nothing follows it
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
    lengths = [T, 192, 128, 64]                       # real tokens per row; the rest is <pad>

    docs = [d for d in split_docs(val_ids, eos) if len(d) > T][:len(lengths)]
    tokens = torch.full((len(lengths), T), pad, dtype=torch.long)
    for r, (d, L) in enumerate(zip(docs, lengths)):
        tokens[r, :L] = torch.from_numpy(d[:L].astype(np.int64))
    tokens = tokens.to(device)

    with torch.no_grad(), autocast(device):
        logits = model(tokens)

    per_tok = per_token_loss(logits, tokens)          # [B, T-1]
    targets = tokens[:, 1:]
    valid = targets != pad
    n_all, n_valid = targets.numel(), int(valid.sum())

    loss_all = per_tok.mean().item()                         # pads counted
    loss_masked = per_tok[valid].mean().item()               # correct
    loss_wrong_den = (per_tok[valid].sum() / n_all).item()   # right numerator, wrong denominator
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

    # Right-padding is safe for attention: causal masking means real tokens never see the pads.
    with torch.no_grad():
        alone = model(tokens[3:4, :lengths[3]])
        padded = model(tokens[3:4])[:, :lengths[3]]
    diff = (alone - padded).abs().max().item()
    print(f"\n  causal check: row 3's logits alone vs. with pads after it, max |diff| = {diff:.2e}")
    print("  (~0: pads sit to the RIGHT, and causal attention only looks LEFT)")


# --------------------------------------------------------------------------- 6
def section_boundaries(model, val_ids, tok, meta, cfg, device, n_windows=16):
    banner("6. DOCUMENT-BOUNDARY MASK")
    eos, T = meta["eos_id"], cfg.context_length
    arr = np.asarray(val_ids)

    # Windows that each contain at least one boundary, at a random position inside.
    eos_pos = np.flatnonzero(arr == eos)
    eos_pos = eos_pos[(eos_pos > T) & (eos_pos < len(arr) - T)]
    rng = np.random.default_rng(0)
    picks = rng.choice(eos_pos, n_windows, replace=False)
    offsets = rng.integers(16, T - 16, n_windows)
    rows = [arr[p - o:p - o + T] for p, o in zip(picks, offsets)]
    tokens = torch.from_numpy(np.stack(rows).astype(np.int64)).to(device)

    with torch.no_grad(), autocast(device):
        logits = model(tokens)

    per_tok = per_token_loss(logits, tokens)          # [B, T-1]
    inputs, targets = tokens[:, :-1], tokens[:, 1:]
    cross = inputs == eos                              # BAD: <eos> -> first token of next doc
    into = targets == eos                              # fine: last token -> <eos>
    rest = ~(cross | into)

    loss_unmasked = per_tok.mean().item()
    loss_masked = per_tok[~cross].mean().item()
    lib_loss, lib_n = lm_loss(logits, tokens, eos_id=eos, mask_boundaries=True)
    assert int(lib_n) == int((~cross).sum()) and abs(lib_loss.item() - loss_masked) < 1e-3

    print(f"  Batch: {n_windows} windows of T={T}, each containing a document boundary")
    print(f"\n  {'':<36}{'positions':>10}{'loss':>10}")
    print(f"  {'without boundary mask':<36}{per_tok.numel():>10,}{loss_unmasked:>10.4f}")
    print(f"  {'with boundary mask':<36}{int((~cross).sum()):>10,}{loss_masked:>10.4f}")
    print(f"\n  Mean loss by kind of position:")
    print(f"  {'<eos> -> next doc (masked)':<36}{int(cross.sum()):>10,}{per_tok[cross].mean().item():>10.4f}")
    print(f"  {'last token -> <eos> (kept)':<36}{int(into.sum()):>10,}{per_tok[into].mean().item():>10.4f}")
    print(f"  {'everything else':<36}{int(rest.sum()):>10,}{per_tok[rest].mean().item():>10.4f}")

    print("\n  First 4 boundaries, with the loss on each pair:")
    print(f"  {'input':<20}->  {'target':<20}{'loss':>8}")
    for r in range(4):
        p = int(cross[r].nonzero()[0])
        for q in (p - 1, p):
            i, t = inputs[r, q].item(), targets[r, q].item()
            note = "  kept" if t == eos else "  MASKED"
            print(f"  {tok_str(tok, i, meta):<20}->  {tok_str(tok, t, meta):<20}{per_tok[r, q].item():>8.3f}{note}")

    share = 1 / meta["avg_tokens_per_doc"]
    print("\n  The masked pairs are among the hardest positions: nothing in document 1 predicts how")
    print("  document 2 starts. Removing them lowers the average slightly. In a normal random batch")
    print(f"  only ~{share:.2%} of positions are boundaries, so the number barely moves; the reason")
    print("  to mask is that these targets teach a relationship that doesn't exist.")
    print("  Note: the loss mask doesn't stop ATTENTION from crossing the boundary. Tokens after")
    print("  <eos> can still attend to document 1; a document-aware attention mask would fix that.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-train", action="store_true", help="skip run B in section 4")
    args = parser.parse_args()

    torch.manual_seed(1337)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    meta, tok = load_meta(), load_tokenizer()
    cfg = ModelConfig(vocab_size=meta["vocab_size"])
    V = cfg.vocab_size
    val_ids = TokenData("val").ids

    model = MiniLLM(cfg).to(device).eval()
    tokens = TokenData("val").get_batch(B, cfg.context_length, device=device)

    logits = section_shapes(model, tokens, V, device)
    section_shift_strings(tokens, tok, meta, val_ids)
    section_untrained_loss(logits, tokens, V)
    model_a = section_shift_bug(cfg, device, TokenData("train"), args.skip_train)
    section_padding(model_a, val_ids, tok, meta, cfg, device)
    section_boundaries(model_a, val_ids, tok, meta, cfg, device)

    print("\nStages 2a + 2b done.")


if __name__ == "__main__":
    main()
