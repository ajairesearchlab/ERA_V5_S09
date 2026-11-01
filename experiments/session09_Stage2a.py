"""
Session 9 assignment, Stage 2a: shapes, the next-token shift, and the untrained loss.

Run from the mini_llm folder:
    uv run python -m experiments.session09               # all four sections (~1-2 min)
    uv run python -m experiments.session09 --skip-train  # sections 1-3 only (seconds)

Sections
    1. Every tensor shape between tokens and the scalar loss
    2. The shift, printed as STRINGS (incl. one document boundary)
    3. Untrained loss and perplexity vs ln(V)
    4. Why the shift bug is dangerous: a short training run, correct vs. no shift
"""
import argparse
import math

import torch
import torch.nn.functional as F

from config import ModelConfig
from data.dataset import TokenData, load_meta, load_tokenizer
from model.model import MiniLLM

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

    # The identity that defines a correct shift: target[t] == input[t+1]
    assert torch.equal(inputs[:, 1:], targets[:, :-1]), "shift identity violated"
    print("\n  check: targets[:, t] == inputs[:, t+1] for every t  -> OK")
    return logits


# --------------------------------------------------------------------------- 2
def section_shift_strings(tokens, tok, meta, n_show=12):
    banner("2. THE SHIFT, AS STRINGS (each row: this input predicts that target)")
    inputs, targets = tokens[:, :-1], tokens[:, 1:]

    def table(row, start, stop, note_fn=None):
        print(f"  {'pos':>4}  {'input':<20}->  {'target':<20}")
        for p in range(start, stop):
            i, t = inputs[row, p].item(), targets[row, p].item()
            note = note_fn(i, t) if note_fn else ""
            print(f"  {p:>4}  {tok_str(tok, i, meta):<20}->  {tok_str(tok, t, meta):<20}{note}")

    print("\nRow 0, first positions:")
    table(0, 0, n_show)

    eos = meta["eos_id"]
    hits = (inputs == eos).nonzero()
    if len(hits):
        row, p = hits[0].tolist()
        lo, hi = max(0, p - 4), min(inputs.shape[1], p + 4)

        def note(i, t):
            if t == eos:
                return "  <- fine: learns to END a document"
            if i == eos:
                return "  <- BAD: predicts doc 2 from doc 1 (mask in Stage 2b)"
            return ""

        print(f"\nRow {row}, around the document boundary at position {p}:")
        table(row, lo, hi, note)
    else:
        print("\n(no <eos> inside this batch's inputs; rerun for another sample)")


# --------------------------------------------------------------------------- 3
def section_untrained_loss(logits, tokens, V):
    banner("3. UNTRAINED LOSS AND PERPLEXITY")
    flat_logits = logits[:, :-1].reshape(-1, V).float()
    flat_targets = tokens[:, 1:].reshape(-1)

    loss = F.cross_entropy(flat_logits, flat_targets)
    # Same number by hand: -log softmax at the correct class, averaged
    manual = -(flat_logits.log_softmax(-1).gather(1, flat_targets[:, None])).mean()

    sigma = flat_logits.std().item()
    uniform = math.log(V)
    predicted = uniform + sigma ** 2 / 2   # E[log Z] for Gaussian logits, E[z_y] ~ 0

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
    return history


def section_shift_bug(cfg, device):
    banner("4. WHY THE SHIFT MATTERS: 150 training steps, correct vs. no shift")
    train = TokenData("train")
    print("  training run A (correct shift) ...")
    good = short_train(cfg, train, device, shift=True)
    print("  training run B (no shift: predict your own token) ...")
    bad = short_train(cfg, train, device, shift=False)

    print(f"\n  {'step':>6}  {'A correct':>12}  {'B no shift':>12}")
    for (s, a), (_, b) in zip(good, bad):
        print(f"  {s:>6}  {a:>12.4f}  {b:>12.4f}")
    print("\n  Both start near ln(V): the bug is invisible at initialisation.")
    print("  B falls far faster because copying the input is trivial, not because it learned language.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-train", action="store_true", help="skip section 4")
    args = parser.parse_args()

    torch.manual_seed(1337)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    meta, tok = load_meta(), load_tokenizer()
    cfg = ModelConfig(vocab_size=meta["vocab_size"])
    V = cfg.vocab_size

    model = MiniLLM(cfg).to(device).eval()
    tokens = TokenData("val").get_batch(B, cfg.context_length, device=device)

    logits = section_shapes(model, tokens, V, device)
    section_shift_strings(tokens, tok, meta)
    section_untrained_loss(logits, tokens, V)
    if not args.skip_train:
        section_shift_bug(cfg, device)

    print("\nStage 2a done.")


if __name__ == "__main__":
    main()
