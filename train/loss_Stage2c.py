"""
Reusable language-model loss: shift by one, mask what must not count, cross-entropy.

    tokens  [B, T]
    inputs  = tokens[:, :-1]          positions that make a prediction
    targets = tokens[:, 1:]           what actually came next   -> [B, T-1]

Masks (set the target to IGNORE so it contributes nothing):
    pad_id          : any target that is <pad>
    mask_boundaries : any target whose input is <eos>  (predicting doc 2 from doc 1)

F.cross_entropy(..., ignore_index=IGNORE) divides by the number of NON-ignored
targets, i.e. the correct denominator from Session 9 Section 6.

Two ways to compute the same loss:
    lm_loss               materialises all logits [B, T, V] at once
    chunked_lm_loss       never holds more than [chunk, V] logits (Session 9 Section 10)
"""
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

IGNORE = -100


def shift_targets(tokens, pad_id=None, eos_id=None, mask_boundaries=False):
    """tokens [B, T] -> targets [B, T-1], with masked positions set to IGNORE."""
    inputs = tokens[:, :-1]
    targets = tokens[:, 1:].clone()
    if pad_id is not None:
        targets[targets == pad_id] = IGNORE
    if mask_boundaries:
        assert eos_id is not None, "mask_boundaries needs eos_id"
        targets[inputs == eos_id] = IGNORE
    return targets


def lm_loss(logits, tokens, pad_id=None, eos_id=None, mask_boundaries=False):
    """Mean loss over contributing positions, and how many positions contributed."""
    targets = shift_targets(tokens, pad_id, eos_id, mask_boundaries)
    V = logits.size(-1)
    loss = F.cross_entropy(
        logits[:, :-1].reshape(-1, V).float(),
        targets.reshape(-1),
        ignore_index=IGNORE,
    )
    n_tokens = (targets != IGNORE).sum()
    return loss, n_tokens


def per_token_loss(logits, tokens):
    """Unreduced loss for every (row, position): [B, T-1]. For inspection, not training."""
    V = logits.size(-1)
    return F.cross_entropy(
        logits[:, :-1].reshape(-1, V).float(),
        tokens[:, 1:].reshape(-1),
        reduction="none",
    ).view(tokens.size(0), -1)


# --------------------------------------------------------------------------- chunked
def _chunk_ce_sum(h, weight, targets):
    """One chunk: its logits exist only inside this function."""
    logits = F.linear(h, weight)                                   # [chunk, V]
    return F.cross_entropy(logits.float(), targets, ignore_index=IGNORE, reduction="sum")


def chunked_cross_entropy(h, weight, targets, chunk_size=1024):
    """
    h [N, D], weight [V, D], targets [N] -> mean loss over non-ignored targets.

    Same number as F.cross_entropy(F.linear(h, weight), targets), but each chunk's
    logits are thrown away after its loss is computed. checkpoint() makes the
    backward pass RECOMPUTE that chunk's logits when it needs them, so the full
    [N, V] tensor never exists, forward or backward. Cost: one extra matmul per chunk.

    Sum the chunk losses, then divide ONCE by the total valid count. Averaging the
    chunk means instead would weight chunks with many masked targets incorrectly.
    """
    n_valid = (targets != IGNORE).sum().clamp(min=1)
    total = h.new_zeros((), dtype=torch.float32)
    for s in range(0, h.size(0), chunk_size):
        total = total + checkpoint(
            _chunk_ce_sum, h[s:s + chunk_size], weight, targets[s:s + chunk_size],
            use_reentrant=False,
        )
    return total / n_valid


def chunked_lm_loss(model, tokens, chunk_size=1024, pad_id=None, eos_id=None, mask_boundaries=False):
    """Drop-in alternative to model(tokens) + lm_loss, for use in training loops."""
    h = model.hidden_states(tokens)[:, :-1]                        # [B, T-1, D]
    targets = shift_targets(tokens, pad_id, eos_id, mask_boundaries)
    loss = chunked_cross_entropy(h.reshape(-1, h.size(-1)), model.lm_head.weight,
                                 targets.reshape(-1), chunk_size)
    return loss, (targets != IGNORE).sum()
