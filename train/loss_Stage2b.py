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
"""
import torch
import torch.nn.functional as F

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
