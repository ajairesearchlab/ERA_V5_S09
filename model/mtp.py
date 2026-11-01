"""
MiniLLM with extra output heads for multi-token prediction (Session 9, Section 13).

    shared trunk -> h [B, T, D] --+--> head 1 (lm_head)        predicts token t+1
                                  +--> head 2 (extra_heads[0]) predicts token t+2
                                  ...

Each extra head is a plain [V, D] linear layer on the SAME hidden state: the
simplest form of MTP. (DeepSeek-V3 adds a small transformer block per extra head.)

forward() still returns head-1 logits only, so generation code works unchanged.
"""
import torch.nn as nn

from model.model import MiniLLM


class MiniLLMMTP(MiniLLM):
    def __init__(self, cfg, n_future: int = 2):
        # The trunk and head 1 are built first and consume the RNG exactly like
        # MiniLLM, so with the same seed they start identical to a baseline model.
        super().__init__(cfg)
        self.n_future = n_future
        self.extra_heads = nn.ModuleList(
            nn.Linear(cfg.d_model, cfg.vocab_size, bias=False) for _ in range(n_future - 1)
        )
        for head in self.extra_heads:
            nn.init.normal_(head.weight, mean=0.0, std=0.02)

    def head_weights(self):
        """[W_1, W_2, ...], each [V, D]; W_k predicts token t+k."""
        return [self.lm_head.weight] + [h.weight for h in self.extra_heads]

    def param_breakdown(self):
        parts = super().param_breakdown()
        total = parts.pop("TOTAL")
        parts["extra heads"] = sum(p.numel() for p in self.extra_heads.parameters())
        parts["TOTAL"] = total + parts["extra heads"]
        return parts
