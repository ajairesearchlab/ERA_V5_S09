"""
RMSNorm and SwiGLU FFN (Session 9, Section 2).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """x / rms(x) * g   -- no mean-centring, no bias."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # g, D parameters

    def forward(self, x):                                          # [B, T, D]
        x_f = x.float()                                            # normalise in fp32 for stability
        rms_inv = torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x_f * rms_inv).type_as(x) * self.weight


class SwiGLU(nn.Module):
    """down( silu(gate(h)) * up(h) )  -- three matrices instead of two."""

    def __init__(self, d_model: int, hidden: int):
        super().__init__()
        self.gate = nn.Linear(d_model, hidden, bias=False)  # "how much"
        self.up = nn.Linear(d_model, hidden, bias=False)    # "what"
        self.down = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x):                                   # [B, T, D]
        return self.down(F.silu(self.gate(x)) * self.up(x))  # [B, T, hidden] -> [B, T, D]
