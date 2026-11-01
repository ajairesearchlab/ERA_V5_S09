"""
Causal multi-head self-attention with RoPE.

Uses torch's fused scaled_dot_product_attention, so the [T, T] score matrix
is never materialised in HBM (same "fusion" idea as Session 9, Section 9).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_rope_cache(context_length: int, head_dim: int, theta: float):
    """cos/sin tables of shape [T, head_dim]."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))  # [hd/2]
    pos = torch.arange(context_length).float()                                      # [T]
    freqs = torch.outer(pos, inv_freq)                                              # [T, hd/2]
    emb = torch.cat([freqs, freqs], dim=-1)                                         # [T, hd]
    return emb.cos(), emb.sin()


def apply_rope(x, cos, sin):
    """x: [B, H, T, hd]. Rotates each (i, i+hd/2) pair by a position-dependent angle."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    rotated = torch.cat([-x2, x1], dim=-1)
    return x * cos + rotated * sin


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x, cos, sin):                                   # x: [B, T, D]
        B, T, D = x.shape

        def split_heads(t):                                           # [B, T, D] -> [B, H, T, hd]
            return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q = split_heads(self.q_proj(x))
        k = split_heads(self.k_proj(x))
        v = split_heads(self.v_proj(x))

        cos, sin = cos[:T].to(q.dtype), sin[:T].to(q.dtype)          # [T, hd], broadcasts over B, H
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )                                                             # [B, H, T, hd]
        out = out.transpose(1, 2).contiguous().view(B, T, D)          # merge heads -> [B, T, D]
        return self.o_proj(out)
