"""
MiniLLM: a small decoder-only transformer.

    tokens [B,T]
      -> tok_emb                      [B,T,D]
      -> N x Block (pre-norm):
             h = h + attn(norm(h))
             h = h + ffn(norm(h))      [B,T,D]
      -> final norm                   [B,T,D]   <- "hidden state" fed to the head
      -> lm_head                      [B,T,V]   <- logits

Self-test (random tokens, no data needed):
    python -m model.model
"""
import math

import torch
import torch.nn as nn

from config import ModelConfig
from model.attention import CausalSelfAttention, build_rope_cache
from model.layers import RMSNorm, SwiGLU


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = CausalSelfAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.ffn = SwiGLU(cfg.d_model, cfg.ffn_hidden)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.attn_norm(x), cos, sin)   # residual stream: add, never overwrite
        x = x + self.ffn(self.ffn_norm(x))
        return x


class MiniLLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if cfg.tie_weights:
            self.lm_head.weight = self.tok_emb.weight     # one [V, D] matrix used both ways

        cos, sin = build_rope_cache(cfg.context_length, cfg.head_dim, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # GPT-2 style: shrink the layers that write into the residual stream,
        # so the stream's variance doesn't grow with depth.
        for name, p in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("down.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers))

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def hidden_states(self, tokens):
        """tokens [B,T] -> final normalised hidden state [B,T,D]."""
        T = tokens.shape[1]
        assert T <= self.cfg.context_length, f"T={T} exceeds context_length={self.cfg.context_length}"
        h = self.tok_emb(tokens)
        for block in self.blocks:
            h = block(h, self.rope_cos, self.rope_sin)
        return self.final_norm(h)

    def forward(self, tokens, return_hidden: bool = False):
        h = self.hidden_states(tokens)
        logits = self.lm_head(h)
        return (logits, h) if return_hidden else logits

    # ------------------------------------------------------------------ utils
    def param_breakdown(self):
        """Unique parameter counts per component (a tied matrix is counted once)."""
        seen, parts = set(), {}

        def count(name, module):
            n = 0
            for p in module.parameters():
                if id(p) not in seen:
                    seen.add(id(p))
                    n += p.numel()
            parts[name] = n

        count("token embedding", self.tok_emb)
        count("attention (all layers)", nn.ModuleList(b.attn for b in self.blocks))
        count("ffn / SwiGLU (all layers)", nn.ModuleList(b.ffn for b in self.blocks))
        count("norms", nn.ModuleList([*(b.attn_norm for b in self.blocks),
                                      *(b.ffn_norm for b in self.blocks), self.final_norm]))
        count("lm head", self.lm_head)
        parts["TOTAL"] = sum(parts.values())
        return parts


if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = ModelConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MiniLLM(cfg).to(device)

    print(f"device: {device}")
    for k, v in model.param_breakdown().items():
        print(f"  {k:<28}{v:>12,}")

    tokens = torch.randint(0, cfg.vocab_size, (2, cfg.context_length), device=device)
    with torch.no_grad():
        logits, hidden = model(tokens, return_hidden=True)
    print(f"tokens {tuple(tokens.shape)}  hidden {tuple(hidden.shape)}  logits {tuple(logits.shape)}")
    print("self-test OK")
