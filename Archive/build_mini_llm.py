"""
One-file installer for the mini_llm Stage 1 project.

Run on the Windows laptop, in the folder where you want the project:
    python build_mini_llm.py

It creates:
    mini_llm/            (all Stage 1 files)
    mini_llm_stage1.zip  (the same files, zipped)
Existing files are not overwritten unless you pass --force.
"""
import shutil
import sys
from pathlib import Path

FILES = {}

FILES["config.py"] = r'''"""
Central configuration for the mini LLM project.

Everything that later sessions may want to change lives here, so experiments
can override fields instead of editing model code.
"""
from dataclasses import dataclass, asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "prepared"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"


@dataclass
class DataConfig:
    # Hugging Face source (streamed, never fully downloaded)
    hf_dataset: str = "HuggingFaceFW/fineweb-edu"
    hf_subset: str = "sample-10BT"
    text_field: str = "text"

    # How much data we keep locally
    train_tokens: int = 20_000_000      # ~40 MB on disk as uint16
    val_tokens: int = 500_000
    tokenizer_train_docs: int = 50_000  # docs used to learn the BPE merges

    # Tokenizer
    vocab_size: int = 8192
    pad_token: str = "<pad>"            # id 0
    eos_token: str = "<eos>"            # id 1, appended after every document

    seed: int = 1337


@dataclass
class ModelConfig:
    vocab_size: int = 8192
    context_length: int = 256           # T
    d_model: int = 384                  # D
    n_layers: int = 8
    n_heads: int = 6                    # head_dim = 384 / 6 = 64
    ffn_hidden: int = 1024              # SwiGLU width (~ 8/3 * D, rounded to 64)
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6
    dropout: float = 0.0
    tie_weights: bool = False           # Session 9 compares tied vs untied

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, "d_model must divide by n_heads"
        assert (self.d_model // self.n_heads) % 2 == 0, "head_dim must be even for RoPE"

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def to_dict(self):
        return asdict(self)
'''

FILES["requirements.txt"] = r'''# PyTorch is installed SEPARATELY (see README step 3) because the
# RTX 5060 (Blackwell, sm_120) needs a CUDA 12.8+ build.
numpy>=1.26
datasets>=2.19
huggingface_hub>=0.23
tokenizers>=0.19
tqdm>=4.66
'''

FILES[".gitignore"] = r'''.venv/
__pycache__/
*.pyc
data/prepared/
checkpoints/
.ipynb_checkpoints/
'''

FILES["check_env.py"] = r'''"""
Step 1 of Stage 1: verify Python, PyTorch, CUDA and the RTX 5060 work together.

Run:  python check_env.py
"""
import platform
import sys


def main():
    print(f"Python      : {sys.version.split()[0]}  ({platform.system()} {platform.release()})")

    try:
        import torch
    except ImportError:
        sys.exit("FAIL: PyTorch is not installed. See README step 3.")

    print(f"PyTorch     : {torch.__version__}")
    print(f"Built CUDA  : {torch.version.cuda}")

    if not torch.cuda.is_available():
        sys.exit("FAIL: CUDA not available. Either the NVIDIA driver is missing or a "
                 "CPU-only PyTorch was installed. Reinstall with the cu128 command in the README.")

    name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    arch = f"sm_{major}{minor}"
    arch_list = torch.cuda.get_arch_list()
    free, total = torch.cuda.mem_get_info(0)

    print(f"GPU         : {name}")
    print(f"Compute cap.: {arch}")
    print(f"VRAM        : {free / 2**30:.2f} GiB free / {total / 2**30:.2f} GiB total")
    print(f"Build archs : {', '.join(arch_list)}")

    if arch not in arch_list and f"compute_{major}{minor}" not in arch_list:
        sys.exit(f"FAIL: this PyTorch build has no kernels for {arch}. "
                 "Install a CUDA 12.8+ build (README step 3).")

    # Real work on the GPU in bf16, the precision we'll train in.
    a = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
    c = (a @ b).float()
    torch.cuda.synchronize()
    print(f"bf16 matmul : OK (checksum {c.abs().mean().item():.3f})")
    print(f"bf16 support: {torch.cuda.is_bf16_supported()}")

    # Fused attention kernel we use in the model
    q = torch.randn(1, 6, 256, 64, device="cuda", dtype=torch.bfloat16)
    torch.nn.functional.scaled_dot_product_attention(q, q, q, is_causal=True)
    torch.cuda.synchronize()
    print("SDPA attn   : OK")

    for mod in ("numpy", "datasets", "tokenizers", "tqdm"):
        try:
            m = __import__(mod)
            print(f"{mod:<12}: {getattr(m, '__version__', 'ok')}")
        except ImportError:
            print(f"{mod:<12}: MISSING -> pip install -r requirements.txt")

    print("\nPASS: environment is ready.")


if __name__ == "__main__":
    main()
'''

FILES["stage1_check.py"] = r'''"""
Step 3 of Stage 1: push one REAL batch of FineWeb-Edu through the untrained
model and print every tensor on the way from text to logits.

Run:  python stage1_check.py

Before running, try to predict each shape yourself:
    tokens, embeddings, hidden after block 1, final hidden, logits.
"""
import torch

from config import ModelConfig
from data.dataset import TokenData, load_meta, load_tokenizer
from model.model import MiniLLM

B = 4  # batch size for this check


def show(name, t, meaning):
    print(f"  {name:<22}{str(tuple(t.shape)):<20}{str(t.dtype).replace('torch.', ''):<10}{meaning}")


def main():
    torch.manual_seed(1337)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    meta = load_meta()
    tok = load_tokenizer()
    cfg = ModelConfig(vocab_size=meta["vocab_size"])
    T, D, V = cfg.context_length, cfg.d_model, cfg.vocab_size

    print("=" * 78)
    print("DATA")
    print("=" * 78)
    for k in ("source", "vocab_size", "pad_id", "eos_id", "train_tokens", "val_tokens",
              "documents_encoded", "avg_tokens_per_doc"):
        print(f"  {k:<20}{meta[k]}")

    val = TokenData("val")
    tokens = val.get_batch(B, T, device=device)

    print("\nRow 0 of the batch, first 40 tokens as strings:")
    row = tokens[0, :40].tolist()
    print("  " + " | ".join(tok.id_to_token(i) for i in row))
    n_eos = (tokens == meta["eos_id"]).sum().item()
    print(f"\n<eos> tokens inside this batch: {n_eos}  "
          f"(each one is a document boundary inside a window -> Session 9 masking)")

    print("\n" + "=" * 78)
    print(f"MODEL   B={B}  T={T}  D={D}  V={V}  layers={cfg.n_layers}  heads={cfg.n_heads}")
    print("=" * 78)
    model = MiniLLM(cfg).to(device).eval()
    for k, v in model.param_breakdown().items():
        print(f"  {k:<28}{v:>12,}")

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    print(f"\n  {'tensor':<22}{'shape':<20}{'dtype':<10}meaning")
    with torch.no_grad(), torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
        show("tokens", tokens, "B sequences x T token ids")
        h = model.tok_emb(tokens)
        show("embeddings", h, "one D-vector per token")
        for i, block in enumerate(model.blocks):
            h = block(h, model.rope_cos, model.rope_sin)
            if i in (0, cfg.n_layers - 1):
                show(f"after block {i + 1}", h, "same shape: blocks refine, never resize")
        h = model.final_norm(h)
        show("final hidden h", h, "input to the output head")
        logits = model.lm_head(h)
        show("logits", logits, "one score per vocab token, per position")

        # cross-check against the plain forward()
        logits_ref = model(tokens)
    assert torch.allclose(logits.float(), logits_ref.float(), atol=1e-2), "step-by-step != forward()"

    lf = logits.float()
    print(f"\nlogit stats: mean {lf.mean():+.4f}  std {lf.std():.4f}  "
          f"min {lf.min():+.3f}  max {lf.max():+.3f}")
    print(f"logits / hidden size ratio: V/D = {V}/{D} = {V / D:.1f}x")
    print(f"logits tensor bytes (bf16): {logits.numel() * 2 / 2**20:.1f} MiB for B={B}, T={T}")

    if device == "cuda":
        print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 2**20:.1f} MiB")

    print("\nStage 1 PASS: text -> tokens -> hidden -> logits works end to end.")


if __name__ == "__main__":
    main()
'''

FILES["data/__init__.py"] = r'''# data package: prepare_data.py builds the dataset, dataset.py loads it.
'''

FILES["data/prepare_data.py"] = r'''"""
Step 2 of Stage 1: build the dataset.

  1. Stream FineWeb-Edu (sample-10BT) from Hugging Face. Nothing big is downloaded.
  2. Train our own 8K byte-level BPE tokenizer on the first N documents.
  3. Tokenise documents, append <eos> after each one, and write flat uint16 files:
        data/prepared/val.bin    (first ~0.5M tokens)
        data/prepared/train.bin  (next ~20M tokens)
  4. Save tokenizer.json and meta.json.

Run from the project root:
    python -m data.prepare_data
    python -m data.prepare_data --train-tokens 50000000   # more data later
    python -m data.prepare_data --force                   # rebuild everything
"""
import argparse
import json
import os
import time

import numpy as np
from tqdm import tqdm

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")  # quiet on Windows

from config import DATA_DIR, DataConfig  # noqa: E402


def stream_documents(cfg: DataConfig):
    """Yield raw text documents one at a time from the HF stream."""
    from datasets import load_dataset

    ds = load_dataset(cfg.hf_dataset, name=cfg.hf_subset, split="train", streaming=True)
    for row in ds:
        text = row[cfg.text_field]
        if text and text.strip():
            yield text


def train_tokenizer(docs, cfg: DataConfig):
    from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

    tok = Tokenizer(models.BPE())
    tok.normalizer = normalizers.NFC()
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=cfg.vocab_size,
        min_frequency=2,
        special_tokens=[cfg.pad_token, cfg.eos_token],  # -> ids 0 and 1
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tok.train_from_iterator(docs, trainer=trainer, length=len(docs))

    assert tok.token_to_id(cfg.pad_token) == 0, "pad must be id 0"
    assert tok.token_to_id(cfg.eos_token) == 1, "eos must be id 1"
    assert tok.get_vocab_size() <= 65535, "ids must fit in uint16"
    return tok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-tokens", type=int, default=None)
    parser.add_argument("--val-tokens", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="rebuild even if files exist")
    args = parser.parse_args()

    cfg = DataConfig()
    if args.train_tokens:
        cfg.train_tokens = args.train_tokens
    if args.val_tokens:
        cfg.val_tokens = args.val_tokens

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tok_path = DATA_DIR / "tokenizer.json"
    train_path, val_path = DATA_DIR / "train.bin", DATA_DIR / "val.bin"

    if train_path.exists() and val_path.exists() and not args.force:
        print(f"Data already prepared in {DATA_DIR}. Use --force to rebuild.")
        return

    t0 = time.time()
    stream = stream_documents(cfg)

    # ---- 1. First N docs: used to train the tokenizer (and then encoded too) ----
    print(f"Streaming {cfg.tokenizer_train_docs:,} docs from {cfg.hf_dataset}/{cfg.hf_subset} ...")
    first_docs = [d for _, d in zip(tqdm(range(cfg.tokenizer_train_docs), unit="doc"), stream)]

    print(f"Training {cfg.vocab_size}-token BPE tokenizer ...")
    tok = train_tokenizer(first_docs, cfg)
    tok.save(str(tok_path))
    eos_id = tok.token_to_id(cfg.eos_token)
    print(f"Saved tokenizer -> {tok_path}")

    # ---- 2. Encode docs until both targets are met ----
    target = cfg.val_tokens + cfg.train_tokens
    chunks, total, n_docs = [], 0, 0
    bar = tqdm(total=target, unit="tok", unit_scale=True, desc="Encoding")

    def encode_batch(batch):
        nonlocal total, n_docs
        for enc in tok.encode_batch(batch):
            ids = np.asarray(enc.ids + [eos_id], dtype=np.uint16)  # doc + boundary marker
            chunks.append(ids)
            total += len(ids)
            n_docs += 1
            bar.update(len(ids))

    BATCH = 1000
    for i in range(0, len(first_docs), BATCH):
        encode_batch(first_docs[i:i + BATCH])
        if total >= target:
            break
    del first_docs

    batch = []
    while total < target:
        try:
            batch.append(next(stream))
        except StopIteration:
            print("\nWARNING: stream ended before target was reached.")
            break
        if len(batch) == BATCH:
            encode_batch(batch)
            batch = []
    if batch and total < target:
        encode_batch(batch)
    bar.close()

    # ---- 3. Split and write ----
    all_ids = np.concatenate(chunks)[:target]
    val_ids = all_ids[: cfg.val_tokens]
    train_ids = all_ids[cfg.val_tokens:]
    val_ids.tofile(val_path)
    train_ids.tofile(train_path)

    meta = {
        "source": f"{cfg.hf_dataset}/{cfg.hf_subset}",
        "vocab_size": tok.get_vocab_size(),
        "pad_id": tok.token_to_id(cfg.pad_token),
        "eos_id": eos_id,
        "dtype": "uint16",
        "train_tokens": int(len(train_ids)),
        "val_tokens": int(len(val_ids)),
        "documents_encoded": n_docs,
        "avg_tokens_per_doc": round(total / max(n_docs, 1), 1),
        "train_docs_boundaries": int((train_ids == eos_id).sum()),
        "seconds": round(time.time() - t0, 1),
    }
    (DATA_DIR / "meta.json").write_text(json.dumps(meta, indent=2))

    # Human-readable sanity check: what does tokenisation look like?
    sample = val_ids[:60].tolist()
    print("\nFirst 60 val tokens as strings:")
    print(" | ".join(tok.id_to_token(i) for i in sample))
    print("\nDecoded:")
    print(tok.decode(sample))

    print("\n" + json.dumps(meta, indent=2))
    print(f"\nDONE in {meta['seconds']}s -> {DATA_DIR}")


if __name__ == "__main__":
    main()
'''

FILES["data/dataset.py"] = r'''"""
Loading the prepared token files.

Design choice: get_batch returns ONE tensor `tokens` of shape [B, T].
It does NOT pre-shift into (x, y). The shift belongs to the loss harness,
because Session 9 asks you to verify it yourself.

Also note: windows are sampled at random offsets in a stream of packed
documents, so a window can span an <eos> boundary. That is exactly the
"document boundary" problem from Session 9 Section 6, left visible on purpose.
"""
import json

import numpy as np
import torch

from config import DATA_DIR


def load_meta():
    return json.loads((DATA_DIR / "meta.json").read_text())


def load_tokenizer():
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(DATA_DIR / "tokenizer.json"))


class TokenData:
    def __init__(self, split: str):
        assert split in ("train", "val")
        path = DATA_DIR / f"{split}.bin"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run: python -m data.prepare_data")
        # memmap: the file stays on disk, only touched slices are read into RAM
        self.ids = np.memmap(path, dtype=np.uint16, mode="r")

    def __len__(self):
        return len(self.ids)

    def get_batch(self, batch_size: int, context_length: int, device="cpu", generator=None):
        """Random windows -> LongTensor [B, T]."""
        high = len(self.ids) - context_length
        starts = torch.randint(0, high, (batch_size,), generator=generator)
        rows = [torch.from_numpy(self.ids[s:s + context_length].astype(np.int64)) for s in starts.tolist()]
        tokens = torch.stack(rows)
        if device != "cpu":
            tokens = tokens.pin_memory().to(device, non_blocking=True)
        return tokens
'''

FILES["model/__init__.py"] = r'''# model package: layers.py (RMSNorm, SwiGLU), attention.py (causal MHA + RoPE), model.py (MiniLLM).
'''

FILES["model/layers.py"] = r'''"""
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
'''

FILES["model/attention.py"] = r'''"""
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
'''

FILES["model/model.py"] = r'''"""
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
'''

FILES["README.md"] = r'''# mini_llm

A small decoder-only language model trained from scratch on FineWeb-Edu.
It's built for the course assignments (Session 9 onward) and reused across them.

| Component | Choice |
|---|---|
| Data | `HuggingFaceFW/fineweb-edu`, subset `sample-10BT`, streamed. Default: 20M train + 0.5M val tokens |
| Tokenizer | Our own byte-level BPE, 8,192 tokens. `<pad>` = 0, `<eos>` = 1 (appended after every document) |
| Model | 8 layers, D=384, 6 heads (head_dim 64), context 256, RoPE, RMSNorm, pre-norm, SwiGLU (1024) |
| Size | ~20.5M parameters untied, ~17.3M tied |

## Setup on Windows 11 (one time), from the mini_llm folder

```powershell
nvidia-smi
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

If activation is blocked: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.
In Antigravity, select `.venv` as the Python interpreter.

## Stage 1: run order

```powershell
python check_env.py            # ends in PASS, Compute cap. sm_120
python -m data.prepare_data    # ~10-30 min: stream, train tokenizer, write data\prepared\
python -m model.model          # random-token self-test
python stage1_check.py         # real batch -> logits, every shape printed
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `CUDA not available` | CPU-only torch installed. `pip uninstall torch`, then reinstall with cu128 |
| `no kernels for sm_120` | Torch build too old. Reinstall with `cu128` or newer |
| `ModuleNotFoundError: config` | Run from the `mini_llm` folder, using `python -m data.prepare_data` |
| `train.bin not found` | Run `python -m data.prepare_data` first |
| Stream stalls or HTTP 429 | `huggingface-cli login`, then rerun with `--force` |
'''


def main():
    force = "--force" in sys.argv
    root = Path.cwd() / "mini_llm"
    for rel, content in FILES.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not force:
            print(f"skip (exists)  {rel}")
            continue
        path.write_text(content, encoding="utf-8", newline="\n")
        print(f"wrote          {rel}")

    archive = shutil.make_archive(str(Path.cwd() / "mini_llm_stage1"), "zip", Path.cwd(), "mini_llm")
    print(f"\nProject: {root}\nZip:     {archive}")


if __name__ == "__main__":
    main()
