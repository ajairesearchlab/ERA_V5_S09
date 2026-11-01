"""
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


@dataclass
class TrainConfig:
    """Training-loop settings (Session 10). The batch the optimiser sees is
    micro_batch x accum_steps sequences, independent of what fits in memory."""
    micro_batch: int = 8                # sequences per forward/backward
    accum_steps: int = 4                # micro-batches per optimiser step -> 32 sequences
    steps: int = 600
    lr: float = 1e-3
    warmup: int = 100
    min_lr_frac: float = 0.1            # cosine decays to 10% of peak
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0              # 0 disables clipping (the norm is still measured)
    eval_every: int = 50
    eval_batches: int = 10
    seed: int = 0

    def to_dict(self):
        return asdict(self)
