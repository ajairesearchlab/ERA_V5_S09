"""
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
