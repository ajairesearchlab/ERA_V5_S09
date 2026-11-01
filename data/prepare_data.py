"""
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
