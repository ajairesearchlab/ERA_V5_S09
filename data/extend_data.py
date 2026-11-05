"""
Session 14: grow train.bin (Session 9: 20M tokens) to more tokens WITHOUT retraining the tokenizer.

prepare_data.py --force would retrain the tokenizer. This script instead:
  1. re-streams FineWeb-Edu (sample-10BT) in the same order as prepare_data.py,
  2. encodes it with the SAVED tokenizer.json, appending <eos> after every document,
  3. checks that val.bin and the existing train.bin come out token-for-token identical
     (so every earlier session's numbers stay valid), and only then
  4. keeps the old file as train_20M.bin and writes the longer train.bin + meta.json.

Run from the mini_llm folder:
    uv run python -m data.extend_data                        # 60M training tokens (default)
    uv run python -m data.extend_data --train-tokens 80000000
"""
import argparse
import json
import os
import shutil
import time

import numpy as np
from tqdm import tqdm

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from config import DATA_DIR, DataConfig            # noqa: E402
from data.prepare_data import stream_documents     # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--train-tokens", type=int, default=60_000_000)
    args = p.parse_args(argv)

    cfg = DataConfig()
    meta_path, train_path, val_path = DATA_DIR / "meta.json", DATA_DIR / "train.bin", DATA_DIR / "val.bin"
    if not (meta_path.exists() and train_path.exists() and val_path.exists()):
        raise SystemExit("No prepared data found. Run: uv run python -m data.prepare_data")
    meta = json.loads(meta_path.read_text())
    old_train = np.fromfile(train_path, dtype=np.uint16)        # a copy, so the file is not locked
    old_val = np.fromfile(val_path, dtype=np.uint16)
    if len(old_train) >= args.train_tokens:
        print(f"train.bin already has {len(old_train):,} tokens (>= {args.train_tokens:,}). Nothing to do.")
        return

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(DATA_DIR / "tokenizer.json"))
    eos = tok.token_to_id(cfg.eos_token)
    target = len(old_val) + args.train_tokens
    print(f"Extending train.bin: {len(old_train):,} -> {args.train_tokens:,} tokens "
          f"(re-encoding {target:,} tokens with the saved tokenizer)")

    t0 = time.time()
    chunks, total, n_docs, batch = [], 0, 0, []
    bar = tqdm(total=target, unit="tok", unit_scale=True, desc="Encoding")

    def encode(docs):
        nonlocal total, n_docs
        for enc in tok.encode_batch(docs):
            ids = np.asarray(enc.ids + [eos], dtype=np.uint16)
            chunks.append(ids)
            total += len(ids)
            n_docs += 1
            bar.update(len(ids))

    for doc in stream_documents(cfg):
        batch.append(doc)
        if len(batch) == 1000:
            encode(batch)
            batch = []
            if total >= target:
                break
    if batch and total < target:
        encode(batch)
    bar.close()
    if total < target:
        print(f"WARNING: stream ended at {total:,} tokens, short of {target:,}.")

    all_ids = np.concatenate(chunks)[:target]
    new_val, new_train = all_ids[:len(old_val)], all_ids[len(old_val):]

    # ---- the safety check: earlier data must be reproduced exactly
    same_val = np.array_equal(new_val, old_val)
    same_prefix = np.array_equal(new_train[:len(old_train)], old_train)
    print(f"val.bin reproduced exactly: {same_val}")
    print(f"first {len(old_train):,} train tokens reproduced exactly: {same_prefix}")
    if not (same_val and same_prefix):
        raise SystemExit("Mismatch: NOT writing anything. (Has the HF dataset or tokenizers version changed?)")

    backup = DATA_DIR / f"train_{len(old_train) // 1_000_000}M.bin"
    if not backup.exists():
        shutil.copyfile(train_path, backup)
        print(f"old train.bin kept as {backup.name}")
    new_train.tofile(train_path)

    meta.update({
        "train_tokens": int(len(new_train)),
        "documents_encoded": n_docs,
        "avg_tokens_per_doc": round(total / max(n_docs, 1), 1),
        "train_docs_boundaries": int((new_train == eos).sum()),
        "extended_from_train_tokens": int(len(old_train)),
        "extend_seconds": round(time.time() - t0, 1),
    })
    meta_path.write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    print(f"DONE in {time.time() - t0:.0f}s: train.bin now {len(new_train):,} tokens")


if __name__ == "__main__":
    main()
