"""
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
