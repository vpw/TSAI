"""Lane sampler over the already-tokenized S5 proxy corpus.

S5 tokenized ~3.9 GB of multilingual text with the *same* frozen sarvam1 vocabulary this
submission's codecs are built from, split into capability lanes. Reusing it means the token
ids in the corpus and the rows of the codec table refer to the same tokens, which is the only
way the comparison means anything -- and it saves rebuilding a corpus that already exists.

Layout, per lane: `<lane>.train.bin` (flat uint32 token ids) and `<lane>.val.npz`.
"""

from __future__ import annotations

import json
import os

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The self-contained slice `scripts/prep_subset.py` cuts, which is what ships to the GPU box.
_LOCAL_CORPUS = os.path.join(_ROOT, "data", "corpus")
# The full S5 corpus, used when running on this workstation with S5 still checked out.
_S5_TOKENS = os.path.join(_ROOT, "..", "..", "S5", "assignment", "proxy", "data", "tokens")

# Lanes used for training. Weighted toward Indic relative to S5's own mixture, because the
# claim under test is about byte-level structure and Indic scripts are where UTF-8 byte
# structure actually bites (3 bytes/char, conjuncts, long tokens).
TRAIN_MIX = {
    "general_web": 34,
    "code": 16,
    "stem": 10,
    "indic_A_verified": 20,
    "indic_B_unverified": 10,
    "indic_C_translated": 10,
}

VAL_LANES = ["general_web", "code", "stem",
             "indic_A_verified", "indic_B_unverified", "indic_C_translated"]


def tokens_dir(path: str | None = None) -> str:
    """Where the token lanes live.

    Preference order: explicit argument, then `$KV2_CORPUS`, then the local subset, then the
    full S5 corpus. The GPU box only ever has the subset, so this resolves correctly there
    without any flag being passed.
    """
    if path:
        return os.path.normpath(path)
    env = os.environ.get("KV2_CORPUS")
    if env:
        return os.path.normpath(env)
    if os.path.isdir(_LOCAL_CORPUS):
        return _LOCAL_CORPUS
    return os.path.normpath(_S5_TOKENS)


def token_stats(path: str | None = None) -> dict:
    with open(os.path.join(tokens_dir(path), "token_stats.json"), encoding="utf-8") as fh:
        return json.load(fh)


class Lane:
    def __init__(self, name: str, root: str):
        self.name = name
        self.tokens = np.memmap(os.path.join(root, f"{name}.train.bin"), dtype=np.uint32, mode="r")
        self.size = len(self.tokens)
        self.drawn = 0

    def draw(self, rng, seq_len: int) -> np.ndarray:
        hi = max(self.size - seq_len - 1, 1)
        i = int(rng.integers(0, hi))
        x = np.asarray(self.tokens[i:i + seq_len + 1], dtype=np.int64)
        if len(x) < seq_len + 1:
            x = np.pad(x, (0, seq_len + 1 - len(x)))
        self.drawn += seq_len
        return x


class MixtureSampler:
    """Draws sequences lane-by-lane from a multinomial over the declared mixture.

    The realised mixture is counted and reported rather than assumed -- same discipline as
    S5, and it matters here because every arm must see the *same* token stream. Seeding the
    sampler identically across arms guarantees that.
    """

    def __init__(self, mixture_pct: dict, seq_len: int, seed: int, root: str | None = None):
        root = tokens_dir(root)
        self.seq_len = seq_len
        self.rng = np.random.default_rng(seed)
        self.names = sorted(mixture_pct)
        w = np.array([mixture_pct[n] for n in self.names], dtype=np.float64)
        self.weights = w / w.sum()
        self.lanes = {n: Lane(n, root) for n in self.names}
        self.counts = {n: 0 for n in self.names}

    def batch(self, n_seqs: int) -> np.ndarray:
        picks = self.rng.choice(len(self.names), size=n_seqs, p=self.weights)
        rows = []
        for p in picks:
            name = self.names[p]
            rows.append(self.lanes[name].draw(self.rng, self.seq_len))
            self.counts[name] += self.seq_len
        return np.stack(rows)

    def realised_mixture(self) -> dict:
        total = sum(self.counts.values()) or 1
        return {n: round(100.0 * c / total, 2) for n, c in sorted(self.counts.items())}


def load_val(lane: str, root: str | None = None) -> np.ndarray:
    blob = np.load(os.path.join(tokens_dir(root), f"{lane}.val.npz"))
    key = "tokens" if "tokens" in blob else blob.files[0]
    return np.asarray(blob[key], dtype=np.int64)
