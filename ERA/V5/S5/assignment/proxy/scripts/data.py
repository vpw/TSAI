#!/usr/bin/env python3
"""Lane sampler for the mixture ablation.

Each lane is a memory-mapped flat token array plus a loss mask. Two things matter:

* Sequences are drawn lane-by-lane from a multinomial over the arm's mixture weights, so the
  realised mixture is the declared mixture (it is measured and reported, not assumed).
* Each lane has a `unique_cap`: the sampler only ever reads from the first `unique_cap`
  tokens of that lane. That is what makes the proxy repeat data at the same epoch counts the
  full-scale plan implies, instead of quietly handing every lane fresh tokens.
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROXY = os.path.dirname(HERE)
TOK = os.path.join(PROXY, "data", "tokens")


class Lane:
    def __init__(self, name, unique_cap=None):
        self.name = name
        self.tokens = np.memmap(os.path.join(TOK, f"{name}.train.bin"), dtype=np.uint32, mode="r")
        mp = os.path.join(TOK, f"{name}.trainmask.bin")
        self.mask = np.memmap(mp, dtype=np.uint8, mode="r") if os.path.exists(mp) else None
        self.size = len(self.tokens)
        cap = self.size if unique_cap is None else min(int(unique_cap), self.size)
        self.unique_cap = max(cap, 1)
        self.capped_by_supply = unique_cap is not None and unique_cap > self.size
        self.drawn = 0

    def draw(self, rng, seq_len):
        hi = max(self.unique_cap - seq_len - 1, 1)
        i = int(rng.integers(0, hi))
        x = np.asarray(self.tokens[i:i + seq_len + 1], dtype=np.int64)
        if len(x) < seq_len + 1:                       # tail of a tiny lane
            x = np.pad(x, (0, seq_len + 1 - len(x)))
        m = (np.asarray(self.mask[i:i + seq_len + 1], dtype=np.uint8) if self.mask is not None
             else np.ones(seq_len + 1, dtype=np.uint8))
        if len(m) < seq_len + 1:
            m = np.pad(m, (0, seq_len + 1 - len(m)))
        self.drawn += seq_len
        return x, m

    def epochs(self):
        return self.drawn / self.unique_cap


class MixtureSampler:
    def __init__(self, mixture_pct, caps, seq_len, seed):
        self.seq_len = seq_len
        self.rng = np.random.default_rng(seed)
        self.names = sorted(mixture_pct)
        self.weights = np.array([mixture_pct[n] for n in self.names], dtype=np.float64)
        self.weights /= self.weights.sum()
        self.lanes = {n: Lane(n, caps.get(n, {}).get("unique_cap_tokens")) for n in self.names}
        self.counts = {n: 0 for n in self.names}

    def set_mixture(self, mixture_pct):
        """Used by the transition arms: swap weights mid-run, keeping the same lane objects."""
        w = np.array([mixture_pct.get(n, 0.0) for n in self.names], dtype=np.float64)
        if w.sum() <= 0:
            raise ValueError("empty mixture")
        self.weights = w / w.sum()

    def batch(self, n_seq):
        picks = self.rng.choice(len(self.names), size=n_seq, p=self.weights)
        xs, ms = [], []
        for p in picks:
            name = self.names[p]
            x, m = self.lanes[name].draw(self.rng, self.seq_len)
            xs.append(x)
            ms.append(m)
            self.counts[name] += 1
        return np.stack(xs), np.stack(ms)

    def report(self):
        tot = sum(self.counts.values()) or 1
        return {n: {"sequences": self.counts[n],
                    "realised_pct": round(100.0 * self.counts[n] / tot, 3),
                    "declared_pct": round(100.0 * self.weights[i], 3),
                    "unique_cap_tokens": self.lanes[n].unique_cap,
                    "pool_tokens": self.lanes[n].size,
                    "epochs_realised": round(self.lanes[n].epochs(), 3),
                    "cap_limited_by_pool_size": self.lanes[n].capped_by_supply}
                for i, n in enumerate(self.names)}


def load_val(lane, max_tokens=None):
    p = os.path.join(TOK, f"{lane}.val.npz")
    if not os.path.exists(p):
        return None
    z = np.load(p)
    t, m, b = z["tokens"], z["mask"], z["tbytes"]
    if max_tokens and len(t) > max_tokens:
        t, m, b = t[:max_tokens], m[:max_tokens], b[:max_tokens]
    return {"tokens": np.asarray(t, dtype=np.int64),
            "mask": np.asarray(m, dtype=np.uint8),
            "tbytes": np.asarray(b, dtype=np.int64)}


def token_stats():
    with open(os.path.join(TOK, "token_stats.json")) as f:
        return json.load(f)
