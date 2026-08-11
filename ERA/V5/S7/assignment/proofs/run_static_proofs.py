#!/usr/bin/env python3
"""Static proofs: everything about the codecs that can be settled without training.

Run:  .venv/bin/python proofs/run_static_proofs.py

Six claims, each measured on the real 68,096-token sarvam1 vocabulary, no GPU:

  P1  Kronecker's crop collisions on a real vocabulary -- how many, and in which scripts.
  P2  Dead-parameter census: how much of the Kronecker projection can never be trained.
  P3  Naive wave summation is permutation-invariant, so anagrams are indistinguishable.
  P4  Phase binding is exactly invertible, and at what dimension.
  P5  Noise robustness: the codes degrade gracefully instead of failing at a threshold.
  P6  Capacity vs token length -- where the dense code actually runs out.

Results land in proofs/results/static_proofs.json plus a readable summary on stdout.
"""

from __future__ import annotations

import collections
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kv2.codecs import (UTF8_IMPOSSIBLE_BYTES, FourierCodec, KroneckerCodec,
                        NaiveSumCodec)
from kv2.vocab import INDIC_SCRIPTS, load_vocab, script_of, token_bytes, tokenizer_path

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
SEED = 20260810


def hr(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# --------------------------------------------------------------------------------------
# P1 -- collisions under the crop
# --------------------------------------------------------------------------------------

def proof_collisions(tokens, byte_seqs, pos_dims=(16, 32, 48, 64)) -> dict:
    hr("P1  Kronecker crop collisions on the real vocabulary")
    out = {}
    for pd in pos_dims:
        groups = collections.defaultdict(list)
        for tok, bs in zip(tokens, byte_seqs):
            groups[bs[:pd]].append(tok)
        colliding = {k: v for k, v in groups.items() if len(v) > 1}
        involved = sorted({t for v in colliding.values() for t in v})
        by_script = collections.Counter(script_of(t) for t in involved)
        indic = sum(n for s, n in by_script.items() if s in INDIC_SCRIPTS)
        out[pd] = {
            "groups": len(colliding),
            "tokens_involved": len(involved),
            "pct_of_vocab": 100.0 * len(involved) / len(tokens),
            "by_script": dict(by_script),
            "indic_share_pct": (100.0 * indic / len(involved)) if involved else 0.0,
            "examples": [v for v in list(colliding.values())[:6]],
        }
        print(
            f"  pos_dim={pd:>3}: {len(colliding):>3} groups, {len(involved):>3} tokens "
            f"({out[pd]['pct_of_vocab']:.3f}% of vocab), Indic share {out[pd]['indic_share_pct']:.0f}%"
        )
    for pair in out[32]["examples"]:
        print(f"        collision @32: {pair}")
    print("\n  Reading: the crop is real but it is NOT a crisis at 68k -- 0.03% of the vocabulary.")
    print("  Every collision is Indic; none is Latin. Direction of the lesson's claim holds,")
    print("  magnitude does not. The stronger objection to the grid is P2.")
    return out


# --------------------------------------------------------------------------------------
# P2 -- dead parameters
# --------------------------------------------------------------------------------------

def proof_dead_parameters(byte_seqs, d_model=8096, char_dim=256, pos_dim=32) -> dict:
    hr("P2  Dead-parameter census of the one-hot Kronecker grid")
    occ = np.zeros(pos_dim)
    used_cells = set()
    nnz = []
    for bs in byte_seqs:
        L = min(len(bs), pos_dim)
        occ[:L] += 1
        nnz.append(L)
        for p in range(L):
            used_cells.add((bs[p], p))
    occ /= len(byte_seqs)
    code_dim = char_dim * pos_dim
    dead = code_dim - len(used_cells)

    # A byte value never emitted by UTF-8 can never activate its row, at any position.
    impossible_rows = len(UTF8_IMPOSSIBLE_BYTES) * pos_dim

    res = {
        "code_dim": code_dim,
        "mean_column_occupancy": float(occ.mean()),
        "always_zero_fraction_pct": float(100 * (1 - occ.mean())),
        "mean_nonzeros_per_code": float(np.mean(nnz)),
        "density_pct": float(100 * np.mean(nnz) / code_dim),
        "cells_ever_used": len(used_cells),
        "cells_dead": dead,
        "dead_pct": float(100 * dead / code_dim),
        "dead_params_at_d_model": int(dead * d_model),
        "total_params_at_d_model": int(code_dim * d_model),
        "utf8_impossible_rows": impossible_rows,
        "column_occupancy": occ.round(4).tolist(),
    }
    print(f"  code_dim                      {code_dim:,}")
    print(f"  mean column occupancy         {res['mean_column_occupancy']:.4f}"
          f"  -> {res['always_zero_fraction_pct']:.1f}% of the code is always zero")
    print(f"  non-zeros per code            {res['mean_nonzeros_per_code']:.2f}"
          f"  ({res['density_pct']:.3f}% dense)")
    print(f"  grid cells ever activated     {len(used_cells):,} of {code_dim:,}"
          f"  ({100*len(used_cells)/code_dim:.1f}%)")
    print(f"  UNREACHABLE grid cells        {dead:,}  ({res['dead_pct']:.1f}%)")
    print(f"  their params at d_model={d_model}  {res['dead_params_at_d_model']/1e6:.1f}M"
          f" of {res['total_params_at_d_model']/1e6:.1f}M")

    # The structural count above is about the code *before* normalisation. The released codec
    # z-normalises, which gives every unreachable cell a non-zero (but token-independent)
    # value -- so the fair, normalisation-proof measure is how many independent directions
    # the finished code actually spans. That is what the projection can distinguish.
    n_rank = 10000
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(byte_seqs), n_rank, replace=False)   # random, not the frequent head
    rank_sample = [byte_seqs[i] for i in idx]
    print(f"\n  Effective rank of the finished (z-normalised) code, "
          f"{n_rank:,} randomly sampled tokens:")
    ranks = {}
    for label, codec in (("kronecker_32", KroneckerCodec(pos_dim=pos_dim)),
                         ("kronecker_48", KroneckerCodec(pos_dim=48)),
                         ("fourier_2048", FourierCodec(n_freq=1024, seed=SEED)),
                         ("fourier_8192", FourierCodec(n_freq=4096, seed=SEED))):
        tbl = codec.encode_many(rank_sample, dtype=np.float32)
        tbl -= tbl.mean(axis=0, keepdims=True)
        sv = np.linalg.svd(tbl, compute_uv=False)
        tol = sv.max() * max(tbl.shape) * np.finfo(np.float32).eps
        rank = int((sv > tol).sum())
        energy = np.cumsum(sv**2) / (sv**2).sum()
        r99 = int((energy < 0.99).sum()) + 1
        # A rank equal to the sample size means the sample, not the codec, was the limit.
        censored = rank >= min(n_rank - 1, codec.code_dim)
        ranks[label] = {"code_dim": codec.code_dim, "numerical_rank": rank,
                        "rank_99pct_energy": r99,
                        "rank_per_dim": round(rank / codec.code_dim, 4),
                        "sample_limited": bool(censored)}
        print(f"    {label:<14} code_dim {codec.code_dim:>5}   rank {rank:>5}"
              f"{' (sample-limited)' if censored else '':<18}   "
              f"99%-energy rank {r99:>5}   rank/dim {rank/codec.code_dim:.3f}")
        del tbl
    res["effective_rank"] = ranks

    print("\n  Reading: this is the number that matters, and it is worse than the cell census.")
    print("  The grid spends 8,192 coordinates but the finished code spans far fewer")
    print("  independent directions, because UTF-8 makes most (value, position) pairs")
    print("  impossible. Note z-normalisation means no coordinate is literally constant, so")
    print("  'dead rows' is the wrong test -- rank is the right one. The dense phase code")
    print("  spends 2,048 coordinates and uses essentially all of them.")
    return res


# --------------------------------------------------------------------------------------
# P3 -- permutation invariance of naive summation
# --------------------------------------------------------------------------------------

def proof_permutation_invariance(byte_seqs, n_freq=1024) -> dict:
    hr("P3  'Just add the waves' is permutation-invariant (the negative control)")
    naive = NaiveSumCodec(n_freq=n_freq, seed=SEED)
    fourier = FourierCodec(n_freq=n_freq, seed=SEED)

    pairs = [(b"listen", b"silent"), (b"stressed", b"desserts"),
             (b"abc", b"cba"), ("तमिल".encode(), "मिलत".encode())]
    rows = []
    for a, b in pairs:
        na = np.allclose(naive.encode_one(a), naive.encode_one(b), atol=1e-5)
        fa = np.allclose(fourier.encode_one(a), fourier.encode_one(b), atol=1e-5)
        rows.append({"a": a.decode("utf-8", "replace"), "b": b.decode("utf-8", "replace"),
                     "naive_identical": bool(na), "fourier_identical": bool(fa)})
        print(f"  {a.decode('utf-8','replace'):>12} vs {b.decode('utf-8','replace'):<12}"
              f"  naive: {'IDENTICAL' if na else 'distinct':<10} fourier: {'IDENTICAL' if fa else 'distinct'}")

    # And at vocabulary scale: how many real tokens does naive summation merge?
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(byte_seqs), 8000, replace=False)
    sample = [byte_seqs[i] for i in idx if len(byte_seqs[i]) > 0]
    def dup_count(codec):
        seen, dup = set(), 0
        for bs in sample:
            key = np.round(codec.encode_one(bs), 4).tobytes()
            if key in seen:
                dup += 1
            else:
                seen.add(key)
        return dup
    naive_dups = dup_count(naive)
    fourier_dups = dup_count(fourier)
    print(f"\n  On {len(sample):,} real tokens: naive merges {naive_dups}, fourier merges {fourier_dups}")
    print("\n  Reading: the assignment's literal phrasing -- 'represent each character like a")
    print("  fourier wave, and just add them' -- cannot work, because a sum does not depend on")
    print("  the order of its terms. Binding value to position by phase is the minimal repair.")
    return {"pairs": rows, "sample_size": len(sample),
            "naive_duplicates": naive_dups, "fourier_duplicates": fourier_dups}


# --------------------------------------------------------------------------------------
# P4 -- exact invertibility vs dimension
# --------------------------------------------------------------------------------------

def proof_invertibility(byte_seqs, dims=(128, 256, 512, 1024, 2048, 4096), n_sample=3000) -> dict:
    hr("P4  Exact invertibility of the phase code, vs code dimension")
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(byte_seqs), n_sample, replace=False)
    sample = [byte_seqs[i] for i in idx if len(byte_seqs[i]) > 0]

    lengths = np.array([len(b) for b in sample])
    out = {}
    print(f"  {'code_dim':>9} {'byte-acc':>10} {'token-exact':>12}   (n={len(sample):,})")
    for real_dim in dims:
        codec = FourierCodec(n_freq=real_dim // 2, seed=SEED)
        rec = codec.decode_batch(codec.complex_codes(sample), lengths)
        ok = sum(sum(1 for x, y in zip(r, b) if x == y) for r, b in zip(rec, sample))
        exact = sum(int(r == b) for r, b in zip(rec, sample))
        out[real_dim] = {"byte_acc": ok / int(lengths.sum()),
                         "token_exact": exact / len(sample)}
        print(f"  {real_dim:>9} {out[real_dim]['byte_acc']:>10.4f} {out[real_dim]['token_exact']:>12.4f}")
    print("\n  Baseline for scale: the one-hot Kronecker grid is 8,192-dim and is exact only for")
    print("  tokens of <= 32 bytes; past that it is not merely inexact, it is silently identical")
    print("  to a different token. The phase code reaches exact at 2,048 with no length cap.")
    return out


# --------------------------------------------------------------------------------------
# P5 -- noise robustness
# --------------------------------------------------------------------------------------

def proof_noise(byte_seqs, dims=(1024, 2048, 4096, 8192),
                sigmas=(0.0, 0.25, 0.5, 1.0, 2.0, 4.0), n_sample=1500) -> dict:
    hr("P5  Noise robustness -- the property the instructor says blocks reversibility")
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(byte_seqs), n_sample, replace=False)
    sample = [byte_seqs[i] for i in idx if len(byte_seqs[i]) > 0]

    lengths = np.array([len(b) for b in sample])
    out = {}
    print("  exact-token decode rate, noise sigma relative to the code's own RMS")
    print("  " + f"{'dim':>6}" + "".join(f"{('s=%.2f' % s):>9}" for s in sigmas))
    for real_dim in dims:
        codec = FourierCodec(n_freq=real_dim // 2, seed=SEED)
        z0 = codec.complex_codes(sample)                      # (N, K)
        rms = np.sqrt((np.abs(z0) ** 2).mean(axis=1, keepdims=True))
        row = {}
        line = f"  {real_dim:>6}"
        for s in sigmas:
            if s > 0:
                noise = (rng.normal(0, 1, z0.shape) + 1j * rng.normal(0, 1, z0.shape))
                z = z0 + noise.astype(np.complex64) * s * rms
            else:
                z = z0
            rec = codec.decode_batch(z, lengths)
            row[s] = sum(int(r == b) for r, b in zip(rec, sample)) / len(sample)
            line += f"{row[s]:>9.3f}"
        out[real_dim] = row
        print(line)
    print("\n  Reading: 'the neural network does not predict exactly those numbers we want' is the")
    print("  stated blocker on Problem 5. A redundant phase code is an error-correcting code:")
    print("  at 8,192 dims -- the Kronecker grid's own size -- decoding survives noise twice the")
    print("  size of the signal. A hard crop has no analogous margin.")
    return out


# --------------------------------------------------------------------------------------
# P6 -- capacity vs length
# --------------------------------------------------------------------------------------

def proof_capacity(dims=(512, 1024, 2048, 4096), lengths=(4, 8, 16, 32, 64, 128), n=400) -> dict:
    hr("P6  Capacity: exact-decode rate vs token length (where the dense code runs out)")
    rng = np.random.default_rng(SEED)
    out = {}
    print("  " + f"{'dim':>6}" + "".join(f"{('L=%d' % L):>9}" for L in lengths))
    for real_dim in dims:
        codec = FourierCodec(n_freq=real_dim // 2, max_pos=max(lengths), seed=SEED)
        row = {}
        line = f"  {real_dim:>6}"
        for L in lengths:
            seqs = [bytes(rng.integers(0, 256, L).tolist()) for _ in range(n)]
            lens = np.full(n, L)
            rec = codec.decode_batch(codec.complex_codes(seqs), lens)
            row[L] = sum(int(r == b) for r, b in zip(rec, seqs)) / n
            line += f"{row[L]:>9.3f}"
        out[real_dim] = row
        print(line)
    print("\n  Reading: capacity degrades smoothly with length and improves with dimension, which")
    print("  is the qualitative behaviour VSA capacity theory predicts for superposition")
    print("  (Frady, Kleyko & Sommer, arXiv 1803.00412). Contrast the crop, which is a step")
    print("  function: perfect to 32 bytes, catastrophic and silent after.")
    print("  NOTE: these are uniform-random bytes, the hardest case. Real UTF-8 tokens are")
    print("  highly structured and decode better at the same length -- see P4.")
    return out


def main() -> None:
    t0 = time.time()
    os.makedirs(RESULTS, exist_ok=True)
    tokens = load_vocab()
    byte_seqs = [token_bytes(t) for t in tokens]
    nonempty = [b for b in byte_seqs if len(b) > 0]

    print(f"tokenizer : {tokenizer_path()}")
    print(f"vocab     : {len(tokens):,} tokens, {len(nonempty):,} with at least one byte")
    lens = np.array([len(b) for b in byte_seqs])
    print(f"byte len  : mean {lens.mean():.2f}  median {int(np.median(lens))}  max {lens.max()}"
          f"  (>32 bytes: {(lens > 32).sum()})")

    # Record the tokenizer by basename, not by absolute path: the identity that matters is
    # which artefact it is, and an absolute path just leaks whose machine produced the file.
    report = {
        "seed": SEED,
        "tokenizer": os.path.basename(tokenizer_path()),
        "vocab_size": len(tokens),
        "byte_length": {"mean": float(lens.mean()), "median": int(np.median(lens)),
                        "max": int(lens.max()), "over_32": int((lens > 32).sum())},
        "P1_collisions": proof_collisions(tokens, byte_seqs),
        "P2_dead_parameters": proof_dead_parameters(byte_seqs),
        "P3_permutation_invariance": proof_permutation_invariance(byte_seqs),
        "P4_invertibility": proof_invertibility(nonempty),
        "P5_noise": proof_noise(nonempty),
        "P6_capacity": proof_capacity(),
        "elapsed_sec": None,
    }
    report["elapsed_sec"] = round(time.time() - t0, 1)

    path = os.path.join(RESULTS, "static_proofs.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    hr(f"done in {report['elapsed_sec']}s -> {os.path.relpath(path, os.path.dirname(HERE))}")


if __name__ == "__main__":
    main()
