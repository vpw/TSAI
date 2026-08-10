#!/usr/bin/env python3
"""Cut a compact, self-contained corpus subset for the GPU run.

S5's tokenized corpus is 3.9 GB, most of which this ablation never touches: at 30M tokens per
arm and six lanes, the largest lane needs ~10M of its 224M tokens. Uploading the whole thing
would dominate the wall-clock cost of the experiment.

This writes `data/corpus/` containing, per lane, a contiguous slice sized to a stated multiple
of what the mixture will actually draw, plus the untouched validation split. Contiguous rather
than random so the slice is reproducible from a two-number description (lane, token count) and
so document boundaries inside it stay intact.

    .venv/bin/python scripts/prep_subset.py --tokens-per-arm 30000000 --headroom 2.0
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kv2.data import _S5_TOKENS, TRAIN_MIX, VAL_LANES, token_stats

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "data", "corpus")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens-per-arm", type=int, default=30_000_000)
    ap.add_argument("--headroom", type=float, default=2.0,
                    help="multiple of the drawn budget to keep, so lanes are not over-repeated")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    # Always cut from the full S5 corpus, never from a previously-cut subset -- otherwise a
    # second run would silently slice a slice.
    src = os.path.normpath(_S5_TOKENS)
    stats = token_stats(src)
    total_w = sum(TRAIN_MIX.values())

    manifest = {
        "source": src,
        "tokenizer": stats["tokenizer"],
        "vocab_size": stats["vocab_size"],
        "train_dtype": stats["train_dtype"],
        "tokens_per_arm": args.tokens_per_arm,
        "headroom": args.headroom,
        "mixture": TRAIN_MIX,
        "lanes": {},
    }

    grand = 0
    for lane, weight in sorted(TRAIN_MIX.items()):
        drawn = args.tokens_per_arm * weight / total_w
        want = int(drawn * args.headroom)
        arr = np.memmap(os.path.join(src, f"{lane}.train.bin"), dtype=np.uint32, mode="r")
        take = min(want, len(arr))
        dst = os.path.join(args.out, f"{lane}.train.bin")
        np.asarray(arr[:take], dtype=np.uint32).tofile(dst)
        grand += take
        epochs = drawn / take
        manifest["lanes"][lane] = {
            "weight_pct": weight,
            "tokens_drawn_per_arm": int(drawn),
            "tokens_kept": int(take),
            "source_tokens": int(len(arr)),
            "epochs_over_kept_slice": round(epochs, 3),
            "truncated_by_supply": take < want,
        }
        print(f"  {lane:<24} keep {take/1e6:>7.1f}M of {len(arr)/1e6:>7.1f}M   "
              f"draws {drawn/1e6:>5.1f}M/arm  ({epochs:.2f} epochs)")

    for lane in VAL_LANES:
        s = os.path.join(src, f"{lane}.val.npz")
        if os.path.exists(s):
            shutil.copy2(s, os.path.join(args.out, f"{lane}.val.npz"))

    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    size = sum(os.path.getsize(os.path.join(args.out, f)) for f in os.listdir(args.out))
    print(f"\n  {grand/1e6:.1f}M train tokens + val splits -> {args.out}")
    print(f"  on disk: {size/1e6:.0f} MB")


if __name__ == "__main__":
    main()
