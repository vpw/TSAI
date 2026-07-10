"""Grid-search per-language oversampling weights for word-level BPE and
SentencePiece BPE, optimizing for the assignment's actual objective:

  1. English's X1 must land at <= ~1.2 (explicit requirement).
  2. Given that, minimize the worst-case gap between English and whichever
     of Hindi/Telugu/Marathi is furthest from it -- that worst outlier is
     what sets X_max/X_min alongside English and drives the score.

English's repeat count is held at 1 (already the minimum possible) since
the lever here is increasing hi/te/mr's relative training weight, not
decreasing English's -- and the first pass showed Telugu, specifically, as
the hardest to pull up, so its range is swept widest.
"""

import contextlib
import itertools
import json
import os

from common import DEFAULT_VOCAB_SIZE, LANGS, TOKENIZER_DIR, load_corpora
from compute_metrics import compute_variant_stats
from train_sentencepiece import train_sentencepiece_variant
from train_tokenizer import train_variant

EN_REPEAT = 1
HI_RANGE = [1, 2, 3]
TE_RANGE = [6, 8, 10, 12, 14, 16, 18, 20, 24, 28]
MR_RANGE = [1, 2, 3, 4]

TARGET_X1_CEILING = 1.2
VARIANTS = ["word", "sentencepiece"]
# Only the HF-based "word" variant exposes this knob; lowering it from the
# default 2 lets the trainer learn rarer, English-specific merges instead of
# always reusing a handful of highly frequent shared pieces, which is what
# closes most of the gap to English's <=1.2 target.
WORD_MIN_FREQUENCY = 1

STATS_DIR = TOKENIZER_DIR.parent
SWEEP_PATH = STATS_DIR / "tune_sweep.json"


@contextlib.contextmanager
def suppress_native_stderr():
    """SentencePiece's C++ trainer logs directly to fd 2, bypassing
    sys.stderr, so silencing it needs an OS-level fd swap."""
    stderr_fd = os.dup(2)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull_fd, 2)
    try:
        yield
    finally:
        os.dup2(stderr_fd, 2)
        os.close(stderr_fd)
        os.close(devnull_fd)


def evaluate(variant: str, texts: dict[str, str], repeat_counts: dict[str, int], vocab_size: int):
    if variant == "sentencepiece":
        with suppress_native_stderr():
            train_sentencepiece_variant(texts, repeat_counts, vocab_size, TOKENIZER_DIR / "sentencepiece")
    else:
        min_frequency = WORD_MIN_FREQUENCY if variant == "word" else 2
        tokenizer = train_variant(variant, texts, repeat_counts, vocab_size, min_frequency=min_frequency)
        out_dir = TOKENIZER_DIR / variant
        out_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save(str(out_dir / "tokenizer.json"))

    stats = compute_variant_stats(variant, texts)
    x1 = stats["per_language"]["en"]["X"]
    worst_gap = max(abs(stats["per_language"][l]["X"] - x1) for l in ["hi", "te", "mr"])
    return stats, x1, worst_gap


def is_better(candidate: dict, current: dict | None) -> bool:
    if current is None:
        return True
    cand_ok = candidate["X1"] <= TARGET_X1_CEILING
    cur_ok = current["X1"] <= TARGET_X1_CEILING
    if cand_ok and not cur_ok:
        return True
    if cand_ok == cur_ok:
        return candidate["worst_gap"] < current["worst_gap"]
    return False


def main() -> None:
    texts = load_corpora()
    results = {v: [] for v in VARIANTS}
    best: dict[str, dict | None] = {v: None for v in VARIANTS}

    combos = list(itertools.product(HI_RANGE, TE_RANGE, MR_RANGE))
    total = len(combos) * len(VARIANTS)
    print(f"Sweeping {len(combos)} weight combos x {len(VARIANTS)} variants ({total} runs)...")

    done = 0
    for hi, te, mr in combos:
        repeat_counts = {"en": EN_REPEAT, "hi": hi, "te": te, "mr": mr}
        for variant in VARIANTS:
            stats, x1, worst_gap = evaluate(variant, texts, repeat_counts, DEFAULT_VOCAB_SIZE)
            record = {
                "repeat_counts": dict(repeat_counts),
                "X1": x1,
                "worst_gap": worst_gap,
                "spread": stats["spread"],
                "score": stats["score"],
                "per_language": stats["per_language"],
            }
            results[variant].append(record)
            if is_better(record, best[variant]):
                best[variant] = record
            done += 1
        if done % 40 == 0:
            print(f"  ...{done}/{total} runs done")

    for variant in VARIANTS:
        b = best[variant]
        print(f"\n=== best for {variant} ===")
        print(f"  repeat_counts={b['repeat_counts']}")
        print(f"  X1(en)={b['X1']:.4f}  worst_gap={b['worst_gap']:.4f}  spread={b['spread']:.4f}  score={b['score']:.2f}")
        for lang in LANGS:
            pl = b["per_language"][lang]
            print(f"    {lang}: X={pl['X']:.4f}")

    SWEEP_PATH.write_text(json.dumps({"best": best, "results": results}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {SWEEP_PATH}")

    print("\nRe-training winning models to leave on disk...")
    for variant in VARIANTS:
        evaluate(variant, texts, best[variant]["repeat_counts"], DEFAULT_VOCAB_SIZE)
    print("Done: data/tokenizer/word/ and data/tokenizer/sentencepiece/ hold the winning models.")


if __name__ == "__main__":
    main()
