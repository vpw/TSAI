"""Compute per-language fertility and the assignment self-score for a
trained tokenizer, mirroring the instructor's *corrected* reference formula
(`../refsol/instructions.md`, which supersedes the stale `tokenizer.json` /
`SOLUTION.md` / `evaluate_tokenizer.py` files still sitting in that folder --
see PLAN_PHASE2.md for why those are out of date):

  faithful_unit = one contiguous Unicode letter/mark/number run, OR one
                  visible non-space punctuation/symbol character (each
                  counted separately -- this is NOT the same as the earlier
                  "wordish_units" definition, which ignored punctuation
                  entirely and produced roughly half as many units).
  fertility(lang) = token_count(lang) / faithful_unit_count(lang)
  spread           = max(fertility) - min(fertility)
  score            = 1000 / spread

The assignment's explicit constraint is on English specifically ("X1 ...
must be around 1.2 or less"). Under this corrected denominator every
language's fertility lands well under 1.0 (the corrected reference gets
0.58-0.73), so the constraint is essentially always satisfied -- it's kept
here as a reported sanity check, not a binding optimization target. We also
report the English-anchored penalty shape from the grading feedback for
completeness:

  english_penalty = exp(max(0, english_fertility/1.2 - 1))
  adjusted_score  = raw_score / english_penalty
"""

import argparse
import json
import math

import regex
from tokenizers import Tokenizer

from common import TOKENIZER_DIR, langset, load_corpora

STATS_DIR = TOKENIZER_DIR.parent
FAITHFUL_UNIT_PATTERN = regex.compile(r"[\p{L}\p{M}\p{N}]+|[^\s\p{L}\p{M}\p{N}]")

LANG_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "te": "Telugu",
    "mr": "Marathi",
    "bn": "Bengali",
}


def faithful_units(text: str) -> int:
    return len(FAITHFUL_UNIT_PATTERN.findall(text))


def compute_stats(fourth: str, tokenizer_path=None) -> dict:
    langs = langset(fourth)
    texts = load_corpora(langs)
    tok_path = tokenizer_path or (TOKENIZER_DIR / fourth / "tokenizer.json")
    tokenizer = Tokenizer.from_file(str(tok_path))

    rows = {}
    for lang in langs:
        text = texts[lang]
        units = faithful_units(text)
        tokens = len(tokenizer.encode(text).ids)
        rows[lang] = {
            "language": LANG_NAMES[lang],
            "faithful_units": units,
            "token_count": tokens,
            "fertility": tokens / units,
        }

    sorted_langs = sorted(rows, key=lambda l: rows[l]["fertility"])
    f_min = rows[sorted_langs[0]]["fertility"]
    f_max = rows[sorted_langs[-1]]["fertility"]
    spread = f_max - f_min
    raw_score = 1000 / spread if spread > 0 else float("inf")

    en_fertility = rows["en"]["fertility"]
    english_penalty = math.exp(max(0.0, en_fertility / 1.2 - 1.0))
    adjusted_score = raw_score / english_penalty

    return {
        "fourth_language": fourth,
        "langs": langs,
        "vocab_size": tokenizer.get_vocab_size(),
        "rows": rows,
        "sorted_by_fertility": sorted_langs,
        "f_min": f_min,
        "f_max": f_max,
        "f_min_lang": sorted_langs[0],
        "f_max_lang": sorted_langs[-1],
        "spread": spread,
        "score": raw_score,
        "english_fertility": en_fertility,
        "english_meets_1_2": en_fertility <= 1.2,
        "english_penalty_factor": english_penalty,
        "adjusted_score": adjusted_score,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fourth", default="mr", choices=["mr", "bn"])
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    stats = compute_stats(args.fourth)
    print(f"\n=== fourth={args.fourth} (vocab_size={stats['vocab_size']}) ===")
    for lang in stats["langs"]:
        r = stats["rows"][lang]
        print(f"  {lang:3s} {r['language']:10s} units={r['faithful_units']:6d} tokens={r['token_count']:6d} fertility={r['fertility']:.6f}")
    print(f"  sorted: {stats['sorted_by_fertility']}")
    print(f"  spread = {stats['f_max']:.6f} - {stats['f_min']:.6f} = {stats['spread']:.6f}")
    print(f"  raw score = 1000 / spread = {stats['score']:.2f}")
    print(f"  English fertility = {stats['english_fertility']:.6f} (<=1.2: {stats['english_meets_1_2']})")
    print(f"  English penalty factor = {stats['english_penalty_factor']:.6f}")
    print(f"  adjusted score = {stats['adjusted_score']:.2f}")

    out_path = args.out or (STATS_DIR / f"stats_{args.fourth}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
