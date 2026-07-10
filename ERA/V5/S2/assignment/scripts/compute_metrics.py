"""Compute per-language X ratios and the assignment self-score for each of
the 4 trained tokenizer variants (byte/char/word BPE + SentencePiece BPE),
so they can be compared side by side.

X_lang = (unique words in the language's article)
         / (distinct BPE token IDs needed to spell all of those unique words)

Score = 1000 / (X_max - X_min) across the four languages, for a given
tokenizer variant.
"""

import json

import regex
import sentencepiece as spm
from tokenizers import Tokenizer

from common import LANGS, TOKENIZER_DIR, load_corpora

STATS_PATH = TOKENIZER_DIR.parent / "stats.json"
WORD_PATTERN = regex.compile(r"[\p{L}\p{M}\p{N}]+")

HF_VARIANTS = ["byte", "char", "word"]
ALL_VARIANTS = HF_VARIANTS + ["sentencepiece"]


def unique_words(text: str) -> list[str]:
    return sorted(set(WORD_PATTERN.findall(text)))


class HFEncoder:
    def __init__(self, variant: str):
        path = TOKENIZER_DIR / variant / "tokenizer.json"
        self.tokenizer = Tokenizer.from_file(str(path))

    def encode_ids(self, word: str) -> list[int]:
        return self.tokenizer.encode(word).ids

    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()


class SPEncoder:
    def __init__(self):
        path = TOKENIZER_DIR / "sentencepiece" / "spm.model"
        self.sp = spm.SentencePieceProcessor(model_file=str(path))

    def encode_ids(self, word: str) -> list[int]:
        return self.sp.encode(word, out_type=int)

    def vocab_size(self) -> int:
        return self.sp.vocab_size()


def load_encoder(variant: str):
    if variant == "sentencepiece":
        return SPEncoder()
    return HFEncoder(variant)


def compute_variant_stats(variant: str, texts: dict[str, str]) -> dict:
    encoder = load_encoder(variant)
    per_language = {}
    for lang in LANGS:
        words = unique_words(texts[lang])
        distinct_ids = set()
        for word in words:
            distinct_ids.update(encoder.encode_ids(word))
        x_value = len(words) / len(distinct_ids)
        per_language[lang] = {
            "unique_words": len(words),
            "distinct_tokens_used": len(distinct_ids),
            "X": x_value,
        }

    sorted_langs = sorted(per_language, key=lambda l: per_language[l]["X"])
    x_min = per_language[sorted_langs[0]]["X"]
    x_max = per_language[sorted_langs[-1]]["X"]
    score = 1000 / (x_max - x_min) if x_max != x_min else float("inf")

    return {
        "vocab_size": encoder.vocab_size(),
        "per_language": per_language,
        "sorted_by_X": sorted_langs,
        "X_min": x_min,
        "X_max": x_max,
        "spread": x_max - x_min,
        "score": score,
    }


def main() -> None:
    texts = load_corpora()
    results = {}
    for variant in ALL_VARIANTS:
        stats = compute_variant_stats(variant, texts)
        results[variant] = stats
        print(f"\n=== {variant} (vocab_size={stats['vocab_size']}) ===")
        for lang in LANGS:
            pl = stats["per_language"][lang]
            print(f"  {lang}: words={pl['unique_words']:6d} tokens_used={pl['distinct_tokens_used']:6d} X={pl['X']:.4f}")
        print(f"  sorted: {stats['sorted_by_X']}  spread={stats['spread']:.4f}  score={stats['score']:.2f}")

    STATS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {STATS_PATH}")


if __name__ == "__main__":
    main()
