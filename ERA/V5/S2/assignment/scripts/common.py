"""Shared corpus loading + per-language weighting used by the tokenizer
training/tuning scripts, so every run is trained on a comparably-weighted
mixture and results are directly comparable.

Generic over the language set (`langs`) so the same helpers serve both the
Marathi-as-4th-language run and the Bengali-as-4th-language run.
"""

import pathlib
import unicodedata

BASE_LANGS = ["en", "hi", "te"]
FOURTH_LANG_CANDIDATES = ["mr", "bn"]

CORPUS_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "corpus"
TOKENIZER_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "tokenizer"

DEFAULT_ALPHA = 0.3
DEFAULT_VOCAB_SIZE = 10000


def langset(fourth: str) -> list[str]:
    return BASE_LANGS + [fourth]


def load_corpora(langs: list[str]) -> dict[str, str]:
    texts = {}
    for lang in langs:
        path = CORPUS_DIR / f"{lang}.txt"
        texts[lang] = unicodedata.normalize("NFC", path.read_text(encoding="utf-8"))
    return texts


def compute_repeat_counts(texts: dict[str, str], alpha: float = DEFAULT_ALPHA) -> dict[str, int]:
    """XLM-R-style exponential smoothing: target share ~ size**alpha instead
    of ~ size, expressed as an integer repeat count per language so the
    larger/simpler corpora (English) don't dominate merge learning."""
    sizes = {lang: len(text) for lang, text in texts.items()}
    total = sum(sizes.values())
    smoothed = {lang: sizes[lang] ** alpha for lang in sizes}
    smoothed_total = sum(smoothed.values())

    raw_share = {lang: sizes[lang] / total for lang in sizes}
    target_share = {lang: smoothed[lang] / smoothed_total for lang in sizes}
    factor = {lang: target_share[lang] / raw_share[lang] for lang in sizes}

    min_factor = min(factor.values())
    return {lang: max(1, round(factor[lang] / min_factor)) for lang in sizes}


def weighted_lines(texts: dict[str, str], repeat_counts: dict[str, int]):
    """Yield each language's text, repeated per its weight, as separate
    training 'documents'."""
    for lang, text in texts.items():
        for _ in range(repeat_counts[lang]):
            yield text
