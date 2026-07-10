"""Export everything the static frontend needs into site/data/*.json:

  vocab_word.json  - id->token array + ordered merge list (rank = index) for
                      the winning word-level BPE tokenizer, plus per-token
                      script classification and which language(s)' unique
                      word lists actually use each token id.
  stats.json        - the X1-X4 table for all four trained variants (byte,
                      char, word, sentencepiece), copied through so the
                      Overview tab can show the comparison, not just the
                      winner.
  samples.json      - a short quick-pick sample paragraph per language plus
                      basic corpus metadata (char/word counts, source title),
                      for the Coverage Inspector tab.
  run_config.json   - the winning run's exact knobs (repeat_counts,
                      min_frequency, vocab_size) so the Methodology tab shows
                      numbers pulled from the real artifact, not hardcoded
                      prose.

Everything here is static data computed once at build time; the browser only
ever loads JSON and runs BPE merges client-side (see site/bpe.js).
"""

import json
import pathlib

import regex
from tokenizers import Tokenizer

from common import CORPUS_DIR, LANGS, TOKENIZER_DIR, load_corpora

SITE_DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "site" / "data"
STATS_PATH = TOKENIZER_DIR.parent / "stats.json"
WORD_PATTERN = regex.compile(r"[\p{L}\p{M}\p{N}]+")

LANG_NAMES = {"en": "English", "hi": "Hindi", "te": "Telugu", "mr": "Marathi"}
SOURCE_TITLES = {"en": "India", "hi": "भारत", "te": "భారతదేశం", "mr": "भारत"}

# Winning config, confirmed by scripts/tune_weights.py (see PLAN.md "Redo results").
WINNING_REPEAT_COUNTS = {"en": 1, "hi": 1, "te": 12, "mr": 1}
WINNING_MIN_FREQUENCY = 1
WINNING_VOCAB_SIZE = 10000


def classify_script(token: str) -> str:
    for ch in token:
        cp = ord(ch)
        if 0x0900 <= cp <= 0x097F:
            return "devanagari"
        if 0x0C00 <= cp <= 0x0C7F:
            return "telugu"
        if ("A" <= ch <= "Z") or ("a" <= ch <= "z"):
            return "latin"
    if token.strip() == "":
        return "whitespace"
    if any(ch.isdigit() for ch in token):
        return "digit"
    return "punct/other"


def build_vocab_export(texts: dict[str, str]) -> dict:
    path = TOKENIZER_DIR / "word" / "tokenizer.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    vocab: dict[str, int] = raw["model"]["vocab"]
    merges = raw["model"]["merges"]

    id_to_token = [None] * len(vocab)
    for tok, idx in vocab.items():
        id_to_token[idx] = tok

    tokenizer = Tokenizer.from_file(str(path))
    token_langs: list[set] = [set() for _ in id_to_token]
    for lang in LANGS:
        words = sorted(set(WORD_PATTERN.findall(texts[lang])))
        for word in words:
            for tid in tokenizer.encode(word).ids:
                token_langs[tid].add(lang)

    tokens_export = []
    for idx, tok in enumerate(id_to_token):
        tokens_export.append(
            {
                "id": idx,
                "token": tok,
                "script": classify_script(tok),
                "langs": sorted(token_langs[idx]),
                "isMerged": idx >= (len(id_to_token) - len(merges)),
            }
        )

    return {
        "vocab_size": len(id_to_token),
        "tokens": tokens_export,
        "merges": [list(pair) for pair in merges],
    }


def build_stats_export() -> dict:
    return json.loads(STATS_PATH.read_text(encoding="utf-8"))


def build_samples_export(texts: dict[str, str]) -> dict:
    out = {}
    for lang in LANGS:
        text = texts[lang]
        words = WORD_PATTERN.findall(text)
        # ~60 words is enough to see interesting merge behavior without
        # overwhelming the highlighted-span view.
        cutoff = 0
        word_count = 0
        for m in regex.finditer(r"[\p{L}\p{M}\p{N}]+", text):
            word_count += 1
            if word_count >= 60:
                cutoff = m.end()
                break
        sample = text[:cutoff] if cutoff else text[:400]
        out[lang] = {
            "name": LANG_NAMES[lang],
            "source_title": SOURCE_TITLES[lang],
            "char_count": len(text),
            "word_count": len(words),
            "sample": sample,
        }
    return out


def build_run_config() -> dict:
    return {
        "variant": "word",
        "repeat_counts": WINNING_REPEAT_COUNTS,
        "min_frequency": WINNING_MIN_FREQUENCY,
        "vocab_size": WINNING_VOCAB_SIZE,
    }


def main() -> None:
    SITE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    texts = load_corpora()

    vocab_export = build_vocab_export(texts)
    (SITE_DATA_DIR / "vocab_word.json").write_text(
        json.dumps(vocab_export, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[vocab_word.json] {len(vocab_export['tokens'])} tokens, {len(vocab_export['merges'])} merges")

    stats_export = build_stats_export()
    (SITE_DATA_DIR / "stats.json").write_text(
        json.dumps(stats_export, ensure_ascii=False), encoding="utf-8"
    )
    print("[stats.json] copied")

    samples_export = build_samples_export(texts)
    (SITE_DATA_DIR / "samples.json").write_text(
        json.dumps(samples_export, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("[samples.json] written")

    run_config = build_run_config()
    (SITE_DATA_DIR / "run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("[run_config.json] written")

    total_bytes = sum((SITE_DATA_DIR / f).stat().st_size for f in ["vocab_word.json", "stats.json", "samples.json", "run_config.json"])
    print(f"Total site/data size: {total_bytes / 1024:.1f} KB")


if __name__ == "__main__":
    main()
