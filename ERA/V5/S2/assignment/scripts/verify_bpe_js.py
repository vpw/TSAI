"""Generate fixtures of (text -> expected token ids) from the real HF word
tokenizer, for cross-checking site/bpe.js's from-scratch JS re-implementation
in Node before it's trusted in the browser. Not part of the deployed site.
"""

import json
import pathlib

from tokenizers import Tokenizer

from common import CORPUS_DIR, LANGS, TOKENIZER_DIR, load_corpora

OUT_PATH = pathlib.Path(__file__).resolve().parent.parent / "site" / "data" / "_bpe_fixtures.json"

EDGE_CASES = [
    "test_variable place3 don't full-width ellipsis... 12.5%",
    "hello \U0001F642 zzzqx",
    "India, officially the Republic of India, is a country in South Asia.",
    "",
    "   ",
    "a",
]


def main() -> None:
    texts = load_corpora()
    tokenizer = Tokenizer.from_file(str(TOKENIZER_DIR / "word" / "tokenizer.json"))

    cases = list(EDGE_CASES)
    for lang in LANGS:
        cases.append(texts[lang][:500])

    fixtures = []
    for text in cases:
        enc = tokenizer.encode(text)
        fixtures.append({"text": text, "ids": enc.ids, "tokens": enc.tokens})

    OUT_PATH.write_text(json.dumps(fixtures, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(fixtures)} fixtures -> {OUT_PATH}")


if __name__ == "__main__":
    main()
