"""Generate fixtures of (text -> expected token ids/tokens -> decoded text)
from the real HF Metaspace tokenizer, for cross-checking site/bpe.js's
from-scratch JS re-implementation in Node before it's trusted in the
browser. Not part of the deployed site.
"""

import argparse
import json
import pathlib

from tokenizers import Tokenizer

from common import TOKENIZER_DIR, langset, load_corpora
from verify_roundtrip import KNOWN_UNSAFE_PROBES, PROBES

OUT_PATH = pathlib.Path(__file__).resolve().parent.parent / "site" / "data" / "_bpe_fixtures.json"

EDGE_CASES = [
    "test_variable place3 don't full-width ellipsis... 12.5%",
    "hello \U0001F642 zzzqx",
    "India, officially the Republic of India, is a country in South Asia.",
    "",
    "   ",
    "a",
    " a",
    "  a",
    "a  b",
    "a\nb",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fourth", default="bn", choices=["mr", "bn"])
    args = parser.parse_args()
    fourth = args.fourth
    langs = langset(fourth)
    texts = load_corpora(langs)
    tokenizer = Tokenizer.from_file(str(TOKENIZER_DIR / fourth / "tokenizer.json"))

    cases = list(EDGE_CASES) + list(PROBES) + list(KNOWN_UNSAFE_PROBES)
    for lang in langs:
        cases.append(texts[lang][:500])

    fixtures = []
    for text in cases:
        enc = tokenizer.encode(text)
        decoded = tokenizer.decode(enc.ids)
        fixtures.append({"text": text, "ids": enc.ids, "tokens": enc.tokens, "decoded": decoded})

    OUT_PATH.write_text(json.dumps(fixtures, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(fixtures)} fixtures -> {OUT_PATH}")


if __name__ == "__main__":
    main()
