"""Round-trip fidelity self-test: decode(encode(text)) must preserve the
same visible non-whitespace characters, for every language and a battery of
adversarial probe strings (URLs, refs, hyphenated/underscored tokens,
mixed-script lines). This is the exact gate the phase-1 submission failed
("Score is 0 because the tokenizer does not satisfy the faithful roundtrip
gate").

Run this after training, before exporting/shipping a tokenizer. Non-zero
exit on any failure.
"""

import argparse
import sys
import unicodedata

from tokenizers import Tokenizer

from common import TOKENIZER_DIR, langset, load_corpora

# The exact sample shape from the grading feedback, plus generic adversarial
# probes covering URL/markdown punctuation, hyphens/underscores, digits,
# mixed scripts, and (documented, not asserted-safe) an emoji case.
PROBES = [
    "https://hi.wikipedia.org/wiki/भारत#cite_ref-1",
    "https://en.wikipedia.org/wiki/India_(disambiguation)",
    "See [India (disambiguation)](https://en.wikipedia.org/wiki/India_(disambiguation) \"India (disambiguation)\")",
    "Category: ./Category:Articles_with_short_description",
    "co-operative_societies-2024",
    "GDP: $3.7 trillion (2023) [1][2]",
    "भारत (आधिकारिक नाम: भारत गणराज्य) — Republic of India",
    "1,428,627,663 people (2023 est.)",
    "పేజీ ID: 12345#section-3",
]

# Characters genuinely outside every training corpus (e.g. emoji) still fall
# back to [UNK] and are NOT expected to round-trip -- this is surfaced
# explicitly rather than silently, matching the phase-1 Coverage Inspector
# behavior for unk_token.
KNOWN_UNSAFE_PROBES = ["🇮🇳 India"]


def strip_ws(text: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFKC", text) if not ch.isspace())


def check(tokenizer: Tokenizer, text: str) -> tuple[bool, str, str]:
    enc = tokenizer.encode(text)
    decoded = tokenizer.decode(enc.ids)
    orig_nw = strip_ws(text)
    dec_nw = strip_ws(decoded)
    return orig_nw == dec_nw, orig_nw, dec_nw


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fourth", default="mr", choices=["mr", "bn"])
    args = parser.parse_args()

    langs = langset(args.fourth)
    tok_path = TOKENIZER_DIR / args.fourth / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tok_path))
    texts = load_corpora(langs)

    failures = []

    print(f"=== round-trip check: fourth={args.fourth} ===")
    for probe in PROBES:
        ok, orig, dec = check(tokenizer, probe)
        status = "OK" if ok else "FAIL"
        print(f"  [{status}] {probe!r}")
        if not ok:
            print(f"         orig(no-ws): {orig!r}")
            print(f"         dec (no-ws): {dec!r}")
            failures.append(probe)

    for lang in langs:
        sample = texts[lang][:2000]
        ok, orig, dec = check(tokenizer, sample)
        status = "OK" if ok else "FAIL"
        print(f"  [{status}] corpus sample [{lang}] (first 2000 chars)")
        if not ok:
            failures.append(f"corpus:{lang}")

    print("\n--- known-unsafe probes (documented, not asserted) ---")
    for probe in KNOWN_UNSAFE_PROBES:
        ok, orig, dec = check(tokenizer, probe)
        print(f"  [{'OK' if ok else 'expected-fail (UNK)'}] {probe!r} -> decoded {dec!r}")

    if failures:
        print(f"\nFAILED: {len(failures)} probe(s) did not round-trip: {failures}")
        return 1
    print("\nAll round-trip probes passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
