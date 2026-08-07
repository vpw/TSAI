#!/usr/bin/env python3
"""Derive stop-word lists for the languages the S4 cleaner has none for.

The S4 quality stage checks "does this document contain at least 2 common words of its
language". `stopword_set()` falls back to the English list when a language is missing, so a
Kannada document is checked for English stop-words, finds none, and is dropped. On the S5
top-up slice that silently rejected ~95% of kan/guj/mal/pan/ory.

The lists are counted out of the corpus rather than written from memory: the most frequent
short word forms in a language's own documents are its function words, and a corpus-derived
list is the right thing for a rule that only asks "does this look like the language at all".

Reads the raw stage of the current run, writes `data/run/derived_stopwords.json`.
"""
import collections
import gzip
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW_STAGE = os.path.join(ROOT, "data", "stage", "00-raw.jsonl.gz")
OUT = os.path.join(ROOT, "data", "run", "derived_stopwords.json")

# Tokenise exactly as quality.py does -- whitespace split, then strip the punctuation it
# strips. Using \w+ here would be wrong: Python's \w does not match Indic combining vowel
# signs, so it shatters every Indic word at each matra and the "words" come out as single
# consonants.
WORD_RE = re.compile(r"\S+")
STRIP = ".,;:!?()[]{}\"'“”‘’।॥"
TOP_N = 25
MAX_LEN = 8          # function words are short; this keeps content nouns out
MIN_DOCS_SEEN = 200


def words_of(text):
    for w in WORD_RE.findall(text):
        w = w.strip(STRIP)
        if w and len(w) <= MAX_LEN:
            yield w


def main():
    want = sys.argv[1:] or ["kan", "guj", "mal", "pan", "ory"]
    counts = collections.defaultdict(collections.Counter)
    doc_freq = collections.defaultdict(collections.Counter)
    n_docs = collections.Counter()

    with gzip.open(RAW_STAGE, "rt") as f:
        for line in f:
            r = json.loads(line)
            lang = r.get("claimed_lang")
            if lang not in want:
                continue
            n_docs[lang] += 1
            words = list(words_of(r.get("text", "")[:20000]))
            counts[lang].update(words)
            doc_freq[lang].update(set(words))

    out = {}
    for lang in want:
        if n_docs[lang] < MIN_DOCS_SEEN:
            print(f"  {lang}: only {n_docs[lang]} docs seen, skipping")
            continue
        # Rank by document frequency, not raw count: a word repeated many times inside one
        # page is boilerplate, a word appearing in most pages is a function word.
        ranked = [w for w, _ in doc_freq[lang].most_common(TOP_N * 3)]
        keep = [w for w in ranked if not w.isdigit() and not w.isascii()][:TOP_N]
        out[lang] = keep
        cov = doc_freq[lang][keep[0]] / n_docs[lang] if keep else 0
        print(f"  {lang}: {n_docs[lang]:5d} docs -> {len(keep)} stop-words "
              f"(top word in {cov:.0%} of docs)")
        print(f"      {' '.join(keep[:12])}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump({"method": "document-frequency rank over the run's own raw stage, "
                             f"words of <= {MAX_LEN} chars, top {TOP_N}",
                   "docs_seen": dict(n_docs), "stopwords": out}, f,
                  ensure_ascii=False, indent=1)
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
