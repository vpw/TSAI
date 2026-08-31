# India Wikipedia Faithful BPE Tokenizer

This is a reproducible ERA V5 Assignment 2 submission: one standard Hugging Face BPE tokenizer with a shared 10,000-token vocabulary for the India Wikipedia pages in English, Hindi, Telugu, and Kannada.

## Published result

| Language | Tokens | Faithful units | Fertility |
|---|---:|---:|---:|
| English | 114,928 | 186,367 | 0.616676 |
| Hindi | 52,411 | 88,359 | 0.593160 |
| Telugu | 24,582 | 36,292 | 0.677339 |
| Kannada | 10,136 | 12,293 | 0.824534 |

```text
Spread = 0.824534 - 0.593160 = 0.231375
Score  = 1000 / 0.231375 = 4322.00
Hindi penalty factor = exp(max(0, 0.593160 / 1.2 - 1)) = 1.000000
Hindi-adjusted score = 4322.00 / 1.000000 = 4322.00
```

Because Hindi fertility is below the 1.2 threshold, the exponential Hindi
penalty is neutral and does not reduce the score.

The evaluator defines a faithful unit as one contiguous Unicode letter/mark/number run or one visible non-space punctuation/symbol character.

The training weights adapt the reference method by script rather than blindly assigning Maithili's weight to Kannada:

```json
{"en": 3, "hi": 4, "te": 4, "kn": 4}
```

English has the largest source corpus, so it remains at 3. Hindi and Telugu receive 4 to protect their Indic-script merge capacity. The reference could give Maithili 2 because Maithili shares Devanagari with Hindi; Kannada uses a separate script and therefore receives 4, like Telugu. These weights are declared before training and are not tuned to minimize the final spread.

## Setup and verification

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python verify_submission.py
```

The final command verifies that:

- `tokenizer.json` loads with `Tokenizer.from_file`;
- the vocabulary contains exactly 10,000 tokens;
- punctuation, brackets, URLs, apostrophes, number separators, and multilingual samples survive `decode(encode(text))` after the tokenizer's documented NFKC normalization;
- unseen Unicode characters are represented through standard UTF-8 BPE byte fallback instead of being replaced by a disappearing `[UNK]` token;
- all four faithful corpus snapshots round-trip to the same normalized visible text;
- the independently recomputed metrics equal `metrics.json`.

## Reproduce from the included snapshots

```bash
.venv/bin/python train_tokenizer.py
.venv/bin/python evaluate_tokenizer.py
```

Rebuilding is deterministic for the pinned dependency versions and produces the published score. To refresh Wikipedia data first, run:

```bash
.venv/bin/python build_wiki_faithful_markdown.py
```

Refreshing the pages can change the corpus and therefore the metrics. For grading, use the included corpus snapshots.

## Authoritative submission files

- `tokenizer.json` — standard, self-contained tokenizer with BPE byte fallback, NFKC normalization, Metaspace pre-tokenization, and chained Metaspace/ByteFallback decoding
- `vocab.txt` — all 10,000 tokens ordered by ID
- `metrics.json` — token counts, faithful-unit counts, ratios, spread, raw score, Hindi penalty, and adjusted score
- `corpus/*.faithful.md` — exact wiki-faithful Markdown snapshots
- `corpus/*.faithful.txt` — the same snapshots used as tokenizer input
- `corpus/*.raw.html` — archived Wikipedia REST HTML used for the conversion
- `corpus/*.meta.json` — source URLs, generation metadata, and faithful-unit counts
- `build_wiki_faithful_markdown.py` — corpus construction
- `train_tokenizer.py` — training
- `evaluate_tokenizer.py` — independent evaluator
- `verify_submission.py` — acceptance and reproducibility checks

The browser widget uses `tokenizer-data.js`, generated directly from the same
standard `tokenizer.json` and `metrics.json` supplied to the grader. Obsolete
custom-tokenizer artifacts have been removed from this submission.

`train_tokenizer.py` also generates `tokenizer-data.js`, which embeds that exact standard tokenizer, published metrics, and short corpus samples for the widget. This preserves the downloadable `tokenizer.json` as the authoritative grader artifact while allowing the page to initialize without module imports or runtime `fetch()` calls.
