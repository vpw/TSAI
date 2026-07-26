# Run report — cleaning a slice of ai4bharat/sangraha

Generated 2026-07-26T12:12:02Z from `data/run/stats.json`. Full run. Total runtime 4.2 min on 12 CPU cores, no GPU.

## Headline

| | |
|---|---:|
| Dataset | `ai4bharat/sangraha` @ `8b813c3f62d3` |
| Licence | CC-BY-4.0 |
| Documents in → out | 64,712 → 60,174 |
| Tokens in → out (sarvam1) | 48,008,963 → 42,889,153 |
| Document retention | 92.99% |
| Token retention | 89.34% |
| Shards emitted / admitted | 13 / 13 |

## Yield descent

| Stage | Docs out | Dropped | % of raw |
|---|---:|---:|---:|
| 0. Raw slice | 64,712 | — | 100.0% |
| 1. Extract *(inherited)* | 64,712 | — | 100.0% |
| 2. Normalize | 64,712 | — | 100.0% |
| 3. Language ID | 64,268 | 444 | 99.31% |
| 4a. Quality (Gopher/C4) | 60,943 | 3,325 | 94.18% |
| 4b. Quality (classifier) | 60,909 | 34 | 94.12% |
| 5. Deduplicate | 60,631 | 278 | 93.69% |
| 6. PII scrub | 60,631 | — | 93.69% |
| 7. Decontaminate | 60,174 | 457 | 92.99% |
| 8. Manifest | 60,174 | — | 92.99% |

Session reference curve, for shape comparison: 100 → 92 → 88 → 61 → 44 → 43 → 42 → 42 → 42.

## Stage 2 — normalize

- Noise characters removed: **117** ({'bidi': 46, 'replacement': 48, 'control': 7, 'private_use': 15, 'zero_width': 1})
- Indic joiners kept: {'ZWJ': 231}
- Documents changed by NFC: 276; with HTML entities: 7
- Ghost conversation markers found: **0** across 0 documents; 0 rewritten to the canonical format
- Garbage vocab slots: 2 → 0 (scanned 40,000 docs; byte-fallback slots counted separately: 102)
- `clean_text()` correctness fixtures: **6/6 pass**

## Stage 3 — language ID

Detector: fastText lid.176.ftz (176 languages). Verdicts: `{'MATCH': 61206, 'CODE_SWITCHED': 3062, 'ROMANIZED': 6, 'LOW_CONFIDENCE': 277, 'MISMATCH': 161}`

- Mislabelled documents caught by runtime detection: **161**
- Caught by trusting the folder path: **0**
- Code-switched, flagged: 3,062; romanised Indic: 6
- Documents a naive ISO 639-1 vs 639-3 comparison would have wrongly rejected (the `te`/`tel` bug): **64,308**

### The corrupted denominator

| Language | If folder trusted | After detection | Delta |
|---|---:|---:|---:|
| asm | 5,006,212 | 4,953,536 | -52,676 |
| az | 0 | 7,631 | +7,631 |
| ben | 0 | 41,973 | +41,973 |
| cy | 0 | 2,886 | +2,886 |
| eng | 7,907,145 | 7,800,181 | -106,964 |
| fi | 0 | 820 | +820 |
| fr | 0 | 149 | +149 |
| fy | 0 | 3,390 | +3,390 |
| guj | 0 | 2,597 | +2,597 |
| hin | 25,093,832 | 25,086,607 | -7,225 |
| hr | 0 | 884 | +884 |
| id | 0 | 29,219 | +29,219 |
| it | 0 | 516 | +516 |
| ku | 0 | 2,304 | +2,304 |
| mal | 0 | 621 | +621 |
| mar | 0 | 4,146 | +4,146 |
| mg | 0 | 2,633 | +2,633 |
| nl | 0 | 214 | +214 |
| no | 0 | 7,321 | +7,321 |
| ory | 0 | 1,305 | +1,305 |
| pan | 0 | 1,015 | +1,015 |
| ro | 0 | 3,021 | +3,021 |
| san | 0 | 11,497 | +11,497 |
| sq | 0 | 2,941 | +2,941 |
| sr | 0 | 984 | +984 |
| tam | 0 | 6,583 | +6,583 |
| tel | 10,001,774 | 10,012,138 | +10,364 |
| tk | 0 | 1,660 | +1,660 |
| tl | 0 | 9,712 | +9,712 |
| tr | 0 | 1,893 | +1,893 |
| urd | 0 | 8,360 | +8,360 |
| uz | 0 | 226 | +226 |

## Stage 4a — quality filter, English-tuned vs script-aware

Same nine rules, same documents, two calibrations. English-tuned keeps **11,820** documents (18.4%); script-aware keeps **60,943** (94.8%). The gap is **49,163 documents / 35,627,196 tokens** of good text an English-tuned filter destroys.

| Rule | Fails · English-tuned | Fails · script-aware |
|---|---:|---:|
| `mean_word_length` | 14 | 4 |
| `symbol_to_word_ratio` | 18 | 18 |
| `lines_end_in_terminal_punct` | 30,329 | 1,412 |
| `duplicate_line_fraction` | 100 | 100 |
| `top_2gram_fraction` | 21 | 21 |
| `common_stopwords_present` | 50,165 | 1,636 |
| `bullet_line_ratio` | 153 | 153 |
| `ellipsis_line_ratio` | 52 | 52 |
| `document_word_count` | 807 | 241 |

| Language | Docs | Kept · English-tuned | Kept · script-aware |
|---|---:|---:|---:|
| hin | 36,233 | 955 (2.6%) | 35,431 (97.8%) |
| tel | 15,054 | 743 (4.9%) | 13,015 (86.5%) |
| eng | 10,435 | 10,083 (96.6%) | 10,083 (96.6%) |
| asm | 2,546 | 39 (1.5%) | 2,414 (94.8%) |

## Stage 4b — trained classifier gate

fastText supervised, dim=64, wordNgrams=2, epoch=8, weak labels from structural weak labels (boilerplate, link, digit, variety, sentence regularity). Held-out exact accuracy **0.8379**, within-one **0.9838** on 8,000 documents. Dropped 34 below score 3.0.

> The labels are weak/heuristic, not LLM-generated. The classifier reproduces the mechanism and its cost profile, not FineWeb-Edu's quality.

## Stage 5 — deduplication

k=5 words, n=126 permutations, b=18 × r=7, LSH threshold 0.6617, drop at true Jaccard ≥ 0.67.

- Exact duplicates: **265** (0 across shards)
- Near-duplicates: **13** (2 across shards)
- LSH candidate pairs evaluated: 945
- Caught by per-shard local passes: 276; by one global pass: 278; **only the global pass: 2**
- Tokens removed: 125,880
- Index memory model: 0.09 GiB for this run; 762.94 GiB at 500M documents

## Stage 6 — PII

13,350 spans masked across 5,124 documents ({'NAME': 12953, 'EMAIL': 157, 'PHONE': 215, 'IPV4': 18, 'AADHAAR': 6, 'CREDENTIAL': 1}). Masking saved -0.58% of tokens on the 500 documents measured.

Precision and recall, against hand-annotated fixtures:

| Dial | Precision | Recall | F1 | Traps wrongly masked |
|---:|---:|---:|---:|---:|
| 0.10 | 0.9375 | 1.0 | 0.9677 | 1 |
| 0.30 | 0.9375 | 1.0 | 0.9677 | 1 |
| 0.45 | 0.9375 | 1.0 | 0.9677 | 1 |
| 0.60 | 0.6522 | 1.0 | 0.7895 | 8 |
| 0.80 | 0.6522 | 1.0 | 0.7895 | 8 |

> Precision and recall come from hand-annotated fixtures, including the session's own PII example -- a web crawl has no ground truth. The corpus numbers above are a tally of what fired, with no precision claim attached to them. The two are reported separately on purpose.

> The name layer cannot tell a private individual from a public figure, and in an Indic web corpus the most frequent person-names are public figures, deities and mythological characters. Masking those costs real knowledge; keeping them risks real people. The exclusion list holds 46 entries including ambedkar, gandhi, kalam, nehru, tagore, which is a blunt instrument and is reported as one.

## Stage 7 — decontamination

813,113 distinct 13-gram fingerprints; a document is dropped at 2 distinct matches. **457 documents dropped** (0.7537%).

| Eval set | Split | Examples | n-grams | Hits |
|---|---|---:|---:|---:|
| MMLU (`cais/mmlu`) | test | 3,000 | 86,705 | 0 |
| GSM8K (`openai/gsm8k`) | test | 1,319 | 127,455 | 0 |
| HellaSwag (`Rowan/hellaswag`) | validation | 3,000 | 167,794 | 0 |
| MILU-Hindi (`murthyrudra/milu-cleaned`) | test | 3,000 | 134,775 | 1,369 |
| MILU-Telugu (`murthyrudra/milu-cleaned`) | test | 3,000 | 154,852 | 189 |
| MILU-Bengali (`murthyrudra/milu-cleaned`) | test | 3,000 | 141,532 | 0 |

Canary: minted 3, recovered by scan True, present in shipped corpus 0.

## Stage 8 — manifest

13 shards, 13 admitted. 42,889,153 tokens measured with `sarvam1`; `words × 1.3` would have claimed 32,689,942 (**-23.78% error**).

| Language | Tokens shipped | Fertility (tokens/word) | vs the assumed 1.3 |
|---|---:|---:|---:|
| asm | 4,507,883 | 4.155 | 3.2× |
| tel | 8,591,899 | 2.481 | 1.9× |
| eng | 7,290,369 | 1.539 | 1.2× |
| hin | 22,499,002 | 1.419 | 1.1× |

Determinism: two independent runs over the same input produced identical shard ids and hashes — **True** (`shard_023a339380c1`).

## Integrity assertions

- No noise character survives: **True** (0 found in 60,174 documents)
- Indic joiners preserved: **True** (1 raw documents carried one; 1 shipped documents still do)
- Shard ids reproduce across runs: **True**
- clean_text() fixtures: **6/6**
