# Run report — cleaning a slice of ai4bharat/sangraha

Generated 2026-07-31T14:40:19Z from `data/run/stats.json`. Full run. Total runtime 40.7 min on 12 CPU cores, no GPU.

## Headline

| | |
|---|---:|
| Dataset | `ai4bharat/sangraha` @ `8b813c3f62d3` |
| Licence | CC-BY-4.0 |
| Documents in → out | 277,285 → 255,218 |
| Tokens in → out (sarvam1) | 192,310,750 → 176,939,773 |
| Document retention | 92.04% |
| Token retention | 92.01% |
| Shards emitted / admitted | 52 / 52 |

## Yield descent

| Stage | Docs out | Dropped | % of raw |
|---|---:|---:|---:|
| 0. Raw slice | 277,285 | — | 100.0% |
| 1. Extract *(inherited)* | 277,285 | — | 100.0% |
| 2. Normalize | 277,284 | 1 | 100.0% |
| 3. Language ID | 276,820 | 464 | 99.83% |
| 4a. Quality (Gopher/C4) | 256,635 | 20,185 | 92.55% |
| 4b. Quality (classifier) | 256,528 | 107 | 92.51% |
| 5. Deduplicate | 255,772 | 756 | 92.24% |
| 6. PII scrub | 255,772 | — | 92.24% |
| 7. Decontaminate | 255,218 | 554 | 92.04% |
| 8. Manifest | 255,218 | — | 92.04% |

Session reference curve, for shape comparison: 100 → 92 → 88 → 61 → 44 → 43 → 42 → 42 → 42.

## Stage 2 — normalize

- Noise characters removed: **863** ({'zero_width': 23, 'bidi': 100, 'control': 119, 'replacement': 597, 'private_use': 24})
- Indic joiners kept: {'ZWNJ': 1674, 'ZWJ': 7427}
- Documents changed by NFC: 18,341; with HTML entities: 24
- Ghost conversation markers found: **4** across 4 documents; 0 rewritten to the canonical format
- Garbage vocab slots: 2 → 1 (scanned 40,000 docs; byte-fallback slots counted separately: 107)
- `clean_text()` correctness fixtures: **6/6 pass**

## Stage 3 — language ID

Detector: fastText lid.176.ftz (176 languages). Verdicts: `{'MATCH': 264495, 'CODE_SWITCHED': 12325, 'LOW_CONFIDENCE': 384, 'ROMANIZED': 55, 'MISMATCH': 25}`

- Mislabelled documents caught by runtime detection: **25**
- Caught by trusting the folder path: **0**
- Code-switched, flagged: 12,325; romanised Indic: 55
- Documents a naive ISO 639-1 vs 639-3 comparison would have wrongly rejected (the `te`/`tel` bug): **276,885**

### The corrupted denominator

| Language | If folder trusted | After detection | Delta |
|---|---:|---:|---:|
| ar | 0 | 3,970 | +3,970 |
| ben | 26,741,110 | 26,755,198 | +14,088 |
| eng | 0 | 126,243 | +126,243 |
| es | 0 | 10,265 | +10,265 |
| fi | 0 | 5,050 | +5,050 |
| guj | 20,001,573 | 19,967,263 | -34,310 |
| hin | 30,000,523 | 29,890,959 | -109,564 |
| kan | 19,049,453 | 19,005,273 | -44,180 |
| la | 0 | 782 | +782 |
| mal | 20,001,500 | 19,991,493 | -10,007 |
| mar | 25,000,207 | 25,006,368 | +6,161 |
| mg | 0 | 37,759 | +37,759 |
| no | 0 | 3,505 | +3,505 |
| ory | 14,959,282 | 14,945,603 | -13,679 |
| pan | 13,803,249 | 13,794,576 | -8,673 |
| san | 0 | 2,455 | +2,455 |
| sq | 0 | 5,992 | +5,992 |
| tam | 22,753,852 | 22,747,297 | -6,555 |
| tel | 0 | 9,189 | +9,189 |
| tk | 0 | 1,509 | +1,509 |

## Stage 4a — quality filter, English-tuned vs script-aware

Same nine rules, same documents, two calibrations. English-tuned keeps **11,540** documents (4.2%); script-aware keeps **256,635** (92.7%). The gap is **245,791 documents / 170,316,178 tokens** of good text an English-tuned filter destroys.

| Rule | Fails · English-tuned | Fails · script-aware |
|---|---:|---:|
| `mean_word_length` | 2,858 | 66 |
| `symbol_to_word_ratio` | 48 | 48 |
| `lines_end_in_terminal_punct` | 104,520 | 6,983 |
| `duplicate_line_fraction` | 879 | 879 |
| `top_2gram_fraction` | 161 | 161 |
| `common_stopwords_present` | 260,289 | 12,027 |
| `bullet_line_ratio` | 444 | 444 |
| `ellipsis_line_ratio` | 305 | 305 |
| `document_word_count` | 6,334 | 586 |

| Language | Docs | Kept · English-tuned | Kept · script-aware |
|---|---:|---:|---:|
| ben | 32,254 | 344 (1.1%) | 31,114 (96.5%) |
| hin | 47,271 | 1,185 (2.5%) | 45,739 (96.8%) |
| mar | 40,951 | 2,924 (7.1%) | 39,572 (96.6%) |
| tam | 32,742 | 2,831 (8.6%) | 26,463 (80.8%) |
| kan | 29,547 | 1,507 (5.1%) | 27,449 (92.9%) |
| guj | 23,774 | 1,117 (4.7%) | 22,209 (93.4%) |
| mal | 31,813 | 1,151 (3.6%) | 27,253 (85.7%) |
| pan | 17,532 | 319 (1.8%) | 17,072 (97.4%) |
| ory | 20,936 | 162 (0.8%) | 19,764 (94.4%) |

## Stage 4b — trained classifier gate

fastText supervised, dim=64, wordNgrams=2, epoch=8, weak labels from structural weak labels (boilerplate, link, digit, variety, sentence regularity). Held-out exact accuracy **0.8704**, within-one **0.9834** on 8,000 documents. Dropped 107 below score 3.0.

> The labels are weak/heuristic, not LLM-generated. The classifier reproduces the mechanism and its cost profile, not FineWeb-Edu's quality.

## Stage 5 — deduplication

k=5 words, n=126 permutations, b=18 × r=7, LSH threshold 0.6617, drop at true Jaccard ≥ 0.67.

- Exact duplicates: **732** (0 across shards)
- Near-duplicates: **24** (0 across shards)
- LSH candidate pairs evaluated: 62,406
- Caught by per-shard local passes: 756; by one global pass: 756; **only the global pass: 0**
- Tokens removed: 345,852
- Index memory model: 0.39 GiB for this run; 762.94 GiB at 500M documents

## Stage 6 — PII

21,299 spans masked across 10,317 documents ({'NAME': 19181, 'EMAIL': 956, 'PHONE': 1064, 'IPV4': 60, 'AADHAAR': 37, 'CREDENTIAL': 1}). Masking saved -0.07% of tokens on the 500 documents measured.

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

813,113 distinct 13-gram fingerprints; a document is dropped at 2 distinct matches. **554 documents dropped** (0.2166%).

| Eval set | Split | Examples | n-grams | Hits |
|---|---|---:|---:|---:|
| MMLU (`cais/mmlu`) | test | 3,000 | 86,705 | 0 |
| GSM8K (`openai/gsm8k`) | test | 1,319 | 127,455 | 0 |
| HellaSwag (`Rowan/hellaswag`) | validation | 3,000 | 167,794 | 0 |
| MILU-Hindi (`murthyrudra/milu-cleaned`) | test | 3,000 | 134,775 | 1,423 |
| MILU-Telugu (`murthyrudra/milu-cleaned`) | test | 3,000 | 154,852 | 0 |
| MILU-Bengali (`murthyrudra/milu-cleaned`) | test | 3,000 | 141,532 | 425 |

Canary: minted 3, recovered by scan True, present in shipped corpus 0.

## Stage 8 — manifest

52 shards, 52 admitted. 176,939,773 tokens measured with `sarvam1`; `words × 1.3` would have claimed 113,298,171 (**-35.97% error**).

| Language | Tokens shipped | Fertility (tokens/word) | vs the assumed 1.3 |
|---|---:|---:|---:|
| mal | 17,344,450 | 3.158 | 2.4× |
| kan | 17,324,073 | 2.477 | 1.9× |
| tam | 20,267,898 | 2.443 | 1.9× |
| ory | 13,843,496 | 2.364 | 1.8× |
| ben | 25,597,280 | 2.09 | 1.6× |
| guj | 18,319,443 | 1.985 | 1.5× |
| mar | 23,957,748 | 1.927 | 1.5× |
| pan | 13,267,275 | 1.64 | 1.3× |
| hin | 27,018,110 | 1.459 | 1.1× |

Determinism: two independent runs over the same input produced identical shard ids and hashes — **True** (`shard_e84bba20ec66`).

## Integrity assertions

- No noise character survives: **True** (0 found in 255,218 documents)
- Indic joiners preserved: **True** (56 raw documents carried one; 15 shipped documents still do)
- Shard ids reproduce across runs: **True**
- clean_text() fixtures: **6/6**
