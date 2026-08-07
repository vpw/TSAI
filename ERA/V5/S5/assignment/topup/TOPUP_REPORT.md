# S5 cleaning top-up

Same eight-stage pipeline as S4, pointed at the lane the ledger says is starved: the
Indic **verified** tier, which the plan runs at 2.5 epochs. Verified tokens are only
worth having if they cover the languages the model must serve, so this pass takes the
Tier-1 languages S4 never touched, plus a second Hindi shard so cross-shard dedup has
something to do.

## Result

| | |
|---|---:|
| Documents in → out | 277,285 → 255,218 |
| Tokens in → out | 192,310,750 → **176,939,773** |
| Token retention | 92.01% |
| Shards admitted | 52/52 |

- This pass: **176.9M** clean tokens
- S4 pass: 43.5M
- Cumulative: **220.4M** against the stated target of 320B (8% of a 4T run) = **0.0689%**

That percentage is the honest one. A single workstation pass is a rounding error against
a 320B target; what it buys is a *measured* per-language yield curve to plan the real run
with, and it caught a defect that would have scaled.

## Per-language yield, and the defect this pass found

The S4 quality stage asks "does this document contain at least 2 common words of its
language". `stopword_set()` fell back to the **English** list for any language it had
no list for — so a Kannada page was checked for English stop-words, found none, and was
dropped. Five languages had no list.

The `before` column is measured on the smoke slice (9,329 docs) that exposed the
defect; the `after` column is this full pass. They are different sample sizes, so read
the column pair as the size of the effect, not as a paired test.

| Language | stop-word list in S4 | docs (full pass) | kept before fix | kept after fix |
|---|---|---:|---:|---:|
| `ben` | yes | 32,254 | 96.6% | 96.5% |
| `guj` | **no** | 23,774 | 4.4% | 93.4% |
| `hin` | yes | 47,271 | 96.9% | 96.8% |
| `kan` | **no** | 29,547 | 5.1% | 92.9% |
| `mal` | **no** | 31,813 | 3.8% | 85.7% |
| `mar` | yes | 40,951 | 95.6% | 96.6% |
| `ory` | **no** | 20,936 | 2.9% | 94.4% |
| `pan` | **no** | 17,532 | 5.7% | 97.4% |
| `tam` | yes | 32,742 | 80.6% | 80.8% |

The split is exactly along whether S4 shipped a list: the four languages that had one
are unchanged, the five that did not go from 3-6% to 86-97%. On the smoke slice whole-
run token retention went from **53.16% to 91.90%**; the full pass lands at **92.01%**,
against S4's 90.51% on its own four languages — so the fixed cleaner now treats nine
languages the way S4 treated four.

The lists were counted out of the corpus by document frequency rather than written
from memory (`scripts/derive_stopwords.py`, `data/run/derived_stopwords.json`), and a
missing list is now recorded in the stage stats instead of being absorbed into the
English fallback.

Tamil is the one language with a list that still keeps noticeably less (80.8%). Its
list is also the shortest S4 shipped, at 15 entries against 21-34 for the others —
the same defect in milder form, and the next thing to fix.

Languages still with no list after this pass: **none**.

## Decontamination and caveats

- Eval firewall: **813,113 distinct 13-grams** across the MILU Hindi / Telugu / Bengali splits; **554 documents dropped** as contaminated (0.217%, 937,610 tokens), canary recovered.
  The smoke pass had lost the Bengali split to datasets-server rate limiting; the full
  pass fetched all three, so Bengali contamination is measured here, not assumed.
- `claimed_lang` for Odia is written `ory`, the pipeline's ISO-639-3 label, while
  Sangraha's folder is `ori`. The folder name is not trusted for anything else either.
- Token counts are measured with the sarvam1 tokenizer, not estimated from word counts:
  the run reports 176.9M measured against 113.3M from the words×1.3 rule of thumb, a
  **36% gap**. Indic fertility is why the ledger counts tokens and not words.

