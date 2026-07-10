# S2 Assignment — BPE Tokenizer Plan (v2: redo after correcting the scoring objective)

*Status: redo complete. Winning tokenizer found — word-level BPE, tuned weights, X1(en)=1.21, score=4512. See "Redo results" below.*

## Corrected scoring objective

The assignment says:

> (Total English Vocab, say 5000 words)/(Total English tokens) **must be around 1.2 or less**, let's call this X1
> Similarly ratios for your Hindi, Telugu and another language is X2, X3, X4
> Sort X1, X2, X3, X4 ... score = 1000/(X_max − X_min)

This means **English's ratio (X1) has an explicit hard target (≤ ~1.2)** — it's not just one of four interchangeable values to average out. The scoring objective is therefore two-layered, not a single symmetric "minimize spread":

1. **Get X1 (English) to ≤ ~1.2** — satisfy the explicit constraint first.
2. **Given that, minimize the gap between English and the *worst* (most divergent) of the other three languages** — since that worst outlier is what sets X_max or X_min alongside English and drives the spread/score.

This supersedes the earlier framing (treat all four symmetrically, just minimize `X_max − X_min` wherever the cluster naturally lands). That framing produced results where English itself exceeded 1.2 — technically a decent spread/score, but a direct violation of the assignment's explicit English requirement.

## What's already built (kept, being re-tuned — not rewritten)

- `scripts/fetch_corpus.py` — fetches the India Wikipedia article in en/hi/te/mr (localized titles resolved via English-article langlinks). Working as-is.
- `scripts/common.py` — shared corpus loading + `compute_repeat_counts(texts, alpha)`, an XLM-R-style exponential-smoothing oversampler (share ~ size**alpha) used to keep smaller corpora from being starved of merge budget.
- `scripts/train_tokenizer.py` — trains byte-level, char-level, and word-level BPE variants (HF `tokenizers`), all vocab_size=10000, all on the same weighted mixture.
- `scripts/train_sentencepiece.py` — trains a SentencePiece BPE variant on the same weighted mixture.
- `scripts/compute_metrics.py` — computes, per variant and per language, `X_lang = unique_words / distinct_token_ids_used` (Unicode-aware word extraction via `regex`'s `[\p{L}\p{M}\p{N}]+`, needed because Devanagari/Telugu combining marks break Python's built-in `\w`), sorts, and computes `score = 1000/(X_max−X_min)`.

## First-pass results (single alpha=0.3, uniform across hi/te/mr) — being redone

| Variant | vocab reached | X1 (en) | worst-of-3 | spread | score |
|---|---|---|---|---|---|
| byte-level BPE | 5,464 (couldn't reach 10k) | 2.54 | 4.14 (hi) | 1.60 | 625 |
| char-level BPE | 10,000 | 2.25 | 1.44 (te) | 0.82 | 1,224 |
| word-level BPE | 10,000 | **1.33** (already > 1.2) | 0.83 (te) | 0.50 | 2,006 |
| SentencePiece BPE | 10,000 | **1.41** (already > 1.2) | 0.93 (te) | 0.48 | 2,085 |

Byte-level and char-level are clearly dominated on every axis (worse X1, worse spread, worse score) — dropped from further tuning, kept only as report/comparison baselines. Word-level and SentencePiece are the finalists, but **both currently fail the explicit English ≤1.2 requirement**, and in both, Telugu is the worst-case outlier, sitting well below where it needs to be.

## Redo strategy

1. **Restrict active tuning to word-level BPE and SentencePiece BPE.** Byte/char stay in the report as "why we didn't use these" baselines, not part of the tuning loop.
2. **Replace the single shared `alpha` with per-language weights**, tuned individually rather than one symmetric exponent applied to all of hi/te/mr. Rationale: a single alpha pulled English up past 1.2 while still leaving Telugu (not just "the Indic languages" generally) as the specific worst case — Hindi/Marathi were already closer to the target, so a uniform knob overcorrects for the wrong language.
3. **Target function to optimize is no longer plain spread.** For a candidate weight vector, compute:
   - `X1` (English) — must land at ≤ ~1.2.
   - `worst_gap = max(|X_lang − X1|)` over Hindi/Telugu/Marathi — this is what to minimize, subject to the X1 constraint.
4. **Turn this into a small scripted sweep** (new `scripts/tune_weights.py`, reusing `common.py` + the existing train/metrics scripts): try a grid or simple local search over per-language weights, retrain word-level + SentencePiece for each candidate, record X1 and worst_gap, and report the best candidate per tokenizer variant.
5. **Re-run `compute_metrics.py` under the winning weights** for both finalists, confirm English ≤1.2 and inspect the resulting worst-case gap and score, then pick whichever variant wins.

## Redo results

Two sweeps were run (`scripts/tune_weights.py`, English's repeat count held at 1 since it's already the minimum; grid search over Hindi/Telugu/Marathi repeat counts):

- **First sweep** (HF `BpeTrainer` default `min_frequency=2`): plateaued at X1≈1.27 for word-level BPE, 1.26–1.31 for SentencePiece, regardless of hi/te/mr weights beyond a point — increasing Telugu's weight was the only lever that moved anything, and it saturated around te≈10–20x.
- **Key additional lever found:** lowering `min_frequency` from 2→1 in the HF `BpeTrainer` (word-level BPE only — SentencePiece has no equivalent knob) lets the trainer learn rarer, English-specific merges instead of always reusing a handful of highly-shared pieces. This alone dropped X1 from 1.27 to **1.21** at the same weight point.
- **Second sweep** (`min_frequency=1` for word-level BPE) + a fine-grained local search confirmed a stable optimum:

**Winner: word-level BPE**, `repeat_counts = {en: 1, hi: 1, te: 12, mr: 1}`, `min_frequency=1`, `vocab_size=10000` (fully reached).

| lang | unique words | distinct tokens used | X |
|---|---|---|---|
| en | 3,116 | 2,572 | **1.2115** |
| hi | 2,250 | 1,992 | 1.1295 |
| te | 1,565 | 1,581 | 0.9899 |
| mr | 2,249 | 1,879 | 1.1969 |

Spread = 0.2216, **score = 4512** (vs. 2,006 in the first pass, and 3,022 for the best-tuned SentencePiece alternative, which has no `min_frequency` lever and plateaus at X1≈1.31).

Note X1=1.2115 is a confirmed local optimum (fine grid search around it found nothing better) — just barely above a strict 1.20 cutoff, but squarely "around 1.2" per the assignment's own wording. Byte-level and char-level BPE remain excluded from tuning (kept only as report baselines — see first-pass table above).

`data/tokenizer/{byte,char,word,sentencepiece}/` and `data/stats.json` reflect this final state. Full sweep data (all 240+28 combos tried) is in `data/tune_sweep.json`.

## Step 4: frontend widget — done

Built as a static site in `site/` (`index.html`, `app.js`, `bpe.js`, `styles.css`, `data/*.json`):

- **Overview & Score** — the required X1–X4 table, sorted values, spread/score formula with real numbers, plus a comparison table against the byte/char/SentencePiece baselines.
- **Vocab Explorer** — searchable/paginated browser over all 10,000 learned tokens (script filter, per-language "used by" filter, base-vs-merged filter). Satisfies the assignment's "let the viewer browse/inspect the full vocabulary" requirement.
- **Coverage Inspector** — paste-any-text live tokenizer: quick-pick per-language samples or custom text, runs a from-scratch client-side BPE encoder, highlights token boundaries over the original text, reports word/token counts, fertility, and flags any character the vocab doesn't cover.
- **Methodology** — corpus construction, the corrected scoring objective, the weighting + `min_frequency` strategy, and the X-vs-fertility distinction, pulling numbers live from `run_config.json` rather than hardcoded prose.

Mechanism: `scripts/export_frontend_data.py` exports the winning word-level `tokenizer.json` into compact static JSON (`site/data/vocab_word.json`, `stats.json`, `samples.json`, `run_config.json`). `site/bpe.js` is a from-scratch JS re-implementation of HF's word-level BPE encode path (Unicode-aware Whitespace pre-tokenization + character-pair merge-rank algorithm) — no server-side tokenizer needed at runtime.

Verification performed:
- `scripts/verify_bpe_js.py` generates fixtures from the real Python tokenizer (all 4 languages + edge cases: underscores, hyphens, emoji/uncovered chars, empty string); a Node script loads `bpe.js` against the same `vocab_word.json` and confirmed **10/10 fixtures match token-for-token**, including the emoji-gets-silently-dropped behavior.
- Full site smoke-tested with headless Chrome (screenshots of all 4 tabs) confirming real data renders correctly, live tokenization works, and Devanagari/Telugu glyphs render properly (fonts present on this machine).

## Deferred

- Step 5: Netlify deployment.

## Verification

- After re-tuning: inspect the updated `data/stats.json`, confirm English X1 ≤ ~1.2 for the chosen variant, and confirm the worst-of-three gap to English has shrunk relative to the sweep alternatives (not just that overall spread/score improved).
- Spot-check a handful of Telugu/Hindi/Marathi vocab entries once weights are more aggressively skewed, to make sure the token list still looks like genuine subwords rather than degenerate artifacts of extreme oversampling.
