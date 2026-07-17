# BPE Tokenizer Widget — India Wikipedia (ERA V5, Session 2)

A single 10,000-token Byte Pair Encoding (BPE) vocabulary trained jointly across the India
Wikipedia article in **English, Hindi, Telugu, and Bengali**, shipped as a static HTML+JS widget
that computes the assignment's fertility ratios/self-score and lets the viewer browse the full
learned vocabulary and inspect tokenization live.

This is **phase 2** of the submission. Phase 1 scored 0 — see [The challenge](#the-challenge-what-went-wrong-in-phase-1)
below for what broke and how this version fixes it.

## Live site

Not yet redeployed after the phase-2 fix (see `PLAN_PHASE2.md`). Deploy `site/` to Netlify
(drag-and-drop at `app.netlify.com/drop`, or connect the repo) and record the URL here.

## The challenge: what went wrong in phase 1

The assignment requires `decode(encode(text))` to preserve every visible non-whitespace
character — a "faithful round-trip" gate. Phase 1 used a `Whitespace` pre-tokenizer with no
`decoder` configured. `Whitespace` splits text into disjoint regex-matched chunks and throws away
everything in between (including the separator characters themselves), so `decode()` just joined
pretoken strings back together with a single generic space. Every original separator —
`:` `/` `.` `#` `-` — was permanently lost. Feedback sample:

```
'https://hi.wikipedia.org/wiki/भारत#cite_ref-1'
  -> decoded 'h tt p s : / / hi . wi k i pe di a . or g / wi k i / भारत cit ere f - 1'
```

This is an architecture bug, not a tuning problem — no amount of vocabulary size or weighting
could have fixed it. Score was 0.

Two references were then consulted: an instructor reference (`../refsol/`, later clarified by an
added `instructions.md`) and a peer implementation. Both independently pointed at the same fix:
replace the `Whitespace` pre-tokenizer with HuggingFace's **`Metaspace`** pre-tokenizer
(`replacement="▁", prepend_scheme="always"`) plus a **matching `Metaspace` decoder**, and drop the
destructive normalizer down to `NFKC` only. Metaspace never discards a character — every space
becomes a `▁` marker embedded directly in the token stream, so `decode()` reconstructs the
original exactly by reversing that one substitution. `../refsol/instructions.md` also corrected
the scoring formula's denominator (see [Scoring](#scoring-formula) below) — the earlier
letter/number-only "wordish units" count was too strict; punctuation must count as its own
faithful unit too.

Full write-up of this investigation, the two references compared side by side, and the Marathi
vs. Bengali 4th-language experiment is in `PLAN_PHASE2.md`. `MODIFY.md` holds the original grading
feedback that kicked this off. `SESSION_REPORT.md` is the (superseded) phase-1 record.

## Design decisions

- **Corpus = faithful Markdown, not plain prose.** Wikipedia's plain `explaintext` extracts strip
  out virtually every URL, header, and reference mark. A tokenizer that never sees a `/` or `#`
  during training has no vocabulary entry for it and falls back to `[UNK]` the moment a probe
  string contains one — even with a correct architecture. So the corpus fetcher pulls the REST
  HTML for each language's article and converts it to Markdown (BeautifulSoup + `markdownify`),
  preserving links (with URLs), references, tables, and headers.
- **One BPE model, jointly trained, not four separate ones.** The assignment's 10,000-token budget
  is a combined vocabulary; training one multilingual BPE model on a weighted mixture of all four
  languages' text (rather than training separately and concatenating) is what makes the shared
  budget actually contestable between languages.
- **Word-boundary-restricted merges via Metaspace, not raw byte-level BPE.** Byte-level BPE was
  tried in phase 1 and discarded — Devanagari/Telugu characters are 3 bytes each in UTF-8, so a
  large share of the shared vocabulary budget went to reassembling raw byte sequences into whole
  characters before any real subword structure could be learned, and it couldn't even reach the
  full 10k budget. Metaspace pre-tokenization keeps merges within word-like spans, which is also
  what makes per-language oversampling (below) a reliable lever.
- **Per-language oversampling (`repeat_counts`) to control budget allocation**, using an XLM-R-style
  exponential-smoothing default (`common.compute_repeat_counts`) as a starting point, then a grid
  search (`tune_weights.py`) over modest integer weights (1×–6×) to directly minimize the spread
  between languages' fertility.
- **`min_frequency=1`** on the BPE trainer. The HF default (`min_frequency=2`) means a merge is
  only learned if a symbol pair co-occurs at least twice; that consistently starved English's long
  tail of once-occurring distinctive words in favor of shared cross-lingual pieces. Dropping it to
  1 was a bigger lever than further reweighting.
- **Bengali over Marathi as the 4th language.** Both were fetched, trained, and scored end-to-end
  through the identical pipeline for a fair comparison. Bengali's article is substantially larger
  and, combined with its Brahmic-script overlap with Hindi, gives the joint tokenizer more to work
  with; it produced a tighter fertility spread (0.0220 vs. 0.0280) and a materially higher score.
  See `PLAN_PHASE2.md` for the full head-to-head numbers.
- **Static data, no backend.** Netlify hosts static files only. All training/tuning happens
  offline in Python; the browser only ever fetches precomputed JSON (`site/data/*.json`) and runs
  a from-scratch JS re-implementation of the tokenizer's encode/decode path client-side — nothing
  is computed server-side at request time.
- **The JS tokenizer is verified byte-for-byte against the Python one**, not just assumed correct
  (`verify_bpe_js.py` generates fixtures from the real HF tokenizer; a Node harness checks
  `site/bpe.js` matches token-for-token and decode-for-decode). This mattered directly: the
  original bug was a decode-side mismatch, so shipping a JS encoder that reimplements the same
  mistake independently would have reproduced the failure invisibly.

### Scoring formula

```
faithful_unit(lang)  = count of: one contiguous Unicode letter/mark/number run,
                        OR one visible non-space punctuation/symbol character
                        (each punctuation character counts separately)
fertility(lang)      = token_count(lang) / faithful_unit_count(lang)
spread               = max(fertility) - min(fertility)
score                = 1000 / spread
```

The assignment text anchors English specifically ("English ratio must be around 1.2 or less").
Under the corrected faithful-unit denominator every language lands well under 1.0 regardless of
weighting, so this is reported as a sanity check rather than an active constraint; an
English-anchored penalty (`exp(max(0, fertility_en/1.2 - 1))`) is also computed and shown for
completeness (`compute_metrics.py`).

## Repository layout

```
scripts/            Python: corpus fetch -> train -> tune -> verify -> export
data/corpus/         fetched Wikipedia Markdown per language (+ .meta.json)
data/tokenizer/{mr,bn}/tokenizer.json   trained HF tokenizer, one per 4th-language candidate
data/tune_sweep_{mr,bn}.json            full grid-search sweep results
site/                the deployable static widget (HTML/CSS/JS + precomputed JSON)
../refsol/           instructor's reference solution (read-only reference material)
```

### Python scripts (`scripts/`)

All scripts are run from inside `scripts/` (they import each other as local modules) against the
project's `.venv`.

| Script | Purpose |
|---|---|
| `common.py` | Shared corpus loading + `compute_repeat_counts` (XLM-R-style exponential-smoothing oversampling), generic over which 4th language is being run. |
| `fetch_corpus.py` | Fetches the India Wikipedia article's REST HTML in en/hi/te/mr/bn, converts to faithful Markdown, resolves localized titles via English Wikipedia's `langlinks`. Writes `data/corpus/{lang}.txt` + `.meta.json`. |
| `train_tokenizer.py` | Trains the shared BPE tokenizer: `NFKC` normalizer, `Metaspace` pre-tokenizer/decoder, `min_frequency=1`, `vocab_size=10000`, on a weighted mixture of the four corpora. Writes `data/tokenizer/{mr,bn}/tokenizer.json`. |
| `tune_weights.py` | Grid search over per-language oversampling weights (`repeat_counts`), minimizing fertility spread directly. Writes `data/tune_sweep_{mr,bn}.json` and re-saves the winning tokenizer. |
| `compute_metrics.py` | Computes per-language fertility, spread, raw score, and the English-anchored penalty/adjusted score for a trained tokenizer. |
| `verify_roundtrip.py` | Round-trip fidelity gate: asserts `decode(encode(x))` preserves visible non-whitespace characters across adversarial probes (URLs, refs, hyphens, mixed scripts — including the exact phase-1 failing sample) plus corpus samples. Non-zero exit on failure; run before shipping a tokenizer. |
| `verify_bpe_js.py` | Generates fixtures (`site/data/_bpe_fixtures.json`) from the real Python tokenizer's encode/decode output, used to cross-check the from-scratch `site/bpe.js` JS implementation in Node before it's trusted in-browser. |
| `export_frontend_data.py` | Exports everything the static frontend needs into `site/data/*.json` — `vocab.json` (full id→token + merges + per-token script/lang tagging), `stats.json` (both mr and bn candidates, for the head-to-head comparison), `samples.json` (per-language sample text for the Coverage Inspector), `run_config.json` (the winning run's exact knobs), `fidelity_probes.json` (the same probes `verify_roundtrip.py` uses, for the live in-browser Fidelity Check tab). |

Typical end-to-end run for one 4th-language candidate:

```bash
cd scripts
python fetch_corpus.py                    # once, covers all 5 candidate languages
python tune_weights.py --fourth bn         # grid search + saves winning tokenizer
python verify_roundtrip.py --fourth bn     # must exit 0 before shipping
python verify_bpe_js.py --fourth bn        # regenerate JS cross-check fixtures
python export_frontend_data.py --winner bn
```

### JS implementation (`site/`)

Static site, no build step, no backend — open `site/index.html` behind any static file server and
it fetches `site/data/*.json` directly.

- **`bpe.js`** — a from-scratch client-side re-implementation of the HF `tokenizers` Metaspace BPE
  encode/decode path: NFKC-normalize, Metaspace pre-tokenize (space → `▁`, prepend one at the
  start), standard lowest-merge-rank BPE loop per pretoken, then decode by reversing the `▁`
  substitution. Exposed as `BpeEncoder` with `.encode(text)`, `.decode(ids)`, and `.roundTrip(text)`.
  Kept in lockstep with `train_tokenizer.py` and verified token-for-token against it via
  `verify_bpe_js.py`'s fixtures.
- **`app.js`** — loads the precomputed JSON, renders all six tabs, and drives the live
  Coverage Inspector / Fidelity Check by calling into `bpe.js` directly in the browser.
- **`index.html` / `styles.css`** — the single-page shell with six tab sections:
  1. **Overview & Score** — the required per-language fertility table, sorted values, spread, raw
     and adjusted self-score, plus the Marathi-vs-Bengali comparison table.
  2. **Fidelity Check** — runs the round-trip probes live in the browser using the shipped
     `bpe.js`, not a precomputed table — direct, reproducible evidence the round-trip bug is fixed.
  3. **Analysis** — the narrative reasoning: root cause of the phase-1 failure, what the
     instructor's reference and peer implementation revealed, the architecture fix, the corpus
     change, the fertility-formula correction, and the weight-tuning result — all numbers pulled
     live from the same JSON as the Overview tab, not hardcoded prose.
  4. **Vocab Explorer** — searchable/filterable/paginated browser over all 10,000 learned tokens
     (script filter, per-language usage filter, merged-vs-base filter). This satisfies the
     "browse the full vocabulary" requirement.
  5. **Coverage Inspector** — paste or pick sample text, tokenize it live client-side, see
     rainbow-highlighted token boundaries, word/token/fertility counts, any uncovered characters,
     and a round-trip check on that exact input.
  6. **Methodology** — the corpus/architecture/scoring/tuning recipe, values pulled from
     `run_config.json` rather than hardcoded.

## Running locally

Requires Python 3.12+ (only needed to regenerate data — the site itself is static) and `uv`.

```bash
# one-time: Python env for the scripts (only needed to retrain/regenerate data)
uv venv .venv
uv pip install -r requirements.txt --python .venv/bin/python

# serve the static site (from the site/ directory)
cd site
python3 -m http.server 8731
```

Then open `http://127.0.0.1:8731/`. Opening `index.html` directly via `file://` will not work —
`fetch()` cannot load local JSON over that protocol; a static file server (or Netlify in
production) is required.

## Verification performed

- `scripts/verify_roundtrip.py` — all adversarial probes pass, including the exact phase-1
  grading-feedback sample, for both the Marathi and Bengali tokenizers.
- `scripts/verify_bpe_js.py` + a Node harness — `site/bpe.js` matches the real Python tokenizer
  token-for-token and decode-for-decode across edge cases, round-trip probes, and corpus samples.
- Full site smoke test: served locally, all six tabs exercised via headless Chrome — Overview
  shows the real score and mr/bn comparison, Fidelity Check shows all probes passing live
  in-browser, Vocab Explorer lists all 10,000 tokens with correct script/merge tagging, Analysis
  and Methodology render real numbers pulled from the shipped JSON, no console errors.
