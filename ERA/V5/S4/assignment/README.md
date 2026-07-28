# S4 — Data Cleaning and Deduplication

ERA V5, Session 4. A real cleaning pipeline applied to a real corpus, with the measured results
presented as a single self-contained widget.

**Live widget:** _(add the Netlify URL here after deploying `site/`)_

## What this is

The assignment asks four things: count and describe the cleaning strategies the session covered,
pick a 10–100M-token dataset, actually apply the cleanups, and present it as a widget.

- **Strategies.** The session gives two different lists of eight, and they are not the same eight —
  the pipeline map includes Extract and folds format discipline into Normalize; the closing
  commitment drops Extract and makes format discipline first-class. The union is **9 strategies**,
  decomposed into **30 techniques**, plus **8 cross-cutting concerns**. Eight remains the headline,
  because the instructor called it "a minimal set… the minimum that you have to do."
- **Dataset.** [`ai4bharat/sangraha`](https://huggingface.co/datasets/ai4bharat/sangraha), CC-BY-4.0
  — the Indic web crawl this session names as the one that shipped with *zero* deduplication, and
  the headline dataset of the S3 assignment. A ~48M-token slice across five files, two pools and
  four languages.
- **The run.** All eight stages, offline, on CPU. Every number in the widget is emitted by the
  pipeline into `data/run/stats.json` and injected into the page at build time; none is typed by
  hand.

## Headline findings

- **The English-tuned quality filter destroys good Indic text.** The same nine Gopher/C4 rules run
  twice — English-tuned and script-aware — differ enormously. Two rules do nearly all of it:
  `common_stopwords_present`, which looks for English function words and scores clean Hindi and
  Telugu at zero, and `lines_end_in_terminal_punct`, which knows `. ! ?` but not the danda `।`.
- **Sangraha really does contain heavy duplication.** Verbatim 300-word documents repeated across
  the shard, which only a dedup pass finds — the stage the session says it never ran.
- **`words × 1.3` is not a token count.** Measured fertility runs to ~4.3 tokens/word for Assamese
  against the flat 1.3 V4 assumed.
- **Sangraha is cleaner at the character level than V4's corpus was.** Noise appears in roughly one
  document in a thousand. That is reported as a finding, and the cleaner's correctness is
  demonstrated separately against fixtures rebuilt from the session's own worked examples.

See `RUN_REPORT.md` for the full numbers.

## Layout

```
scripts/            the pipeline, one module per stage
  common.py           shared plumbing, source list, tokenizers
  fetch_corpus.py     0 · acquire and slice the corpus
  clean_fixtures.py       clean_text() correctness fixtures
  normalize.py        2 · NFC, noise stripping, joiner preservation, ghost tags
  langid_stage.py     3 · runtime detection vs. the folder's claim
  quality.py          4a · nine Gopher/C4 rules, English-tuned vs script-aware
  quality_clf.py      4b · trained classifier gate
  dedup.py            5 · exact + MinHash/LSH, local vs global
  pii.py              6 · regex layer + name layer
  decontaminate.py    7 · n-gram eval fingerprints + canaries
  manifest.py         8 · per-shard provenance, gating, determinism
  driver.py           runs the chain, assembles stats.json, asserts integrity
  taxonomy.py         the strategy taxonomy (static content for the widget)
  build_site.py       renders site/index.html from stats.json + taxonomy
  site_template.html  the page itself
site/index.html     the deliverable — single file, no external requests
data/run/           stats.json, manifests.json, shards/  (generated)
data/raw/           downloaded parquet + cached eval sets  (gitignored)
```

## Reproducing

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
curl -L -o models/lid.176.ftz \
  https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz

.venv/bin/python scripts/driver.py --smoke   # ~2k docs, a few minutes
.venv/bin/python scripts/driver.py           # the full ~48M-token slice
PYTHONPATH=scripts .venv/bin/python scripts/build_site.py
```

Downloads ~1.9 GB of parquet on first run. The pipeline streams in batches — it needs about 2 GB of
RAM, not the whole shard. The eval-set fingerprints are fetched from the HF datasets-server, which
rate-limits unauthenticated requests; they are cached under `data/raw/eval/` and resumed across
runs, so a first run may need to be repeated once to fill them.

## Deploying

`site/` is one static file with no build step and no external requests, so it opens from `file://`
as well as over HTTPS. Drag the folder onto <https://app.netlify.com/drop> and record the URL at the
top of this file.

## Attribution

Sangraha is CC-BY-4.0, which permits this use with attribution. Every shard manifest records the
licence, the pinned source revision, and the sha256 of both the input file and the emitted shard.
No unredacted personal data is included in the widget.
