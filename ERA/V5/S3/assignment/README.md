# S3 Assignment — Data Plan for a 40B India-First Model

**Deployed report: https://era-v5-s3-vardhan.netlify.app/**

## What's here

- `site/` — **the deliverable**: a single self-contained `index.html` report answering the
  four assignment questions. Deploy by dragging the `site/` folder onto
  https://app.netlify.com/drop (no build step, no backend). Light + dark mode supported.
- `DESIGN.md` — the full worked-out numbers and reasoning behind the report (kept out of the
  submission because longer submissions score lower).
- `research/` — analyses of the ERA V4 references (LightningLM, BrahmicTokenizer-131K,
  Kronecker embeddings) + implications for this assignment + a longer study plan.
- `resources/` — session content, transcript, reference URLs (Kronecker arxiv ID corrected
  to 2605.29459; URLS.md originally duplicated the tokenizer's ID).

## Headline numbers in the report

8T pre-training tokens (200/param) · 12 Indic languages in 2 tiers · 10% always-on batch
share (Indic + India-perspective English) · 15% anneal reserve · SFT 1M / preference 400K /
RLVR 80K · fertility targets 1.0 (code) – 1.9 (Malayalam), weighted ≈1.22 · vocab
**196,608 = 3×2¹⁶** derived from the fertility targets via BrahmicTokenizer-style LP slot
allocation, made affordable by Kronecker embeddings.

## Deployment

Live at https://era-v5-s3-vardhan.netlify.app/ — deployed by dragging `site/` onto
https://app.netlify.com/drop. To update it, rebuild `site/index.html` and drag the folder
again; there is no build step, no backend and no CLI in the loop.
