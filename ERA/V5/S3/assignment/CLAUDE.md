# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this directory is

Session 3 (S3) assignment of the ERA V5 course (The School of AI). There is no build/test tooling here — this is a research + writing assignment whose deliverable is a **short static report page deployed to Netlify**.

Two parallel goals (per `AGENTS.md`):

1. **The assignment** (`S3-assignment.md`): design the data strategy for a hypothetical 40B-parameter, Gemma-4-class, India-first model that is strong at coding, agentic work, and Indic languages. The report must answer four things:
   - data composition for pre-training, post-training (SFT/preference), and RL/alignment — what to collect and why;
   - the cleaning pipeline for those objectives;
   - how to evaluate the model against the objectives;
   - target tokenizer **fertility** per language/domain (which Indic languages, code, science, math, agentic), and from those numbers the **tokenizer vocab size**.

   Grading rewards depth of thought and penalizes length — keep the report short and dense with concrete numbers.

2. **The research task**: build understanding of the ERA V4 reference material (`resources/URLS.md`): the LightningLM 108B/120B MoE model, the Brahmic tokenizer, and Kronecker embeddings. Analyses live in the `research/` subdirectory (separate from the report).

## Layout

- `S3-assignment.md` — the assignment statement (also embedded at the end of `resources/s3-session.md`).
- `resources/s3-session.md` — session content (data collection/sourcing); the authoritative summary of concepts the report should draw on.
- `resources/s3-transcript.md` — full lecture transcript (~156KB); mine it for insights and concrete numbers not in the session summary.
- `resources/URLS.md` — links to the ERA V4 papers/pages. Note: the Brahmic tokenizer and Kronecker embeddings entries currently point at the same arxiv ID.
- `research/` — analyses of the referenced material (research goal).
- `site/` — the deployable report (static HTML/JS only; Netlify has no server side).

## Conventions

- Deployment target is Netlify as a static site: no backend, everything client-side or precomputed.
- The sibling S2 assignment (`../../S2/assignment/`) followed the same pattern (static `site/` dir, report markdown alongside); its `SESSION_REPORT.md` documents the S2 fertility/ratio work that S3's fertility question builds on.
- Fertility numbers claimed in the report should be grounded in the S2 tokenizer results where possible rather than invented.
