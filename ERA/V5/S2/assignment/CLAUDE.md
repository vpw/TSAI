# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository status

This repository currently contains only assignment specs (`AGENTS.md`, `S2-assignment.md`) — no code has been written yet. When implementing, this file should be updated with actual build/run/test commands and real architecture notes once the project structure exists.

## What this assignment requires

This is session 2 (S2) of the ERA V5 course (The School of AI). The deliverable is a **single HTML + JS artifact** (a JS framework such as React is allowed if bundled appropriately for the scope) that must be deployable as-is to Netlify.

Task: build a BPE (Byte Pair Encoding) tokenizer widget covering India's Wikipedia page in English, Hindi, Telugu, and one additional language of choice, with these constraints:

- Combined vocabulary across all languages: 10,000 tokens total.
- For each language, compute the ratio X = (vocab size of that language, e.g. ~5000 words) / (token count for that language after BPE). Each language's target ratio is ≤ ~1.2.
- Label the four ratios X1–X4, sort them, and compute the self-score as `1000 / (X_max - X_min)`.

The submitted widget must:
1. Display the per-language ratios, token statistics, calculations, and the resulting self-score.
2. Let the viewer browse/inspect the full learned token list (the tokenizer's vocabulary).
3. Be hosted (Netlify or elsewhere) with the URL included in the submission.

## Structure expectations

- Even if multiple exercises/sections exist, the final submission is **one HTML+JS file** (plus any supporting JS bundle), organized as tabs or sections for each part — not separate deployed pages.
- Since deployment target is Netlify as a static site, avoid introducing a server-side backend; tokenizer training/inference should run client-side or be precomputed and shipped as static data consumed by the page.
