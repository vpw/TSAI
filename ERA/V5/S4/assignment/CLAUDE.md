# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this directory is

Session 4 (S4) assignment of the ERA V5 course (The School of AI). The session topic is
**Data Cleaning and Deduplication**. The deliverable is a **single HTML + JS widget deployed
to Netlify** — same pattern as S2.

Task (per `S4-assignment.md`):
1. Count and describe the cleaning "strategies" covered in the session.
2. Find a 10-100M (token/doc scale) dataset to clean — check the S3 assignment's leftover
   dataset candidates first (`../../S3/assignment/`).
3. Apply those cleanup strategies to the chosen dataset.
4. Build a widget that shows: how many strategies and what they are, which dataset was
   picked, what was cleaned and why/how, any other strategy/concern handled, and final
   statistics.
5. Deploy to Netlify and share the link.

## Layout

- `S4-assignment.md` — the assignment statement.
- `TODO.md` — working checklist for this session; keep it updated as steps complete.
- `resources/s4-session.md` — lesson writeup, including descriptions of the interactive
  widgets on the session page.
- `resources/s4-transcript.md` — full live-class transcript (~150KB); mine it for concrete
  numbers and callouts not in the session summary.
- `resources/s4-widget-data.md` — real data already extracted from every interactive widget
  on the session page (all 8 pipeline stages + all 10 widgets: exact stage yields, V4 defect
  callouts, MinHash/LSH worked example, PII/language-ID cases, manifest gating logic). Use
  this instead of re-scraping the live page.
- Not yet created: the deployable widget (`site/` or similar), and any scripts/data used to
  actually clean the chosen dataset.

## Conventions

- Deployment target is Netlify as a static site, same as S2/S3: single HTML+JS deliverable
  (a JS framework is fine if bundled for the scope), no server-side backend — any dataset
  cleaning should run offline/precomputed, with results shipped as static data for the widget.
- Widget data was already pulled with the `extract-widget-data` skill — don't re-run browser
  extraction against the live session page; `resources/s4-widget-data.md` has the real numbers.
- 8 pipeline stages is the headline strategy count (extract → normalize → language ID →
  quality filter → deduplicate → PII scrub → decontaminate → manifest). Still open per
  `TODO.md`: whether sub-techniques (shingling, MinHash, LSH) should be counted/described
  separately in the writeup.
- Grading rewards concrete numbers and depth over length — keep the widget dense, not padded
  (same bias observed in S2/S3 grading).
