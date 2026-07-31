# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this directory is

Session 5 (S5) assignment of the ERA V5 course (The School of AI). The session topic is
**Data Mixtures and Curriculum**. Unlike every prior session in this course, the deliverable
here is **not** a Netlify widget/app — it's a **GitHub repo README.md** containing the
written plan. The instructor was explicit about this being a deliberate change from the
S2–S4 pattern.

Task (per `S5-assignment.md`): draft the mixture-and-curriculum plan for "V5" as a written
spec that:
- States a share of the fixed token budget for every capability slot (general web, code,
  reasoning, agentic, long-context, Indic, etc.).
- Splits the Indic slot across verified / unverified / translated / synthetic tiers rather
  than giving one headline number.
- Names the agentic, reasoning, and long-context slots explicitly and points each at real
  datasets from the inventory.
- Fixes the protected always-on floor the selector (OPUS) cannot cross.
- Declares the anneal reserve held back for the cooldown phase.
- Lays out difficulty and reasoning-length bands with a concrete example each.
- Commits to (ideally runs) a 1B/3B-scale proxy experiment and metric to validate the mixture
  before it's trusted at full scale.

Grading rewards numbers defended against the real dataset inventory (no padding a lane with
data that doesn't exist) and penalizes length/wishful accounting — see the "Evaluation
strategy" section of `S5-assignment.md` for the exact rubric.

## Layout

- `S5-assignment.md` — the assignment statement and evaluation rubric.
- `TODO.md` — working checklist for this session; keep it updated as steps complete.
- `resources/s5-session.md` — lesson writeup, including descriptions of the interactive
  widgets on the session page (mixture composer, benchmark explainer, dataset inventory,
  OPUS live view).
- `resources/s5-transcript.md` — full live-class transcript (~145KB); mine it for concrete
  numbers and reasoning not in the session summary.
- Not yet created: a widget-data extraction pass (per `TODO.md`, the session page's
  interactive widgets have real live content — dataset inventory counts, mixture composer
  defaults, sample tasks — that the prose alone won't capture; extract before drafting
  numbers), and the actual plan/README to submit.

## Conventions

- Submission target is a GitHub README.md this time — no `site/` directory, no Netlify
  deploy for this session.
- The mixture plan should tie its Indic and dataset-lane numbers back to real supply: the S4
  cleaning work (`../../S4/assignment/`) feeds the "cumulative cleaning target" this session
  references, and the dataset inventory widget (once extracted) is the source of truth for
  what data actually exists per lane.
- Same "short and dense, no padding" grading bias as prior sessions — this rubric explicitly
  penalizes length and rewards defended numbers over long submissions.
- Before drafting numbers, use the `extract-widget-data` skill against the live session page
  to pull real widget content (same approach used for S4's `s4-widget-data.md`), rather than
  inventing plausible-sounding figures.
