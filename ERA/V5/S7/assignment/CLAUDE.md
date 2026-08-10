# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this directory is

Session 7 (S7) assignment of the ERA V5 course (The School of AI). The session topic is
**Embeddings and Model Internals** — where Session 6's packed batch of integer token ids finally
meets the model. It covers: the gather/scatter-add mechanics of an embedding lookup and why
Zipfian frequency makes different vocabulary rows train at wildly different effective rates; the
real parameter/memory cost of the token-facing matrices (V4's 131,072 x 8,096 setup is 1.06B
params, 16.98 GB of AdamW training state, per matrix); weight tying; low-rank factorized
embeddings (ALBERT-style); **Kronecker factorization** — the byte-level, vocabulary-independent
embedding scheme Session 2 deferred to this class (93.75% parameter reduction, but silently
collides distinct tokens once they share the first `pos_dim=32` UTF-8 bytes, which bites Indic
scripts hard); the V4 "frozen embedding + mixture shift" incident and why a compressed input path
is a *less* capable adapter, not a cheaper one; and positional encoding's absolute-table wall
(cannot extrapolate past `max_position`), setting up Session 8's rotary/ALiBi treatment.

**Unlike every prior session, the assignment is not a data/pipeline deliverable — it's an open
research problem.** Per `S7-assignment.md`, the instructor (planning to write a "Kronecker
Embedding V2" paper) poses five separate, independent embedding-design problems and asks you to
pick **one**:

1. Embeddings that store mathematical structure (e.g. the embedding of `9` composes under `+`/`*`
   so `9+9`'s embedding-math resembles `18`'s) — append new dimensions for this, keep 32 for
   existing word/subword content.
2. The natural extension of Kronecker to represent images and audio, not just text.
3. Removing the fixed 32-byte-position window — a dynamic scheme that doesn't crop tokens longer
   than 32 bytes.
4. A "real" Fourier alternative to Kronecker — representing each character as a Fourier wave and
   summing to form a word.
5. An invertible/reversible Kronecker (same embedding always decodes back to the same token),
   which would let the model drop the output head entirely and scale to a ~1M vocabulary.

Deliverable: state which problem you solved, prove the solution works (train a small transformer
model as evidence — the instructor explicitly says the agent can figure out the model/training
part), and write it up as a README (a webapp with graphs/animations is welcome but not required)
plus the code that proves it. Grading is pass/fail-style via a GitHub README or app link, checked
in incognito for public accessibility — no numeric rubric table this session (contrast with S5's
budget rubric and S6's 1000-point breakdown).

## Layout

- `S7-assignment.md` — the assignment statement, verbatim (short — just the 5 problems + submission
  format, no evaluation rubric this time).
- `resources/s7-session.md` — full lesson writeup (14 sections), including the gather/scatter-add
  mechanics, the exact parameter/memory arithmetic for dense vs. tied vs. factorized vs. Kronecker
  embeddings, the Kronecker construction (`kappa(b) = (1/sqrt(L)) * vec(sum_p c[byte_p] ⊗
  p[position_p])`, projection `Linear(8192, d_model)`), the 32-byte collision risk with a concrete
  Hindi collision example, the V4 frozen-embedding incident, and the positional-encoding sections
  that hand off to Session 8. Widget captions are summarized inline in each section (this session's
  widgets are illustrative demos described in prose, not separate hosted pages — see Conventions).
- `resources/s7-transcript.md` — full live-class transcript (~145KB, pulled from the session's
  linked Google Doc); mine it for implementation details, instructor asides, and the exact framing
  of the five assignment problems not fully captured in the session summary.
- Not yet created: a decision on which of the 5 problems to pursue, the small transformer
  implementation + training run that proves the chosen idea, and the README/writeup to submit.

## Conventions

- Submission target is a GitHub README (or app) link — same lightweight submission style as S5,
  not S6's evidence-bundle codebase, though the assignment does require runnable proof code.
- This session's numeric grounding (parameter counts, memory costs, collision examples) is already
  in the session prose itself (Section 7-8 of `resources/s7-session.md`) — the lesson widgets are
  demonstrative/interactive (sliders, live training runs) rather than the sole source of numbers
  needed to defend a plan, unlike S4-S6. Widget-data extraction via the `extract-widget-data`
  skill is optional grounding here, not a blocking prerequisite, unless a chosen problem needs a
  specific widget's exact default (e.g. the collision-count byte-budget lab in Section 8).
- The five problems are explicitly independent — "don't try and mix them." Pick one and go deep
  rather than spreading thin across several.
- The assignment explicitly expects agent-driven proof: "ask your agent to write a small
  transformer model and train it. It will figure out itself." Toy-scale is fine; the point is
  proving the mechanism, not achieving SOTA.
- Ties back to V4/V5 decisions from prior sessions: the Kronecker codec and its `pos_dim=32`
  window (S7 §7-8), the embedding_policy_id ledger field (S6), and the V4 frozen-embedding
  mixture-shift incident (S5) are all live context a chosen solution should be aware of, even
  though this assignment's deliverable is a standalone research proof rather than a V5-integrated
  plan.
