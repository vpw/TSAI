# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this directory is

Session 8 (S8) assignment of the ERA V5 course (The School of AI). The session topic is
**Modern Attention Variants** — where Session 7's `[B, T, D]` token vectors finally meet each other.
It builds standard scaled dot-product attention from scratch (Q/K/V projections, the causal mask,
why softmax(-∞)=0 enables parallel training), then names attention's "two bills" — quadratic
*compute* (`~T²` score comparisons) and linear *KV-cache memory* (`~T`, per-user, unshared) — and
tours the field's responses to each: removing softmax to get a fixed-size linear-attention state
(and why that state needs the **delta rule** to correct rather than just accumulate); **sparse/
top-k attention** (keep softmax, read fewer keys, but scoring-then-pruning doesn't remove the
scoring cost without a cheap proposal step); **RoPE** (position as rotation, so query-key dot
products depend only on relative distance `i−j`) and **DroPE** (V4's reported 8K→256K, 32×
extension, applied *before* annealing — "defined at 256K" ≠ "competent at 256K"); the full **KV
cache formula** (`2 × layers × kv_heads × head_dim × T × batch × bytes_per_number`, worked example
6.44GB/user → 51.54GB/8 users at 48 layers, 8 KV heads, head_dim 128, bf16, T=32,768); **GQA/MQA**
(shares KV heads across query heads — lowers the cache *slope*, doesn't stop the *growth*);
**sequence compression + top-k block selection** (DeepSeek-style, via a cheap low-rank indexer);
**depth schedules** mixing fixed-state and sparse-attention layers (V4's `DDDGDDDG` motif — 8×
larger KV cache but only ~1.41× more compute going from 1-in-8 to 8-in-8 sparse layers); and a
**Memory Stream** for state that survives chunk boundaries (one stop-gradient summary vector,
~6% injected). Closes on "long context is a system, not a number" (six conditions must all hold)
and the two-roads framing (extend a short-trained model vs. build/train long natively) that the V5
architecture decision inherits.

**Like S7, the assignment is not a data/pipeline deliverable — it's a build-and-defend web app.**
Per `S8-assignment.md`, build (with the agent) a visual explainer app covering every attention
mechanism from the session, **ordered by actual historical launch date** (not teaching order, not
grouped by family) — the instructor's stated point is to make the field's shifting priorities
visible on a timeline (exactness → memory → length → memory again → sparsity/compression). Every
mechanism needs honest pros/cons: what it buys, what it costs, when you'd actually pick it.
Minimum coverage: standard attention, absolute learned positions, sinusoidal, RoPE, ALiBi, MQA,
GQA, sliding window, attention sinks, NTK-aware scaling, YaRN, linear attention, delta rule/Gated
DeltaNet, MLA, sparse/top-k attention, DeepSeek's compressed sparse attention, DroPE — plus credit
for a well-sourced mechanism not covered in class. **Dates must be checked against primary sources**
(paper/release), not trusted from an agent's confident-sounding memory — the instructor explicitly
cites correcting a wrong date from Session 7 live at the top of this class, and grades a
1000-point bonus question specifically on what the chronological ordering reveals.

## Layout

- `S8-assignment.md` — the assignment statement, verbatim (brief + questions + rubric points +
  submission requirements — 1000 pts for the link/repo, 1000 bonus for the timeline-insight
  question, 250 optional for a public share).
- `resources/s8-session.md` — full lesson writeup (18 sections), including the attention equation
  build-up, the two-bills framing, the no-softmax/delta-rule derivation, RoPE's rotation math,
  DroPE's evidence-boundary framing, the full KV-cache formula with worked numbers, GQA/MQA,
  sequence compression + indexer-based top-k, the `DDDGDDDG` depth schedule, the Memory Stream
  gating rule, the six-condition "long context is a system" checklist, and the two-roads closer.
  Widget captions summarized inline per section (see Conventions — this session's widgets are
  illustrative/interactive demos, not the sole source of load-bearing numbers).
- `resources/s8-transcript.md` — full live-class transcript (~118KB, pulled from the session's
  linked Google Doc); mine it for implementation asides, the instructor's own framing of the
  chronology requirement, and the specific Session-7-date-correction anecdote referenced in the
  assignment.
- Not yet created: the mechanism-by-mechanism chronology research (dates + primary sources), the
  web app itself (design + implementation), and the README documenting sources.

## Conventions

- Submission target is a **live app link (Netlify/Vercel/equivalent) + GitHub repo + README with
  chronology sources** — same two-artifact shape as S7 (live link/app + code), but this session's
  deliverable is explicitly a polished, shareable web app ("something you would actually send to a
  friend"), not primarily a research writeup. The live link must pass an incognito-window
  accessibility check (stated directly on the assignment page's submission form).
- This session's numeric grounding (cache formula worked example, schedule compute/memory
  multipliers, Memory Stream injection scale) is already in the lesson prose itself
  (`resources/s8-session.md` §10-14) — the widgets are interactive demos of formulas/tradeoffs
  already stated in text, not the sole source of a number needed to defend a plan. Treat
  `extract-widget-data` as optional grounding here, not a blocking prerequisite, unless a specific
  widget's exact default value becomes load-bearing for the app's own numbers. The app's own
  visualizations are built from scratch by the agent, not by reusing the lesson's own widgets.
- The hard part of this assignment is **historical accuracy, not code**: every mechanism's launch
  date must be checked against its actual paper/release. The instructor explicitly flags this as
  the place agents most confidently get things wrong, and ties a 1000-point bonus question to what
  the correctly-ordered timeline reveals. Treat date-sourcing as a first-class research task before
  or alongside building the app, and cite sources in the README.
- **Use the `arxiv-library` skill for the chronology research**, on the assumption that every
  mechanism's primary source is on arXiv. Three steps per mechanism: (1) search/discover the actual
  paper via the skill's arxiv MCP layer (metadata, dates, canonical IDs) rather than trusting recall
  of a title or date; (2) download the PDF into the local library so it's a checkable artifact, not
  a claim; (3) index it (via the skill's `rag-toolkit` layer) so specific claims — exact publication
  date, the mechanism's original name, what problem it says it's solving — can be pulled back out
  with a citation instead of re-summarized from memory. Indexing is "may be needed," not mandatory
  for every paper — reach for it when a date or claim needs verifying against the PDF text itself,
  not as a blanket first step for all ~17 mechanisms.
- Resubmission is allowed (due Sat Aug 22, 2026) — same as S7's "follow-up submissions" pattern, so
  an initial pass covering the minimum list can be extended later (e.g. the bonus not-covered-in-
  class mechanism) without penalty.
- Ties back to prior-session decisions this session explicitly hands off from/to: Session 7's
  embedding pipeline (`token id → fixed byte codec → code → trainable projection → [B,T,D]`) is the
  stated starting point; Session 7's absolute-position-table wall is the problem RoPE/DroPE solve;
  and this session's own §16 "What V5 has to decide" open questions (schedule ratio, extend-vs-
  build-native, sparsity budget re-measurement) are explicitly framed as inputs to a later V5
  architecture review, not this assignment's scope.
