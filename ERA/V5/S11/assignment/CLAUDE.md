# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this directory is

Session 11 (S11) assignment of the ERA V5 course (The School of AI). The session topic is
**Optimizers and Learning-Rate Schedules** — Session 10 left `optimizer.step()` as an
unexplained line; this session asks *a gradient gives a direction, so what decides the
distance?* The lesson runs 15 sections, and every method it introduces (learning rate,
momentum, per-parameter second moment, Adam, weight decay, warmup, schedule, batch
scaling, muP, Muon) is a progressively better answer to that one question, each correcting
a specific failure of the one before it.

The arc: gradient descent's single shared learning rate fails because weights don't share
a curvature (§3); momentum's exponential moving average cancels what alternates and
accumulates what's consistent (§4); per-parameter learning rates (`η/√v`) give every
weight the same step size regardless of gradient magnitude (§5); **Adam** combines both
averages plus **bias correction**, which matters most at step 1 (3.16η vs 1.00η) (§6);
**AdamW**'s decoupled weight decay makes `ηλ` one setting, not two, with a 1/(ηλ)-step
EMA timescale on the finished weights (§7); the optimizer is **half of a model's 16
bytes/weight training memory** (§8); **warmup** exists because early gradients are
correlated (all wrong in the same direction on a freshly initialized model), producing
Adam's largest possible step — 19.2e-3 update-to-weight ratio unwarmed vs 2.83e-3 warmed
(§9); **schedules** — cosine needs the run length fixed in advance, WSD doesn't and
supports checkpoint-and-continue (§10); batch size and learning rate scale together,
linearly under gradient descent and by √batch under Adam, up to a **critical batch size**
(§11); **muP** cancels width's effect on the best learning rate, so a sweep at width 256
transfers to width 4,096 (§12); **Muon/MuonClip/Hyperball** treat a weight matrix as a
map between spaces rather than a bag of independent numbers, at a real but modest
(1.1x-1.4x, tuning-dependent) speedup over a *well-tuned* AdamW (§13); and §14's V5
decisions (AdamW + decoupled decay + warmup + WSD, four questions left open).

**The numbers that matter for this assignment** (all from the lesson,
`resources/s11-session.md`):

- §6 Adam, one weight, η=0.001, 5 hand-computed steps from gradients `0.50, 0.40, 0.60,
  0.45, 0.55`: `m,v,m̂,v̂,step,w` given to 4-6 decimals at each t (see the table in §6).
  This is the exact worked example assignment item 1 has to reproduce and match against
  PyTorch's own Adam.
- §6 bias correction: at t=1 with g=0.5, the step is **3.16η without correction** vs
  **1.00η with it** — the number item 2's "first 20 steps both ways" plot should reduce to
  as t→1.
- §9 warmup: gradient behaviour → step size table, same sign **1.000η**, noisy (mean
  zero) **0.281η**; the largest update-to-weight ratio the lesson's own run sees is
  **19.2e-3 unwarmed vs 2.83e-3 warmed** — the quantity item 3 has to log per layer.
- §10 schedules: cosine vs WSD vs constant+EMA, all described qualitatively (no single
  numeric loss value given in the prose) — item 4's two real losses at step 200 are the
  student's own measurement, not a lesson number to match.
- §12 muP: width → best η table, `256: 3.0e-3, 512: 1.5e-3, 1024: 7.5e-4, 2048: 3.8e-4,
  4096: 1.9e-4` — roughly halving each time width doubles under the standard
  parameterization. Item 5 sweeps widths 256/512/1024 directly against this table and
  extrapolates to 4096.

**Submission format confirmed (2026-09-10):** GitHub README.md (publicly accessible). This follows the S10 pattern of linking a public GitHub repo from the assignments tab. The README must include detailed write-ups for all five items with supporting code. Key directive from the assignment: **"Tune both sides before accepting a comparison. Almost every optimizer claim that failed to replicate was a well tuned method measured against a badly tuned one."**

**The thing this session is actually testing** is a direct continuation of S10's
instrumentation discipline, aimed specifically at the optimizer: *"Almost every optimizer
claim that failed to replicate was a well tuned method measured against a badly tuned
one."* Item 1 exists so a hand-rolled Adam and PyTorch's own agree to several decimals —
a wrapper around a formula is not evidence the formula was understood. Item 3's
update-to-weight ratio is the diagnostic the lesson names explicitly as "the quantity to
monitor." Item 4 pits two schedules against each other honestly (same steps, same stop
point) rather than picking the one with more flattering marketing. Item 5's muP-style
sweep is the same "tune both sides" discipline applied to hyperparameter transfer across
width.

## Layout

- `S11-assignment.md` — the assignment statement (Section 15 of the lesson, verbatim),
  plus a note that no separate submission block exists yet in the Axiom `/assignments`
  tab as of 2026-09-07.
- `resources/s11-session.md` — full lesson writeup, all 15 sections, captured verbatim
  from the lesson page (`get_page_text`). MathJax renders each formula twice (spelled-out
  then glyph run) and tables are flattened to rows — an artifact of capturing rendered
  text, not an editing choice. Widget captions noted inline; live widget state not
  captured (judged unnecessary — see Conventions).
- `resources/s11-transcript.md` — full live-class transcript (~123KB, 489 lines),
  fetched via `curl .../export?format=txt` (link-shared doc, no browser needed, same
  route S10 used). Header dates it **2026/09/05 06:43 IST**, opens with *"today's session
  is on optimizers and learning rate schedules,"* ends cleanly at *"Meeting ended after
  02:23:59."*
- Not yet created: the notebook(s), the write-up/README, and the GitHub repo.

## Conventions

- **Submission is GitHub README.md** (confirmed 2026-09-10) — follow S10's pattern: public
  repo linked from Axiom assignments tab. The README includes write-ups for all five
  items with supporting code and results.
- **Every number in the write-up must come from a cell that actually ran.** Carried
  straight from S9/S10. The lesson's own worked numbers (the §6 five-step Adam table, the
  §9 warmup ratios, the §12 width/η table) are *expected values or transfer targets* to
  check the harness against — state them as predictions, then show the run agreeing or
  explain the gap. Item 5 in particular is explicitly asking the student to extrapolate
  past what they measured (widths 256-1,024 → a stated value and confidence at 4,096),
  so the write-up needs to be honest about that being an extrapolation, not a fourth
  measurement.
- **This assignment needs real training runs, not just arithmetic** — unlike S10's item 6
  (pure bit-format arithmetic), every one of S11's five items needs a model actually
  training: item 1 needs PyTorch's Adam running on a real weight to compare against by
  hand; items 3 and 4 need full training loops with warmup/schedule logic; item 5 needs
  three separate small runs at three widths. Decide model/scale and where the loop runs
  (this machine has no GPU, carried from S9/S10) before drafting a plan — likely
  candidate is the same nanoGPT-scale approach S10 settled on, since these are short
  runs (per-item step counts are all in the hundreds, not the thousands) and width
  256-1,024 sweeps are the kind of thing `era-v5-gpu-run` or even CPU can plausibly
  handle at this scale.
- **Reuse S9/S10's build pipeline rather than hand-editing a notebook.** `../../S10/assignment/tools/`
  (copied from S9, `build_readme.py` already extended for multi-result-file reads) has
  `notebook_src*.py` (`# %%` cells) → `tools/py2nb.py` → `tools/run_nb.py` (writes
  `results*.json`) → `tools/dump_log.py` → `tools/build_readme.py`. Copy the tools,
  don't rewrite them, and extend `build_readme.py`'s placeholder set for this session's
  five items rather than S10's six.
- **Widget-data extraction judged unnecessary for this session.** Every widget described
  in `resources/s11-session.md` (gradient descent in 2D, momentum, per-parameter LR,
  Adam five-step, L2-vs-decoupled-decay, optimizer memory, warmup ratio, schedule
  comparison, LR transfer) animates a formula or table already given in full in the
  prose, and every graded number in the assignment comes from the student's own run
  against a real model, not from reading a widget's live state. Reach for
  `extract-widget-data` only if a specific widget value becomes load-bearing and the
  prose table doesn't already state it.
- **The `arxiv-library` skill applies only if the write-up ends up citing sources.** The
  lesson names several dated results in passing — Muon/MuonClip's Kimi K2 run, the 2025
  AdamW-EMA finding, the 2026 constant-LR-with-weight-averaging result, Hyperball (June
  2026) — but none of the five assignment items ask for a literature comparison the way
  S8's did. If the write-up does reference one of these by name/date, discover it via the
  skill's arxiv MCP layer rather than trusting recall, download the PDF into the local
  library, and index via `rag-toolkit` only if a specific claim needs pulling out of the
  PDF text with a citation.
- **No GPU on this machine** (carried from S9/S10: `.venv` is torch cpu-only, if a
  `.venv` even gets created here — check what S10 left behind first). Item 5's three
  width sweeps (256/512/1,024) are the most compute-sensitive item; decide whether they
  fit on CPU at a small enough model/step count or need the same EC2/Colab route S10
  used. Use the `era-v5-gpu-run` skill rather than re-deriving a GPU workflow if a GPU
  run turns out to be needed.
- Ties back to prior sessions explicitly: §6's "the two bars beneath each weight" (m and
  v) is the Session 10 widget's optimizer state, deferred there to this session; §7's
  Kronecker factors point back to Session 7's embeddings; §11's batch-size discussion
  extends Session 10's gradient accumulation and its 2024 averaging bug directly; the
  session closes by handing off to **Session 12** ("across many GPUs").
