# Task

This directory is part of the assignments for the ERA V5 course of The School of AI (TSAI).
Specifically this is for the eleventh session (S11).

The `S11-assignment.md` file lists the exercise in full — please refer to that for the
details. **Session 11 has no assignment entry in the Axiom LMS's Assignments tab yet**
(checked 2026-09-07; the tab lists S1-S10 only), so there is no confirmed due date, point
value, or submission-format statement. Section 15 of the lesson is the whole spec for
now — treat the eventual submission shape as an open decision, not an assumption carried
from S10.

# Details

The session is **Optimizers and Learning-Rate Schedules**: Session 10 left
`optimizer.step()` unexplained — a gradient tells a weight which direction to move, not
how far, and this session is the history of increasingly better answers to "how far,"
each one fixing a specific failure of the last.

The arc: plain gradient descent fails because different weights (and different
directions in the same loss surface) don't share a curvature, so one learning rate can't
serve all of them (§3); momentum's exponential moving average (β1=0.9) cancels
alternating gradients and accumulates consistent ones (§4); a second EMA over the
*squared* gradient (β2=0.999) gives each parameter its own learning rate, `η/√v`, so a
rarely-updated embedding still takes a full step when its gradient finally arrives (§5);
**Adam** combines both, plus **bias correction** — exact, not approximate, and it matters
most at step 1 because β2 is so much closer to 1 than β1 (3.16η vs 1.00η without/with
correction) (§6); **AdamW** decouples weight decay from the Adam denominator, making the
*product* `ηλ` the real setting — it sets the EMA timescale of the finished weights
(1/(ηλ) steps) (§7); the optimizer state (m, v, plus fp32 master weights) is **half** of
a model's 16-bytes-per-weight training memory (§8); **warmup** exists because a freshly
initialized model's gradients are all pointing the same wrong way, producing Adam's
largest possible step (update-to-weight ratio 19.2e-3 unwarmed vs 2.83e-3 warmed) (§9);
**schedules** trade off cosine (needs the run length fixed in advance) against WSD
(doesn't, and supports checkpoint-and-resume) (§10); batch size and learning rate scale
together — linearly for gradient descent, by √batch for Adam, up to a critical batch
size where more samples stop helping (§11); **muP** removes width's effect on the best
learning rate so a small sweep transfers to a large model (§12); and **Muon** /
**MuonClip** / **Hyperball** treat a weight matrix as a linear map rather than a bag of
independent scalars, for a real but modest speedup once both sides are tuned equally
(§13). §14 states V5's own choices (AdamW + decoupled decay + warmup + WSD, with four
questions still open) and §15 is the assignment.

Numbers worth having in hand: the §6 five-step hand-computed Adam trace (gradients 0.50,
0.40, 0.60, 0.45, 0.55 at η=0.001, giving m/v/m̂/v̂/step/w at each step) is the exact
worked example item 1 has to reproduce and match against PyTorch's own optimizer; the §9
warmup table (same-sign gradient → 1.000η step, noisy gradient → 0.281η step) is the
mechanism behind item 3's update-to-weight ratio; the §12 width→η table (256: 3.0e-3 down
to 4096: 1.9e-4, roughly halving per doubling) is what item 5's sweep is checked against
and extrapolated from.

Full writeup: `resources/s11-session.md`. Live-class transcript:
`resources/s11-transcript.md` (2026-09-05, ~123KB).

**What the assignment actually asks for** — five items, all requiring a real optimizer
running against a real model, not just arithmetic:

1. **Reproduce Adam by hand.** One weight, five gradients — compute `m, v, m̂, v̂` and the
   resulting step yourself, then check each against PyTorch's `torch.optim.Adam`. They
   should agree to several decimal places.
2. **Disable bias correction and plot the first twenty steps both ways.** Report the
   number of steps after which the difference (bias-corrected vs not) stops mattering.
3. **Log the update-to-weight ratio for every layer**, and identify the step at which
   warmup stops changing it.
4. **Train the same model twice for 300 steps**, once under cosine and once under WSD,
   stopping both at step 200. Report both losses and state which model you would keep.
5. **Sweep the learning rate at widths 256, 512 and 1,024**, plot loss against learning
   rate, mark the three minima, state the value you'd use at width 4,096 and how
   confident you are in it.

**The graded discipline is the same "tune both sides" honesty as S10's instrumentation
point, now aimed at the optimizer specifically**: *"Almost every optimizer claim that
failed to replicate was a well tuned method measured against a badly tuned one."* Item 1
is a correctness check on understanding, not on PyTorch. Item 3's ratio is the lesson's
own named monitoring quantity. Item 4 is explicitly an apples-to-apples comparison (same
model, same step budget, same stopping point). Item 5 asks for an honest confidence
statement about extrapolating past what was actually measured, which is itself a form of
the same discipline.

**Two practical constraints before planning.** (a) There is **no GPU confirmed available
on this machine yet for S11** (S9/S10 ran CPU-only or on provisioned cloud GPUs) — decide
where the five items run, especially item 5's three separate width sweeps, which is the
most compute-sensitive item here. (b) No submission format is confirmed (see above) —
don't commit to a specific deliverable shape (single notebook vs README-plus-repo vs
something else) until the Assignments tab actually posts S11, or the user says otherwise.

**If the write-up ends up citing sources** (Muon/MuonClip's Kimi K2 run, the 2025
AdamW-EMA result, Hyperball), use the `arxiv-library` skill rather than recall: discover
via its arxiv MCP layer, download the PDF into the local library so the source is a
checkable file, and index via `rag-toolkit` when a specific claim needs pulling out of
the PDF text with a citation. This is optional here — none of the five items require a
literature comparison the way S8's did.

As a capable agent, plan to: (1) read `resources/s11-session.md` §§3-13 for the mechanics
each item has to demonstrate, and the transcript for the instructor's own framing/
worked-example asides, (2) settle the open decisions in `TODO.md` (where the runs
happen, what model/scale, submission format once confirmed), (3) reuse
`../../S10/assignment/tools/` (the S9-derived `py2nb.py`/`run_nb.py`/`dump_log.py`/
`build_readme.py` pipeline) rather than hand-editing a notebook, (4) run every item
top to bottom in a fresh runtime so every number traces to a cell that actually ran,
(5) write up the five items with their explanations, (6) push and verify once a
submission format exists. `TODO.md` tracks progress on these steps.

## References
Refer CLAUDE.md if it exists
