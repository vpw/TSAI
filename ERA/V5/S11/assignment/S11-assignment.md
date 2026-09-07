# Session 11 - Assignment

Source: Section 15 of the Session 11 lesson page ("Optimizers and Learning-Rate
Schedules"), captured verbatim.

1. Reproduce Adam by hand. Take one weight and five gradients, compute m, v, m̂, v̂ and the
   resulting step yourself, then check each against PyTorch. They should agree to several
   decimal places.
2. Disable bias correction and plot the first twenty steps both ways. Report the number
   of steps after which the difference stops mattering.
3. Log the update-to-weight ratio for every layer, and identify the step at which warmup
   stops changing it.
4. Train the same model twice for 300 steps, once under cosine and once under WSD, and
   stop both at step 200. Report both losses and state which model you would keep.
5. Sweep the learning rate at widths 256, 512 and 1,024, plot loss against learning rate,
   and mark the three minima. State the value you would use at width 4,096 and how
   confident you are in it.

Tune both sides before accepting a comparison. Almost every optimizer claim that failed
to replicate was a well tuned method measured against a badly tuned one.

## Submission block

**Not yet posted.** Checked the course's `/assignments` tab (2026-09-07): it lists
Session 1 through Session 10 only ("Session 10 - Assignment QnA", due 2d ago, marked
**Late**, 1000 pts, awaiting review) — there is no "Session 11 - Assignment QnA" entry
yet. So there is currently no due date, point value, submission-format statement, or
rubric to capture for S11 — Section 15's five items above are the whole spec available
right now. Re-check `/assignments` once the instructor posts the graded entry, since past
sessions' submission format has varied (Netlify widget/site, GitHub README-only, full
repo with evidence bundle) and shouldn't be assumed from S10's shape.
