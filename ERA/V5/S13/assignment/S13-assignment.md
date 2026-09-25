# Session 13 — Assignment

Source: Axiom assignment page
(`https://axiom.theschoolofai.in/courses/cmq97i5kn032208o8xu5dab4q/assignments/cmu6k8tf608i509l7m9gaotpr`),
captured 2026-09-24. Identical to §18 of the lesson.

## Brief (verbatim)

> Train a 20M LLM for 50M tokens on Google Colab (or anything else of your choice). Fix Batch
> size that you can run. Train again with Reversibility (report which variant worked for you,
> mid-point, euler, etc) Train again with Reversibility, but push it to the maximum batch size.
>
> Report final loss, speed (token/s), memory peak and other findings. Submit detailed README.md
> (github link), repo must have the ipynb notebooks.

## Axiom submission block

- Title: **Session 13 - Assignment QnA**
- Status: Available. **Due Sat, Sep 26, 2026, 7:00 AM.** 1000 points. Resubmission allowed.
- One field: **GitHub README.md** link (1000 pts), with the checkbox *"I tested this link in an
  incognito window — it's publicly accessible (not private)."*
- The Rubric tab publishes no criteria beyond the brief.

## The instructor's framing in the live class

From `resources/s13-transcript.md` (class of 2026-09-19), near the end:

- *"Right now I want you to get used to reversibility."* Fix the batch size first: *"decide the
  biggest batch size that is possible for you to run a 20 million for 50 million tokens, so get
  the speed and everything set."*
- *"There are at least two variants: there's Euler and there's midpoint. Test both of them and
  see which one is working for you."* Choose between them *"based on how the loss trajectory
  is"*, at the **same** batch size as the baseline.
- Then *"push the batch size and see how much can you push in a single batch."*
- Report *"the final loss for all the runs that you have, the speed, basically token per second
  that you're hitting for the model, and the memory peak and other findings."*
- Reversibility brings two restrictions: *"we can't use weight decay, we cannot use dropout."*
- A student proposed also calculating the **cost** with and without reversibility on the cloud;
  the instructor agreed (*"Of course. Correct."*).
- The instructor measured on a Mac that doubling sequence length (512 → 1024) moved RAM use only
  from 52% to 57% with reversibility.

The three runs the brief names:

1. **Baseline**: standard residual transformer, ~20M parameters, 50M tokens, at a fixed batch
   size that fits.
2. **Reversible, same batch**: the same model and token budget with a reversible residual update.
   Test at least Euler and midpoint, and report which one worked.
3. **Reversible, max batch**: the chosen variant, with the batch pushed to the largest that fits.
