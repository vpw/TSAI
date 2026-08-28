# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this directory is

Session 9 (S9) assignment of the ERA V5 course (The School of AI). The session topic is
**Loss Functions & Output Heads** — the bridge from Session 8's hidden state to a single scalar the
optimiser can push down. The lesson runs 24 sections: the rest of the transformer block (residual
stream, FFN/SwiGLU, RMSNorm, pre-norm — 202.4M params per layer at `d_model = 4,096`); the **output
head** (`z = h · W_vocabᵀ`, also called the unembedding or LM head) and **weight tying**; softmax
turning logit *gaps* into ratios; **cross-entropy as one question** — what probability did you give
the truth — with its general form, entropy, and the collapse `D_KL(p‖q) = H(p,q)` for a one-hot
target; the gradient `softmax(z) − onehot(y)`, which **sums to exactly zero** (the fact §11's
stability problem rests on) and is **dense over all 131,072 rows**; where targets come from
(next-token shift, and the four quiet bugs around it — padding, document packing, the shift
direction, and off-by-one); **perplexity** as the readable form of the loss; the head as a second
memory problem; a GPU interlude; **four implementations of one objective**; head stability;
adaptive softmax; **multi-token prediction**; then the post-training bridge — auxiliary
pre-training losses, SFT as masked cross-entropy, Bradley-Terry reward models, RLHF/PPO, **DPO**
(the reward model cancels), **GRPO** (the value network cancels too), the DPO variant family, and
**distillation** (forward vs reverse KL). Closes with §22's loss map — every loss in the course as
a KL against something different — and §23's V5 decisions.

**The numbers that matter for this assignment** (all from the lesson, `resources/s9-session.md`):

- `V = 131,072`, `D = 4,096` → dense head = `131,072 × 4,096` = **536,870,912 = 536.9M params**.
- That is **exactly** the dense embedding table Session 7 threw away, and it is **16.0×** the whole
  compressed input side. S7's 93.75% saving across both ends degrades to 46.9% if the head stays dense.
- **Weight tying is closed to us.** S7's input side is a fixed byte codec plus one projection —
  there is no `[V, D]` input table to tie to. The assignment still asks for the tied-vs-untied
  parameter comparison, so report it as the counterfactual it is.
- Untrained-model anchor: loss `= ln(131,072) =` **11.784 nats**, perplexity **131,072**. This is
  the assignment's cheapest sanity check — if an untrained model doesn't sit near vocab size, there
  is a bug upstream of everything else.
- Perplexity is **not comparable across tokenizers** (it is per-token, and a tokenizer defines the token).
- The logits *tensor*, not the parameters, is the real bill: **16 GiB retained for backward** in the
  naive path, **64 GiB at 256K context**.

**Unlike S7 and S8, this is not a build-and-defend web app — it's a notebook deliverable.** Per
`S9-assignment.md`: one Colab notebook, moved to GitHub, that runs top to bottom, plus a short
write-up. Part 1 is a loss harness made "correct and observable" (seven required numbers). Part 2
adds a second output head predicting token `t+2` and reports both losses and their sum. The graded
artifact is a **GitHub README.md link** that must be publicly accessible in an incognito window,
with the `.ipynb` and/or training logs in the same repo to back the README's numbers.

**The thing this session is actually testing** is that you can catch a silent bug in the few lines
between the model output and the scalar. The instructor's warning is explicit and repeated from
last session: *a target shift in the wrong direction can produce a beautiful loss curve*. That is
why the assignment demands the shift be verified by printing **token strings side by side, not
ids** — the failure mode is invisible in a wall of integers and raises no exception.

## Layout

- `S9-assignment.md` — the assignment statement, verbatim from the assignment page (brief +
  submission block). 1000 points, due Sat Aug 29 2026, resubmission allowed. The Rubric tab exists
  but is **empty** — no criterion rows, only the 1000-point total.
- `resources/s9-session.md` — full lesson writeup, all 24 sections, captured verbatim from the
  lesson page. Inline MathJax appears twice per formula (spelled-out form then rendered glyph run)
  and tables are flattened to rows — an artifact of capturing rendered text, not an editing choice.
- `resources/s9-transcript.md` — full live-class transcript. **Not yet extracted** (pending
  download approval). The lesson page notes the studio recording was corrupted and a cropped GMeet
  version was uploaded in its place.
- Not yet created: the notebook itself, the write-up/README, and the GitHub repo.

## Conventions

- **Submission target is a GitHub README.md link** — a single link, not the two-artifact
  (live app + repo) shape of S7/S8. No deployment step, no hosting decision. The repo must contain
  the notebook (`.ipynb`) and/or training logs so the README's numbers are backed by a runnable
  artifact rather than asserted. The link must pass an **incognito-window accessibility check** —
  the submission form has a dedicated checkbox for it, same as every prior session.
- **Every number in the write-up must come from a cell that actually ran**, not from the lesson and
  not from recall. This is the S9 equivalent of S8's date-sourcing discipline: the assignment is
  graded on observability, so a number without a printed cell behind it is the failure mode. The
  lesson's own figures (536.9M, 11.784, 131,072) are the *expected* values to check the harness
  against — state them as predictions, then show the run agreeing or explain the gap.
- **Widget-data extraction is optional here, not a prerequisite.** The lesson's interactive widgets
  (the vocabulary/dot-product explorer, the drag-a-logit gradient demo, the SFT masking toggle) are
  demonstrations of formulas already stated in full in the prose, and this assignment's numbers come
  from the student's own notebook run rather than from any widget's default state. Reach for
  `extract-widget-data` only if a specific widget's exact value becomes load-bearing.
- **The `arxiv-library` skill applies only if the write-up makes external factual claims.** The core
  deliverable is measurement, not citation, so there is no S8-style dating task here. If the write-up
  does name a paper (z-loss in OLMo/Chameleon, Bradley-Terry, DPO, GRPO, adaptive softmax, MTP), find
  it via the skill's arxiv MCP layer rather than trusting a remembered title or date, download the
  PDF into the local library so the source is a checkable file, and index it via `rag-toolkit` when a
  specific claim needs pulling out of the PDF text with a citation.
- **Known ambiguity — "Part 3" does not exist.** The lesson page's §24 says to submit *"the seven
  numbers from Part 1, the two losses from Part 2, and your demonstration from Part 3"*, but neither
  the lesson nor the assignment page defines a Part 3; the assignment page's copy drops the clause
  and ends at Part 2. Treat the assignment page as authoritative (Parts 1 and 2 only), and consider
  asking the instructor rather than inventing a third part.
- Resubmission is allowed (due Sat Aug 29, 2026), so an initial pass covering Parts 1 and 2 can be
  extended later without penalty.
- Ties back to prior sessions explicitly: Session 7's byte-codec input side is *why* weight tying is
  unavailable and why the dense head is the dominant remaining parameter block; Session 8's hidden
  state `[B, T, D]` is the input this session consumes; Session 10 picks up at loss → gradients.
