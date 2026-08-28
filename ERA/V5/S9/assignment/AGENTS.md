# Task

This directory is part of the assignments for the ERA V5 course of The School of AI (TSAI).
Specifically this is for the ninth session (S9).

The `S9-assignment.md` file lists the exercise in full — please refer to that for the details.
Unlike S7 and S8, this is **not** a web app or a research writeup. It is a **notebook deliverable**:
one Colab notebook, moved to GitHub, that runs top to bottom, plus a short write-up. The graded
artifact is a single **GitHub README.md link**, publicly accessible in an incognito window, with the
`.ipynb` and/or training logs in the same repo backing every number the README claims.

# Details

The session covers the path from a hidden state to a scalar loss: the rest of the transformer block
(residual stream, FFN/SwiGLU, RMSNorm, pre-norm); the **output head** `z = h · W_vocabᵀ` (a.k.a.
unembedding / LM head) and weight tying; softmax turning logit *gaps* into ratios; **cross-entropy**
as the single question *what probability did you assign the truth*, its general form via entropy and
KL, and the collapse to `−log q(y)` for a one-hot target; the gradient `softmax(z) − onehot(y)`,
which sums to **exactly zero** and is **dense over the whole vocabulary**; where targets come from
(the next-token shift) and the four quiet bugs there — padding, document packing, shift direction,
off-by-one; **perplexity** as the readable form of the loss; the head as a second memory problem;
a GPU interlude; four implementations of one objective (materialise / fuse / chunk / shard);
head stability and z-loss; adaptive softmax; **multi-token prediction**; then the post-training
bridge — SFT as masked cross-entropy, Bradley-Terry reward models, RLHF/PPO, DPO, GRPO, the variant
family, and distillation (forward vs reverse KL). §22 maps every loss in the course as a KL against
something different.

Numbers worth having in hand: `V = 131,072`, `D = 4,096`, dense head = **536.9M params** (exactly
the embedding table S7 discarded, and 16× the compressed input side); untrained loss
`ln(131,072) =` **11.784**, perplexity **131,072**; naive logits tensor **16 GiB** retained for
backward, **64 GiB** at 256K context. **Weight tying is unavailable to V5** — S7's input side is a
byte codec plus a projection, so there is no `[V, D]` table to tie to.

Full writeup: `resources/s9-session.md`. Live-class transcript: `resources/s9-transcript.md`.

**What the assignment actually asks for:**

1. **Part 1 — the loss harness.** Take the three-line `cross_entropy` snippet and make it *correct
   and observable*. Seven required outputs: (a) every tensor shape, with one line naming what each
   dimension is; (b) the shift verified by printing **token strings**, inputs beside targets — not
   ids; (c) padding masked, with the count of contributing tokens shown to change; (d) two documents
   packed into one sequence with the boundary masked, loss shown before and after, difference
   explained; (e) perplexity, demonstrating an untrained model sits near vocabulary size; (f) tied
   vs untied head parameter counts on this configuration; (g) peak memory for ordinary
   cross-entropy vs a **chunked version written by hand**, both numbers and the ratio.
2. **Part 2 — one extra head.** Add a second output head predicting token `t+2`. Report both losses
   separately and their sum, and explain what happens to the second head's loss over training
   relative to the first.
3. **Submit** a GitHub README.md link (incognito-accessible) with the seven numbers from Part 1 and
   the two losses from Part 2, with the notebook/logs in the repo.

**The graded skill is catching a silent bug**, not writing code. The instructor's warning, repeated
from last session: a target shift in the wrong direction produces a beautiful loss curve. Many
serious training bugs live in the few lines between the model output and the scalar, and they do not
raise an exception. Print the strings.

**Note a discrepancy before planning:** the lesson page's §24 asks for "your demonstration from
Part 3", but no Part 3 is defined anywhere; the assignment page drops that clause. Treat the
assignment page (Parts 1 and 2) as authoritative — don't invent a third part.

**If the write-up ends up citing papers** (z-loss in OLMo/Chameleon, Bradley-Terry, DPO, GRPO,
adaptive softmax, MTP), use the `arxiv-library` skill rather than recall: discover via its arxiv MCP
layer, download the PDF into the local library so the source is a checkable file, and index via
`rag-toolkit` when a claim needs pulling out of the PDF text with a citation. This is a smaller task
than S8's — the deliverable here is measurement, not citation.

As a capable agent, plan to: (1) read `resources/s9-session.md` §§3-13 for the mechanics the harness
has to demonstrate, and the transcript for the instructor's framing of the bug-hunting point,
(2) decide the model/tokenizer the harness runs on and whether it uses the real `V = 131,072`
configuration or a small proxy with the real numbers computed alongside, (3) build the notebook so
each required number is printed by a cell that ran, (4) run it end-to-end top to bottom in a fresh
runtime, (5) write the README with the nine numbers and the explanations, (6) push to GitHub and
verify the link in an incognito window. TODO.md tracks progress on these steps.

## References
Refer CLAUDE.md if it exists
