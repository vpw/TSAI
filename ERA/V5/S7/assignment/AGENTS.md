# Task
This directory is a part of the assignments for the ERA V5 course of The school of AI (TSAI).
Specifically this is for the seventh session (S7).

The S7-assignment.md file lists the exercise in full — please refer to that for the details.
Unlike every prior session, this is not a data/pipeline deliverable graded on a numeric rubric.
It's an open research problem: the instructor poses five separate, independent embedding-design
problems (extending Kronecker factorization, which this session's lesson introduces) and asks you
to pick **one**, build a small transformer to prove your solution works, and write it up.

# Details
The session covers embeddings and model internals: why an embedding lookup is a gather (not a
matmul) and a scatter-add on the backward pass, why Zipfian token frequency makes different rows
of the table train at wildly different effective rates, the real parameter/memory cost of the
token-facing matrices at V5's scale (over a billion params, ~17GB of AdamW training state per
matrix), weight tying, low-rank factorized embeddings, and the centerpiece: **Kronecker
factorization**, a byte-level embedding scheme whose parameter count doesn't depend on vocabulary
size at all, at the cost of silently colliding tokens that share their first 32 UTF-8 bytes (which
hits Indic scripts, with their 3-bytes-per-character encoding and multi-codepoint conjuncts, much
harder than English). The session closes on positional encoding's absolute-table wall (can't
extrapolate past the trained max length), handing off to Session 8's rotary/ALiBi treatment.
Full writeup: resources/s7-session.md. Live-class transcript: resources/s7-transcript.md.

The five problems (pick exactly one, don't mix):
1. Embeddings that encode mathematical structure, so arithmetic on embeddings mirrors arithmetic
   on the numbers they represent (e.g. embed(9) + embed(9) ≈ embed(18)).
2. Extend Kronecker to represent images and audio, not just text tokens.
3. Remove the fixed 32-byte-position window so tokens longer than 32 bytes aren't silently
   cropped/collided — make the window dynamic.
4. A genuine Fourier-wave alternative to Kronecker: represent each character as a wave and sum
   waves to form a word's embedding.
5. Make Kronecker invertible (embedding -> unique token, not just token -> embedding), which would
   let a model drop its output head and scale to ~1M vocabulary tokens.

The instructor is explicit that he's evaluating these as candidate ideas for a "Kronecker
Embedding V2" paper he's planning to write — treat the choice and the proof seriously, not as a
box-ticking exercise. He's also explicit that the agent (me) should be trusted to write and train
a small transformer to prove whichever idea is chosen — toy scale is fine, proving the mechanism
is the point.

As a capable agent, plan to: (1) read resources/s7-session.md and the transcript for the exact
mechanics of Kronecker and its failure modes, (2) help pick which of the 5 problems to pursue
(a judgment call — surface tradeoffs, but this may need the user's steer), (3) design and
implement a small proof-of-concept transformer + training run that demonstrates the chosen idea
works, (4) write a tight README (optionally a small demo webapp with graphs/animations), and
(5) push/share the GitHub link. TODO.md tracks progress on these steps.

## References
Refer CLAUDE.md if it exists
