# Task

This directory is part of the assignments for the ERA V5 course of The School of AI (TSAI).
Specifically this is for the eighth session (S8).

The `S8-assignment.md` file lists the exercise in full — please refer to that for the details.
Like S7, this is not a data/pipeline deliverable graded on a numeric rubric table. It's a
build-and-defend web app: cover every attention mechanism from the session, arranged
**chronologically by actual launch date** (not teaching order, not grouped by family), each with
honest pros/cons (what it buys, what it costs, when you'd choose it).

# Details

The session covers modern attention: standard scaled dot-product attention (Q/K/V projections, the
causal mask, why `softmax(-∞)=0` lets training run in parallel while generation stays
left-to-right), then the two costs attention creates as context grows — quadratic *compute*
(`~T²` comparisons) and linear, per-user, unshared *KV-cache memory* (`~T`) — and the mechanisms
invented to attack one or both: removing softmax to get a fixed-size linear-attention state (which
then needs the **delta rule** to *correct* rather than just accumulate into that state); sparse/
top-k attention (keeps softmax, reads fewer keys, but needs a cheap proposal step or scoring
everyone first defeats the point); **RoPE** (encodes relative position as rotation, so query-key
scores depend only on `i−j`) and **DroPE** (V4's reported 8K→256K, 32× extension applied *before*
annealing — being mathematically definable at a length isn't the same as the model being
competent there); the full KV-cache formula
(`2 × layers × kv_heads × head_dim × T × batch × bytes`, worked to 6.44GB/user, 51.54GB/8 users);
**GQA/MQA** (shares KV heads, lowers the cache growth *slope*, doesn't stop the *growth*); sequence
compression + a cheap low-rank indexer for top-k block selection (DeepSeek-style); depth
**schedules** that mix fixed-state and sparse-attention layers across a model (V4's `DDDGDDDG`
motif); and a **Memory Stream** for state that survives chunk boundaries. Closes on "long context
is a system, not a number" (six conditions must all hold together) and a two-roads framing (extend
a short-trained model vs. build/train long natively).

Full writeup: `resources/s8-session.md`. Live-class transcript: `resources/s8-transcript.md`.

**What the assignment actually asks for:**

1. A web app (Netlify/Vercel/equivalent) that visually explains attention, starting from standard
   scaled dot-product attention before anything that modifies it.
2. Every mechanism placed **in chronological order by real launch date** — this is the part the
   instructor says he cares about most, because the ordering itself is meant to reveal how the
   field's priorities shifted over time (exactness → memory → length → memory again → sparsity/
   compression returning). This isn't decoration; it's graded directly via a 1000-point bonus
   question asking what the timeline shows that a list wouldn't.
3. Honest pros/cons per mechanism — what it buys, what it gives up, when you'd actually pick it.
   A mechanism can be right for a 2K chatbot and wrong for a 1M-token agent without being "bad."
4. Minimum coverage list: standard attention, absolute learned positions, sinusoidal, RoPE, ALiBi,
   MQA, GQA, sliding window, attention sinks, NTK-aware scaling, YaRN, linear attention, the delta
   rule and Gated DeltaNet, MLA, sparse and top-k attention, DeepSeek's compressed sparse
   attention, and DroPE. A well-sourced mechanism *not* covered in class earns credit if added to
   the same standard (date, motivation, mechanism, advantage, cost, timeline placement).
5. **Every date must be checked against a primary source** (the actual paper or release), not
   trusted from an agent's confident recall — the instructor explicitly says he got a mechanism's
   date wrong in Session 7 and had to correct it live at the top of this class. This is the
   highest-risk part of the assignment and should be treated as real research, not filler.
6. Submit: live link (incognito-accessible), GitHub repo, and a README listing chronology sources.

**Use the `arxiv-library` skill for step 5's research**, assuming every mechanism's primary source
is on arXiv: (1) search/discover the paper via the skill's arxiv MCP layer instead of trusting a
remembered title or date, (2) download the PDF into the local library so the source is a checkable
file, not a claim, (3) index it via the skill's `rag-toolkit` layer when a specific date or claim
needs pulling straight out of the PDF text with a citation — indexing is there if needed, not a
required step for every one of the ~17 mechanisms.

As a capable agent, plan to: (1) read `resources/s8-session.md` and the transcript for the exact
mechanics and the instructor's framing of the chronology requirement, (2) research and verify a
launch date + primary source for every mechanism in the minimum list using `arxiv-library` (this is
the part most likely to need real citation-checking, not assumption), (3) design the app's
structure — a timeline is the organizing principle, not a mechanism-by-mechanism card list — and
decide what's interactive vs. static (the assignment explicitly says a clear static page beats a
broken clever one), (4) build and deploy it, (5) write the README with sources, and (6) push/share
the GitHub + live links. TODO.md tracks progress on these steps.

## References
Refer CLAUDE.md if it exists
