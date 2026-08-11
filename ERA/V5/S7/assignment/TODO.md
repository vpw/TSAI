# S7 TODO — Embeddings and Model Internals

Solving **Problem 4**: a real Fourier alternative to the byte-level Kronecker embedding.
See `analysis/FIVE-PROBLEMS.md` for why this problem over the other four.

`S7-assignment.md` = the task, `resources/s7-session.md` = lesson,
`resources/s7-transcript.md` = live-class transcript, `resources/s7-widget-data.md` = widget data.

- [x] Session scaffolding: `CLAUDE.md`, `AGENTS.md`, session + transcript resources.
- [x] Widget extraction — all 13 widgets, including the verbatim `KroneckerEmbedding` module
      source (widget 8) and the `embedding_policy_id` JSON (widget 13).
- [x] Choose the problem — Problem 4, follow-ups planned 5 → 1 → 2, recorded with citations.
- [x] Codec modules (`kv2/codecs.py`): `KroneckerCodec`, `NaiveSumCodec` (the negative control),
      `FourierCodec` with an exact unbinding decoder.
- [x] Invariant tests — 23 passing (`tests/test_codecs.py`).
- [x] Static proofs on the real 68k vocab (`proofs/run_static_proofs.py`), ~23 min on CPU.
      **Found and corrected a real error:** the first version claimed "6,193 dead projection
      rows". The shipped codec z-normalises, so never-activated cells are not zero and nothing
      is literally dead — the harness reporting `dead rows 0 of 8,192` caught it. Restated as
      unreachable grid cells plus effective rank.
- [x] Trained ablation — 6 arms x 40M tokens on the T4, ~3 GPU-hours.
      `kronecker_48` OOMs at `--micro-seqs 16` (1.67 GB code table); reran at 8.
- [x] `RESULTS.md`, `README.md`, and `site/index.html` (visual summary).
- [x] **T4 stopped and verified.**
- [ ] Push and share the public link — needs a decision on destination (see below) and the
      user's PAT. Then confirm the link opens in an incognito window.

## The result, in one line

`fourier_2048` reaches 1.2999 macro bits/byte against `kronecker_32`'s 1.3015 — parity, at a
quarter of the input-path parameters. Both codes span ~1,550 effective directions; the grid
spends 8,192 coordinates to do it, the phase code spends 2,048.

## Push destination — needs a decision

S6 was submitted as a **standalone public repo** (`github.com/vpw/era-v5-s6`) rather than a
branch on `github.com/vpw/TSAI`, because the submission form requires a link that opens in an
incognito window. Same choice applies here. Note `s7-embeddings` currently sits on top of
`s3-data-plan`, so it carries commit `fc33133` as well as the four S7 commits; a standalone repo
sidesteps that entirely.

There is no `gh` CLI and no credential helper on this machine — the user authenticates by pasting
a PAT at the password prompt, so push commands get handed over rather than run.

## Follow-up submissions (resubmission is allowed)

1. **Problem 5 (reversibility)** — the unbinding decoder already inverts exactly and survives
   noise as large as the signal; the open part is replacing the output head and comparing against
   the softmax-bottleneck baselines.
2. **Problem 1 (math structure)** — a CRT/phase block reuses the same machinery; the honest
   result is that no single space is homomorphic for both `+` and `x`, so blocks get appended.
3. **Problem 2 (image/audio)** — the codec needs only a discrete alphabet and a position index,
   so quantised patches substitute for bytes unchanged.

## Loose ends worth a second pass

- `fourier_8192` being the *worst* structured arm is unexplained beyond "redundancy plus a fixed
  token budget". A seed sweep or a longer run would settle it.
- The Fourier frequencies are random and unlearned. A learned frequency schedule is the obvious
  next experiment.
- Everything is a 40M-token proxy. The dense table still wins by 4%, and the seed paper's
  "Kronecker beats BPE-tied" claim was not reproduced at this scale.
