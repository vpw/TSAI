# S7 TODO — Embeddings and Model Internals

Solving **Problem 4**: a real Fourier alternative to the byte-level Kronecker embedding.
See `analysis/FIVE-PROBLEMS.md` for why this problem over the other four.

`S7-assignment.md` = the task, `resources/s7-session.md` = lesson,
`resources/s7-transcript.md` = live-class transcript, `resources/s7-widget-data.md` = widget data.

- [x] Session scaffolding: `CLAUDE.md`, `AGENTS.md`, session + transcript resources.
- [x] Widget extraction — all 13 widgets in `resources/s7-widget-data.md`, including the
      verbatim `KroneckerEmbedding` module source (widget 8) and the `embedding_policy_id`
      JSON (widget 13).
- [x] Choose the problem — Problem 4, with follow-ups planned in the order 5 → 1 → 2.
      Recorded with citations in `analysis/FIVE-PROBLEMS.md`.
- [x] Codec modules (`kv2/codecs.py`): `KroneckerCodec` (faithful baseline),
      `NaiveSumCodec` (the assignment's literal "just add them" — the negative control),
      `FourierCodec` (phase binding + exact unbinding decoder).
- [x] Invariant tests — 23 passing (`tests/test_codecs.py`).
- [x] Ablation harness (`train_arm.py`) reusing S5's Llama-shaped decoder; the embedding
      module is the only variable across arms.
- [~] Static proofs on the real 68k vocab (`proofs/run_static_proofs.py`) — P1/P3/P4/P5/P6
      done, P2's effective-rank measurement still running.
      **Found and corrected a real error here:** the first version claimed "6,193 dead
      projection rows". The shipped codec z-normalises, so never-activated cells are not
      zero and nothing is literally dead. The harness reporting `dead rows 0 of 8,192`
      caught it. Restated as (a) unreachable grid cells before normalisation and
      (b) effective rank of the finished code, which is the normalisation-proof measure.
- [~] Trained ablation on the T4 — 6 arms x 40M tokens, running.
      (`kronecker_32`, `fourier_2048`, `fourier_8192`, `dense`, `naive_2048`, `kronecker_48`)
- [ ] `RESULTS.md` from `scripts/make_report.py`, then the README.
- [ ] **Stop the T4 and verify it reaches `stopped`** — it bills by the second.
- [ ] Push to GitHub, confirm public in an incognito window, share the link.

## Follow-up submissions (resubmission is allowed)

1. **Problem 5 (reversibility)** — the unbinding decoder already inverts the code exactly and
   survives noise as large as the signal; the open part is replacing the output head and
   comparing against the softmax-bottleneck baselines.
2. **Problem 1 (math structure)** — a CRT/phase block reuses the same machinery; the honest
   result is that no single space is homomorphic for both `+` and `x`, so blocks get appended.
3. **Problem 2 (image/audio)** — the codec needs only a discrete alphabet and a position
   index, so quantised patches substitute for bytes unchanged.
