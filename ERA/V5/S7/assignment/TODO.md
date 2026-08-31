# S7 TODO — Embeddings and Model Internals

Solving **Problem 4**: a real Fourier alternative to the byte-level Kronecker embedding.
See `analysis/FIVE-PROBLEMS.md` for why this problem over the other four.

`S7-assignment.md` = the task, `resources/s7-session.md` = lesson,
`resources/s7-transcript.md` = live-class transcript, `resources/s7-widget-data.md` = widget data.

## Round 1 — submitted

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
- [x] Pushed and submitted — `github.com/vpw/era-v5-s7` (standalone public repo, same model as
      S6; `origin` here is still `github.com/vpw/TSAI`, branch `s7-embeddings`).

### The result, in one line

`fourier_2048` reaches 1.2999 macro bits/byte against `kronecker_32`'s 1.3015 — parity, at a
quarter of the input-path parameters. Both codes span ~1,550 effective directions; the grid
spends 8,192 coordinates to do it, the phase code spends 2,048.

---

## Round 2 — instructor feedback, resubmission plan

**Verdict received:** "Outstanding investigation: the naïve-wave refutation, real-tokenizer
collision count, rank audit, decoding/noise curves and parameter-aware six-arm LM study set a
very high standard." Four specific fixes requested. Resubmission is allowed; this is a
correction pass on the existing repo, **not** a new problem.

**Decision: Problem 5 (reversibility) is deferred until this pass lands.** A corrected S7 is
worth more than a new problem stacked on an un-errorbarred headline. S8 is due **2026-08-22**,
so keep this bounded — roughly 5 T4-hours plus doc edits.

Three of the four points were already in this file's "loose ends"; the feedback promotes them
from nice-to-have to load-bearing.

### 4a. Pin the tokenizer artifact  — *cheapest, do first, no GPU*

Every number in the repo (the 22/68,096 collision count, every bpb figure) depends on the
sarvam1 68,096-token tokenizer, which currently lives **outside** this repo in S4
(`ERA/V5/S4/assignment/models/tokenizer-sarvam1.json` — verify the path before wiring it up).
A grader cloning `era-v5-s7` in incognito cannot reproduce anything.

- [ ] Decide: commit the JSON into the repo, or add a fetch script that pins URL + SHA-256.
      (S5 deliberately *untracked* a duplicated tokenizer in `8547ef5` — check that reasoning
      before re-committing a copy.)
- [ ] Whichever route: re-run `proofs/run_static_proofs.py` from a **clean clone** and confirm
      the collision count and rank numbers come out identical.
- [ ] Record the SHA-256 in the README reproduction section (§6).

### 4b. Seed sweep on the main LM comparison — *~5 T4-hours*

The headline gap is 0.12% (`fourier_2048` 1.2999 vs `kronecker_32` 1.3015) from a **single**
seed — inside run-to-run noise, so it cannot support the word "parity" as written. The 3.43%
matched-parameter gap (`kronecker_32` vs `fourier_8192`) is large enough to likely survive, but
should get the same treatment.

- [ ] 3 seeds x `fourier_2048` and `kronecker_32` (2 already exist at seed 20260810, so 4 new
      runs, ~30 min each). Reuse `scripts/run_ablation.sh`; T4 is sm_75 → fp16 not bf16.
- [ ] If budget allows, extend to `fourier_8192` so the matched-parameter claim gets a bar too.
- [ ] Restate the headline as mean ± spread in `RESULTS.md`, `README.md` §4 and
      `site/index.html`. If the gap does not clear the spread, **say so** — "indistinguishable
      at this scale" is a fine result and matches how the three failed predictions were handled.
- [ ] Remember to stop the T4 and verify (`scripts/aws_gpu.sh`; awscli lives in S5's venv).

### 4c. Tighten the "no length cap" claim — *doc edit, no compute*

Kronecker performs a **hard crop** at byte 32 — information is destroyed. The phase code crops
nothing, but a fixed 2,048-dim vector has **finite phase capacity**, and our own §4 curve
("a slope rather than a cliff") already measures the blur. So "no length cap" overclaims.

- [ ] Rewrite as "no hard crop, but finite phase capacity — accuracy degrades smoothly with
      token length" across `README.md` (§3, §4, §5), `RESULTS.md` and `site/index.html`.
- [ ] Point the sentence at the capacity curve that already backs it, so the correction reads
      as precision rather than retreat.

### 4d. Explicit length handling — *small code change + one run*

Token byte-length is currently implicit: a 4-byte and a 60-byte token both emerge as one vector
with no signal distinguishing them.

- [ ] Add explicit length conditioning to `FourierCodec` — a length scalar / bucket embedding,
      or 1/sqrt(L) normalisation (the construction already carries a 1/sqrt(L) factor; check
      whether it is doing this job already before adding a second mechanism).
- [ ] Document it in README §3 regardless of whether it wins.
- [ ] Optional 7th arm if T4 time is left after 4b — it plausibly explains part of the
      long-token (>32 byte) slice, where support is only 382/149/157 positions.

### Ordering

4a → 4c (both free, land immediately) → 4b (the GPU spend) → 4d (code, folded into the same
GPU session if time allows). Then push to `github.com/vpw/era-v5-s7` and re-share the link.

**Pushing:** no `gh` CLI and no credential helper on this machine — the user pastes a PAT at the
password prompt, so *prepare* the push commands rather than running them.

---

## Follow-up submissions (after Round 2 lands)

1. **Problem 5 (reversibility)** — the unbinding decoder already inverts exactly and survives
   noise as large as the signal; the open part is replacing the output head and comparing against
   the softmax-bottleneck baselines.
2. **Problem 1 (math structure)** — a CRT/phase block reuses the same machinery; the honest
   result is that no single space is homomorphic for both `+` and `x`, so blocks get appended.
3. **Problem 2 (image/audio)** — the codec needs only a discrete alphabet and a position index,
   so quantised patches substitute for bytes unchanged.

## Loose ends worth a second pass

- `fourier_8192` being the *worst* structured arm is unexplained beyond "redundancy plus a fixed
  token budget". The 4b seed sweep may settle it.
- The **sparsity confound**: Kronecker lights <=32 of 8,192 rows per token, the phase projection
  must unmix a dense mixture. A top-k sparsified phase code at 8,192 dims would separate
  "sparse activation helps" from "the grid helps". Untested; named in README §8.
- `fourier_512` / `fourier_1024` to trace the efficiency frontier rather than sampling two points.
- The Fourier frequencies are random and unlearned. A learned frequency schedule is the obvious
  next experiment.
- Everything is a 40M-token proxy. The dense table still wins by 4%, and the seed paper's
  "Kronecker beats BPE-tied" claim was not reproduced at this scale.
