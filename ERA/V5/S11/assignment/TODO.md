# S11 TODO — Optimizers and Learning-Rate Schedules

## ▶ STATUS (2026-09-07): scaffolded, not yet started

No due date yet — **S11 is not in the Axiom Assignments tab** as of this check. Section
15 of the lesson (`S11-assignment.md`) is the whole spec: five items, all needing a real
training run, no pure-arithmetic item like S10's bit-format one.

- [x] **Session verified** (2026-09-07). The lesson page's own heading reads "Session 11:
      Optimizers and Learning-Rate Schedules" — matches this folder, no drift.
- [x] **Assignment captured** → `S11-assignment.md`, Section 15 verbatim (5 items). No
      separate submission block exists yet — checked the course's `/assignments` tab,
      which lists Session 1 through Session 10 only (S10 itself shown "Late", awaiting
      review). Re-check once S11 is posted there.
- [x] **Lesson captured** → `resources/s11-session.md`, all 15 sections verbatim, via
      `get_page_text` on the lesson page (single call, page was well under the ~50KB cap).
- [x] **Transcript captured** (2026-09-07) → `resources/s11-transcript.md`, 123KB, 489
      lines, via `curl .../export?format=txt` on the Google Doc the user supplied
      directly (`16nNdlSP5XRprvgbFSo50RZmi_1bcTGoRWki5jNWsJyc`) — no browser download
      needed, doc is link-shared. Header dates it **2026/09/05 06:43 IST**, opens on
      *"today's session is on optimizers and learning rate schedules"*, ends cleanly at
      *"Meeting ended after 02:23:59."*
- [x] **Widget extraction judged unnecessary** (2026-09-07). Every widget in this lesson
      (2D gradient descent, momentum, per-parameter LR, Adam five-step, L2-vs-decoupled
      decay, optimizer memory, warmup ratio, schedule comparison, LR transfer) animates a
      formula or table already given in full in the prose, and all five assignment items
      require the student's own model/training run rather than a widget's live state.
- [ ] **Branch cut** — not yet done. Following the pattern (S10 branched from S9's
      branch before its first commit), cut `s11-optimizers` (or similar) from
      `s10-training-loop` before the first S11 commit specific to this session's work.
- [x] **Working set written** — `CLAUDE.md`, `AGENTS.md`, this file (2026-09-07).

## ▶ OPEN DECISIONS — settle with the user before building

- [x] **D1. Where do the five items run? → CPU on this machine** (decided 2026-09-07).
      Unlike S10's MFU item, none of these five items need a real accelerator's measured
      peak FLOP/s — they're optimizer/schedule diagnostics, not throughput measurements.
      Even item 5's widest config (n_embd=1,024, n_layer=4) is only ~50M params
      (attention+MLP ≈ 12·d² per layer), comfortably CPU-trainable for the few-hundred-
      step budgets these items need. Escalate to `era-v5-gpu-run` (Lane A EC2 T4 or Lane
      B Colab) only if a specific run proves too slow in practice — most likely candidate
      would be item 5's three-width sweep if the LR grid is large.
- [x] **D2. What model? → S10's nanoGPT** (decided 2026-09-07). The transcript settles
      this directly (`resources/s11-transcript.md` line 463): *"train the same model
      twice for 300 steps. Take basically a small model, the model that you have taken
      last assign[ment]."* That's an explicit pointer to reuse, not pick fresh. S10's
      nanoGPT (`../../S10/assignment/notebook_src_nanogpt.py`, the `GPTConfig`/`GPT`
      classes — a from-scratch char-level GPT, not a copy of Karpathy's repo) already
      exposes `n_embd` as a clean width knob: baseline `GPTConfig(vocab_size=65,
      n_embd=128, n_layer=4, n_head=4, seq_len=128, batch_size=8)` on tinyshakespeare-char.
      `n_head=4` divides 256/512/1,024 evenly (head_dim 64/128/256), so item 5's sweep is
      just varying `n_embd` in `{256, 512, 1024}` with everything else held fixed. Items
      1-4 use the baseline `n_embd=128` config for continuity with S10's own numbers.
      S10's proxy transformer is not reused here — the instructor's line names one model,
      and nanoGPT was the instructor's own explicit pick in S10's transcript too.
- [ ] **D3. Submission format** — cannot be settled until S11 appears in the Axiom
      Assignments tab with its own submission block. Don't assume S10's GitHub-README
      shape without checking; it's the likely precedent but not confirmed. Re-check
      `/assignments` periodically or when told the assignment has been posted.

## Part 1 — the five items

Not started. Each will need a cell that actually runs, per the standing "no number
without a cell behind it" convention (S9 onward).

- [x] **1. Reproduce Adam by hand** (2026-09-07). Reused the lesson's own worked example
      (`w0=1.0`, grads `0.50, 0.40, 0.60, 0.45, 0.55`, η=0.001) in `notebook_src.py`.
      Three-way check: by-hand `m, v, m̂, v̂, step, w` vs the lesson's own printed table
      (max abs diff **4.6e-7**, consistent with the lesson rounding to 6 decimals) vs
      `torch.optim.Adam` in float64 (max abs diff **0.0**, exact agreement — plain Adam,
      no weight decay, matches Section 6's formula precisely). Results in
      `results.json["item1"]`, full trace in `logs/run.log`.
- [x] **2. Bias correction ablation** (2026-09-07). Extended item 1's gradient sequence
      to 20 steps (same first 5, seeded noisy continuation around the same mean) and ran
      Adam by hand twice, correction on vs off. **Finding — the "tens of steps" guess
      above was wrong**, and the corrected math shows why: the closed-form step-size
      ratio is `sqrt(1-β2^t)/(1-β1^t)`, independent of the gradients, and it only enters
      ±5% of 1 at **t = 2,327** — driven entirely by β2=0.999's slow decay (β1=0.9's
      factor is already ≈1 within ~50 steps). At step 20 the ratio is still **0.16**
      (uncorrected steps ~6x larger than corrected), so `w` has drifted to 0.980
      (corrected) vs 0.881 (uncorrected) — a real, unclosed gap, not a rounding
      difference. This is a disagreement-is-a-finding result: the difference does *not*
      stop mattering within the 20 plotted steps, and the honest answer to "how many
      steps" is ~2,300, on the same O(1/(1-β2))=1,000-step order the lesson names for
      β2's memory span. Plot: `assets/item2_bias_correction.png` (trajectories +
      closed-form ratio curve with the crossing point marked). Results in
      `results.json["item2"]`.
- [x] **3. Per-layer update-to-weight ratio through warmup** (2026-09-07). First item to
      use the real nanoGPT model (D2: `n_embd=128, n_layer=4, n_head=4`, tinyshakespeare,
      813K params). Trained two 300-step AdamW runs (peak η=3e-4, warmup=60 steps vs
      none), logging `||update||/||weight||` per weight-matrix tensor from step 1.
      **Metric fix during development:** initially tracked all 36 named tensors
      including LayerNorm biases, which init at exactly 0 — dividing by their ~0 norm
      produced ratios up to ~1e9 even though training itself was healthy (loss 4.2→2.5,
      no NaN). Fixed by restricting the ratio to weight matrices only (`dim>=2`, 18
      tensors) — the same tensors §7 already excludes from decay, for the same reason
      (norm scales/biases aren't "how large is this weight" in a meaningful sense).
      **Finding:** no-warmup's ratio peaks immediately at step 1 (1.51e-2) — exactly
      §9's mechanism, a freshly-initialized model's gradients all pointing the same
      wrong way — then decays. With-warmup's ratio instead *rises* to its own peak
      (9.7e-3) right as the ramp completes (~step 59), then both curves converge into
      the same noisy steady-state band (~4-6e-3) by **step 90**. Warmup doesn't avoid a
      large step, it relocates it to when gradients have decorrelated somewhat, and
      caps its size (1.5e-2 → 0.97e-2, a smaller 1.6x reduction than the lesson's own
      6.8x, plausibly a scale effect — worth noting honestly rather than forcing a
      match). Per-layer breakdown shows `wte.weight` (the embedding) staying elevated
      longest, consistent with §5's point about embeddings having uneven per-parameter
      gradient frequency. Plot: `assets/item3_warmup_ratio.png`. Results in
      `results.json["item3"]` (full per-layer ratio log included, not just the 5
      plotted layers).
- [x] **4. Cosine vs WSD, 300-step budget, compared at step 200** (2026-09-07). Same
      init and same batch sequence for both (only the schedule differs). Interpreted
      "stop at 200" per §10's own framing: cosine pre-commits its decay curve to the
      full 300 steps, so stopping at 200 catches it mid-decay (LR=1.22e-4, vs its floor
      of 1.5e-5 which it wouldn't reach until step 299); WSD never pre-commits — it
      holds flat until told to stop, so stopping it at 200 makes steps 180-200 the decay
      window, landing exactly on the floor (1.68e-5) right at 200. **Result: WSD wins**
      — mean loss over the last 10 steps 2.597 (WSD) vs 2.617 (cosine), final-step 2.608
      vs 2.621. The two loss curves track almost identically for the first ~180 steps
      (same batches, similar LR) and only separate in WSD's steep final decay window —
      a modest but real gap, matching the structural argument: WSD's decay is aimed at
      the actual stopping point, cosine's isn't. Plot: `assets/item4_cosine_vs_wsd.png`.
      Results in `results.json["item4"]`.
- [ ] **5. Width sweep at 256/512/1,024 + muP-style extrapolation.** Three short LR
      sweeps, one per width, loss-vs-LR curves, mark the minimum of each. Compare the
      three minima's positions against the lesson's own width→η table (§12: 3.0e-3 /
      1.5e-3 / 7.5e-4 at these three widths) as a sanity check, then state a value for
      width 4,096 (lesson's own table says ~1.9e-4) and an honest confidence level given
      this is an extrapolation past the measured widths, not a fourth data point.

## Ship

- [ ] **S0. Scaffold notebook(s), tooling, template.** Copy `tools/` from
      `../../S10/assignment/tools/` (already extended for multi-result-file reads).
      Decide one combined `notebook_src.py` vs per-item sources based on how D1/D2 land.
- [ ] **S1. Run everything top to bottom** in a fresh runtime per D1's chosen environment.
- [ ] **S2. Write the README/write-up** once a submission format is confirmed (D3),
      answering all five items with numbers traced to cells that ran.
- [ ] **S3. Push and verify**, format and destination TBD pending D3.
- [ ] **S4. Submit**, once S11 has a graded entry in the Axiom Assignments tab.

## Standing conventions (carried from S6-S10, don't re-decide)

**Build pipeline — the notebook is generated, not hand-edited.**
`tools/py2nb.py` (writes `.ipynb` from `# %%` source) → `tools/run_nb.py` (executes via
nbclient, writes `results*.json`) → `tools/dump_log.py` → `tools/build_readme.py`
(template + results → README, non-zero exit on any unresolved placeholder — this is
what makes "no number without a cell behind it" mechanical). Copy from
`../../S10/assignment/tools/`, don't rewrite.

**Push destination — subtree split, not a nested `.git`** (settled 2026-08-14, reused
every session since):

```
git subtree split --prefix=ERA/V5/S11/assignment -b s11-standalone
git push https://github.com/vpw/era-v5-s11.git s11-standalone:main
```

(Repo name is a guess following the `era-v5-s<n>` pattern of `era-v5-s10` etc. — confirm
or create when actually pushing.) Keep `CLAUDE.md` / `AGENTS.md` / `TODO.md` in the
split (user-confirmed at S7). No `gh` CLI or credential helper on this machine — hand the
push command to the user rather than running it, they paste a PAT at the prompt.
