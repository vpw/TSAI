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

- [ ] **D1. Where do the five items run?** Every item needs a real model actually
      training — item 1 needs PyTorch's Adam on a real weight to hand-check, items 3-4
      need full loops with warmup/schedule logic over a few hundred steps, item 5 needs
      three separate short runs at widths 256/512/1,024. No GPU confirmed on this
      machine (carried from S9/S10: `.venv` was torch cpu-only there). Unlike S10's MFU
      item, none of these five items *need* a real accelerator's peak FLOP/s — they need
      a working optimizer and enough steps to see a trend, which a small enough model may
      get on CPU within reasonable wall-clock time. Options: (a) CPU on this machine at a
      tiny scale (fastest to start, no provisioning, but check items 3-5's step counts
      don't take unreasonably long); (b) reuse `era-v5-gpu-run`'s Lane A (existing EC2
      T4) or Lane B (Colab) if CPU proves too slow, especially for item 5's three-width
      sweep. Recommend starting on CPU and only escalating if a specific item is too slow.
- [ ] **D2. What model?** S10 settled on nanoGPT (instructor's own pointer in the
      transcript) plus its own S9 proxy transformer, run side by side. For S11, item 5's
      width sweep (256/512/1,024) needs a model whose width is a first-class,
      trivially-variable hyperparameter — check whether nanoGPT's config exposes
      `n_embd` cleanly for this, or whether reusing/adapting S9's proxy transformer
      (`V=10,000, D=256` baseline, 4 layers/heads) is simpler to resize across three
      widths. Items 1-4 don't have this constraint and can run on whichever model D1/D2
      settle on.
- [ ] **D3. Submission format** — cannot be settled until S11 appears in the Axiom
      Assignments tab with its own submission block. Don't assume S10's GitHub-README
      shape without checking; it's the likely precedent but not confirmed. Re-check
      `/assignments` periodically or when told the assignment has been posted.

## Part 1 — the five items

Not started. Each will need a cell that actually runs, per the standing "no number
without a cell behind it" convention (S9 onward).

- [ ] **1. Reproduce Adam by hand.** One weight, five gradients (can reuse the lesson's
      own `0.50, 0.40, 0.60, 0.45, 0.55` at η=0.001 as the worked check, or a fresh set) —
      compute `m, v, m̂, v̂`, step by hand, then verify against `torch.optim.Adam` to
      several decimal places.
- [ ] **2. Bias correction ablation.** Same setup, bias correction on vs off, first 20
      steps, plotted. Report the step count after which the two trajectories converge —
      the lesson's own §6 table implies this should happen quickly since β2=0.999's bias
      factor `(1-β2^t)` is still the slower-decaying one; expect the answer to be on the
      order of tens of steps, to be measured rather than assumed.
- [ ] **3. Per-layer update-to-weight ratio through warmup.** Log it from step one (per
      §14's V5 decision), identify the step where warmup stops visibly changing the
      ratio's ceiling. Compare shape against the lesson's own 19.2e-3 → 2.83e-3 figures
      (not the same model/scale, so an exact match isn't expected — a similar order-of-
      magnitude drop is).
- [ ] **4. Cosine vs WSD, 300-step budget, compared at step 200.** Same model, same init,
      same data — the only difference is the schedule. Report both losses at step 200 and
      state which checkpoint you'd actually keep, with reasoning (WSD's structural
      argument for checkpoint-ability vs any measured loss difference at this short a
      budget).
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
