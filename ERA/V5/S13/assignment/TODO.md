# S13 TODO — Reversibility (Distributed Training II)

## ▶ STATUS (2026-09-25): notebook built and gated; full run executing on the EC2 T4.

**Due Sat 2026-09-26 07:00** (1000 pts, resubmission allowed, one GitHub README link field).

## ▶ SCAFFOLDING

- [x] **Session verified** (2026-09-24). The lesson heading reads "Session 13: Distributed
      Training II, Model and Pipeline Parallel", which matches this folder.
- [x] **Assignment captured** → `S13-assignment.md`: the brief verbatim, the Axiom block, and the
      instructor's framing from the transcript. The Rubric tab adds nothing.
- [x] **Lesson captured** → `resources/s13-session.md`, all 18 sections, in two passes
      because the page is over the 50K cap. §16 has an Addendum with the paper's exact variant
      equations and Tables 3–4.
- [x] **Transcript captured** → `resources/s13-transcript.md`, 97KB, class of 2026-09-19,
      via `curl .../export?format=txt`, the same route as S10–S12.
- [x] **Paper verified**: arXiv **2512.02056** (Gal, Eliasof, Turek, Ascher, Treister,
      Haber; submitted 2025-11-27), read from the PDF.
- [x] **Widget extraction judged unnecessary** (see CLAUDE.md Conventions).
- [x] **Working set written**: `CLAUDE.md`, `AGENTS.md`, this file.
- [x] **Branch cut**: `s13-reversibility`, from `s12-distributed-zero`.
- [x] Copied `tools/` (unchanged), `.gitignore` (`data/` ignored), `requirements.txt` from S12;
      `.venv` via uv (torch 2.14 CPU) for smoke tests.

## ▶ DECISIONS — settled 2026-09-25 (user: "go with your recommendations, EC2 T4")

- [x] **D1. GPU lane.** EC2 T4 (`era-v5-gpu-run` Lane A: scriptable, unattended, ~$1–2 total)
      or Colab (Lane B: the brief's suggestion, but the user drives the browser). The T4 has no
      bf16, so everything runs in fp16 + GradScaler with the residual stream kept in fp32.
- [x] **D2. Data and tokenizer.** Recommended: TinyStories with a BPE trained on it (vocab
      8,192), so ~20M params are mostly transformer rather than embedding table. The GPT-2 50K
      vocab at d=256 would be 12.9M of embedding alone. Alternative: a FineWeb-Edu slice
      (more "LLM", noisier curves at this size).
- [x] **D3. Model shape.** Recommended: **deep and narrow**, d=256, L=24, 4 heads, T=512, tied
      embeddings, ≈21.0M params. Reversibility saves memory in proportion to depth, so a deep
      model shows the effect. Alternative: d=384, L=10 (≈20.8M), more conventional, with a
      smaller reversible advantage.
- [x] **D4. What "Euler" means.** The paper has no plain "Euler" variant. It says midpoint(a)
      "behaves in expectation just like forward Euler", and its Hamiltonian scheme is
      "symplectic Euler". Plan: run **midpoint** (eq. 2.4), **midpoint(a)** with Lightning LM's
      h = 0.25, a = 0.5 (eq. 3.6), and the **Hamiltonian / symplectic-Euler** two-stream
      scheme (eq. 2.8–2.9) as "Euler". Add leapfrog (eq. 2.6) as a cheap short-run extra if time
      allows. State the interpretation in the README.
- [x] **D5. Variant bake-off budget.** Full 50M tokens for every variant (cleanest, ~20–25 min
      each on T4), or a shorter screen (e.g. 10M tokens) with only the winner run to 50M.
      **Settled: full 50M for all four variants, leapfrog included.** The benchmark showed
      ~23 min per reversible run, so the full set fits comfortably before the deadline.
- [x] **D6. GPU access** (2026-09-25). This box had no key authorized on `vardhan-gpu-1`
      (the GPU scripts assume `~/.ssh/id_rsa`, which exists only on the other machine). With
      the user's OK, EC2 Instance Connect pushed a temporary key once, and this box's
      `~/.ssh/id_ed25519.pub` was then appended to ubuntu's `authorized_keys`. Use
      `SSH_KEY=~/.ssh/id_ed25519` with `gpu_ssh.sh`.

## ▶ BUILD

- [x] Data prep: download, train the tokenizer, pre-tokenize 50M train tokens plus a held-out
      val split into a uint16 memmap. Cache it so every arm reads identical data in identical
      order.
- [x] Model: nanoGPT-style pre-LN block f(p) = Attn(LN₁ p) + MLP(LN₂(p + Attn(LN₁ p))), with a
      pluggable residual rule: `standard | midpoint | midpoint_a | leapfrog | hamiltonian`.
- [x] Reversible stack as a custom `torch.autograd.Function`. The forward runs under
      `no_grad` and saves only the two boundary states. The backward walks down the stack,
      rebuilds p_{ℓ−1} by the inverse rule, reruns f with grad, and accumulates. Handle the
      first step (p₋₁ = p₀) for the two-step rules.
- [ ] **Correctness gates**, as notebook asserts, run on CPU/GPU before any long run:
  - [ ] Gradient check: for each variant, the reversible backward's grads equal plain
        autograd through the same architecture (fp32, tight tolerance).
  - [ ] Reconstruction error per layer: rebuilt vs forward states, fp32 vs fp16-autocast.
  - [ ] Memory vs depth (a few steps at L = 4…48): baseline linear, reversible flat. This
        reproduces the paper's Fig. 3 shape.
- [x] Prototype checks (2026-09-25, CPU, fp64/fp32, d=64, L=24): the reversible backward
      matches autograd to 1e-16 (midpoint, hamiltonian), 4e-15 (leapfrog). **midpoint(a) at
      a=0.5 is the exception:** 3e-9 even in fp64, and its fp32 rebuilt input is off by
      **15%**. Its inverse divides by a, doubling error per layer, as predicted from the paper's
      own |a|=1 stability condition. Lightning LM's h=0.25, a=0.5 is backward-unstable at depth.
- [x] T4 benchmark (B=32): standard 59.9K tok/s at 5.45 GiB; reversible 37.1K tok/s (+61%)
      at 2.05 GiB. The first max-batch search OOM'd because one step doesn't allocate Adam
      state. The search now runs 2 steps, and the notebook sets `expandable_segments`.
- [x] Smoke-tested end to end on CPU and on the T4 (`S13_SMOKE=1`).
- [ ] Max-batch search (OOM caught, cache emptied, binary search) for baseline and reversible.
      Compare the ratio with the paper's ~10×.

## ▶ RUNS (50M tokens each unless D5 says otherwise)

- [ ] R1. Baseline at fixed batch B (the largest power-of-two-ish batch that fits the baseline).
- [ ] R2. Reversible variants at the same B: midpoint, midpoint(a), "Euler" (Hamiltonian).
      Choose the winner from the loss trajectory and stability.
- [ ] R3. Winner at max batch B_max. Scale the LR from the baseline's (state the rule, with a
      quick probe if needed, per S11's "tune both sides").
- [ ] For every run: final train/val loss, steady-state tokens/s, `max_memory_allocated` and
      `max_memory_reserved`, wall-clock, and $ cost at the instance's hourly rate.

## ▶ WRITE-UP AND SHIP

- [ ] README via `README.tmpl.md` → `build_readme.py`: the results table, loss curves, memory
      vs depth, the variant verdict with reasoning, the confound in the max-batch loss, and how
      the ~10× and 30–50% claims held up.
- [ ] Commit the notebooks, `results.json`, `logs/nbexec.log`, `assets/*.png`.
- [ ] The user creates `github.com/vpw/era-v5-s13`. Subtree split and push, verify
      anonymously, submit in Axiom before 2026-09-26 07:00.
