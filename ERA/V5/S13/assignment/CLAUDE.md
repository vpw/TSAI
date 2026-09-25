# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this directory is

Session 13 (S13) assignment of the ERA V5 course (The School of AI). The session topic is
**Distributed Training II: Model and Pipeline Parallel**. Session 12 divided what is *stored*
(ZeRO). This session divides what is *computed* (tensor, sequence, pipeline and context
parallelism), then ends on a technique that removes most activations altogether:
**reversibility**. The lesson has 18 sections. **The assignment is about §16–§17 only.** §§1–15
are the distributed-training context and nothing in the deliverable depends on them.

The arc in brief:
- **§0/prelude:** ZeRO never divides activations.
- **§1:** the running 30.2B / 96-layer / d=5120 model, whose 16 bytes/weight give 450.0 GiB of
  state. Activations at ~34 bytes/token/hidden-unit give 127.5 GiB for one 8,192-token sequence.
- **§3–4:** TP (Megatron column/row split, 56.4 GB traffic per sequence, so NVLink-only), with
  SP alongside.
- **§5–6:** PP, with the bubble at (p−1)/(m+p−1) and 1F1B / interleaved / zero-bubble schedules.
- **§7:** CP (ring, all-gather, Ulysses, head-tail balancing).
- **§8–10:** communication tables, topology-aware placement, Llama 3 405B's layouts.
- **§11–13:** DeepSeek V3 / V4 / V4.1-Flash, self-study.
- **§14:** the fixed order for choosing a layout.
- **§15:** V5's open questions.
- **§16:** reversible residual streams.
- **§17:** what reversibility changes. The binding term moves from activations to the training
  state.

**The mechanism the assignment is about (§16, arXiv 2512.02056, Gal et al., Nov 2025).** Treat
depth as time. Replace the residual update p_ℓ = p_{ℓ−1} + f(p_{ℓ−1}), which cannot be inverted,
with one that can, such as midpoint: p_{ℓ+1} = p_{ℓ−1} + 2h·f(p_ℓ), whose inverse is
p_{ℓ−1} = p_{ℓ+1} − 2h·f(p_ℓ). The forward pass keeps only the boundary states. The backward pass
walks down the stack, rebuilds each layer's input from its output, reruns the block with grad
enabled, and backprops through it. The consequences:
- Activation memory becomes **independent of depth**.
- Compute rises by **~30–50%** (one extra block forward per layer).
- The paper's batch gains are **~10×** on five cards (H100: 26 → 257).
- Throughput rises **+101%** at 96 layers, because of the larger batch.
- **Dropout must be 0**, since the rebuild must reproduce the forward exactly.
- The rule is **only marginally stable** in h and the blend coefficient. Lightning LM used
  h = 0.25 and a = 0.5.
- The instructor adds **no weight decay** as a second restriction.
- When memory is not the binding constraint, reversibility is simply **slower** (Lightning LM's
  2B model on 80 GB cards).

The exact update and inverse for every variant (midpoint, midpoint(a), leapfrog, Hamiltonian /
symplectic Euler) are in `resources/s13-session.md` §16 *Addendum*, taken from the PDF.

**What the assignment asks for** (`S13-assignment.md`, due **Sat 2026-09-26 07:00**, 1000 pts):
1. Train a **~20M-param LLM for 50M tokens** at a **fixed batch size** that fits.
2. Train it again **reversibly at the same batch**, testing **at least midpoint and "Euler"**,
   and report which worked, judged from the loss trajectory.
3. Train reversibly again at the **maximum batch that fits**.
4. For every run, report **final loss, tokens/s, peak memory**, and other findings. The class
   also agreed on cost with vs without reversibility.

The deliverable is a GitHub repo with a detailed README **and the ipynb notebooks**.

**How this differs from S12.** S12 was a CPU simulator, where the correctness oracle was the
lesson's own tables. **S13 is a real GPU training measurement**, closer to S9–S11. Peak memory
and tokens/s only mean something on an accelerator, and 50M tokens × 20M params is ~6e15 FLOPs
per run, days on this 2-core CPU box. The oracle discipline still applies, pointed at different
targets:
- The reversible backward must produce the **same gradients** as plain autograd through the same
  architecture.
- Reconstructed activations must match the forward's, and the error should be reported.
- The memory-vs-depth shape should reproduce the paper's Fig. 3 (baseline linear, reversible
  flat).
- The max-batch ratio should be compared with the paper's ~10×, with the gap explained rather
  than forced.

## Layout

- `S13-assignment.md`: the brief verbatim, the Axiom submission block, and the instructor's
  framing from the live class (test both Euler and midpoint, no dropout / no weight decay, the
  cost comparison).
- `resources/s13-session.md`: full lesson, all 18 sections. The page is ~58.9K chars, over
  `get_page_text`'s 50K cap, so it was read in two passes: the full page, then the page DOM
  replaced with just the §15-onward text. Tables are re-laid-out as markdown and KaTeX written
  as plain math. §16 has an **addendum with the paper's exact variant equations and its
  Tables 3–4**, taken from the PDF because the lesson states only plain midpoint.
- `resources/s13-transcript.md`: live-class transcript, **2026/09/19 06:37 IST**, 97KB, 213
  lines. It ends at *"Meeting ended after 01:53:48"*. Fetched with `curl .../export?format=txt`
  on the link-shared Google Doc (id `1-4dU02rjBmaRgiRRiNHiASKkfgoHNOsfDilyEiCG8qY`), the same
  route S10–S12 used. The assignment discussion starts around character offset 79,700.
- `notebook_src.py` → `S13.ipynb` (executed on the T4, 2026-09-25, 136.5 min) → `results.json`
  → `README.tmpl.md` + `tools/build_readme.py` → `README.md` (166 values filled). Also
  `logs/nbexec.log` (dump_log) and `logs/progress.txt` (live eval lines from the run).
  `assets/` has five plots and `tokenizer.json`. `data/` is the runtime cache (gitignored).
- **Status: run complete, README built (2026-09-25).** Leapfrog won (1.766 vs baseline
  1.829). midpoint(a) at a=0.5 failed, as predicted, with gradient cosine 0.009 after
  training. Max batch was 3.27× (logits-bound, 40 MiB/seq vs 1 MiB of boundary states) and
  brought no throughput gain on the T4. See TODO.md for the table.

## Conventions

- **Submission is a GitHub repo link** (the Axiom field is labelled "GitHub README.md"), public,
  with the notebooks committed inside it. Ship via subtree split to `github.com/vpw/era-v5-s13`.
  The user must create that repo first, because there is no `gh` CLI or token on this box. See
  the `era-v5-toolchain-environment` memory for the split and push commands and the
  anonymous-reachability check.
- **Every number in the README must come from a cell that actually ran**, through the
  `notebook_src.py → .ipynb → results.json → build_readme.py → README.md` pipeline. Copy `tools/`
  from S12 unchanged. Commit `logs/nbexec.log`, not gitignored.
- **This session needs a GPU.** Use the `era-v5-gpu-run` skill, which covers two lanes (EC2 T4
  `vardhan-gpu-1`, or Colab). **The T4 is sm_75 and has no bf16**, so measure in fp16 +
  GradScaler and say so. The reversible stream is numerically sensitive: keep the residual
  stream p in fp32 and run only f(·) under autocast, or reconstruction drift will corrupt the
  gradients. Measure the drift either way; it is a finding.
- **Fairness rules for the comparison.** The instructor's rule: every arm gets dropout = 0 and
  weight_decay = 0, including the baseline, so the only difference is the residual rule. Hold
  data order, tokenizer, seq len, LR schedule and seed fixed. The max-batch run changes the step
  count for the same 50M tokens, so its loss is confounded with batch/LR. S11's lesson was
  "tune both sides before accepting a comparison": scale the LR sensibly and say plainly what the
  loss difference does and does not show.
- **Measure speed and memory honestly.** Report tokens/s over the steady state: exclude warmup
  and compile steps, call `torch.cuda.synchronize()` before timing, and exclude eval. Get peak
  memory from `torch.cuda.max_memory_allocated()` after `reset_peak_memory_stats()` per run, and
  report `max_memory_reserved` too. Find max batch by searching with OOM caught and the cache
  emptied between tries, then give it a small safety margin.
- **Widget-data extraction judged unnecessary.** The one widget on §16 visualizes the §17 tables,
  which the prose states in full, and none of the deliverable's numbers come from widgets.
- **Citations.** The README will cite the reversible-LLM paper (**arXiv 2512.02056**, verified
  via the arXiv API 2026-09-24, submitted 2025-11-27), and probably Reformer / RevNets and
  Korthikanti et al. 2022 for the 34-bytes figure. Following the standing `arxiv-library`
  convention: discover and verify IDs and dates rather than recalling them, keep the PDF as a
  checkable file, and pull claims from the PDF text. **On this box the `arxiv-library` corpus
  and RAG layers do not exist, and its search script 406s.** Use
  `curl -A "Mozilla/5.0" 'https://export.arxiv.org/api/query?search_query=ti:"<title>"'` and
  `pdftotext` instead. Lightning LM is The School of AI's own report, not an arXiv paper; cite
  it as the lesson does.
- **Branch `s13-reversibility`**, cut from `s12-distributed-zero` 2026-09-24. Push it to
  `origin` (SSH) when there is something to push.
- **Ties back:** S12's `weight_decay 0.0` was flagged in that class as *"we cannot use weight
  decay with reversibility, we'll discuss that next session"*. S10/S11's nanoGPT and S11's
  LR-sweep lesson carry over directly.
