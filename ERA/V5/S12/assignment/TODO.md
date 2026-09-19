# S12 TODO — Distributed Training I, Data Parallel and ZeRO

## ▶ STATUS (2026-09-19): committed as `a16b9f1`; only the repo push and Axiom submission remain

All four arrangements implemented and verified against the lesson's published numbers.
**Due Sat 2026-09-19 07:00** (1000 pts, resubmission allowed, single GitHub Link field).

Headline results at the assignment's 32 virtual GPUs, nanoGPT (813,440 params), 40 steps:

| arrangement | bytes/weight | comm | 30B @ 32 GPUs |
| --- | --- | --- | --- |
| data parallel | 16.0000 | 1.9375P | 447.0 GiB |
| ZeRO-1 | 4.3750 | 1.9375P | 122.2 GiB |
| ZeRO-2 | 2.4375 | 1.9375P | 68.1 GiB |
| ZeRO-3 | 0.5000 | 2.9062P | 14.0 GiB |

All four are **bit-identical** — same losses, same weights, every parameter, max diff 0.0.
Notebook runtime ~2.7 minutes on CPU.

## ▶ SCAFFOLDING

- [x] **Session verified** (2026-09-17). The lesson page's own heading reads "Session 12:
      Distributed Training I, Data Parallel and ZeRO" — matches this folder, no drift.
- [x] **Assignment captured** → `S12-assignment.md`, Section 14 verbatim, plus the Axiom
      submission block (due 2026-09-19 07:00, 1000 pts, resubmission allowed, one GitHub
      Link field with the incognito-accessibility checkbox, "0/1 answered") and the
      instructor's fuller framing from the live class. The Rubric tab exists but publishes
      no criteria beyond the brief.
- [x] **Lesson captured** → `resources/s12-session.md`, all 14 sections, via `get_page_text`
      on the lesson page (single call, page well under the ~50KB cap). Tables re-laid-out as
      markdown; numbers unchanged.
- [x] **Transcript captured** (2026-09-17) → `resources/s12-transcript.md`, 106KB, 369
      lines, via `curl .../export?format=txt` on the Google Doc the user supplied directly
      (`16mavqsTRYdaTE5ktiK7rHGodRfv-jbYioLl0MdAaaYA`) — no browser download needed, doc is
      link-shared. Header dates it **2026/09/12 06:46 IST**, opens on *"today might be a
      short session but this is one of the most important session when it comes to speeding
      up our training runs,"* ends cleanly at *"Meeting ended after 02:05:05."*
- [x] **Widget extraction judged unnecessary** (2026-09-17). Every widget (data parallelism,
      the three collectives, the ring all-reduce, communication cost, ZeRO stages, the memory
      wall, bucket overlap, precision) animates a table or formula the prose already states
      in full, and the §6/§7 tables give the exact per-stage and per-world-size numbers the
      simulator is checked against.
- [x] **Working set written** — `CLAUDE.md`, `AGENTS.md`, this file (2026-09-17).
- [x] **Branch cut** (2026-09-17) — `s12-distributed-zero`, from `s11-optimizers`.
- [x] **Build pipeline copied** (2026-09-17) — `tools/`, `.gitignore`, `requirements.txt`
      from S11. **`build_readme.py` needed no changes at all**: it does a generic
      dotted-path lookup into a single `results.json`, which is exactly this session's
      shape. `.venv` created with uv (torch 2.14.0, CPU).

## ▶ DECISIONS — all settled 2026-09-17

- [x] **D1. "How the computation changes" = communication volume and time, not FLOPs.**
      The lesson supplies a byte-volume model (multiples of P, §4-§6) and a time model
      (bytes/bandwidth vs compute-per-step, §5), and never asks for FLOPs. The fabric counts
      bytes on every collective; §13 applies §5's bandwidths for time. Measured 2P/2P/2P/3P
      exactly (as 1.9375P/2.9062P — see the ring note below).
- [x] **D2. Threaded simulation + a real gloo spot-check.** A pure-Python fabric of 32
      workers with byte-counted collectives is the deliverable; §3c spawns 4 real
      `torch.distributed` gloo processes and compares tensor by tensor (max diff 1.2e-07,
      float32 rounding — gloo reduces in ring order, the fabric via `stack().mean(0)`).
- [x] **D3. nanoGPT**, reused unchanged from S10/S11 (813,440 params, 36 tensors, wte and
      lm_head tied). Sharded **per layer group** rather than one flat buffer — both give
      exactly 1/N, but stage 3's per-layer gather-use-discard cycle needs a rank's slice of
      a *layer* to be well defined. Every group divides evenly by 32, so no padding.
- [x] **D4. Yes — project analytically to 30B.** §12 evaluates the same accounting
      function at 30e9 and reproduces **all sixteen numbers** of the lesson's §7 ladder to
      within 0.02 GiB, twelve of them at world sizes never simulated. Labelled throughout as
      projection, distinct from the measured toy run.
- [x] **Ring-cost note (worth remembering).** The lesson quotes all-reduce as 2P, but a ring
      all-reduce actually moves `2(N-1)/N · P` per GPU — **1.9375P at N=32**, 1.75P at N=8.
      2P is the large-N limit. The notebook reports the exact counted figure and shows it
      converging on 2P rather than rounding itself into agreement with the table, and the
      assertions are written against the exact form. Same for 3P → 2.9062P.

## ▶ BUILD

- [x] **Virtual GPU layer** (§2) — 32 workers, a world-size/rank abstraction, and the three
      collectives (all-reduce, reduce-scatter, all-gather) implemented over them with byte
      counters on every transfer.
- [x] **Verify the collective identity** (§3a) — reduce-scatter + all-gather produces
      exactly what all-reduce produces (§4), bit-identical, which is what stages 1 and 2
      being communication-free rests entirely on. Plus a real 4-rank gloo cross-check (§3c).
- [x] **Demo model + training loop** (§4-§5) on the virtual GPU layer, data-parallel
      baseline first. Largest gap between any two of the 32 weight copies across all 40
      steps: exactly **0.000000** — the §3 widget's own readout, reproduced.
- [x] **ZeRO-1 / ZeRO-2 / ZeRO-3** (§6-§8) implemented over the same layer, each removing one more
      class of duplicated state.
- [x] **Memory accounting per stage** (§9) — measured values printed against the closed
      form (`16`, `4 + 12/N`, `2 + 14/N`, `16/N`), matching to floating-point exactness, and
      against the lesson's §6 table evaluated at **N=8, a world size never run**.
- [x] **Communication accounting per stage** (§9), measured, and shown to land on 2P/2P/2P/3P.
- [x] **The 4-bytes-per-weight floor** (§11) — DP and ZeRO-1 never fit a 74.5 GiB card at
      *any* world size, because 111.8 GiB of replicated weights+gradients is
      world-size-independent. The boundary lands at exactly 20B parameters. This is the
      sharpest single result the simulator produces.
- [x] **Loss equivalence check** (§10) — all four arrangements bit-identical: same losses,
      same weights, every parameter, max difference 0.0. ZeRO changes only where state
      lives, never the mathematics. This is what "matches what ZeRO does" means most
      strictly, and it is the notebook's strongest single result.

## ▶ EXTRAS — all four built (§13-§15)

- [x] **Memory-wall sweep (§11)** — world size 1→1024, log-log plot against the 74.5 GiB
      card rule. DP and ZeRO-1 asymptote *above* the line; ZeRO-2 fits from 32, ZeRO-3 from 8.
- [x] **Bucketing and overlap (§13)** — reproduces the lesson's **83%** hidden-transfer
      figure at bucket=2 on an H100 step, and §5's 34%/77% comm-fraction test exactly.
      **Gap worth noting:** the lesson's claim that B200's best bucket is 2 layers cannot be
      reproduced without a per-transfer startup cost, which the lesson never quantifies.
      Rather than pick a value that forces agreement, the notebook sweeps it and reports that
      the claim holds for α ≥ 29.0 ms.
- [x] **CPU offload (§14)** — ZeRO-2 + optimizer offload, GPU GiB saved vs PCIe GB/step at
      60 GB/s. At 8 GPUs it is what takes 104.8 GiB down to 62.9 and under the card.
- [x] **MXFP8 (§15)** — 16.0000 → 14.0625 bytes/weight, 12.1% reduction, 392.9 GiB at 30B.
      All three match the lesson exactly.

## ▶ WRITE-UP AND SUBMISSION

- [x] **README.md** built from `README.tmpl.md` via `tools/build_readme.py` — 456 lines,
      **87 values substituted from `results.json`**, zero unresolved placeholders (the tool
      exits non-zero on any). No number in the write-up is typed by hand.
- [x] **Per-stage pros and cons** — README §16, a for/against/use-it-when block per stage,
      each claim tied to a measured number, plus the ZeRO-2-on-32 vs ZeRO-3-on-8 decision
      for V5 and what would settle it.
- [x] **Sources cited and verified** — README has a references table with arXiv IDs looked
      up against the arXiv API rather than recalled: ZeRO **1910.02054** (Rajbhandari,
      2019-10-04), ZeRO-Offload **2101.06840** (2021-01-18), ZeRO-Infinity **2104.07857**
      (2021-04-16), PyTorch FSDP **2304.11277** (Zhao, 2023-04-21). **Gotcha:** the
      `arxiv-library` skill's local corpus (`/u/references/papers/arxiv`) does not exist on
      this machine, and `search_arxiv.py` fails with HTTP 406 — the arXiv API rejects its
      User-Agent. Querying `export.arxiv.org/api/query` over https with a browser UA works.
      The TorchTitan MXFP8 41% figure is cited as the lesson reports it, not independently
      verified.
- [x] **Run the notebook top to bottom in a fresh runtime** — `tools/run_nb.py` with
      `allow_errors=False`, so any raising cell fails the build instead of landing in the
      repo with a traceback. That is what makes the notebook's ~25 assertions load-bearing.
- [x] **Committed** (2026-09-19) — `a16b9f1` on `s12-distributed-zero`, 21 files, working
      tree clean. **Deviation from S11's config worth knowing:** `logs/nbexec.log` is
      committed here. S11's `.gitignore` excluded it, which contradicted `dump_log.py`'s own
      docstring ("the log is committed so the README's numbers can be checked without
      opening the notebook") and the README's layout table. S12 commits it.
- [ ] **Create the GitHub repo** — `s12-standalone` subtree split → `github.com/vpw/era-v5-s12`,
      following the S10/S11 pattern. **Verify in an incognito window** before submitting.
- [ ] **Push `s12-distributed-zero` to `origin`** (github.com/vpw/TSAI). Note
      `s11-optimizers` was never pushed either (S10's branch was) — worth clearing together.
- [ ] **Submit the link in Axiom** before **Sat 2026-09-19 07:00**, and tick the
      accessibility checkbox.

## ▶ CARRY-OVER FROM S11

- [x] **S11 submitted in Axiom** (confirmed by the user 2026-09-17). Nothing open there.
