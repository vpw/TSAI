# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this directory is

Session 12 (S12) assignment of the ERA V5 course (The School of AI). The session topic is
**Distributed Training I: Data Parallel and ZeRO** — Session 11 ended by handing off "across
many GPUs," and this session picks that up with one question: *what do you do when the thing
you are training is larger than the machine you are training it on?* The lesson runs 14
sections and the whole argument is settled by arithmetic in §1, then unpacked.

The arc: a weight costs **16 bytes** to train (2 weight + 2 gradient + 4 fp32 master copy +
8 optimizer moments), so V5's 30B parameters cost **447.0 GiB** before a single activation
exists — six 80 GB cards (§1); the vocabulary is fixed (GPU / node / world size / rank /
interconnect / collective / **P = 60 GB**, one fp16 copy of the parameters — every
communication cost in the session is a multiple of P) (§2); **data parallelism** replicates
the model everywhere and averages gradients, which makes the distributed run mathematically
identical to a single-GPU run on an N-times-larger batch (§3); the averaging is an
**all-reduce**, and the key identity is that **reduce-scatter + all-gather = all-reduce at
the same 2P cost** (§4); communication is judged as a *fraction of step time*, and that
fraction **rises as cards get faster** — 34% on 64xH100 vs 77% on 64xB200 for the same
120 GB (§5); **ZeRO** (Microsoft, 2019) removes the duplication in three stages — stage 1
shards the 12 optimizer bytes, stage 2 also shards the gradients, stage 3 also shards the
weights — and stages 1 and 2 are *free* in communication because they are the two halves of
the all-reduce that was already being paid for, while stage 3 adds an all-gather in both
passes, 2P → 3P (§6); applying that to 30B gives the **memory ladder**, where DP and ZeRO-1
never fit on a 74.5 GiB card *at any world size* because both leave 4 bytes/weight
replicated (§7); **offload** trades GPU memory for PCIe bandwidth (~60 GB/s) (§8);
**DeepSpeed** implements all three stages plus offload via JSON, **FSDP2** implements stage 3
natively in PyTorch via `fully_shard()` and DTensor (§9); **bucketing and overlap** hide the
transfer under the backward pass, which runs last-layer-first, so the last layer's gradients
can leave while the first layers are still computing (§10); **MXFP8 on Blackwell** saves only
12.1% of stored state but ~41% of step time (§11); §12 is what V4 actually ran (DeepSpeed
ZeRO-2, 8 GPUs, bf16) and §13 is V5's decisions with four open questions.

**The numbers that matter for this assignment** (all from `resources/s12-session.md`):

- **16 bytes/weight** decomposed as 2 (bf16 weight) + 2 (bf16 gradient) + 4 (fp32 master) +
  8 (two fp32 optimizer moments). Everything else in the session is this table divided by a
  world size.
- **The per-stage bytes-per-weight formula**, which the simulator has to reproduce exactly.
  Reading it off the §6 table at N=8 and checking it against the §7 ladder at N=16/32/64
  gives:

  | arrangement | bytes/weight at world size N | at N=8 | at N=32 | communication |
  | --- | --- | --- | --- | --- |
  | data parallelism | `2 + 2 + 12` | 16.00 | 16.00 | 2P |
  | ZeRO-1 | `2 + 2 + 12/N` | 5.50 | 4.375 | 2P |
  | ZeRO-2 | `2 + (2 + 12)/N` | 3.75 | 2.4375 | 2P |
  | ZeRO-3 | `16/N` | 2.00 | 0.50 | 3P |

  Multiplied by 30e9 and converted to GiB these give the §7 ladder exactly (ZeRO-2 at 32
  GPUs = 68.1 GiB, ZeRO-3 at 8 = 55.9 GiB, and so on). **The assignment's "32 virtual GPUs"
  lands precisely on the §7 ladder's 32-GPU column** — 447.0 / 122.2 / 68.1 / 14.0 GiB — so
  the simulator has four published targets to be checked against, not just a plausible trend.
- **P = 60 GB** (30B x 2 bytes), and the communication column above: 2P for DP/ZeRO-1/ZeRO-2,
  3P for ZeRO-3. That is the "computation changes" half of the assignment measured honestly.
- **The card is 74.5 GiB** (80 GB). DP and ZeRO-1 never fit; ZeRO-2 fits from 32 GPUs;
  ZeRO-3 fits from 8. The 4 bytes/weight floor (weights + gradients replicated) is 111.8 GiB
  at 30B and is world-size-independent — that invariant is the single sharpest thing a
  simulator can demonstrate.
- §5's cost model: time = bytes / bandwidth, NVLink ~450 GB/s intra-node, InfiniBand
  ~50 GB/s inter-node; 2P = 0.27 s vs 2.40 s. Compute-per-step 7.10 s (64xH100) / 3.12 s
  (64xB200) for a 1M-token step.

**Submission format (confirmed 2026-09-17, entry already live):** GitHub repo link, one
field, 1000 points, **due Sat 2026-09-19 07:00**, resubmission allowed. The notebook goes
*inside* the repo — the form has a single GitHub Link field and no file upload. Same shape
as S9/S10/S11.

**Build status (2026-09-17): complete and green; only the repo push and Axiom submission
remain.** The measured results at the assignment's 32 virtual GPUs, nanoGPT (813,440 params,
40 steps), all matching the closed form to floating-point exactness:

| arrangement | bytes/weight | communication | 30B @ 32 GPUs |
| --- | --- | --- | --- |
| data parallel | 16.0000 | 1.9375P | 447.0 GiB |
| ZeRO-1 | 4.3750 | 1.9375P | 122.2 GiB |
| ZeRO-2 | 2.4375 | 1.9375P | 68.1 GiB |
| ZeRO-3 | 0.5000 | 2.9062P | 14.0 GiB |

**All four arrangements are bit-identical** — same losses, same weights, every parameter,
max difference exactly 0.0. That is the strongest result here and the strictest reading of
the instructor's "make sure it matches what ZeRO does".

Three findings worth carrying forward, because they are places the work went *past* the
lesson rather than merely reproducing it:

1. **The lesson's 2P is a large-N limit.** A ring all-reduce moves `2(N-1)/N · P` per GPU —
   1.9375P at N=32, 1.75P at N=8. The notebook asserts against the exact form and shows it
   converging on 2P, rather than rounding itself into agreement with the table.
2. **The §6 table reproduces at N=8, a world size the notebook never runs at.** That was the
   design point: agreement there cannot have been fitted. §12 extends it to all sixteen
   numbers of the §7 ladder, twelve at world sizes never simulated, within 0.02 GiB.
3. **One lesson claim could not be reproduced directly.** §10's "best bucket on B200 is two
   layers" needs a per-transfer startup cost the lesson never quantifies; without one,
   smaller is always better. The notebook sweeps it and reports that the claim holds for
   α ≥ 29.0 ms, rather than picking a value that forces the expected answer. The 83%
   hidden-transfer figure and §5's 34%/77% both come out exactly.

**What makes this session's deliverable different from every prior one.** S9-S11 all
measured a real model actually training; this one asks for a **simulator** — 32 virtual GPUs
built out of CPU threads (explicitly sanctioned in class), a demo model running on top of
them, and ZeRO-1/2/3 implemented over that. The graded claim is therefore not "my run was
fast" but "**my simulation matches what ZeRO really does**" — the instructor's own words in
the closing minutes were *"ask it to make sure that it matches what zero does."* That makes
the §6/§7 tables above the checkable ground truth, and it makes the README's per-stage
**pros and cons** discussion (the instructor's other explicit ask) load-bearing rather than
decorative. The assignment text's "explains that YOU have understood these concepts (and not
your agent)" is unusually pointed and should shape how the write-up is voiced: mechanism and
reasoning in the student's own framing, not a generated tour.

## Layout

- `S12-assignment.md` — the assignment statement (Section 14 of the lesson, verbatim), the
  Axiom submission block (due date, points, the single GitHub-link field), and the
  instructor's fuller framing of it from the live class.
- `resources/s12-session.md` — full lesson writeup, all 14 sections, captured from the
  lesson page via `get_page_text` (single call, page well under the ~50KB cap). Tables were
  re-laid-out as markdown for readability; numbers unchanged. Widget captions noted inline;
  live widget state not captured (judged unnecessary — see Conventions).
- `resources/s12-transcript.md` — full live-class transcript (106KB, 369 lines), fetched via
  `curl .../export?format=txt` on the link-shared Google Doc the user supplied, the same
  route S10 and S11 used. Header dates it **2026/09/12 06:46 IST**, opens on *"today might
  be a short session but this is one of the most important session when it comes to speeding
  up our training runs,"* ends cleanly at *"Meeting ended after 02:05:05."*
- `notebook_src.py` — **the source of truth.** `# %%` cells, 51 of them, 36 assertions.
  Edit this, never `S12.ipynb`; the notebook is generated from it.
- `S12.ipynb` — generated by `tools/py2nb.py`, executed by `tools/run_nb.py` in a fresh
  kernel with `allow_errors=False`. ~2.8 minutes on CPU.
- `README.tmpl.md` → `README.md` — 456 lines, **87 values substituted from `results.json`**
  by `tools/build_readme.py`, which exits non-zero on any unresolved placeholder. No number
  in the write-up is typed by hand.
- `results.json`, `logs/nbexec.log`, `assets/*.png` — generated evidence.
  `assets/gloo_check.py` is written by the notebook itself for the §3c cross-check.
  `assets/tinyshakespeare.txt` is fetched at runtime and gitignored.
- `tools/` — copied from S11 unchanged. **`build_readme.py` needed no modification**: its
  dotted-path lookup into a single `results.json` is exactly this session's shape.
- `.venv/` — uv-created, torch 2.14.0, CPU. Gitignored.
- Not yet created: the standalone GitHub repo (`s12-standalone` subtree split →
  `github.com/vpw/era-v5-s12`, following the S10/S11 pattern).

## Conventions

- **Submission is a GitHub repo link** (confirmed 2026-09-17) with the notebook committed
  inside it and a detailed README. Follow S9/S10/S11's pattern: public repo, verify in an
  incognito window before ticking the accessibility checkbox.
- **Every number in the write-up must come from a cell that actually ran.** Carried straight
  from S9/S10/S11. The twist this session is that the lesson's numbers are not just
  *expected values* — they are the **correctness oracle**. The §6 bytes-per-weight row and
  the §7 ladder column at N=32 are what the simulator's own memory accounting must
  reproduce; print both side by side and show the match (or explain the gap) rather than
  asserting agreement.
- **Decide early what "computation changes" means here**, because the assignment phrase is
  ambiguous and the answer shapes the whole notebook. The lesson gives a
  *communication-volume* cost model (multiples of P, §4-§6) and a *time* model
  (bytes/bandwidth against a compute-per-step, §5) — it does not ask for FLOPs. The
  defensible reading is: report bytes moved per step per GPU (2P/2P/2P/3P, measured in the
  simulator by counting what the collectives actually transfer, not by quoting the table),
  plus wall-clock per stage from the threaded run, plus the §5-style communication-as-a-
  fraction-of-compute ratio. **Settled this way 2026-09-17** — the fabric counts bytes on
  every collective and §13 applies §5's bandwidths for time. Simulator wall-clock is
  reported but labelled as Python overhead, not a hardware prediction.
- **This is a simulation, so CPU is the target, not a GPU.** The instructor sanctioned CPU
  threads explicitly (*"you can easily do it on your own computer. It's a small program"*),
  and this machine has no GPU (carried from S9/S10/S11). Do **not** reach for
  `era-v5-gpu-run` here unless something specific forces it — the deliverable is 32 virtual
  workers, which is a threading/accounting exercise. Note that Python threads share one
  address space, so "memory per GPU" has to be *accounted* (shard bookkeeping) rather than
  read off the process RSS; be explicit in the write-up about which numbers are measured
  and which are computed from the shard map.
- **Reuse S9-S11's build pipeline rather than hand-editing a notebook.**
  `../../S11/assignment/tools/` has `py2nb.py` (`# %%` cells → notebook), `run_nb.py`
  (executes, writes `results.json`), `dump_log.py`, `build_readme.py` (fills
  `README.tmpl.md` placeholders from the results). Copy the tools and extend
  `build_readme.py`'s placeholder set for this session; don't rewrite them.
- **Widget-data extraction judged unnecessary for this session** (2026-09-17). Every widget
  (data parallelism, the three collectives, the ring all-reduce, the communication-cost
  bars, ZeRO stages, the memory wall, bucket overlap, precision) animates a table or formula
  the prose already states in full, and the §6/§7 tables give the exact per-stage and
  per-world-size numbers the simulator is checked against. Reach for `extract-widget-data`
  only if a specific widget value becomes load-bearing and the prose doesn't state it.
- **The `arxiv-library` skill applies here** — unlike S11, this session's write-up is very
  likely to cite real dated work: the **ZeRO paper (Microsoft, 2019)**, ZeRO-Offload /
  ZeRO-Infinity, FSDP's own paper, and the TorchTitan MXFP8 result (41% faster pre-training
  on B200, loss matching bf16 over 1,500 steps). On the assumption the relevant paper is on
  arXiv: (1) discover it via the skill's arxiv MCP layer rather than trusting a remembered
  title or date, (2) download the PDF into the local library so the source is a checkable
  file, not a claim, and (3) index it via the skill's `rag-toolkit` layer when a specific
  number or claim needs pulling straight out of the PDF text with a citation. Indexing is
  available if needed, not mandatory for every paper.
  **Two gotchas hit on 2026-09-17, both worth knowing before reaching for this skill here:**
  (a) the skill's local corpus (`/u/references/papers/arxiv`) **does not exist on this
  machine** — it belongs to a different instance, so the catalog and RAG layers are
  unavailable; (b) `search_arxiv.py` fails with **HTTP 406** because the arXiv API rejects
  its User-Agent, and it takes `--dir`, not `--root`. What works is querying
  `https://export.arxiv.org/api/query?search_query=ti:"<title>"` directly with a browser
  User-Agent. The four IDs the README cites were verified that way: ZeRO **1910.02054**
  (Rajbhandari, 2019-10-04), ZeRO-Offload **2101.06840**, ZeRO-Infinity **2104.07857**,
  PyTorch FSDP **2304.11277**.
- **`training-stack-docs` is the other relevant skill** for this session specifically — it
  is version-pinned on FSDP/DeepSpeed/Trainer config semantics, which is exactly what §9 and
  §12 describe. Use it rather than recall when the write-up states what a DeepSpeed stage
  flag or an FSDP2 call actually does.
- **Branch `s12-distributed-zero`**, cut from `s11-optimizers` 2026-09-17. Note that
  `s11-optimizers` itself was never pushed to `origin` (S10's branch was) — worth clearing
  alongside S12's push.
- Ties back and forward: §1's 16-bytes-per-weight table is Session 11 §8's optimizer-memory
  point restated as the premise of a whole session; §3's global-batch product extends S10's
  gradient accumulation and S11 §11's batch-size discussion; §12's `weight_decay 0.0` is
  flagged in class as *"we cannot use weight decay with reversibility, we'll discuss that
  next session"* — a forward pointer to Session 13.
