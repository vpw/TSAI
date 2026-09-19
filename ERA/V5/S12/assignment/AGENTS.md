# Task

This directory is part of the assignments for the ERA V5 course of The School of AI (TSAI).
Specifically this is for the twelfth session (S12).

The `S12-assignment.md` file lists the exercise in full — please refer to that for the
details, including the Axiom submission block. **Session 12 already has a live entry in the
Axiom Assignments tab** (checked 2026-09-17): due **Sat 2026-09-19 07:00**, 1000 points,
resubmission allowed, and a single **GitHub Link** field with a public-accessibility
checkbox. The notebook is committed inside that repo — there is no separate upload field.

# Details

The session is **Distributed Training I: Data Parallel and ZeRO**. Session 11 closed by
handing off "across many GPUs"; this session answers the question that creates — what you do
when the model is bigger than the card it has to train on.

The whole session is settled by one piece of arithmetic in §1 and then unpacked. A weight
costs **16 bytes** to train: 2 for the bf16 weight, 2 for its gradient, 4 for the fp32 master
copy (because adding tiny updates to a 16-bit number loses them to rounding), and 8 for the
optimizer's two fp32 running averages. V5's 30 billion parameters therefore cost **447.0
GiB** before a single activation exists, against an 80 GB (74.5 GiB) card.

The arc from there: §2 fixes the vocabulary (GPU, node, world size, rank, interconnect,
collective, and **P = 60 GB**, one fp16 copy of the parameters — every communication cost in
the session is a multiple of P). §3 is **data parallelism**: full replica per GPU, different
data, average the gradients, and because everyone starts identical and applies an identical
update they stay identical — which makes the distributed run mathematically the same as a
single-GPU run on an N-times-larger batch. §4 names the **collectives** and states the
identity everything after it depends on: **reduce-scatter followed by all-gather *is* an
all-reduce**, and a ring all-reduce is internally exactly those two phases, hence 2P. §5
measures communication as a **fraction of step time** and shows the fraction *rising* as
cards get faster (34% on 64xH100, 77% on 64xB200, for the same 120 GB). §6 is **ZeRO**
(Microsoft, 2019): stage 1 shards the 12 optimizer bytes, stage 2 also the gradients, stage 3
also the weights — and 1 and 2 are *free* in communication because they are the two halves of
the all-reduce already being paid, while 3 adds an all-gather in each pass (2P → 3P). §7 is
the **memory ladder**, where DP and ZeRO-1 never fit at *any* world size because both leave 4
bytes/weight replicated (111.8 GiB at 30B, world-size-independent). §8 is **offload** (GPU
memory bought with PCIe bandwidth). §9 is **DeepSpeed vs FSDP2**. §10 is **bucketing and
overlap** — the backward pass runs last-layer-first, so the last layer's gradients can leave
while the first layers still compute. §11 is **MXFP8 on Blackwell** (only 12.1% of stored
state, but ~41% of step time). §12 is what V4 actually ran, §13 is V5's open questions, §14
is the assignment.

Numbers worth having in hand, because they are the **correctness oracle** for this
assignment rather than mere context — per-GPU bytes per weight at world size N:
`DP = 16`, `ZeRO-1 = 4 + 12/N`, `ZeRO-2 = 2 + 14/N`, `ZeRO-3 = 16/N`. At N=8 that is
16.00 / 5.50 / 3.75 / 2.00 (the §6 table), and at **N=32 — the assignment's own world
size** — it reproduces the §7 ladder column exactly: **447.0 / 122.2 / 68.1 / 14.0 GiB** for
a 30B model. Communication per step per GPU is 2P / 2P / 2P / 3P.

Full writeup: `resources/s12-session.md`. Live-class transcript:
`resources/s12-transcript.md` (2026-09-12, 106KB).

**What the assignment actually asks for** — one build, not a list of items:

1. **Create 32 virtual GPUs** — CPU threads on this machine are explicitly sanctioned in
   class (*"you can easily do it on your own computer. It's a small program"*), or a Colab
   GPU.
2. **Write a demo model that runs on top of them.**
3. **Simulate ZeRO-1, ZeRO-2 and ZeRO-3** over that arrangement.
4. **Show how the memory and the computation change** across the stages.
5. **Submit the notebook plus a GitHub repo with a detailed README** that demonstrates the
   student's own understanding.

**The graded discipline here is different from S9-S11.** Those sessions measured a real model
training; this one asks for a *simulator*, so the claim being graded is not "my run was fast"
but "**my simulation matches what ZeRO really does**" — the instructor's own closing words
were *"ask it to make sure that it matches what zero does,"* and separately *"you understand
the pros and cons of each of the stage."* So: print the simulator's own memory accounting
next to the §6/§7 published numbers and show the match; count the bytes the collectives
actually move rather than quoting 2P/3P from the table; and give each stage an honest
pros/cons treatment. The brief's *"explains that YOU have understood these concepts (and not
your agent)"* is pointed, and should shape how the README is voiced.

**Two practical notes before planning.** (a) The target is **CPU**, not a GPU — do not reach
for `era-v5-gpu-run` here without a specific reason; this is a threading and accounting
exercise. (b) Python threads share one address space, so "memory per GPU" must be
*accounted* from the shard map rather than read off process RSS — be explicit in the write-up
about which numbers are measured and which are computed.

**The write-up will almost certainly cite real dated work** (the ZeRO paper, Microsoft 2019;
ZeRO-Offload/ZeRO-Infinity; FSDP; the TorchTitan MXFP8 result). Use the `arxiv-library` skill
rather than recall: discover via its arxiv MCP layer, download the PDF into the local library
so the source is a checkable file, and index via `rag-toolkit` when a specific date or claim
needs pulling out of the PDF text with a citation. For anything about what a DeepSpeed stage
flag or an FSDP2 call actually *does*, use `training-stack-docs` rather than recall.

As a capable agent, plan to: (1) read `resources/s12-session.md` §§1-7 for the arithmetic the
simulator must reproduce and §§9-10 for what the real implementations do, plus the transcript
for the instructor's framing, (2) settle the open decisions in `TODO.md` (what "computation
changes" means concretely, threads vs processes, what the demo model is), (3) reuse
`../../S11/assignment/tools/` (`py2nb.py` / `run_nb.py` / `dump_log.py` / `build_readme.py`)
rather than hand-editing a notebook, (4) run the notebook top to bottom in a fresh runtime so
every number traces to a cell that actually ran, (5) write the README with the per-stage
pros/cons and the published-vs-simulated comparison, (6) push the repo, verify it in an
incognito window, and submit the link before the 2026-09-19 07:00 deadline. `TODO.md` tracks
progress on these steps.

## References
Refer CLAUDE.md if it exists
