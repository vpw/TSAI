# S8 chronology research — mechanism launch dates + primary sources

**Status: all 20 rows now carry an exact, sourced date — no approximations remain.** The last
approximate one (row 12, NTK-aware scaling) was pinned to 2023-06-29 on 2026-08-18 via Wayback,
after the user supplied the original Reddit permalink.

Method: every date below was pulled from arXiv's own submission metadata (v1 timestamp) via the
`arxiv-library` skill — not from memory. Papers are downloaded into the local library at
`/u/references/papers/arxiv/` (tagged `era-v5-s8`) so every row is a checkable artifact. Three rows
(ConvS2S/learned positions, NTK-aware scaling, DroPE) required verifying a *citation inside another
paper*, not just that paper's own arXiv metadata. For those, the bibliography entry was extracted
two independent ways — a RAG hybrid-search hit against the indexed PDF, and a raw `pdftotext` dump
of the relevant pages (bypassing RAG chunking entirely) — and the two were required to agree before
treating the citation as settled. The cited target paper was then independently re-verified against
arXiv's own metadata (title/authors/date), never taken on the citing paper's word alone.

**One correction made during this pass, left visible rather than silently fixed (row 19,
DroPE):** my first search concluded DroPE had no public primary source. That conclusion was wrong —
reached by not checking `../../resources/lightninglm.md` first, which documents that
`resources/s8-session.md`'s "V4" is a real published model with a real paper. Checking that paper's
own bibliography surfaced DroPE's actual source in about two minutes. Left the correction narrated
in place instead of just replacing the row, since "check every date, don't trust a confident first
answer" is the entire point of this research pass.

**Sorted chronologically — this order is what the app's timeline should follow.**

| # | Mechanism | ArXiv ID | Date (v1) | Title | Notes |
|---|---|---|---|---|---|
| 1 | Absolute learned positional embeddings | 1705.03122 | 2017-05-08 | Convolutional Sequence to Sequence Learning (Gehring et al.) | Predates "Attention Is All You Need" by ~5 weeks. Verified by RAG-querying 1706.03762 directly: its own text says "We also experimented with using learned positional embeddings [9] instead" — reference [9] is this ConvS2S paper, confirmed by pulling its bibliography entry from the PDF. This is the source the Transformer paper itself credits, not an independent guess. |
| 2 | Standard scaled dot-product attention | 1706.03762 | 2017-06-12 | Attention Is All You Need (Vaswani et al.) | Also the origin of sinusoidal positional encoding (next row) — same paper, same date. |
| 3 | Sinusoidal positional encoding | 1706.03762 | 2017-06-12 | Attention Is All You Need (Vaswani et al.) | Same paper as row 2. The two techniques are both introduced here; list separately in the app but note the shared source/date. |
| 4 | Sparse / top-k attention | 1904.10509 | 2019-04-23 | Generating Long Sequences with Sparse Transformers (Child, Gray, Radford, Sutskever) | Earliest widely-credited primary source for factorized/sparse attention patterns reducing full O(n²) attention. |
| 5 | MQA (Multi-Query Attention) | 1911.02150 | 2019-11-06 | Fast Transformer Decoding: One Write-Head is All You Need (Shazeer) | Single-author Google paper; predates GQA by ~3.5 years. |
| 6 | Sliding window attention | 2004.05150 | 2020-04-10 | Longformer: The Long-Document Transformer (Beltagy, Peters, Cohan) | Earliest primary source credited for the sliding-window pattern in the modern LLM sense. Mistral 7B (2310.06825, 2023-10-10) is the mechanism's most famous production LLM deployment but is not the originating paper — flagged so the app doesn't misattribute origin to Mistral. |
| 7 | Linear attention (softmax removed) | 2006.16236 | 2020-06-29 | Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention (Katharopoulos, Vyas, Pappas, Fleuret) | Standard credited source for the kernel-feature-map reformulation that turns attention into an RNN-style fixed-size recurrent state. |
| 8 | Delta rule for linear attention (origin) | 2102.11174 | 2021-02-22 | Linear Transformers Are Secretly Fast Weight Programmers (Schlag, Irie, Schmidhuber) | **Disputed priority, worth flagging in the app.** This is the actual origin of applying a delta-rule-style corrective update to a linear-attention fast-weight state — it predates the paper usually cited as "the DeltaNet paper" (row 16) by over three years, and even predates RoPE (next row). The 2024 DeltaNet paper's contribution is a *parallel-scan training algorithm* for the delta rule, not the delta rule itself. |
| 9 | RoPE (Rotary Position Embedding) | 2104.09864 | 2021-04-20 | RoFormer: Enhanced Transformer with Rotary Position Embedding (Su et al.) | — |
| 10 | ALiBi (Attention with Linear Biases) | 2108.12409 | 2021-08-27 | Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation (Press, Smith, Lewis) | — |
| 10b | **BONUS — FlashAttention (not covered in class)** | 2205.14135 | 2022-05-27 | FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness (Dao, Fu, Ermon, Rudra, Ré) | See the dedicated bonus section below — this is a *third* answer to the two-bills problem, distinct from every other mechanism in this table. |
| 11 | GQA (Grouped-Query Attention) | 2305.13245 | 2023-05-22 | GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints (Ainslie et al.) | Sits nearly two years after ALiBi/MQA — worth calling out in the app: the "memory" question (GQA) and the "length" question (ALiBi, 2021) were not solved back-to-back, there's a real gap where the field was doing other things. |
| 12 | NTK-aware scaling (RoPE context extension) | *no arXiv paper* | **2023-06-29** (Reddit) | "NTK-Aware Scaled RoPE allows LLaMA models to have extended (8k+) context size without any fine-tuning..." — u/bloc97, r/LocalLLaMA | **Non-arXiv origin; date now exact — upgraded 2026-08-18 from the earlier "~2023-06" estimate (user supplied the post URL).** Origin first confirmed via the YaRN paper's own bibliography (RAG-queried 2309.00071): "bloc97. NTK-Aware Scaled RoPE allows LLaMA models to have extended (8k+) context size without any fine-tuning and minimal perplexity degradation., 2023a." with a reddit.com URL — there is no formal paper. **Exact date verified two independent ways** (reddit.com itself is bot-walled — both `curl` and a real browser hit "Prove your humanity", so neither could read it live): (1) a Wayback Machine snapshot of the post (`web.archive.org/web/20230629143320`) contains Reddit's own `shreddit-post` element with `created-timestamp="2023-06-29T08:21:29.413000+0000"` and `author="bloc97"`; (2) the Wayback CDX index's earliest capture of that URL is `20230629082150` — 08:21:50 UTC, 21 seconds after the embedded creation timestamp, and an archive capture cannot predate the post it captured, so it independently bounds the date to the same day. Post ID `14lz7j5`, permalink `https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/`. **Second finding from this pass:** bloc97 made *two* distinct contributions, and papers cite them separately — NTK-Aware Scaled RoPE (this row) and "Add NTK-Aware interpolation *by parts* correction" (NTK-By-Parts), a later refinement. YaRN builds on **NTK-By-Parts**, not on plain NTK-Aware; see arXiv 2401.07004 ("Extending LLMs' Context Window with 100 Samples", 2024-01-13), whose reference list carries both entries and whose §2 describes them as separate methods. Don't collapse the two into one timeline entry without saying so. |
| 13 | YaRN | 2309.00071 | 2023-08-31 | YaRN: Efficient Context Window Extension of Large Language Models (Peng, Quesnelle, Fan, Shippole) | Explicitly built as a formalization/improvement over the NTK-aware and Dynamic-NTK methods (row 12) — cite that lineage in the app's copy. |
| 14 | Attention sinks | 2309.17453 | 2023-09-29 | Efficient Streaming Language Models with Attention Sinks (Xiao, Tian, Chen, Han, Lewis) — "StreamingLLM" | — |
| 15 | MLA (Multi-head Latent Attention) | 2405.04434 | 2024-05-07 | DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model (DeepSeek-AI) | — |
| 16 | Delta rule / DeltaNet (parallel training algorithm) | 2406.06484 | 2024-06-10 | Parallelizing Linear Transformers with the Delta Rule over Sequence Length (Yang, Wang, Zhang, Shen, Kim) | See row 8 — this is the parallel-scan algorithm paper, commonly (and here, per the assignment's own phrasing) called "the DeltaNet paper," but the delta rule concept itself is older. |
| 17 | Gated DeltaNet | 2412.06464 | 2024-12-09 | Gated Delta Networks: Improving Mamba2 with Delta Rule (Yang, Kautz, Hatamizadeh) | A "Gated DeltaNet-2" follow-up exists (2605.22791, 2026-05-21, Hatamizadeh/Choi/Kautz) — not the one meant by the assignment's minimum-coverage list, noted here so it isn't confused with the original. |
| 18 | DeepSeek's compressed + sparse attention | 2502.11089 | 2025-02-16 | Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention (Yuan, Gao, Dai, ... Ruan, Liang [DeepSeek-AI]) | This is the real DeepSeek paper behind the compression + top-k block selection + sliding-window branch design described in the lesson (§12) — authors include Chong Ruan and Wenfeng Liang (DeepSeek-AI leadership), confirming DeepSeek-AI authorship. |
| 19 | DroPE | 2512.12167 | 2025-12-13 | Extending the Context of Pretrained LLMs by Dropping Their Positional Embeddings (Gelberg, Eguchi, Akiba, Cetin — Sakana AI) | **Correction (2026-08-18): my first pass on this row was wrong.** I initially concluded DroPE had no findable primary source and was internal to the ERA course. That was an error from stopping the search too early — I should have checked `../../resources/lightninglm.md` (the course's own documentation of "LightningLM," the model `resources/s8-session.md` calls "V4" — see next row) before concluding "not found." That doc names the real paper: `arxiv 2606.07404`, "Reversible Foundations" (Rohan Shravan). I downloaded that paper and RAG-searched it directly for "DroPE," which surfaced §5.3 ("Positional recalibration: DroPE, applied before annealing") crediting the procedure to "Gelberg et al., 2025." I then cross-checked the exact bibliography entry two ways — a RAG hybrid-search hit *and* an independent raw `pdftotext` dump of pages 55-57 (bypassing RAG chunking entirely, since this is the row that most needs certainty) — both returned the identical entry: "Y. Gelberg, R. Eguchi, T. Akiba, and E. Cetin. Extending the context of pretrained LLMs by dropping their positional embeddings, 2025. URL https://arxiv.org/abs/2512.12167. Code: https://github.com/SakanaAI/DroPE." Downloaded 2512.12167 and confirmed title/authors/date directly against arXiv's own metadata (not just the citing paper's text) — matches. The lesson's "defined at 256K ≠ competent at 256K" framing is about the *reported LightningLM run* (V4 applied DroPE at 8K→256K), which is a separate, narrower evidence-boundary point from "does DroPE itself have a primary source" — it does. |
| 20 | *(context, not a separate mechanism)* "V4" / LightningLM | 2606.07404 | 2026-06-08 | Reversible Foundations: Training a 120B Sparse MoE through State-Preserving Scaling (Rohan Shravan, The School of AI) | Not one of the assignment's minimum-coverage mechanisms — listed here because `resources/s8-session.md`'s "V4" is this real, published model (the ERA course's own capstone from its prior cohort — transcript line 464's "the model we trained in V4" confirms this), not a hypothetical. Its `DDDGDDDG` motif, KV-cache numbers, Memory Stream, and DroPE application are all grounded in this paper, not invented for the lesson. Cite it directly if the app reproduces any V4-specific number (the 8K→256K/32× claim, the injection-scale figures, etc.) rather than treating them as course-internal folklore. |

## Bonus mechanism (not covered in class) — FlashAttention, chosen

**FlashAttention** — Dao, Fu, Ermon, Rudra, Ré (Stanford/Buffalo). arXiv `2205.14135`, submitted
**2022-05-27** ("Fast and Memory-Efficient Exact Attention with IO-Awareness"; v1 date confirmed
directly from the PDF title page via `pdftotext`, independent of the arXiv API metadata). Downloaded
and tagged `bonus-mechanism` in the local library.

**Why this is the strongest pick, and why it earns a genuinely separate slot on the timeline** (not
just "another sparse-attention paper"): every other mechanism in this table answers one of the two
bills named in the lesson — quadratic *compute* (approximate/sparsify the scores) or linear
*KV-cache memory* (share/compress/evict what's stored). FlashAttention answers neither bill by
changing what attention computes. It computes the exact same softmax attention, bit-for-bit, and
instead attacks a third cost the lesson doesn't name: **memory bandwidth** — how many times the
attention matrix gets written to and read back from slow HBM versus fast on-chip SRAM. Standard
attention materializes the full `T×T` score matrix in HBM; FlashAttention fuses the whole
score→softmax→weighted-sum pipeline into blocks that stay in SRAM, using online softmax
(running max/sum, no need to see the whole row first) to never materialize the full matrix at all.
Wall-clock speed goes up and memory drops from O(T²) to O(T) *without giving up exactness* — the
one lever every other mechanism here declines to pull, because they all trade some accuracy for
some savings.

**Chronological placement is itself informative for the Q2 bonus-insight answer:** FlashAttention
(2022-05) lands in the gap I already flagged in row 11's note — between ALiBi (2021-08) and GQA
(2023-05), a stretch where it looks like "nothing happened" if you only track the architecture-level
mechanisms in the minimum-coverage list. FlashAttention fills that gap and shows the field wasn't
idle — it was solving the *systems* half of the problem (how fast can you compute the thing you
already have) in parallel with the *algorithmic* half (what should you even compute). Worth stating
explicitly in the app: "exactness vs. cost" isn't only an algorithmic trade-off (sparsify, compress,
approximate) — kernel-level engineering bought real headroom for free, which is part of why the field
could afford to keep training exact attention as long as it did before moving to GQA/MQA/linear
alternatives.

**Pros/cons for the app's honest-tradeoffs framing:** Buys — exact attention (no quality loss),
large wall-clock speedup, and the O(T²)→O(T) memory drop is "free" (no architecture change, drop-in
kernel swap). Costs — it doesn't reduce the O(T²) *compute* (FLOPs are unchanged, only bandwidth
and materialization are); it doesn't touch inference-time KV-cache growth at all (that's still O(T)
per the cache formula in §10, unaffected by this kernel); and it's a hardware-specific
implementation (tiling sizes tuned to specific GPU SRAM sizes), so it needs re-engineering per
hardware generation (there's a FlashAttention-2 and -3 for this reason) rather than being a portable
algorithmic idea like RoPE or GQA. When you'd pick it: essentially always, for exact attention on
supported hardware — it has no accuracy trade-off, so "when would you choose it" is closer to
"why would you ever not," which is itself a useful contrast against every other row in this table.

## Papers downloaded this pass (all tagged `era-v5-s8` in the arxiv-library catalog)

1706.03762, 1705.03122, 2104.09864, 2108.12409, 1911.02150, 2305.13245, 2004.05150, 2310.06825,
2309.17453, 2309.00071, 2006.16236, 2406.06484, 2412.06464, 2102.11174, 2405.04434, 1904.10509,
2502.11089, 2512.12167, 2606.07404, 2205.14135.
