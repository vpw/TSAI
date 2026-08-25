# S8 chronology research — mechanism launch dates + primary sources

**Status: 21 mechanisms, every one with an exact, sourced date — no approximations remain.**
(Was 20; row 12 split into 12/12b on 2026-08-19.)

**Date-audit pass, 2026-08-19 — what the user's flags actually changed.** The audit was not
cosmetic; it produced one corrected date, one new row, and three sourced caveats:

- **Row 1 CORRECTED — 2017-05-08 → 2016-11-07.** The old attribution followed *Attention Is All You
  Need*'s own citation, which points at the *later* of two papers by the same group. The earlier one
  (1611.02344) states the mechanism outright as `e_j = w_j + l_j`. This is the single biggest lesson
  of the whole research pass: *"the paper the famous paper cites"* is a good heuristic and not a
  proof — it tells you what that paper's authors reached for, not what came first.
- **Row 12b ADDED — NTK-By-Parts, 2023-07-07**, on the user's instruction to plot the two NTK methods
  separately. Dated from GitHub's own API (PR + commit timestamps), which is stronger evidence than
  row 12's Wayback inference.
- **Rows 4, 6, 7 stand, but their caveats are now quotations from the papers themselves** rather than
  my characterisations. Row 7's challenger turned out to be real and 19 months earlier
  (Shen et al. 1812.01243, 2018-12-04); what keeps the row where it is, is that the challenger is
  non-causal — verified by full-text scan, not assumed.

The last approximate date (row 12, NTK-Aware) was pinned to 2023-06-29 on 2026-08-18 via Wayback,
after the user supplied the original Reddit permalink.

**Act 5 re-check, 2026-08-19.** Before generating the sources table, all 19 arXiv IDs were re-queried
in one batch against `export.arxiv.org/api/query` and compared field-by-field against the signed-off
data. **All 21 mechanism dates confirmed exactly**; the exact v1 timestamps are now stored per row
(`v1` field in `site/data/mechanisms.js`) rather than being re-derived, so the table asserts a
checkable record instead of a claim. One error surfaced:

- **Context marker CORRECTED — 2026-06-08 → 2026-06-05** (LightningLM / "V4", 2606.07404). Recorded
  three days late, from the listing page rather than the submission timestamp. Not one of the graded
  mechanisms and it does not reorder anything, but it was wrong on the page and is now right.

Worth noting *how* it survived the audit: it is the one row nobody scrutinised, precisely because it
is not a mechanism. The audit's attention went where the grading was.

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
| 1 | Absolute learned positional embeddings | **1611.02344** | **2016-11-07** | A Convolutional Encoder Model for Neural Machine Translation (Gehring, Auli, Grangier, Dauphin) | **CORRECTED 2026-08-19 after the user flagged this row in the date audit — was 1705.03122 / 2017-05-08.** The original attribution followed *Attention Is All You Need*'s own citation ("We also experimented with using learned positional embeddings [9]" → [9] = ConvS2S 1705.03122). That is still what AIAYN cites, but AIAYN cites the *later* of two papers by the same group. Checking the earlier one settles it: 1611.02344 §"position embeddings" says plainly *"we add position embeddings to encode the absolute position of each source word within a sentence. Each source embedding e_j therefore contains a position embedding l_j as well as the word embedding w_j"*, with the explicit form **`e_j = w_j + l_j`** — a learned per-position vector added to the token embedding, i.e. exactly the mechanism, six months before ConvS2S. Date confirmed against arXiv's own metadata (v1 2016-11-07T23:46:45Z). **Antecedent, stated for honesty and deliberately not used as the marker:** that same passage points further back — *"Position embeddings have also been found helpful in memory networks for question-answering and language modeling (Sukhbaatar et al., 2015)"*. Checking 1503.08895 (End-To-End Memory Networks): it has a *fixed* "position encoding (PE)" scheme **and** a separate learned "Temporal Encoding" matrix `T_A(i)`, of which the paper says *"Both T_A and T_C are learned during training."* That is a learned position-index embedding, but it indexes **memory slots / sentences**, not token positions within a sequence — a different granularity from what the app means by absolute positional embeddings. Marker stays at 1611.02344; the dossier states the Sukhbaatar antecedent. |
| 2 | Standard scaled dot-product attention | 1706.03762 | 2017-06-12 | Attention Is All You Need (Vaswani et al.) | Also the origin of sinusoidal positional encoding (next row) — same paper, same date. |
| 3 | Sinusoidal positional encoding | 1706.03762 | 2017-06-12 | Attention Is All You Need (Vaswani et al.) | Same paper as row 2. The two techniques are both introduced here; list separately in the app but note the shared source/date. |
| 4 | Sparse / top-k attention | 1904.10509 | 2019-04-23 | Generating Long Sequences with Sparse Transformers (Child, Gray, Radford, Sutskever) | **Row stands; caveat now sourced from the paper's own §2 Related Work, per the user's audit request for a synopsis of it.** That section names its precursors rather than claiming primacy: *"(Parmar et al., 2018) uses blocks of local attention to apply Transformers to images"* (= Image Transformer, arXiv 1802.05751, v1 **2018-02-15**, verified against arXiv metadata — so restricted attention over local blocks is ~14 months older); *"(Dai et al., 2018) introduces a state reuse 'memory' for modeling long-term dependencies"* (Transformer-XL); plus Reed et al. 2017 and Menick & Kalchbrenner 2018 for parallel/multi-scale image generation, and Huang et al. 2018 for efficient relative attention in MIDI. The paper's own positioning claim is modality-breadth and simplicity, not being first: *"Our work is simpler than many of the techniques above and can be applied equally across images, text, and audio. Many of the above techniques are orthogonal to ours."* So the honest framing for the app: Sparse Transformers is the widely-credited source for **factorized sparse attention at scale**, and it credits Parmar et al. 2018 for local-block attention. |
| 5 | MQA (Multi-Query Attention) | 1911.02150 | 2019-11-06 | Fast Transformer Decoding: One Write-Head is All You Need (Shazeer) | Single-author Google paper; predates GQA by ~3.5 years. |
| 6 | Sliding window attention | 2004.05150 | 2020-04-10 | Longformer: The Long-Document Transformer (Beltagy, Peters, Cohan) | **Row stands; caveat now sourced from Longformer's own related-work text.** Longformer does not claim the pattern: it says *"The model with the most similar attention pattern to ours is Sparse Transformer (Child et al., 2019), which uses a form of **dilated sliding window** of blocks of size 8x8"*. So a dilated sliding window already exists in row 4's paper (2019-04-23), and local-block attention goes back further still to Image Transformer (2018-02-15). Longformer's own stated contributions are a more flexible/maintainable CUDA kernel than BlockSparse, and *"additional task motivated **global attention** patterns suitable for common NLP tasks... essential for good performance in the transfer learning setting"*. Honest framing: Longformer is the source credited for **sliding window + global tokens as a named NLP mechanism**, not for the sliding window pattern itself. Separately, Mistral 7B (2310.06825, 2023-10-10) is the famous production deployment, not the origin — flagged so the app doesn't misattribute it. |
| 7 | Linear attention (softmax removed) | 2006.16236 | 2020-06-29 | Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention (Katharopoulos, Vyas, Pappas, Fleuret) | **Row stands, but the challenger is real and is now verified — this was the strongest of the user's audit flags.** Shen, Zhang, Zhao, Yi & Li, *"Efficient Attention: Attention with Linear Complexities"*, arXiv **1812.01243**, v1 **2018-12-04** (confirmed against arXiv metadata) already reassociates the attention product to get linear complexity — **19 months earlier**. Katharopoulos et al. cite it and characterise it as *"Concurrently with this work, Shen..."*, which is defensible for publication venues but not for arXiv v1 dates. **What decides the row:** the lesson's argument is about attention becoming an RNN with a *fixed-size recurrent state* that later needs the delta rule to correct — and that requires the **causal/autoregressive** formulation. Efficient Attention is a vision paper and never does this: a full-text scan of 1812.01243 finds exactly **one** occurrence of "causal"/"autoregressive" in the entire PDF, and it is inside a bibliography entry (the XLNet citation), not in the method. Katharopoulos et al. contribute the kernel-feature-map framing **and** the causal linear-attention recurrence — the part the whole no-softmax section of the lesson rests on. So the marker stays at 2020-06-29 and the dossier states Shen et al. 2018-12-04 as the earlier linear-complexity attention. |
| 8 | Delta rule for linear attention (origin) | 2102.11174 | 2021-02-22 | Linear Transformers Are Secretly Fast Weight Programmers (Schlag, Irie, Schmidhuber) | **Disputed priority, worth flagging in the app.** This is the actual origin of applying a delta-rule-style corrective update to a linear-attention fast-weight state — it predates the paper usually cited as "the DeltaNet paper" (row 16) by over three years, and even predates RoPE (next row). The 2024 DeltaNet paper's contribution is a *parallel-scan training algorithm* for the delta rule, not the delta rule itself. |
| 9 | RoPE (Rotary Position Embedding) | 2104.09864 | 2021-04-20 | RoFormer: Enhanced Transformer with Rotary Position Embedding (Su et al.) | — |
| 10 | ALiBi (Attention with Linear Biases) | 2108.12409 | 2021-08-27 | Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation (Press, Smith, Lewis) | — |
| 11 | GQA (Grouped-Query Attention) | 2305.13245 | 2023-05-22 | GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints (Ainslie et al.) | Sits nearly two years after ALiBi/MQA, with genuinely nothing from the minimum-coverage list in between — worth calling out in the app: the "memory" question (GQA) and the "length" question (ALiBi, 2021) were not solved back-to-back. |
| 12 | NTK-Aware scaling (RoPE context extension) | *no arXiv paper* | **2023-06-29** (Reddit) | "NTK-Aware Scaled RoPE allows LLaMA models to have extended (8k+) context size without any fine-tuning..." — u/bloc97, r/LocalLLaMA | **Non-arXiv origin; date now exact — upgraded 2026-08-18 from the earlier "~2023-06" estimate (user supplied the post URL).** Origin first confirmed via the YaRN paper's own bibliography (RAG-queried 2309.00071): "bloc97. NTK-Aware Scaled RoPE allows LLaMA models to have extended (8k+) context size without any fine-tuning and minimal perplexity degradation., 2023a." with a reddit.com URL — there is no formal paper. **Exact date verified two independent ways** (reddit.com itself is bot-walled — both `curl` and a real browser hit "Prove your humanity", so neither could read it live): (1) a Wayback Machine snapshot of the post (`web.archive.org/web/20230629143320`) contains Reddit's own `shreddit-post` element with `created-timestamp="2023-06-29T08:21:29.413000+0000"` and `author="bloc97"`; (2) the Wayback CDX index's earliest capture of that URL is `20230629082150` — 08:21:50 UTC, 21 seconds after the embedded creation timestamp, and an archive capture cannot predate the post it captured, so it independently bounds the date to the same day. Post ID `14lz7j5`, permalink `https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/`. **Second finding from this pass:** bloc97 made *two* distinct contributions, and papers cite them separately — NTK-Aware Scaled RoPE (this row) and "Add NTK-Aware interpolation *by parts* correction" (NTK-By-Parts), a later refinement. YaRN builds on **NTK-By-Parts**, not on plain NTK-Aware; see arXiv 2401.07004 ("Extending LLMs' Context Window with 100 Samples", 2024-01-13), whose reference list carries both entries and whose §2 describes them as separate methods. Don't collapse the two into one timeline entry without saying so.  **SPLIT 2026-08-19 (user decision, date audit): this row is now NTK-Aware only; NTK-By-Parts is its own timeline marker, row 12b.** |
| 12b | **NTK-By-Parts** (NTK-Aware interpolation "by parts" correction) | *no arXiv paper* | **2023-07-07** (GitHub PR) | "Add NTK-Aware interpolation \"by parts\" correction" — bloc97, PR #1 to `jquesnelle/scaled-rope` | **NEW ROW, added 2026-08-19 on the user's explicit instruction ("Use two") after the date audit.** Previously collapsed into row 12; the two are genuinely distinct methods and papers cite them separately. **Why it matters:** YaRN builds on *this*, not on plain NTK-Aware — 2309.00071 §1 lists it as one of "two improvements of the 'NTK-aware' interpolation... the 'NTK-by-parts' interpolation method (bloc97, 2023b) which performs the best when fine-tuned on a small amount of longer-context data", and §3.2 says "a variant of the resulting method was released under the name 'NTK-by-parts' interpolation". So the app's YaRN lineage arrow must point here, not at row 12. **Source and date:** YaRN's bibliography gives the URL as `https://github.com/jquesnelle/scaled-rope/pull/1` (the repo was later renamed, so it now resolves to `jquesnelle/yarn/pull/1`). Dated from **GitHub's own API**, two independent timestamps: the PR was opened by user `bloc97` at `2023-07-07T20:40:33Z` (title matches YaRN's citation exactly), and its first commit `Create LlamaPartNTKScaledRotaryEmbedding.py` is authored `2023-07-07T20:24:12Z` — 16 minutes earlier, same day. Merged `2023-07-09T18:12:55Z`. This is *stronger* evidence than row 12's Wayback route: it is the platform's own record rather than an archival inference. **Third method — DECIDED 2026-08-19, dossier text only, no marker.** YaRN also cites a "Dynamic NTK" method by **emozilla** (a different author, not bloc97), reddit `14mrgpr`. The user's ruling is that it stays in the NTK dossier prose and does **not** get a timeline marker. Two reasons this is the right call and not just a shortcut: (a) it is not in the assignment's minimum-coverage list, and (b) I could not date it to the standard every other row here meets — Wayback's earliest capture of that URL is 2023-10-11, months after the fact, which bounds nothing useful, and reddit.com is bot-walled to direct fetches. Plotting a marker whose date I could not verify would breach the one rule this whole research pass exists to enforce. **Closed — not an open question.** |
| 13 | YaRN | 2309.00071 | 2023-08-31 | YaRN: Efficient Context Window Extension of Large Language Models (Peng, Quesnelle, Fan, Shippole) | Explicitly built as a formalization/improvement over the NTK-aware and Dynamic-NTK methods (row 12) — cite that lineage in the app's copy. |
| 14 | Attention sinks | 2309.17453 | 2023-09-29 | Efficient Streaming Language Models with Attention Sinks (Xiao, Tian, Chen, Han, Lewis) — "StreamingLLM" | — |
| 15 | MLA (Multi-head Latent Attention) | 2405.04434 | 2024-05-07 | DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model (DeepSeek-AI) | — |
| 16 | Delta rule / DeltaNet (parallel training algorithm) | 2406.06484 | 2024-06-10 | Parallelizing Linear Transformers with the Delta Rule over Sequence Length (Yang, Wang, Zhang, Shen, Kim) | See row 8 — this is the parallel-scan algorithm paper, commonly (and here, per the assignment's own phrasing) called "the DeltaNet paper," but the delta rule concept itself is older. |
| 17 | **BONUS — Differential Transformer (not covered in class)** | 2410.05258 | 2024-10-07 | Differential Transformer (Ye, Dong, Xia, Sun, Zhu, Huang, Wei — Microsoft Research / Tsinghua) | Replaces the earlier FlashAttention pick — see the dedicated bonus section below for why, and for the sourcing. |
| 18 | Gated DeltaNet | 2412.06464 | 2024-12-09 | Gated Delta Networks: Improving Mamba2 with Delta Rule (Yang, Kautz, Hatamizadeh) | A "Gated DeltaNet-2" follow-up exists (2605.22791, 2026-05-21, Hatamizadeh/Choi/Kautz) — not the one meant by the assignment's minimum-coverage list, noted here so it isn't confused with the original. |
| 19 | DeepSeek's compressed + sparse attention | 2502.11089 | 2025-02-16 | Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention (Yuan, Gao, Dai, ... Ruan, Liang [DeepSeek-AI]) | This is the real DeepSeek paper behind the compression + top-k block selection + sliding-window branch design described in the lesson (§12) — authors include Chong Ruan and Wenfeng Liang (DeepSeek-AI leadership), confirming DeepSeek-AI authorship. |
| 20 | DroPE | 2512.12167 | 2025-12-13 | Extending the Context of Pretrained LLMs by Dropping Their Positional Embeddings (Gelberg, Eguchi, Akiba, Cetin — Sakana AI) | **Correction (2026-08-18): my first pass on this row was wrong.** I initially concluded DroPE had no findable primary source and was internal to the ERA course. That was an error from stopping the search too early — I should have checked `../../resources/lightninglm.md` (the course's own documentation of "LightningLM," the model `resources/s8-session.md` calls "V4" — see next row) before concluding "not found." That doc names the real paper: `arxiv 2606.07404`, "Reversible Foundations" (Rohan Shravan). I downloaded that paper and RAG-searched it directly for "DroPE," which surfaced §5.3 ("Positional recalibration: DroPE, applied before annealing") crediting the procedure to "Gelberg et al., 2025." I then cross-checked the exact bibliography entry two ways — a RAG hybrid-search hit *and* an independent raw `pdftotext` dump of pages 55-57 (bypassing RAG chunking entirely, since this is the row that most needs certainty) — both returned the identical entry: "Y. Gelberg, R. Eguchi, T. Akiba, and E. Cetin. Extending the context of pretrained LLMs by dropping their positional embeddings, 2025. URL https://arxiv.org/abs/2512.12167. Code: https://github.com/SakanaAI/DroPE." Downloaded 2512.12167 and confirmed title/authors/date directly against arXiv's own metadata (not just the citing paper's text) — matches. The lesson's "defined at 256K ≠ competent at 256K" framing is about the *reported LightningLM run* (V4 applied DroPE at 8K→256K), which is a separate, narrower evidence-boundary point from "does DroPE itself have a primary source" — it does. |
| 21 | *(context, not a separate mechanism)* "V4" / LightningLM | 2606.07404 | 2026-06-05 | Reversible Foundations: Training a 120B Sparse MoE through State-Preserving Scaling (Rohan Shravan, The School of AI) | Not one of the assignment's minimum-coverage mechanisms — listed here because `resources/s8-session.md`'s "V4" is this real, published model (the ERA course's own capstone from its prior cohort — transcript line 464's "the model we trained in V4" confirms this), not a hypothetical. Its `DDDGDDDG` motif, KV-cache numbers, Memory Stream, and DroPE application are all grounded in this paper, not invented for the lesson. Cite it directly if the app reproduces any V4-specific number (the 8K→256K/32× claim, the injection-scale figures, etc.) rather than treating them as course-internal folklore. |

## Bonus mechanism (not covered in class) — Differential Transformer, chosen

**Correction, 2026-08-25.** This slot originally held FlashAttention. Instructor feedback on the
first submission was direct and correct: *"flash attention is an IO kernel not a new attention
variant, so it does not earn the uncovered bonus."* That is right — FlashAttention reschedules
memory traffic (HBM ↔ SRAM) and returns bit-identical attention; it never changes what attention
*computes*, only where the numbers sit while it computes them. It stays a genuinely interesting
mechanism (still downloaded and tagged in the local library, `2205.14135`), it just isn't an
*attention variant*, so it doesn't satisfy the assignment's bonus criterion. Replaced below with a
mechanism that does change the computation.

**Differential Transformer** — Ye, Dong, Xia, Sun, Zhu, Huang, Wei (Microsoft Research / Tsinghua).
arXiv `2410.05258`, submitted **2024-10-07T17:57:38Z** (confirmed two independent ways: the
`export.arxiv.org` Atom feed's own `<published>` field, and a `pdftotext` read of the downloaded
PDF's title page — the PDF header reads "Published as a conference paper at ICLR 2025" above the
arXiv stamp, i.e. the same work, later peer-reviewed). Downloaded and indexed in the local library.

**What it actually changes:** split each head's queries and keys into two groups, run two ordinary
causal softmax attention maps in parallel, and subtract the second from the first with a small
learned scalar λ: `DiffAttn(X) = softmax(Q₁K₁ᵀ) − λ·softmax(Q₂K₂ᵀ)`, applied to the same value
matrix. Whatever both maps attend to alike is treated as noise and cancels; whatever only the first
map attends to survives — the same idea as a differential amplifier cancelling common-mode noise.
This is a real change to the attention computation, not a kernel rewrite of an existing one — which
is exactly the property FlashAttention lacked.

**Why this is the strongest replacement pick:** every other mechanism in this table answers one of
the two bills named in the lesson — quadratic *compute* or linear *KV-cache memory* — by trading
some accuracy for some savings. Differential Transformer answers neither bill, and doesn't try to:
it spends *more* arithmetic (two attention computations instead of one, though the paper offsets
this by halving each head's dimension so FLOPs and cache stay roughly on par with a standard head)
to fix a problem the lesson never puts a number on — how much of the attention map is noise rather
than signal. The paper reports gains in long-context key-information retrieval, reduced hallucination
in QA/summarization, and more robust in-context learning against prompt-order permutation.

**Chronological placement is itself informative for the Q2 bonus-insight answer:** 2024-10-07 lands
inside the field's densest efficiency cluster — MLA (2024-05), DeltaNet's parallel-training paper
(2024-06), Gated DeltaNet (2024-12), and NSA (2025-02) all within nine months of each other, every
one of them chasing compute or memory down. Differential Transformer sits in the middle of that
cluster doing the opposite: paying more to get a cleaner signal. Worth stating explicitly in the
app: not everything shipped in the field's busiest cost-cutting window was actually about cost.

**Pros/cons for the app's honest-tradeoffs framing:** Buys — sharper attention maps, better
long-context retrieval, less hallucination, more permutation-robust in-context learning. Costs — two
attention computations per head instead of one (accounting made less transparent by the compensating
head-dimension halving); it is a pretraining-time architectural choice, not something you can adopt
into an existing checkpoint the way you can uptrain into GQA. When you'd pick it: when output
faithfulness is the binding constraint, not throughput or cache size — the mirror image of when
FlashAttention would have been "essentially always."

## Papers downloaded this pass (all tagged `era-v5-s8` in the arxiv-library catalog)

1706.03762, 1705.03122, 2104.09864, 2108.12409, 1911.02150, 2305.13245, 2004.05150, 2310.06825,
2309.17453, 2309.00071, 2006.16236, 2406.06484, 2412.06464, 2102.11174, 2405.04434, 1904.10509,
2502.11089, 2512.12167, 2606.07404, 2205.14135, 2410.05258.
