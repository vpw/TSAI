# Attention, in the Order It Happened

**ERA V5, Session 8 — every attention mechanism covered in class, arranged by the date it actually
launched, not the order it was taught.**

🔗 **Live app:** [cfc9f13c.ht-ml.app](https://cfc9f13c.ht-ml.app/)
📦 **Repo:** [github.com/vpw/era-v5-s8](https://github.com/vpw/era-v5-s8)

**Short answer to the bonus question:** ordering tells you the sequence, spacing tells you the
pressure. The recurring shape on this timeline is that the *idea* arrives years before the *cost*
that makes anyone adopt it — MQA solves the KV-cache problem in 1,293 days before GQA makes anyone
care, and the delta rule waits 1,204 days for someone to work out how to train it in parallel. A
plain ordered list flattens both waits to zero; only the true date-spacing shows the field was
economising in bursts, not steadily. Section 5 below unpacks this with the other seven
observations the app surfaces.

---

## 1. What this is

Twenty-one mechanisms, plus one model (LightningLM, the lesson's "V4") quoted only for the numbers
it grounds — laid out on a single proportional timeline from 2016-11-07 to 2025-12-13, split into
four "lanes" for which of attention's costs each mechanism is trying to pay down: **position**,
**memory** (the KV-cache), **compute** (the quadratic score matrix), and **systems** (bandwidth,
not FLOPs — the bonus mechanism's lane). Every mechanism gets a uniform dossier: what it is, what
it buys, what it costs, when you'd actually pick it, and — on every single one — exactly how its
launch date was checked.

The one-line pitch from the assignment: *vanilla attention was not wrong, it was expensive, and
everything after it is somebody looking at that bill and trying to pay less of it.*

### Running it locally

No build step, no server, no dependencies:

```
open site/index.html
```

or just double-click it — it's a static page that works straight off `file://`.

## 2. The six acts

1. **Act 0 — Attention, built once.** Standard scaled dot-product attention on six tokens, worked
   by hand: score → scale → mask → softmax → weighted sum, with every displayed number computed at
   render time (nothing hard-coded) and a per-cell arithmetic inspector.
2. **Act 1 — The two bills.** An interactive bill meter: quadratic compute vs. linear-but-per-user
   KV-cache memory, with context/concurrency/precision sliders and live KV-head presets
   (MHA/GQA/MQA) reproducing the lesson's own worked numbers (6.44 GB → 51.54 GB across 8 users).
3. **Act 2 — The timeline.** The proportional, real-date-spacing axis itself — four lane bands, an
   "even spacing" toggle that collapses it into a plain list so you can see exactly what that
   flattening destroys, and lineage arcs linking mechanisms that build on each other.
4. **Act 3 — The dossiers.** One uniform six-block writeup per mechanism (idea / buys / costs /
   pick it when / lineage / how the date was checked), opening in place under the timeline.
5. **Act 4 — What the timeline shows.** The bonus-question answer: eight computed observations,
   each with a "show me on the timeline" button that highlights the exact date range it's about.
6. **Act 5 — The sources table.** Every mechanism, its evidence tier, and its full "how this date
   was checked" writeup as one auditable table — the same table reproduced in §4 below, generated
   from the same data file so the two copies cannot drift apart.

## 3. Minimum-coverage checklist

Standard attention, absolute learned positions, sinusoidal, RoPE, ALiBi, MQA, GQA, sliding window,
attention sinks, NTK-aware scaling, YaRN, linear attention, the delta rule, Gated DeltaNet, MLA,
sparse/top-k attention, DeepSeek's compressed sparse attention (NSA), and DroPE — all present.
Plus the bonus, not-covered-in-class pick:

**FlashAttention** (Dao et al., [arXiv:2205.14135](https://arxiv.org/abs/2205.14135), 2022-05-27).
Chosen over Multi-Token Prediction or Mamba because it answers a genuinely distinct *third*
question the other seventeen mechanisms don't touch: it changes no math at all, computes
bit-identical attention, and attacks memory **bandwidth** via IO-aware kernel fusion rather than
FLOPs or KV-cache size. It also fills a real chronological gap in the data — the quietest stretch
on the whole timeline, between ALiBi (2021-08) and GQA (2023-05) — which is itself one of the
eight observations in Act 4.

## 4. Chronology and sources

Every date below is an arXiv **v1 submission timestamp** (never the latest revision — several of
these papers are on v5–v7 by now) unless the source column says otherwise. The two non-paper rows
were dated from the Wayback Machine's CDX index (NTK-Aware, a Reddit post reddit.com blocks to
both `curl` and a browser) and GitHub's own API (NTK-By-Parts, a pull request). Evidence tiers:
**A** = direct arXiv v1 metadata, **B** = identified via another paper's citation and confirmed two
independent ways, **C** = archival reconstruction, no paper exists at all.

This table is generated by [`tools/gen-readme-table.js`](tools/gen-readme-table.js) directly from
[`site/data/mechanisms.js`](site/data/mechanisms.js) — the same file that drives the timeline, the
dossiers, and the sources table in the app itself, so a corrected date can't leave a stale copy
behind. **Do not hand-edit the table; rerun the generator.**

<!-- CHRONOLOGY-TABLE:START -->
| # | Date | Mechanism | Lane | Primary source | Tier | How the date was checked |
|---|------|-----------|------|----------------|------|--------------------------|
| 1 | 2016-11-07 | **Learned absolute positions** | position | [arXiv:1611.02344](https://arxiv.org/abs/1611.02344)<br>`v1 2016-11-07T23:46:45Z` | B | arXiv v1 metadata (2016-11-07T23:46:45Z). **This date was corrected during the audit.** The mechanism is usually credited to ConvS2S (2017-05-08) because that is what *Attention Is All You Need* cites — but AIAYN cites the later of two papers by the same group. This one states it outright: *"we add position embeddings to encode the absolute position of each source word within a sentence"*, form `e_j = w_j + l_j`. |
| 2 | 2017-06-12 | **Scaled dot-product attention** | baseline | [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)<br>`v1 2017-06-12T17:57:34Z` | A | arXiv v1 metadata. The baseline the whole app is built around; nothing in dispute. |
| 3 | 2017-06-12 | **Sinusoidal encoding** | position | [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)<br>`v1 2017-06-12T17:57:34Z` | A | Same paper and same submission as #2 — deliberately listed as its own entry, since the app treats them as two mechanisms that happen to share an origin. |
| 4 | 2019-04-23 | **Sparse / top-k attention** | compute | [arXiv:1904.10509](https://arxiv.org/abs/1904.10509)<br>`v1 2019-04-23T19:29:47Z` | A | arXiv v1 metadata. |
| 5 | 2019-11-06 | **MQA — Multi-Query Attention** | memory | [arXiv:1911.02150](https://arxiv.org/abs/1911.02150)<br>`v1 2019-11-06T00:19:05Z` | A | arXiv v1 metadata. Single-author Google paper — and the timeline's best surprise, landing ~3.5 years before GQA. |
| 6 | 2020-04-10 | **Sliding window attention** | compute | [arXiv:2004.05150](https://arxiv.org/abs/2004.05150)<br>`v1 2020-04-10T17:54:09Z` | A | arXiv v1 metadata. |
| 7 | 2020-06-29 | **Linear attention** | compute | [arXiv:2006.16236](https://arxiv.org/abs/2006.16236)<br>`v1 2020-06-29T17:55:38Z` | A | arXiv v1 metadata. |
| 8 | 2021-02-22 | **Delta rule (origin)** | compute | [arXiv:2102.11174](https://arxiv.org/abs/2102.11174)<br>`v1 2021-02-22T16:51:38Z` | A | arXiv v1 metadata. |
| 9 | 2021-04-20 | **RoPE** | position | [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)<br>`v1 2021-04-20T09:54:06Z` | A | arXiv v1 metadata. |
| 10 | 2021-08-27 | **ALiBi** | position | [arXiv:2108.12409](https://arxiv.org/abs/2108.12409)<br>`v1 2021-08-27T17:35:06Z` | A | arXiv v1 metadata. |
| 11 | 2022-05-27 | **FlashAttention** | systems | [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)<br>`v1 2022-05-27T17:53:09Z` | A | arXiv v1 metadata, **plus** an independent `pdftotext` read of the PDF title page — checked twice because this is the bonus row and carries extra scrutiny. |
| 12 | 2023-05-22 | **GQA — Grouped-Query Attention** | memory | [arXiv:2305.13245](https://arxiv.org/abs/2305.13245)<br>`v1 2023-05-22T17:16:38Z` | A | arXiv v1 metadata. |
| 13 | 2023-06-29 | **NTK-Aware scaling** | position | [r/LocalLLaMA post](https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/) | C | **No paper exists — the weakest evidence class in the set, and still pinned to the minute.** reddit.com is bot-walled to both `curl` and a real browser, so the date came from the Wayback Machine two independent ways: a snapshot carries Reddit's own `created-timestamp="2023-06-29T08:21:29.413+0000"` with `author="bloc97"`, and the CDX index's earliest capture of the URL is 08:21:50 UTC the same day — an archive cannot predate what it archived. |
| 13b | 2023-07-07 | **NTK-By-Parts** | position | [GitHub PR](https://github.com/jquesnelle/yarn/pull/1) | B | **Not a paper and not a Reddit post — a GitHub pull request**, which makes this the best-evidenced non-paper row here. Dated from GitHub's own API, two independent timestamps: the PR was opened by `bloc97` at `2023-07-07T20:40:33Z` and its first commit is authored `2023-07-07T20:24:12Z`, 16 minutes earlier the same day. Merged 2023-07-09. Platform record, not archival inference. The repo was later renamed, so YaRN's cited URL now resolves to `jquesnelle/yarn/pull/1`. |
| 14 | 2023-08-31 | **YaRN** | position | [arXiv:2309.00071](https://arxiv.org/abs/2309.00071)<br>`v1 2023-08-31T18:18:07Z` | A | arXiv v1 metadata. Its own bibliography is also what established that #13 and #13b have no papers, and what separated the two. |
| 15 | 2023-09-29 | **Attention sinks** | memory | [arXiv:2309.17453](https://arxiv.org/abs/2309.17453)<br>`v1 2023-09-29T17:59:56Z` | A | arXiv v1 metadata. |
| 16 | 2024-05-07 | **MLA — Multi-head Latent Attention** | memory | [arXiv:2405.04434](https://arxiv.org/abs/2405.04434)<br>`v1 2024-05-07T15:56:43Z` | A | arXiv v1 metadata. The mechanism is introduced inside a model paper rather than one of its own — normal for DeepSeek. |
| 17 | 2024-06-10 | **DeltaNet (parallel training)** | compute | [arXiv:2406.06484](https://arxiv.org/abs/2406.06484)<br>`v1 2024-06-10T17:24:42Z` | A | arXiv v1 metadata. |
| 18 | 2024-12-09 | **Gated DeltaNet** | compute | [arXiv:2412.06464](https://arxiv.org/abs/2412.06464)<br>`v1 2024-12-09T13:09:04Z` | A | arXiv v1 metadata. A "Gated DeltaNet-2" follow-up exists (2605.22791, 2026-05-21) — **not** the one the brief means; noted so the two are not swapped. |
| 19 | 2025-02-16 | **DeepSeek NSA** | compute | [arXiv:2502.11089](https://arxiv.org/abs/2502.11089)<br>`v1 2025-02-16T11:53:44Z` | A | arXiv v1 metadata. Author list includes DeepSeek-AI leadership, confirming this is the paper behind the lesson's §12 compression + top-k block selection design. |
| 20 | 2025-12-13 | **DroPE** | position | [arXiv:2512.12167](https://arxiv.org/abs/2512.12167)<br>`v1 2025-12-13T04:23:47Z` | B | arXiv v1 metadata — **but this is the row I got wrong first.** My initial search concluded DroPE had no public source, which was simply stopping too early. Found via LightningLM's bibliography, extracted two independent ways (a RAG hit *and* a raw `pdftotext` dump of the reference pages, bypassing chunking entirely), then title/authors/date re-confirmed against arXiv directly. |
| — | 2026-06-05 | **LightningLM ("V4")** | *context* | [arXiv:2606.07404](https://arxiv.org/abs/2606.07404)<br>`v1 2026-06-05T15:48:42Z` | A | arXiv v1 metadata. **This date was corrected during the Act 5 re-check** — it had been recorded as 2026-06-08, three days late, from the listing page rather than the submission timestamp. Every V4-specific number the app quotes — the DDDGDDDG depth schedule, the KV-cache figures, the Memory Stream, the 8K→256K DroPE application — is grounded here. |

**22 rows** — 21 mechanisms plus one model quoted for its numbers. 20 dated from an arXiv **v1** submission record (19 distinct papers; two rows share *Attention Is All You Need*), 2 from sources with no paper at all. Evidence tiers: **A 18 / B 3 / C 1**.

*Generated by `tools/gen-readme-table.js` from `site/data/mechanisms.js`. Do not edit by hand — rerun the generator.*
<!-- CHRONOLOGY-TABLE:END -->

### The method, because that's what's actually graded

Three dates in this table were wrong on the first pass, and the way each one was caught is worth
stating plainly:

- **Learned absolute positions (#1).** The obvious citation is ConvS2S (2017-05-08), because that
  is what *Attention Is All You Need* itself cites for the idea. But AIAYN cites the **later** of
  two papers from the same group — the earlier one
  ([1611.02344](https://arxiv.org/abs/1611.02344), 2016-11-07) states the mechanism outright:
  *"we add position embeddings to encode the absolute position of each source word,"* form
  `e_j = w_j + l_j`. **The general lesson: "the paper the famous paper cites" is a heuristic, not a
  proof** — three rows in this table turned out to have an earlier precursor named inside the
  credited paper's own related-work section.
- **NTK scaling (#13/#13b).** What's usually called "NTK-aware scaling" is actually two separate
  community fixes by the same author (bloc97): NTK-Aware itself (a Reddit post, 2023-06-29,
  Tier C) and NTK-By-Parts (a GitHub PR, 2023-07-07, Tier B) — and YaRN, which does have a paper,
  builds on the *second* one specifically (its own §3.2), not the one usually credited.
- **DroPE (#20).** A first search pass concluded DroPE had no public source at all — wrong, just
  stopped looking too early. It's named in LightningLM's own bibliography
  ([2606.07404](https://arxiv.org/abs/2606.07404)) as Gelberg, Eguchi, Akiba & Cetin (Sakana AI),
  [2512.12167](https://arxiv.org/abs/2512.12167), 2025-12-13 — confirmed two independent ways (a
  RAG hit and a raw `pdftotext` dump of the citing paper's reference pages, which bypasses
  chunking entirely) before being re-confirmed against arXiv's own metadata directly.

Full research notes, including every candidate paper considered and rejected, are in
[`resources/chronology-research.md`](resources/chronology-research.md).

## 5. What the timeline shows that a list doesn't (the bonus question)

Every number below is computed from `site/data/mechanisms.js` at render time in the app itself
(Act 4), not typed in here separately — so it can't quietly go stale if a date is corrected later.

1. **Silence, not agreement, follows the Transformer.** The largest single gap on the entire
   timeline is the ~2 years between the Transformer (2017-06) and the first attempt to make it
   cheaper (Sparse Transformer, 2019-04). Scaling was still paying off by going *bigger*; efficiency
   work starts only once that stops being free.
2. **MQA solves memory in 2019 — GQA doesn't ship until 2023.** Same idea, softened; what changed
   in between wasn't the technique, it was *who pays* — training cost vs. serving cost at scale.
3. **The delta rule sits dormant for three years.** Published 2021-02, before RoPE. The blocker was
   never the idea (correct the state instead of accumulating into it) — it was that read-then-write
   is sequential and GPUs hate sequential. DeltaNet's 2024 contribution is a parallel training
   algorithm, not the rule itself.
4. **The "quiet" stretch between ALiBi and GQA wasn't quiet — it was systems work.** The one entry
   inside that ~21-month gap is FlashAttention, which kept exact attention affordable for years
   longer than it otherwise would have.
5. **2023 happens all at once.** Five mechanisms land inside a four-month window — a pile-up a
   plain list renders as five ordinary rows.
6. **Position is the problem that never stayed solved.** Eight attempts across the full nine-year
   span of the timeline, ending — in the newest entry — with DroPE proposing to delete positional
   embeddings altogether.
7. **The field roughly doubled its pace after 2023.** And the composition shifted: early years are
   almost entirely about compute; the recent half spreads across memory, length, and compute at
   once.
8. **Two of the twenty-one never had a paper.** Both from 2023, both about context length, both
   shipped in production before or without peer review — because the publication cycle was too slow
   for how hard the constraint was binding.

## 6. Repo layout

| Path | Contents |
|---|---|
| `site/index.html` | the app — open directly, no build step |
| `site/assets/act*.js` | one file per act (pipeline, bill meter, timeline, dossiers, insights, sources table) |
| `site/data/mechanisms.js` | the single source of truth — every date, source, tier, and dossier field |
| `tools/gen-readme-table.js` | generates §4's table from `mechanisms.js` |
| `resources/chronology-research.md` | full per-mechanism research notes, candidates considered and rejected |
| `resources/s8-session.md`, `resources/s8-transcript.md` | the lesson writeup and live-class transcript this app is built from |
| `S8-assignment.md` | the assignment brief, verbatim |

## References

- **1706.03762** — Vaswani et al., *Attention Is All You Need*. The baseline everything else in
  this app responds to.
- **2104.09864** — Su et al., *RoFormer* (RoPE). Position as rotation, so Q·K depends only on
  relative distance.
- **2305.13245** — Ainslie et al., *GQA*. Shares KV heads across query-head groups to soften MQA's
  quality trade.
- **2502.11089** — DeepSeek-AI, *Native Sparse Attention*. Compressed + top-k block selection via a
  cheap indexer — this session's most recent mechanism at time of writing.
- **2512.12167** — Gelberg, Eguchi, Akiba & Cetin, DroPE — the newest entry, found via LightningLM's
  bibliography rather than a direct search.

Full sourcing for all 21 mechanisms is in §4's table above and in
[`resources/chronology-research.md`](resources/chronology-research.md).
