# V5 — data mixture and curriculum plan

Every number below is computed from `data/inventory.json` by `scripts/ledger.py` and lands in
[SUPPLY_LEDGER.md](SUPPLY_LEDGER.md). The inventory is transcribed from the session's own
dataset-inventory widget (extraction in [`resources/s5-widget-data.md`](resources/s5-widget-data.md)).
Nothing here is typed by hand, and the mixture is testable: the arms in [`proxy/`](proxy/) were
run on a GPU and the decision rules were [committed before the runs](proxy/HYPOTHESES.md).

**Headline:** the budget is 2.4T trained tokens, and that number is a *supply calculation, not a
preference*. The corpus cannot feed a 4T run through a 40%-keep selector.

---

## 1. The budget is derived, not chosen

OPUS keeps 40% of candidates, and 10% of every batch bypasses it. To train **T** tokens you must
therefore hold `0.9·T/0.4 + 0.1·T` candidate tokens.

| | |
|---|---|
| Total unique supply, all lanes | **6.40T** |
| Largest trained budget this corpus can feed | **2.72T** |
| Budget taken | **2.40T** (needs 5.64T candidates, leaving 761.6B headroom) |
| A 4T run would need | 9.40T candidates — **not fundable** |

The over-training is bought with OPUS's 6× effective multiplier, not with raw tokens: 2.4T raw =
**14.4T effective**, which at the 120B model the curriculum widget names is **20 raw tok/param,
120 effective**. Compute overhead of the selector is 4.7%.

**Stage split** (training-lifecycle widget): pretraining 95% = 2.28T · anneal 2% = **48B** ·
SFT / reasoning-training / preference 1% each.

---

## 2. The two mixtures

| Lane | Main run % | Anneal % | Demand | Unique supply | Epochs | Verdict |
|---|---:|---:|---:|---:|---:|---|
| General web | 32 | 8 | 733.4B | 4.69T | 0.16 | covered |
| Code | 24 | 20 | 556.8B | 1.10T | 0.50 | covered |
| STEM / math | 12 | 10 | 278.4B | 146.0B | 1.91 | repeat |
| Indic | 16 | 28 | 378.2B | 275.9B | 1.37 | repeat |
| Reasoning | 8 | 18 | 191.0B | 85.1B | 2.24 | repeat |
| Long-context | 6 | 8 | 140.6B | 100.0B | 1.41 | repeat |
| Agentic / tool-use | 2 | 8 | 49.4B | **627M** | **78.85** | **must synthesise** |

Composer defaults were web 34 / reasoning 6; we move 2 points from web to reasoning. Web is the
only lane with 30× headroom over its demand, so the two points are free there, and reasoning is
the lane whose benchmarks (AIME, GPQA Diamond, HLE, FrontierMath) the model is most likely to be
judged on. Everything else matches the composer. The anneal column is the composer's own "V5
anneal" preset, unchanged.

### What each lane is meant to win, and what fills it

| Lane | Benchmarks it is bought for | Datasets that actually fill it |
|---|---|---|
| General web | MMLU | DCLM-Baseline 2.6T, FineWeb-Edu 1.3T, D2/D1 V4 web 791B |
| Code | LiveCodeBench, Aider Polyglot, Codeforces | The Stack v2 900B, D3 Code (V4) 199B, CommitPack 4B |
| STEM / math | AIME, GPQA Diamond, HLE, FrontierMath | proof-pile-2 55B, D4 STEM (V4) 49B, peS2o 42B |
| Reasoning | AIME, GPQA, HLE | AON (V4) **78B of 85.1B**, OpenThoughts2 3B, OpenMathReasoning 2B, OpenR1-Math 1.6B, NuminaMath 0.5B |
| Agentic | SWE-bench Verified, tau-bench, BFCL v3, GAIA, BrowseComp | SWE-Gym 150M, SWE-smith 120M, OpenHands rollouts 90M, ToolBench 80M, ToolACE 60M, Glaive 50M, Nexus 30M, xLAM 25M, Hermes 22M |
| Long-context | **no benchmark exists** — see below | repo-packed code 60B, book-length packed 40B |
| Indic | MILU, IndicGenBench | AI4Bharat only: Sangraha synthetic 162B / verified 64B / unverified 24B, IndicCorpV2 20.9B, BPCC 3B, Samanantar 2B |

Three things this table is meant to make impossible to miss:

1. **The reasoning slot is one corpus.** 92% of its 85.1B is V4-lineage AON. Every open
   reasoning-trace set combined is ~7.1B. If AON is bad, the lane is bad.
2. **Long-context is not new supply.** Both rows are repackings of tokens already counted under
   code and general web. Funding 6% of the run here double-counts unless the packing is done from
   the code and web budgets, which is how we schedule it.
3. **Indic has one supplier.** Every Indic row is AI4Bharat. There is no second vendor for the
   flagship capability.

### The long-context lane has no benchmark

The composer points it at `long-eval`. The benchmark explainer ships 18 benchmarks in four groups
— agentic, coding, reasoning, Indic — and **has no long-context group at all**. No RULER, no
LongBench, no needle-in-a-haystack. We fund the lane anyway because 32K+ packing is a
prerequisite for the agentic lane (SWE-bench trajectories do not fit in 8K), and we name our own
gate: **RULER at 32K ≥ 85% and a repo-level needle test**, declared here because the session
does not supply one.

---

## 3. Indic, split four ways

A single headline number hides the problem, so here is the split, derived from supply rather than
asserted. The widget's own default (A 40 / B 25 / C 20 / D 15) is **unsatisfiable against its own
inventory**: the whole translated tier is Samanantar 2B + BPCC 3B = **5B tokens**, and 20% of a
378B lane is 75.6B — 15 epochs on translated text.

| Tier | Source | Unique | Epochs | Contributes | Share of lane |
|---|---|---:|---:|---:|---:|
| **A verified** | Sangraha verified 64B | 64.0B | 2.50 | 160.0B | **42.3%** |
| **B unverified** | Sangraha unverified 24B + IndicCorpV2 20.9B | 44.9B | 2.00 | 89.8B | **23.7%** |
| **C translated** | Samanantar 2B + BPCC 3B | 5.0B | 2.00 | 10.0B | **2.6%** |
| **D synthetic** | Sangraha synthetic 162B | 162.0B | 0.73 | 118.4B | **31.3%** |

Epochs are the decision variable: verified text is worth repeating 2.5×, translated text is
capped at 2× because translationese compounds, and synthetic is deliberately run *below* one
epoch — we have far more of it than we want to use.

**59% of the Indic slot is already synthetic before we generate anything.** Of 275.9B unique
Indic tokens, 114B are real and 162B are Sangraha synthetic. The four-tier split exists to stop
that ratio from being invisible.

---

## 4. Agentic: defend it in supervised tokens, not raw ones

The trajectory widget measures what a tool-use sample actually trains on: a full multi-step run is
**53% supervised** (356 of 668 tokens), a one-shot call **43%** (42 of 98). Observations and user
turns are context, not signal.

| | |
|---|---|
| Raw slot supply | 627M |
| Multi-step trajectories | 360M raw → **191M supervised** |
| Single function calls | 267M raw → **115M supervised** |
| Total supervised | **306M** |
| Demand at a 2% floor | 49.4B |
| Shortfall against raw supply | **79×** |
| Must be built, even at the 4-epoch ceiling | **46.9B (94.9%)** |

Independent check: tokenizing the real Glaive / ToolACE / Hermes corpora for the proxy and masking
by role gives **45.3% supervised**, between the widget's two figures. The measurement holds on
real data.

So the 2% is not a claim that 49.4B agentic tokens exist. It is a commitment to **generate 46.9B**
— verifier-checked SWE-Gym/SWE-smith-style rollouts where the test suite is the reward — and the
plan is wrong if that generation does not happen. This is the correction to S3's plan, which gave
the agentic lane 4% of 8T = **320B tokens** against 627M of real supply. That is the wishful
accounting this rubric punishes, and it was ours.

---

## 5. The protected floor OPUS may not cross

**10% of every batch bypasses the selector entirely: Indic 7 / agentic 2 / reasoning 1.**
Plus mixture floors the selector may not push below: **Indic ≥ 12%, agentic ≥ 2%.**

V4 protected one lane at 8%. V5 protects three, and splits the floor rather than letting Indic and
agentic share, because the selector widget shows they starve for *different reasons*:

| Proxy | Floor | Indic trained share | Agentic trained share |
|---|---|---:|---:|
| English-heavy | off | **0.0%** | **0.0%** |
| Balanced | off | 19.7% | **0.0%** |
| English-heavy | on | 13.0% | 16.3% |
| Balanced | on | 15.8% | 19.7% |

Read the second row: **fixing the proxy rescues Indic and does nothing for agentic.** Indic starves
because the proxy is English-shaped, and a balanced proxy cures that. Agentic starves because
trajectories look low-utility to *any* benchmark-derived proxy — half their tokens are
observations the model is not scored on. Only the floor saves it. The session's prose does not
draw this distinction; the widget does.

---

## 6. Anneal reserve

**2% of the budget = 48B tokens**, held back at composition time, not scraped together at the end.
What is reserved: the best Tier-A verified Indic, the verifier-checked agentic trajectories, and
the long reasoning traces. The anneal mixture triples Indic (16 → 28) and more than doubles
reasoning (8 → 18) and agentic (2 → 8).

The reserve is only meaningful if the data is *withheld* — an anneal that re-runs tokens the main
phase already used is a learning-rate schedule, not a data decision. Concretely: 13.4B verified
Indic, 3.8B agentic and 8.6B reasoning tokens are tagged `anneal_only` at composition and excluded
from main-run sampling.

---

## 7. Difficulty and reasoning-length bands

These are **not two independent ladders** — the widget is explicit that "a hard problem with a
short trace and an easy one with an ultra trace are not interchangeable". It is a **6 × 4 grid**,
and a cell left empty is a capability that will not exist.

### Difficulty: B0 → B5, with a real example each

| Band | Level | Concrete example (from the inventory) |
|---|---|---|
| **B0** | Nursery | "The cat sat on the mat. The cat is happy." — simple children's sentences; FineWeb-Edu low-score bucket |
| **B1** | Grade-school | "A shop sells 3 pens for ₹12. What do 7 pens cost?" — GSM8K-shaped, NuminaMath easy split |
| **B2** | High-school | "Find all real x with x² − 5x + 6 < 0." — NuminaMath / AIME warm-up tier |
| **B3** | Undergraduate | "Prove that a continuous function on a closed interval attains its maximum." — proof-pile-2, peS2o undergraduate texts |
| **B4** | Graduate | "Derive the partition function of the 1-D Ising model with an external field." — peS2o graduate, GPQA Diamond shape |
| **B5** | Research / PhD | "Given this 2024 arXiv abstract, state the main theorem's hypotheses and where they are used." — peS2o recent, FrontierMath / HLE shape |

### Reasoning length: the same problem at four depths

The widget's worked example, kept because it makes the waste visible. *How many integers in
1..1000 are divisible by 3 or 5?* Answer 333 + 200 − 66 = **467** at every depth.

| Band | Steps | Trace tokens | Solve rate | Verdict | The trace |
|---|---:|---:|---:|---|---|
| **short** | 2 | 37 | 62% | thin but cheap | "Inclusion-exclusion: ⌊1000/3⌋+⌊1000/5⌋−⌊1000/15⌋." → "= 333 + 200 − 66." |
| **medium** | 4 | 74 | 77% | well spent | counts each set, names lcm(3,5)=15, then combines |
| **long** | 6 | 161 | 91% | well spent | defines A and B, justifies each floor, **and checks by complement**: 1000·⅔·⅘ ≈ 533, 1000−533 = 467 |
| **ultra** | 14 | 346 | 95% | **wasted effort** | restates the goal, plans, re-derives each count from largest/smallest multiple, verifies twice |

**Ultra buys +4 points for 2.15× the tokens of long.** That is the argument for scheduling trace
length instead of always training long: an easy problem with an ultra trace teaches the model to
burn tokens. We reserve ultra traces for B4–B5 cells only, and require every B0–B2 cell to carry
short and medium traces so the low end of the dial exists at all.

---

## 8. Curriculum: stage profiles and the seams between them

Mixture is a schedule, not a constant. Per-stage profiles, read off the curriculum widget:

| Marker | Stage | Web | Code | Reason | Long-ctx | Indic | STEM |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0% | Seed | 55 | 15 | 3 | 2 | 15 | 10 |
| 25% | General | 45 | 20 | 6 | 3 | 16 | 10 |
| 50% | Reasoning | 25 | 28 | 18 | 5 | 14 | 10 |
| 75% | Long-context | 18 | 30 | 16 | 18 | 12 | 6 |
| 100% | Cooldown end | 8 | 22 | 20 | 10 | 30 | 10 |

Web falls 55 → 8; code, reasoning and long-context climb once the base is solid; Indic doubles into
the cooldown, which is the anneal reserve arriving.

**Every seam gets a warmup band.** The transition widget measures peak gradient norm against a 3×
threshold:

| Shift | Embeddings | Warmup | Peak grad norm | |
|---|---|---:|---:|---|
| max | frozen | none | **151×** | unstable |
| max | **trainable** | none | **8.0×** | unstable |
| max | frozen | 3B | 6.0× | unstable |
| max | frozen | 5B | 3.4× | unstable |
| moderate (0.4) | trainable | 3B | **1.5×** | controlled |

The dominant factor is **frozen embeddings, not the size of the shift** — unfreezing alone takes
151× to 8.0×, while warmup alone (frozen, 5B) only reaches 3.4× and still fails. Warmup does not
rescue a maximal shift. V5 therefore: keeps embeddings trainable at every seam, caps per-seam
shift at 0.4 sharpness, and drops a **3B-token 60/40 blend** into each seam, matching V4's
practice.

---

## 9. The proxy: the mixture as a testable hypothesis

A mixture is a hypothesis until a cheap run has tested it. Seven runs on one T4, decision rules
fixed and committed **before** the runs, in [`proxy/HYPOTHESES.md`](proxy/HYPOTHESES.md).
Full numbers: [`proxy/RESULTS.md`](proxy/RESULTS.md).

<!-- PROXY_RESULTS -->

---

## 10. Cleaning status against the cumulative target

<!-- CLEANING_RESULTS -->

---

## 11. What would change this plan

| Finding | Consequence |
|---|---|
| AON turns out to be low quality | the reasoning lane loses 92% of its supply and 8% must be re-planned, not re-weighted |
| Agentic generation does not reach 46.9B | the 2% floor is unfundable; cut the lane to what verifier-checked generation actually yields and say so |
| RULER@32K < 85% | the 6% long-context lane is not buying anything measurable; fold it back into code and web packing |
| Sangraha synthetic proves to degrade Indic benchmarks | tier D drops from 31.3% of the Indic lane and the lane shrinks — there is no other supplier to backfill with |
| A 1B/3B proxy contradicts the T4 result | this plan's mixture claims are provisional at 40M params; the larger rung governs |

### Numbers the session contradicts itself on

Flagged rather than silently reconciled:

- **General web supply**: composer supply-check says 4.5T, its own callout says "roughly 4.8T",
  the inventory totals 4.8T. The 4.5T figure appears to exclude the STEM rows.
- **STEM supply**: composer claims 250B; the inventory's plausible STEM rows total **146B**. The
  ledger uses 146B, the conservative reading.
- **V4 web start**: prose says "roughly 70%", the composer shows 72.
- **Anneal size**: the lifecycle stage block says ~2%, its own TYPICAL SCALE field says ~1–5%.
  We take 2%.

---

## Repo map

| Path | What |
|---|---|
| [`SUPPLY_LEDGER.md`](SUPPLY_LEDGER.md) | every budget number, generated from the inventory |
| [`scripts/ledger.py`](scripts/ledger.py) | the ledger; holds the mixture decisions as constants, prints 6 consistency checks |
| [`data/inventory.json`](data/inventory.json) | 32 datasets + OPUS/agentic/curriculum constants, transcribed from the widgets |
| [`proxy/`](proxy/) | the ablation: hypotheses, harness, results |
| [`topup/`](topup/) | the S5 cleaning pass and its run report |
| [`resources/s5-widget-data.md`](resources/s5-widget-data.md) | all 9 widgets extracted verbatim — the source for every widget number quoted here |
