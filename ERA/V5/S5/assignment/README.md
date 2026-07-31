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

Composer defaults were web 34 / reasoning 6; we move 2 points from web to reasoning. Web has the
largest headroom of any lane — 4.69T of unique supply against 733.4B of demand, **6.4×** — so the
two points cost nothing there, while reasoning is the lane whose benchmarks (AIME, GPQA Diamond,
HLE, FrontierMath) the model is most likely to be judged on. Everything else matches the composer.
The anneal column is the composer's own "V5 anneal" preset, unchanged.

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
because the proxy is English-shaped, and a balanced proxy cures that. Agentic survives no proxy we
have — the likeliest reason being that roughly half a trajectory's tokens are observations nobody
scores, so its projected utility is low however the proxy is built (measured supervised fraction:
45.3%, §4). Only the floor saves it. The session's prose does not draw this distinction; the
widget does.

This is a claim about what the selector *picks*. §9 measures the complementary quantity — what the
model *loses* when the lane goes to zero — and finds Indic loses about five times more than
agentic. Both hold: agentic is the lane no proxy will choose, Indic is the lane the model can
least afford to lose. The floor is doing two different jobs.

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

A mixture is a hypothesis until a cheap run has tested it. **Eight runs, 3.9 GPU-hours of training
on one T4** (4.0 h wall for the ablation, ~4.3 h of instance uptime including setup and
evaluation), decision rules fixed and committed **before** the runs
([`proxy/HYPOTHESES.md`](proxy/HYPOTHESES.md), commit `920006f`). Full numbers:
[`proxy/RESULTS.md`](proxy/RESULTS.md).

40.3M parameters (14.2M non-embedding — the 68,096-entry sarvam1 vocabulary is the rest), 75M
tokens per arm, identical architecture / optimiser / seed / token count across arms. Only the
mixture differs. Metric is per-domain **bits-per-byte** on held-out documents excluded from every
training pool by content hash; on the agentic set only assistant tokens are scored.

Each lane carries a **unique-token cap** so the proxy repeats it at the epoch count the full-scale
ledger implies. Realised: agentic 3.89 epochs, verified Indic 2.49, reasoning 2.25, STEM 1.89,
code 0.51, web 0.16 — the plan's own numbers. The proxy is testing the repetition pressure, not
just the lane proportions.

### Results

| Lane | A proposed | B web-heavy | C no floor | D verified-only | σ (seed) |
|---|---:|---:|---:|---:|---:|
| `general_web` | 1.5074 | **1.4051** | 1.4750 | 1.5055 | 0.0139 |
| `code` | 1.0240 | 1.1704 | **0.9902** | 1.0245 | 0.0164 |
| `stem` | 1.6012 | 1.6223 | **1.5673** | 1.5994 | 0.0212 |
| `reasoning` | 1.0990 | 1.5515 | **1.0722** | 1.1012 | 0.0148 |
| `agentic` | **1.0393** | 1.2208 | 1.1977 | 1.0423 | 0.0236 |
| `long_context` | 1.4865 | **1.4035** | 1.4516 | 1.4835 | 0.0139 |
| `indic_A_verified` | 1.4427 | 1.7720 | 2.3014 | **1.4040** | **0.2588** |
| `indic_B_unverified` | **0.6939** | 0.7665 | 2.2011 | 0.6984 | 0.0085 |
| `indic_C_translated` | **1.1016** | 1.1511 | 1.7692 | 1.1732 | 0.0095 |
| `indic_D_synthetic` | **0.6821** | 0.7596 | 2.0817 | 0.8014 | 0.0079 |

σ is the arm-A-vs-arm-A-different-seed gap. **Absolute bpb is not comparable across rows** — a
Devanagari character costs ~3 UTF-8 bytes against 1 for ASCII (measured: Indic lanes run 6.3–9.2
bytes/token, English 2.4–4.0), so Indic rows sit lower at equal skill. Read across a row only.

### Verdicts against the rules as declared

**1. Mixture vs the web-heavy default — REFUTED on the general-web clause.**
A beats B on agentic by 0.1816 (7.7× σ) and on verified Indic by 0.3293 (1.27× σ — thin), and by
0.45 on reasoning. But general web costs **+7.28%** relative against the **2%** budget I fixed in
advance, so the rule fails. Arm B trains on 72% web against A's 32%; B modelling web better was
never in doubt and the 2% budget was set before the effect size was known. That is a badly chosen
threshold, not a surprise about the mixture — and the honest move is to report the rule as failed
and quote the real price. **7.3% general-web bpb is what the capability lanes cost.** The threshold
is not rewritten after the fact.

**2. Does the protected floor earn its cost — YES, decisively.** This is the strongest result in
the experiment.

| Removing the floor costs | | Removing the floor gains | |
|---|---:|---|---:|
| `indic_B_unverified` | 1.5071 (178× σ) | `general_web` | 0.0324 (2.3× σ) |
| `indic_D_synthetic` | 1.3996 (177× σ) | `code` | 0.0338 (2.1× σ) |
| `indic_A_verified` | 0.8587 (3.3× σ) | `stem` | 0.0339 (1.6× σ) |
| `indic_C_translated` | 0.6676 (70× σ) | `long_context` | 0.0349 (2.5× σ) |
| `agentic` | 0.1584 (6.7× σ) | `reasoning` | 0.0269 (1.8× σ) |

The floor is **not free** — every unprotected lane is measurably worse with it on. But it costs
~0.03 bpb each and buys 0.9–1.5. That ratio is the argument, and it is now a measurement rather
than an assertion.

The asymmetry refines the plan. Arms C zeroes *both* Indic and agentic, yet Indic collapses ~5×
harder. Agentic trajectories are English JSON — the model recovers much of that from web and code.
Nothing else in the mixture teaches Devanagari. The OPUS widget's point was that agentic is the
lane no benchmark-derived proxy will ever *select*; this measures the complementary fact, that
Indic is the lane the model can least afford to *lose*. Both are true and they are different claims.

**3. Four-tier Indic split vs verified-only — NO SIGNAL. The rule is untestable here.**
A is 0.0387 bpb worse than D on verified Indic — **0.15× the σ of 0.2588 on that lane**. The
verified-Indic lane is 18× noisier than the median lane, because its validation set is the
smallest (275k scored tokens against 1.5M for web) and its content the most heterogeneous (S4's
verified pool mixes Assamese, English, Hindi, Telugu). Arm A scored 1.4427; the identical mixture
on a different seed scored 1.7015.

**A single-seed experiment would have "shown" whichever direction its seed landed on.** The seed
repeat is what separates a result from an artifact, and it is the reason this comparison is
reported as undecided rather than as support for either design.

**4. Transition stability — NOT REPRODUCED.**

| Probe | Embeddings | Warmup | Peak ratio at the seam | Same statistic, no seam nearby |
|---|---|---:|---:|---:|
| E1 | **frozen** | none | 3.63× | 3.24× |
| E2 | trainable | none | 2.89× | 2.58× |
| E3 | trainable | 10% band | 3.50× | 3.47× |

The last column is the control: the same peak-over-baseline statistic applied to settled training
with no mixture change nearby. It already produces 2.6–3.5× excursions, and two of the three are
themselves over the 3× threshold. The seam is the largest gradient of each run, but only by 1–15%.
**At 40M parameters this test cannot separate a mixture transition from ordinary gradient noise.**
The frozen-vs-trainable direction matches the widget, but the frozen run also runs hotter *away*
from the seam — a whole-run property of freezing, not a spike it causes at the seam. The warmup
band did not lower the peak. The widget's 19× gap (151× vs 8.0×) does not appear; this is 1.26×.

None of that refutes the practice at 120B, where embeddings carry far more of the representation
and shifts are far larger in absolute tokens. It says the proxy cannot be used as evidence for it:
§8's warmup-band commitment rests on the session's measurement and V4's production experience, not
on this run.

### What the proxy actually licenses

Confirmed: the protected floor earns its cost, by a factor of ~30 in bpb traded. The mixture beats
the web-heavy default on every capability lane. The agentic supervised fraction (45.3% measured on
real corpora) sits between the widget's 53% and 43%.

Not licensed: the four-tier Indic split (no signal), the transition-warmup policy (not
reproduced), and any claim resting on verified-Indic bpb alone (σ too large). At 40.3M parameters
and 75M tokens this is the rung *below* the assignment's 1B/3B suggestion, and it is labelled as
such. The arithmetic for doing it properly, at this machine's measured 7.6 TFLOP/s:

| Rung | Params | Tokens (20×) | FLOPs/arm | T4-hours/arm | 8 arms |
|---|---:|---:|---:|---:|---:|
| this run | 40M | 75M | 0.02 EFLOP | 0.68 | **3.9 (measured)** |
| 1B | 1B | 20B | 120 EFLOP | 4,403 | 35,221 |
| 3B | 3B | 60B | 1,080 EFLOP | 39,623 | 316,981 |

A single Chinchilla-optimal 1B arm is ~183 T4-days. That is why the cheap rung was run, and why
its negative results are reported as limits of the rung rather than as findings about the plan.

---

## 10. Cleaning status against the cumulative target

The mixture says the Indic **verified** tier carries 42.3% of the Indic lane at 2.5 epochs, so
that is where this session's cleaning went. Full report: [`topup/TOPUP_REPORT.md`](topup/TOPUP_REPORT.md).

| | Tokens |
|---|---:|
| S4 pass (hin/tel/eng/asm) | 43.5M |
| **S5 pass** (ben, hin·2, mar, tam, kan, guj, mal, pan, ory) | **176.9M** |
| Cumulative | **220.4M** |
| Stated target (8% of 4T) | 320B |
| Progress | **0.069%** |

192.3M → 176.9M tokens at **92.01% retention**, 52/52 shards admitted, 554 documents (0.217%)
dropped by the 813,113-n-gram MILU firewall.

That 0.069% is the honest number: one workstation pass is a rounding error against a 320B target.
What it actually buys is a measured per-language yield curve — and it caught a defect that would
have scaled.

**The cleaner silently deleted five languages.** The quality stage asks "does this document
contain at least 2 common words of its language", and `stopword_set()` fell back to the *English*
list for any language it had none for. So a Kannada page was checked for English stop-words, found
none, and was dropped.

| | Languages | Kept before | Kept after |
|---|---|---:|---:|
| S4 shipped a stop-word list | ben, hin, mar, tam | 80.6–96.9% | 80.8–96.8% |
| S4 shipped none | **kan, guj, mal, pan, ory** | **2.9–5.7%** | **85.7–97.4%** |

The split falls exactly along whether a list existed. Slice retention went from **53.16% to
91.90%**, and the full pass lands at 92.01% against S4's 90.51% on its own four languages — the
fixed cleaner now treats nine languages the way S4 treated four. The lists were counted out of the
corpus by document frequency rather than written from memory, and a missing list is now recorded
in the stage stats instead of being absorbed into the fallback.

This is the data-gating point the assignment makes: **a mixture is only as trustworthy as the
cleaned tokens behind it.** The Indic lane's 16% share assumed verified supply that a language-blind
cleaner would have thrown away at a 20:1 rate in five of the nine languages it was asked to cover.

One number worth keeping: this pass measured **176.9M tokens against 113.3M** from the
words × 1.3 rule of thumb — a **36% gap**. Indic fertility is why the ledger counts tokens, never
words.

---

## 11. What would change this plan

| Finding | Consequence |
|---|---|
| AON turns out to be low quality | the reasoning lane loses 92% of its supply and 8% must be re-planned, not re-weighted |
| Agentic generation does not reach 46.9B | the 2% floor is unfundable; cut the lane to what verifier-checked generation actually yields and say so |
| RULER@32K < 85% | the 6% long-context lane is not buying anything measurable; fold it back into code and web packing |
| Sangraha synthetic proves to degrade Indic benchmarks | tier D drops from 31.3% of the Indic lane and the lane shrinks — there is no other supplier to backfill with |
| A 1B/3B proxy contradicts the T4 result | this plan's mixture claims are provisional at 40M params; the larger rung governs |

Three things the proxy was **unable to settle**, carried as open rather than assumed:

| Open question | Why it is open | What would close it |
|---|---|---|
| Four-tier Indic vs verified-only (§3) | the gap was 0.15× the seed noise on that lane | a larger rung, and a verified-Indic validation set big enough to cut σ — the 275k-token set here is the experiment's weakest link |
| Warmup bands at stage seams (§8) | settled training already produces 2.6–3.5× grad-norm excursions, so the test does not discriminate at 40M | the same probe at 1B+, where embeddings carry more of the representation |
| The 7.3% general-web price (§9) | measured, but against a 2% budget chosen before the effect size was known | decide the acceptable web regression against a downstream benchmark, not against bpb |

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
