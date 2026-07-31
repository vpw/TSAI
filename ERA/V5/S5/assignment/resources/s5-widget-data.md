# S5 Session widgets — real extracted data

Source: https://axiom.theschoolofai.in/courses/cmq97i5kn032208o8xu5dab4q/sessions/cmryiscq4002009rhmu56icmp/lesson
Extracted: 2026-07-30. Session title on page confirmed as "Session 5: Data Mixtures and Curriculum".

Nine interactive widgets, one per session section as listed below. Numbers below are transcribed
from the rendered widget state, not from the surrounding prose.

| # | Widget | § |
|---|---|---|
| 1 | V5 mixture composer | 2 |
| 2 | Benchmark explainer | 3 |
| 3 | Dataset inventory | 4 |
| 4 | Training lifecycle timeline | 5 |
| 5 | Agentic training trajectory | 6 |
| 6 | Reasoning effort levels | 7 |
| 7 | OPUS live view | 8 |
| 8 | Curriculum timeline | 9 |
| 9 | Gradient-norm trace | 10 |

---
## Widget 1: "Mixture Composer" (Section 2)

Badge: `Interactive · real budget math`

Header text (verbatim): "A data mixture is a **capability decision composed backward from the
benchmarks you intend to pass**. This is the **main pretraining run**, and it is general-web-heavy
because general web is by far the most abundant data. Move any slider and the rest renormalize so
the budget always sums to 100%. The scarce lanes, agentic and verified Indic and long reasoning,
are held small here on purpose and concentrated later in the short anneal phase."

Panel titles: "One compute budget, split across capability lanes" / "Every band is a slice of the
same fixed token budget" / "MIXTURE PROPORTIONS — total 100% · always renormalized" /
"CAPABILITY LANES · DRAG TO RECOMPOSE".

### Default state — preset "V5 pretraining (main run)"

| Lane | Share | Floor badge |
|---|---:|---|
| Code | 24% | — |
| Agentic / tool-use | 2% | `≥ 2%` |
| Reasoning traces | 6% | — |
| Long-context | 6% | — |
| Indic | 16% | `≥ 12%` |
| STEM / math | 12% | — |
| General web | 34% | — |

Sums to 100%. Slider order top-to-bottom is Code, Agentic, Reasoning, Long-context, Indic,
STEM/math, General web.

Status line: "**Funded lanes:** `Code`, `Indic`, `STEM / math`, `General web` ·
**Starved:** `Agentic / tool-use`, `Reasoning traces`, `Long-context`"

Callout "MAIN PRETRAINING RUN" (verbatim): "General web is the largest lane because it is the
**most abundant data we have**, roughly **4.8T tokens** available, far more than any other lane.
The scarce capabilities, agentic and verified Indic and long reasoning, are small here **on
purpose** and concentrated later in the short anneal phase. Load the **V5 anneal** preset to see
that concentrated final mix."

### "INDIC PROVENANCE (SESSION 3)" sub-panel — "tiers of the Indic slice"

**This is the four-tier split the assignment asks for, as the widget ships it:**

| Tier | Label | Share of the Indic slice |
|---|---|---:|
| A | verified native | **40%** |
| B | unverified crawl | **25%** |
| C | translated | **20%** |
| D | synthetic | **15%** |

Sub-line: "**Tier A** verified native pages: हिंदी सम्वाद, తెలుగు వార్తలు"

### "What this mixture buys" — lane → benchmark map (cyan = a lane that is funded)

| Lane | Benchmarks |
|---|---|
| Code | `LiveCodeBench`, `Aider` |
| Agentic / tool-use | `SWE-bench`, `tau-bench`, `BFCL`, `GAIA`, `BrowseComp` |
| Reasoning + STEM | `AIME`, `GPQA`, `HLE` |
| Long-context | `long-eval` |
| Indic | `MILU`, `IndicGenBench` |
| General web | `MMLU` |

### "Supply check" — "demand = share × run size, versus real supply"

RUN size selector: `1T` / **`2T` (default)** / `5T` / `10T`.

At RUN = 2T:

| Lane | Demand | Supply | Verdict |
|---|---:|---:|---|
| Code | 480B | 1.1T | `covered` |
| Agentic / tool-use | 40B | **0.63B** | `must synthesize` |
| Reasoning traces | 120B | 85B | `needs repetition` |
| Long-context | 120B | 100B | `needs repetition` |
| Indic | 320B | 276B | `needs repetition` |
| STEM / math | 240B | 250B | `covered` |
| General web | 680B | 4.5T | `covered` |

Footer (verbatim): "At a 2T run, **agentic is the binding constraint, it must be synthesized**;
the scarce lanes need repetition and synthetic generation to fill even these small shares."

Note: agentic supply reads **0.63B tokens** against a 40B demand at the *floor* share of 2% —
a 63× shortfall. This is the single most important number in the widget.

### "PROTECTED FLOOR · THE DELIBERATE DECISION" callout (verbatim)

"Two lanes are never starved to zero: `Indic ≥ 12%` and `Agentic ≥ 2%`. Drag either toward zero
and it clamps at the floor while the others absorb the rest. The floor is a small guarantee, not
a large share, long-tail capability is a decision, not a residue."

### "Tie-in to V4, weights are a schedule not a constant" callout (verbatim)

"V4's real stage weights: Web faded `72→18`, Code ramped `13→35`, STEM ramped `7→39`, and the
**Always-On lane stayed pinned at 8%** the whole run. The protected floor here is that Always-On
lane, made explicit."

(Note: the session prose rounds the web start to "roughly 70%"; the widget says **72**.)

### Widget footer (verbatim)

"The pretraining mix is a first-pass hypothesis refined by 1B/3B proxy ablation. Supply numbers
are the approximate token totals from the Dataset Inventory."

### Presets available
`V5 pretraining` (main run) · `V5 anneal` (final short phase) · `Naive web-heavy` (crawl what is cheap)

### Preset "V5 anneal — final short phase"

| Lane | Share |
|---|---:|
| Code | 20% |
| Agentic / tool-use | 8% |
| Reasoning traces | 18% |
| Long-context | 8% |
| Indic | 28% |
| STEM / math | 10% |
| General web | 8% |

Sums to 100%. Status line: "**Funded lanes:** `Code`, `Agentic / tool-use`, `Reasoning traces`,
`Long-context`, `Indic`, `STEM / math`, `General web` · **every lane funded**"

Supply check (run selector still at 2T, so these demands are 2T × the anneal shares — the widget
does not automatically switch the run size when you load the anneal preset):

| Lane | Demand | Supply | Verdict |
|---|---:|---:|---|
| Code | 400B | 1.1T | `covered` |
| Agentic / tool-use | 160B | 0.63B | `must synthesize` |
| Reasoning traces | 360B | 85B | `must synthesize` |
| Long-context | 160B | 100B | `needs repetition` |
| Indic | 560B | 276B | `needs repetition` |
| STEM / math | 200B | 250B | `covered` |
| General web | 160B | 4.5T | `covered` |

### Preset "Naive web-heavy — crawl what is cheap"

| Lane | Share |
|---|---:|
| Code | 18% |
| Agentic / tool-use | 2% |
| Reasoning traces | 4% |
| Long-context | 2% |
| Indic | 6% |
| STEM / math | 8% |
| General web | **60%** |

Status line: "**Funded lanes:** `Code`, `STEM / math`, `General web` · **Starved:**
`Agentic / tool-use`, `Reasoning traces`, `Long-context`, `Indic`"

The protected-floor callout **turns red and changes text** under this preset —
"**PROTECTED FLOOR BREACHED**" (verbatim): "This preset pushes **Indic to 6%** and
**Agentic to 2%**, below the floor of `Indic ≥ 12%` and `Agentic ≥ 2%`. The floor means these
scarce lanes are never starved to zero, not that they take a large share. Dragging a slider
re-imposes the floor."

Supply check at 2T under the naive preset — note that almost everything reads `covered`, because
the cheap mixture demands so little of the scarce lanes:

| Lane | Demand | Supply | Verdict |
|---|---:|---:|---|
| Code | 360B | 1.1T | `covered` |
| Agentic / tool-use | 40B | 0.63B | `must synthesize` |
| Reasoning traces | 80B | 85B | `covered` |
| Long-context | 40B | 100B | `covered` |
| Indic | 120B | 276B | `covered` |
| STEM / math | 160B | 250B | `covered` |
| General web | 1.2T | 4.5T | `covered` |

**This is the trap the session is built around**: the naive mixture is the *only* one whose supply
check is nearly all green, because it asks almost nothing of the lanes that are hard to fill.
"Covered" is a statement about ambition, not about quality.

### Run-size sweep — supply is constant, demand scales

Back on the `V5 pretraining` preset, switching RUN to **10T** leaves every supply figure unchanged
(1.1T / 0.63B / 85B / 100B / 276B / 250B / 4.5T) and scales demand by 5×:

| Lane | Demand @10T | Supply | Verdict |
|---|---:|---:|---|
| Code | 2.4T | 1.1T | `needs repetition` |
| Agentic / tool-use | 200B | 0.63B | `must synthesize` |
| Reasoning traces | 600B | 85B | `must synthesize` |
| Long-context | 600B | 100B | `must synthesize` |
| Indic | 1.6T | 276B | `must synthesize` |
| STEM / math | 1.2T | 250B | `must synthesize` |
| General web | 3.4T | 4.5T | `covered` |

Footer changes to: "At a 10T run, agentic is the binding constraint, it must be synthesized…"

**Consequence for the plan: at 10T only general web is covered. The trained-token budget is itself
bounded by supply, not by compute.** At 2T, three lanes are covered; at 10T, one.

### Controls that did not respond
None outstanding. The `Naive web-heavy` preset button needed three click attempts at different
points inside the button before it registered (it fired at (1300,534), not at the visual centre);
the `V5 anneal` and `V5 pretraining` buttons fired first time. Sliders were read at their preset
values rather than dragged, since the three presets already give three full mixtures.

---
## Widget 2: "Benchmark Explainer" (Section 3)

Badge: `Interactive · real samples`

Header (verbatim): "…benchmark named in Session 5, in one place: what it is, what it tests, a real
sample task, and, critically, the **loss map** of the matching training example. The colouring
shows which tokens the model is actually trained on. The recurring lesson: the model is **never
trained to imitate the environment**. Only its own tokens are green; the issue text, tool returns,
and page observations stay grey."

Footnote (verbatim): "Sample tasks are illustrative and paraphrased. Version-specific task counts
are approximate and being verified. Loss-map colouring is driven by the per-block
**supervised / masked / reward** tag in the data, not painted by hand."

Loss-map legend, used identically on every benchmark:
`green · supervised (in the loss)` / `grey · masked context / observation` /
`violet · reward-only (no token loss)`

### Full benchmark list — 18 benchmarks in 4 groups

**AGENTIC & TOOL-USE (9)**
| Benchmark | One-line description as rendered |
|---|---|
| SWE-bench Verified | Human-validated GitHub issues; write a patch that passes hidden tests |
| SWE-bench Live / Pro | Fresher, contamination-resistant, harder SWE-bench variants |
| Terminal-Bench | Tasks completed in a real terminal / shell |
| tau-bench / tau2-bench | Tool-agent-user interaction under a policy (retail, airline) |
| BFCL v3 | Berkeley Function-Calling Leaderboard: single / parallel / multi-turn |
| WebArena / WorkArena | Self-hosted web sites and enterprise workflows |
| GAIA | General-assistant multi-step questions needing tools + browsing |
| BrowseComp | Hard, verifiable web-browsing for hard-to-locate facts |
| OSWorld | Real computer-use tasks across desktop apps |

**CODING (3)**
| Benchmark | Description |
|---|---|
| LiveCodeBench | Competition-style coding collected over time to avoid contamination |
| Aider Polyglot | Real code-editing across many languages in a diff format |
| Codeforces | Competitive programming mapped to an ELO rating |

**REASONING & MATH (4)**
| Benchmark | Description |
|---|---|
| AIME (2024 / 2025) | Competition math, integer answers, no tools |
| FrontierMath | Research-level math (Epoch AI), extremely hard |
| GPQA Diamond | Graduate-level google-proof science MCQ |
| HLE (Humanity's Last Exam) | Very hard, broad, expert benchmark |

**INDIC (2)**
| Benchmark | Description |
|---|---|
| MILU (AI4Bharat) | Multi-task Indic understanding across many languages and subjects |
| IndicGenBench (Google) | Generation across **29 Indic languages (13 scripts, 4 families)**: summarization, translation, QA |

**IMPORTANT GAP — there is no LONG-CONTEXT group in the explainer.** The mixture composer points
the long-context lane at a benchmark it calls `long-eval`, but no such entry exists in this
widget, and no RULER / needle-in-a-haystack / LongBench entry exists either. The long-context lane
is the only lane in the composer whose benchmark is not defined anywhere in the session. Any plan
that funds a long-context lane has to name its own eval.

### Inspected benchmark 1 of 18 — SWE-bench Verified (Agentic & Tool-use)

- **WHAT IT IS**: "Human-validated real GitHub issues paired with the actual repository; the agent must write a patch."
- **WHAT IT TESTS**: "Repo-level bug fixing: navigate a real codebase, localise the fault, and edit code that makes hidden tests pass."
- **METRIC**: `% resolved (pass@1) · 500 tasks (engineer-verified solvable)`
- **SAMPLE TASK — PROMPT**: "Issue: TypeError when passing a list to Config.foo(); .foo() assumes a str. Repo: django/django @ a1b2c3 (full checkout provided)."
- **EXPECTED ANSWER / SUCCESS CONDITION**: "A unified diff that turns the failing hidden test suite green (all tests pass)."

LOSS MAP — matching training example:

| Block | Colour | Content |
|---|---|---|
| issue text | grey | `TypeError when passing a list to Config.foo()` |
| repo files | grey | `django/django @ a1b2c3 (full checkout)` |
| assistant reasoning | **green** | `foo() calls s.strip(); a list has no .strip(). Guard the type.` |
| assistant patch | **green** | `@@ def foo(self, s):` / `-  return s.strip()` / `+  if isinstance(s, list): s = ' '.join(s)` / `+  return s.strip()` |
| test output | grey | `pytest: 42 passed` |
| REWARD +1 | violet | "the verifier runs the hidden tests and scores +1 if they pass; the outcome carries no token loss" |

Summary line (verbatim): "Supervise **only** the model's patch and its reasoning (green). The issue
text, repo files, and test output are environment context (grey). The pass/fail is a reward-only
signal (violet), never imitated token by token."

### Inspected benchmark 2 of 18 — Codeforces (Coding)

- **METRIC**: `ELO / rating`
- **SAMPLE TASK — PROMPT**: "Div 2: count pairs (i, j) with a[i] + a[j] divisible by k. n <= 2e5."
- **SUCCESS CONDITION**: "A program accepted on all tests within the time limit."

LOSS MAP:

| Block | Colour | Content |
|---|---|---|
| problem statement | grey | `count pairs whose sum is divisible by k` |
| assistant reasoning | **green** | `bucket values by remainder mod k, then combine complementary buckets` |
| assistant program | **green** | `# read a, k; count[r]++ for r=x%k;` / `# ans += count[0]*(count[0]-1)/2 + sum count[r]*count[k-r]/2` |
| REWARD +1 | violet | "accepted verdict maps to a rating; the judge's response is never in the loss" |

Summary: "Supervise the submitted program and reasoning (green). The problem statement is context
(grey). The verdict drives a violet, rating-based reward."

### Inspected benchmark 3 of 18 — HLE / Humanity's Last Exam (Reasoning & Math)

- **SAMPLE TASK — PROMPT**: "An expert-level question drawn from any field, with a short verifiable answer."
- **SUCCESS CONDITION**: "…match to the reference answer." (exact-match)

LOSS MAP:

| Block | Colour | Content |
|---|---|---|
| question | grey | `expert-level question, any field` |
| chain-of-thought | **green** | `the expert derivation toward the answer` |
| final answer | **green** | `the short verifiable answer` |
| REWARD +1 | violet | "exact-match verifier scores +1; the question is never in the loss" |

Summary: "As GPQA: supervise the reasoning and answer (green); the question is context (grey);
the checked answer is a violet reward."

### Controls that did not respond
The right-hand benchmark list does **not** scroll with the mouse wheel, and dragging its scrollbar
thumb had no effect. It does scroll when an item is clicked and the keyboard `Down` key is then
pressed — that is how the full list was enumerated. 3 of 18 benchmarks were opened in detail
(one per group except Indic); the remaining 15 are captured by name, description and group only.

---
## Widget 3: "SOTA Dataset Inventory" (Section 4)

Badge: `Interactive · samples and tokens`

Header (verbatim): "Before you weight a mixture, you have to know what actually exists. Here are the
real datasets feeding each capability slot, sized in **both samples and tokens**, with sources and
tiers. Flip the view: a small-sample agentic trajectory set jumps to the top on tokens while a
big-sample function-call set drops. **Sample counts and token counts tell different stories**, and
the slot totals show you cannot weight a lane past the tokens that exist for it."

Panel: "Real datasets per capability slot — Click any row to inspect it. Toggle the view to
re-sort and re-scale the size bars." View toggle: **`Samples view` (default, size + sort by
samples)** / `Tokens view` (size + sort by tokens).

Legend: `size bar = current view, scaled within its slot` · `tier A` · `tier B` · `tier C` ·
`tier D` · `slot flagged thin = scarcity`

Footnote (verbatim): "Dataset sizes are approximate and being verified; sample vs token counts
intentionally differ. **Sangraha and V4 numbers are confirmed from our sources.**"

Row badges seen: `approx` (most rows), `confirmed` (all V4-lineage rows + all three Sangraha
rows), `synthetic` (Sangraha synthetic only).

### THE FULL INVENTORY — 32 datasets, 6 slots

**CODE — 3 sets · slot supply `1.1T` tokens**

| Dataset | Source | Samples | Tokens | License | Tier |
|---|---|---:|---:|---|---|
| The Stack v2 | BigCode / Software Heritage | 600M | 900B | permissive + opt-out | B |
| D3 Code (V4 corpus) *confirmed* | V4 run | 250M | 199B | V4 lineage | B |
| CommitPack / CommitPackFT | BigCode | 4M | 4B | mixed / permissive | B |

**AGENTIC & TOOL-USE — 9 sets · slot supply `627M` tokens · flagged `SLOT RUNS THIN`**

| Dataset | Source | Samples | Tokens | License | Tier |
|---|---|---:|---:|---|---|
| ToolBench | OpenBMB | 120K | 80M | Apache-2.0 | D |
| Glaive function-calling v2 | Glaive | 113K | 50M | Apache-2.0 | D |
| ToolACE | ToolACE | 110K | 60M | Apache-2.0 | A/D |
| xLAM / APIGen | Salesforce | 60K | 25M | CC-BY (mixed) | A/D |
| Nexus / NexusRaven | Nexusflow | 40K | 30M | CC-BY-4.0 | A |
| SWE-smith | SWE-smith | 26K | 120M | task licenses | A |
| Hermes function-calling | NousResearch | 15K | 22M | Apache-2.0 | A/D |
| OpenHands rollouts | All-Hands / OpenHands | 10K | 90M | mixed | A |
| SWE-Gym | SWE-Gym | 2.4K | 150M | task licenses | A |

Sums to 627M. **The entire agentic slot is smaller than 0.03% of the code slot.**
Note the sample/token inversion the header promises: SWE-Gym has the *fewest* samples (2.4K) and
the *most* tokens (150M); ToolBench has the most samples (120K) and only 80M tokens.

**REASONING & MATH — 5 sets · slot supply `85.1B` tokens**

| Dataset | Source | Samples | Tokens | License | Tier |
|---|---|---:|---:|---|---|
| AON (V4 corpus) *confirmed* | V4 run | 40M | 78B | V4 lineage | A |
| OpenMathReasoning | NVIDIA | 3.2M | 2B | CC-BY-4.0 | A/D |
| OpenThoughts2 | OpenThoughts | 1.1M | 3B | Apache-2.0 | A/D |
| NuminaMath | Numina | 860K | 500M | Apache / CC-BY | A |
| OpenR1-Math (R1-distilled) | Hugging Face | 220K | 1.6B | Apache-2.0 | D |

Sums to 85.1B. **92% of this slot is one V4-lineage corpus (AON, 78B).** The open reasoning-trace
sets together contribute only ~7.1B.

**LONG-CONTEXT — 2 sets · slot supply `100B` tokens**

| Dataset | Source | Samples | Tokens | License | Tier |
|---|---|---:|---:|---|---|
| Repo-packed code (32K+) | packed from code corpora | 1.5M | 60B | permissive | B |
| Book-length corpora (packed) | books + long docs | 400K | 40B | mixed / public-domain | B |

Note: neither is an independent corpus — both are **packings of data already counted in other
slots** (code, books). Long-context supply is a re-shaping of existing tokens, not new supply.

**INDIC — 6 sets · slot supply `276B` tokens · flagged `SLOT RUNS THIN`**

| Dataset | Source | Samples | Tokens | License | Tier |
|---|---|---:|---:|---|---|
| Sangraha (synthetic) *synthetic* | AI4Bharat | 90M | **162B** | CC-BY-4.0 | C |
| Samanantar | AI4Bharat | 49.7M | 2B | CC0 / CC-BY | C |
| Sangraha (verified) *confirmed* | AI4Bharat | 40M | **64B** | CC-BY-4.0 | A |
| BPCC (parallel) | AI4Bharat | 22M | 3B | CC-BY-4.0 | C |
| Sangraha (unverified) *confirmed* | AI4Bharat | 15M | **24B** | CC-BY-4.0 | B |
| IndicCorpV2 | AI4Bharat | 10M | 20.9B | CC-BY / mixed | B |

Sums to 275.9B ≈ 276B. **Every Indic dataset in the inventory is AI4Bharat.** There is no second
supplier for the flagship capability.

**GENERAL WEB & STEM — 7 sets · slot supply `4.8T` tokens**

| Dataset | Source | Samples | Tokens | License | Tier |
|---|---|---:|---:|---|---|
| DCLM-Baseline | DataComp-LM | 2.6B | 2.6T | mixed / CommonCrawl | B |
| FineWeb-Edu | Hugging Face | 1.3B | 1.3T | ODC-By | B |
| D2 Web-Diverse (V4) *confirmed* | V4 run | 780M | 627B | V4 lineage | B |
| D1 Web-Foundation (V4) *confirmed* | V4 run | 200M | 164B | V4 lineage | B |
| D4 STEM (V4) *confirmed* | V4 run | 60M | 49B | V4 lineage | B |
| peS2o | AI2 | 40M | 42B | ODC-By | A |
| proof-pile-2 | EleutherAI | 5M | 55B | mixed | A |

Sums to 4.837T ≈ 4.8T.

### "Running per-slot token total" (labelled `real supply`)

| Slot | Real supply |
|---|---:|
| Code | 1.1T |
| Agentic & tool-use | 627M |
| Reasoning & math | 85.1B |
| Long-context | 100B |
| Indic | 276B |
| General web & STEM | 4.8T |

Caption (verbatim): "You cannot weight a slot beyond the tokens that **exist** for it. Supply, not
preference, is the hard cap on the mixture. General web and STEM dwarf every specialist slot."

### "DESIGN NOTE · WHERE THE SLOTS RUN THIN" callout (verbatim)

**"Indic and Agentic cannot be filled from real supply alone."**

"The Agentic slot holds only about **627M** tokens of real supply, and Indic's trusted,
non-synthetic text is about **114B** against **162B** synthetic. These are the lanes the
**protected sampler and synthetic generation must cover**; you cannot simply up-weight past what
exists."

Two stat tiles: `AGENTIC REAL` = **627M tokens** · `INDIC REAL VS SYNTH` = **114B / 162B**

**The 114B checks out against the table**: 64 (Sangraha verified) + 24 (Sangraha unverified) +
20.9 (IndicCorpV2) + 3 (BPCC) + 2 (Samanantar) = 113.9B real; the remaining 162B is Sangraha
synthetic. So **59% of the Indic slot as inventoried is already synthetic** before we generate a
single new token.

### Inspected rows

**The Stack v2** (Code) — WHAT IT IS: "BigCode's deduplicated permissive-source code corpus built
from Software Heritage." WHY IT IS IN THIS SLOT: "Anchors the Code slot with real repository and
file-level code across hundreds of languages." Samples 600M · Tokens 900B · License
`permissive + opt-out` · Tier B. KEY METADATA / CHECKS: `repo + file granularity`,
`near-dedup applied`, `opt-out honored`.

**OpenHands rollouts** (Agentic) — WHAT IT IS: "Agent rollouts collected from the OpenHands
framework." WHY IT IS IN THIS SLOT: "End-to-end tool-use episodes inside a real coding agent."
Samples 10K · Tokens 90M · License `mixed` · Tier A. CHECKS: `~10K rollouts`, `full episodes`,
`mixed licenses`.

**OpenR1-Math (R1-distilled)** (Reasoning) — WHAT IT IS: "R1-distilled long math reasoning
traces." WHY IT IS IN THIS SLOT: "Long distilled traces: fewer samples, far more tokens each."
Samples 220K · Tokens 1.6B · License `Apache-2.0` · Tier **D**. CHECKS: `~220K samples`,
`very long traces`, `tier D`.

**Sangraha (verified)** (Indic) — WHAT IT IS: "AI4Bharat's human-verified Indic web and
native-source slice." WHY IT IS IN THIS SLOT: "**The trustworthy core of the Indic slot: our
protected lane.**" Samples 40M · Tokens 64B · License `CC-BY-4.0` · Tier A. CHECKS: `~64B tokens`,
`verified`, `confirmed`.

**D2 Web-Diverse (V4)** (General web) — Samples 780M · Tokens 627B · License `V4 lineage` ·
Tier B. CHECKS: `confirmed`, `~627B tokens`.

### Cross-check against the mixture composer

The composer's supply-check column matches this widget exactly for Code (1.1T), Agentic (0.63B =
627M), Reasoning (85B), Long-context (100B) and Indic (276B). The composer splits this widget's
single 4.8T "General web & STEM" slot into **General web 4.5T + STEM/math 250B**. The inventory
does not label which of its 7 rows are STEM; the plausible STEM rows (D4 STEM 49B + peS2o 42B +
proof-pile-2 55B = 146B) do **not** add to 250B, so the composer's STEM supply figure cannot be
reconstructed from the inventory. Flag this as an inconsistency rather than resolving it silently.

### Controls that did not respond
The `Tokens view` toggle was not exercised — the table rows do not respond to the mouse wheel and
the enumeration had to be driven by click-a-row-then-press-`Down`, which re-sorts nothing. Both
the samples and the tokens column are captured for every one of the 32 rows above, so the toggle
would only change ordering and bar scaling, not data.

---
## Widget 4: "Training Stages Lifecycle" (Section 5)

Badge: `Interactive · the whole run`

Header (verbatim): "The whole V5 run as an ordered sequence of stages. Each block's **width is
roughly its token budget**, so pretraining dominates and every post-training stage is visibly tiny.
**Click a stage to expand it** and read its objective, unit, loss and loss map. The supervised
signal changes shape as you move right, from pretraining's fully green map, to SFT's response-only
mask, to a single violet reward at RLVR. The violet marker shows exactly **when reasoning enters**."

Rail label: `Pretraining → anneal → SFT → reasoning → preference → serving`

### The six stages and their token budgets

| # | Stage | Token budget as rendered |
|---|---|---|
| 1 | Pretraining | **~95% of tokens** |
| 2 | Mid-training / Anneal | **~2% of tokens** |
| 3 | SFT | **<1% of tokens** |
| 4 | Reasoning training | **<1% of tokens** |
| 5 | Preference alignment | **<1% of tokens** |
| 6 | Serving | **not training** |

Violet marker sits between stages 2 and 3: "**reasoning enters here** — after the base model
exists". Left edge of the rail is labelled "base model is built here"; right edge "…all tokens".

Footer: "Six stages in order. Pretraining builds the base; everything after is a thin correction
on top, and **reasoning enters at stage 4**." · "Stage budgets are approximate. Losses are
acquaintance-level. Post-training data shapes match the modern open-model standard."

### LOSS-MAP COLORS legend (widget-wide)
- **Supervised token.** Contributes to the cross-entropy loss. (green)
- **Masked context / no loss.** Prompt, problem, or observation. (grey)
- **Reward-only.** Verifier scores the outcome; no token loss. (violet)
- **Selected stage.** The stage currently expanded on the left. (cyan)

### "V4 REALITY" callout (verbatim)

"V4's run went **seed → dense-to-MoE growth → medium → large → mid-training anneal**.
**Reasoning training and agentic RL are exactly the post-training stages V4 barely had**, which V5
builds out."

### Expanded stage 2 / 6 — Mid-training / Anneal · budget `~2% of tokens`

| Field | Content (verbatim) |
|---|---|
| OBJECTIVE | "A short low learning-rate cooldown on a reserved high-quality mix." |
| UNIT · SHAPE OF ONE DATA POINT | "A high-quality document (the best reserved data)." |
| LOSS / REWARD | "Next-token **cross-entropy**, still every token, but only on the best reserved mix." |
| TYPICAL SCALE | "**~1 to 5% of total tokens.**" |
| ONE-LINE RULE | "**Save the best for last.**" |

LOSS MAP FOR ONE EXAMPLE — "loss on every token · best data only", block titled
`RESERVED HIGH-QUALITY DOCUMENT`, every token green:
"In a valid proof, each step follows from the last by a stated rule, and the final line is the
theorem."

Note under the map (verbatim): "Same all-green cross-entropy as pretraining, but the data is
**curated and reserved for the end** and the learning rate is decayed. **Quality of the mix, not
the loss shape, is what changes.**"

### Controls not exercised
Stages 1, 3, 4, 5 and 6 were not expanded (one stage opens at a time; the anneal was chosen because
it is the stage this assignment must declare a reserve for). The `← Prev` / `Next →` buttons were
not used.

---
## Widget 5: "Agentic Trajectory Walk-through" (Section 6)

Badge: `Interactive · step through a real agent run`

Header (verbatim): "A Codex-class agentic training example is not a single answer. It is a long
multi-step tool-calling **trajectory**: plan, call, read the environment back, hit a failure,
recover, and finish. Step through one real run and watch which tokens actually train the model.
**Only the assistant's own tokens turn green.**"

Two tabs: **`Long trajectory`** (~10 steps · tool use + recovery) / **`One-shot call`**
(BFCL-style · schema + 1 call). Controls: `Next step → reveal step N`, `Reveal all`, `Restart`.

THE TASK GIVEN TO THE AGENT (verbatim): "Find every US research grant matching a given project,
then find the people and labs who won each grant, and judge **which winners could buy the piece of
hardware I own**, because it is exactly the machine their funded work needs."

### The supervised/context token counts — the number this widget exists to produce

| View | TOKENS TRAINED ON | TOKENS SEEN | % of seen tokens supervised |
|---|---:|---:|---:|
| **Long trajectory** (fully revealed) | **356** | **668** | **53%** |
| **One-shot call** (fully revealed) | **42** | **98** | **43%** |

So a full ~10-step agent run with a failure and a recovery yields **356 supervised tokens**. A
BFCL-style single function call yields **42**.

A revealed step showing the failure/recovery (verbatim from the trajectory):
```
ASSISTANT tool call (failed)                          trained · call failed
fetch_org_profile({ org: "Ridgeline Structural Biology Lab" })
ERROR 429: rate_limited, org profile service unavailable, retry later
```

### One-shot call contents (verbatim)

Panel title: "The one-shot contrast: schema + a single call — BFCL-style baseline · only the call
is supervised"

| Block | Tag | Content |
|---|---|---|
| FUNCTION SCHEMA | `masked (given)` | `search_grants(query: str, agency: str, topic: str) -> list[{grant_id, title}]` / `"Search US research grants by topic."` |
| USER | `masked (user)` | "Find NSF grants about cryo-EM motion correction." |
| ASSISTANT FUNCTION CALL | **`trained`** | `search_grants({ query: "cryo-EM motion correction", agency: "NSF", topic: "structural biology" })` |

Note under it (verbatim): "This BFCL-style example supervises exactly **one** function call against
a fixed schema. There is **no observation, no failure, no recovery, no plan**. It teaches the
**call format** cheaply, but it never teaches the long chain of decisions a real agent run demands.
That chain only exists in the long trajectory on the other tab."

### Callouts (verbatim)

**THE MASKING RULE**: "Only the assistant's own tokens (its **reasoning**, its **tool-call
arguments**, its **final answer**) are supervised. User turns and tool observations are **masked**,
because the model must never learn to invent the output of a tool it has not really run."

**WHY THIS DATA IS RARE**: "These long trajectories are scarce **Tier-A agentic data**, reserved
for the **annealing stage**. A whole run yields only a few hundred supervised tokens, so every
clean trajectory is expensive to collect."

**CONTRAST TAB**: "Switch to **One-shot call** to see the BFCL-style baseline: a schema plus a
single function call. It teaches the format, but never the long chain of plan, observation and
recovery a real agent needs."

Footnote: "The trajectory is an illustrative example. Token weights are approximate; the masking
rule is exact."

### Why this matters for the budget arithmetic

The inventory's agentic slot is 627M **raw** tokens. At the trajectory's own 53% supervised rate,
only ~330M of those carry any loss at all — and the function-calling-heavy sets (ToolBench, Glaive,
ToolACE, xLAM, Hermes = 237M raw) sit nearer the one-shot 43% rate. Any agentic share must be
defended in supervised tokens, not raw tokens.

---
## Widget 6: "Reasoning-effort tiers" (Section 7)

Badge: `Interactive · the effort dial`

Header (verbatim): "The same problem, solved four ways. Low, medium, high and ultra are not a
single inference flag you flip. They are **earned in training** by exposure to [traces from short]
to long. Slide the effort dial and watch one chain of thought grow, with the solve rate rising then
flattening, so ultra is **wasted effort** on an easy problem."

Sub-caption: "Reasoning tokens are shown green, the supervised chain that trains".

**THE PROBLEM (IDENTICAL AT EVERY TIER)** (verbatim): "How many integers between **1 and 1000**
are divisible by **3 or 5**? The correct answer is **467**."

Answer panel at every tier: `ANSWER 467` — "correct at every tier; only the **solve rate** changes".

### THE FOUR BANDS — the numbers the assignment needs

| Tier | Sub-label | Reasoning steps | THINKING BUDGET / reasoning tokens | Approx solve rate | Effort verdict |
|---|---|---:|---:|---:|---|
| **LOW** | near-direct | 2 | **37 tok** | **62%** | `thin but cheap` — "fast, lowest solve rate" |
| **MEDIUM** | a few steps | 4 | **74 tok** | **77%** | `well spent` — "reasoning still buys accuracy here" |
| **HIGH** | derive + check | 6 | **161 tok** | **91%** | `well spent` — "reasoning still buys accuracy here" |
| **ULTRA** | deliberate + verify | 14 | **346 tok** | **95%** | `wasted effort` — "only +4 pts over HIGH on an easy problem" |

Stat-tile labels: `REASONING TOKENS` ("length of the supervised trace") · `APPROX SOLVE RATE`
("approx, test-time scaling (illustrative)") · `EFFORT VERDICT`.

Dial note (verbatim): "The budget maps to trace length. Drag it and it **snaps to the nearest
tier**, because effort is quantised by what the model was trained on, **not a smooth knob**."

Chart: "Solve rate vs reasoning length — `rises, then plateaus`", legend `accuracy curve` /
`selected tier` / `wasted zone`; the plateau band is shaded and labelled "plateau, extra length
wasted", and ULTRA's marker sits inside it.

### The four chains of thought, verbatim (with per-step token counts)

**LOW — 2 steps, 37 tok**
1. `Inclusion-exclusion: floor(1000/3) + floor(1000/5) - floor(1000/15).` — 25 tok
2. `= 333 + 200 - 66.` — 7 tok

**MEDIUM — 4 steps, 74 tok**
1. `Count multiples of 3 up to 1000: floor(1000/3) = 333.` — 17 tok
2. `Count multiples of 5 up to 1000: floor(1000/5) = 200.` — 17 tok
3. `Divisible by both 3 and 5 means divisible by 15: floor(1000/15) = 66, counted twice.` — 23 tok
4. `Inclusion-exclusion: 333 + 200 - 66 = 467.` — 12 tok

**HIGH — 6 steps, 161 tok**
1. `Let A = multiples of 3 and B = multiples of 5 inside [1, 1000].` — 19 tok
2. `|A| = floor(1000/3) = 333, since 3 times 333 = 999 which is at most 1000.` — 25 tok
3. `|B| = floor(1000/5) = 200, since 5 times 200 = 1000 exactly.` — 21 tok
4. `A and B together means divisible by lcm(3,5) = 15, so |A and B| = floor(1000/15) = 66, since 15 times 66 = 990.` — 39 tok
5. `Inclusion-exclusion: |A or B| = 333 + 200 - 66 = 467.` — 18 tok
6. `Check by complement: divisible by neither is about 1000 times (2/3) times (4/5) = 533, and 1000 - 533 = 467. Consistent.` — 34 tok

**ULTRA — 14 steps, 346 tok** (first 8 captured; the step list scrolls)
1. `Restate the goal: count n with 1 at most n at most 1000 such that 3 divides n or 5 divides n.` — 24 tok
2. `Plan: apply inclusion-exclusion to the two divisibility events, then verify two independent ways.` — 18 tok
3. `Multiples of 3: the largest at most 1000 is 999 = 3 times 333, the smallest is 3, so there are exactly 333.` — 27 tok
4. `Multiples of 5: the largest at most 1000 is 1000 = 5 times 200, the smallest is 5, so there are exactly 200.` — 27 tok
5. `The overlap, divisible by both 3 and 5, means divisible by lcm(3,5) = 15.` — 22 tok
6. `Multiples of 15: the largest at most 1000 is 990 = 15 times 66, so there are exactly 66.` — 22 tok
7. `A naive sum 333 + 200 = 533 double counts the 66 multiples of 15.` — 16 tok
8. `So |A or B| = 533 - 66 = 467.` — 13 tok

### "THE TRAINING REALITY · Effort is a mixture decision" callout (verbatim)

"The training mixture holds reasoning traces **binned by length**, from short to very long, across
**math, code and general problem solving**. The curriculum grades them from short to long. The
later **RLVR** stage turns this exposure into a dial the caller can actually set: low, medium,
high, ultra. **Reserve traces across the full length and difficulty spectrum or a tier will simply
not exist.**"

### "FORWARD POINTER · The dial gets finished later" callout (verbatim)

"Real systems already expose this: OpenAI o-series has a **reasoning_effort** setting and Qwen3
has a **thinking mode** toggle. In this course **the dial is finished in Sessions 17 and 18**,
where RLVR turns graded exposure into a caller-set control."

Footnote: "Traces are illustrative. The accuracy-vs-length shape follows published
**test-time-scaling** results (s1 budget forcing, arXiv:2501.19393; DeepSeek-R1 long
chain-of-thought, arXiv:2501.12948) and is illustrative here."

### Controls that did not respond
The tier buttons in the right-hand "REASONING-EFFORT TIER" row responded only intermittently; the
four large tier buttons in the main panel worked every time and were used instead. The thinking-
budget slider was read at each tier's snapped value rather than dragged (the widget states it
snaps to tiers anyway). The ULTRA chain-of-thought list scrolls internally; steps 9–14 were not
captured, though the totals (14 steps / 346 tok) are.

---
## Widget 7: "OPUS: Dynamic Per-Iteration Selection" (Section 8)

Standalone source: `/widgets/widget_3_opus_selection.html`

Panel: "Candidate batches → project onto proxy → keep the useful ones · Top 40% by projected
utility survive". Legend: `kept by OPUS (top score)` / `rejected (below the keep cut)` /
**`forced by the Always-On lane`**.

**Definition given verbatim**: "OPUS = **Optimizer-induced Projected Utility Selection**. It scores
in the **optimizer-induced update space, not the raw gradient**, using a **ghost + CountSketch
low-rank approximation**, and samples with a **Boltzmann rule** so the kept set stays diverse."
Citation shown: **arXiv:2602.05400, ICML 2026 Oral**. "The 4.7% overhead and the 6× effective
multiplier are from the paper. Selection scores computed live in your browser."

### PROXY TARGET DIRECTION — the two presets

| Preset | English | Reasoning | Sovereign | "aligns with" (cosine with the English web band) |
|---|---:|---:|---:|---:|
| **English-heavy** (default, "V4 production proxy") | 0.70 | 0.53 | 0.33 | **0.876** |
| **Balanced** ("values all bands") | 0.58 | 0.58 | 0.58 | **0.733** |

### One iteration as rendered (English-heavy, keep 40%)

"Iteration 1 · arrived 14 · kept 6 · rejected 8". Per-batch projected utilities:

| Verdict | Domain | Size | Proj. utility |
|---|---|---:|---:|
| KEPT | reasoning | 9M | 0.83 |
| KEPT | code | 9M | 0.82 |
| CUT | agentic | 16M | 0.59 |
| CUT | indic | 11M | 0.52 |
| KEPT | reasoning | 11M | 0.82 |
| KEPT | web | 16M | 0.83 |
| CUT | web | 18M | 0.82 |
| CUT | indic | 12M | 0.60 |
| CUT | agentic | 15M | 0.62 |
| KEPT | reasoning | 10M | 0.88 |
| CUT | agentic | 13M | 0.60 |
| KEPT | code | 10M | 0.83 |
| CUT | web | 11M | 0.79 |
| CUT | indic | 13M | 0.54 |

**Every Indic batch (0.52–0.60) and every agentic batch (0.59–0.62) scores below every kept batch
(0.82–0.88).** This is not a close call — the scarce lanes lose systematically, not occasionally.

### Keep-fraction → effective-token multiplier (slider range 10–90%, step 5)

| Keep-fraction | Effective-token multiplier | Overhead | Widget's own worked line |
|---:|---:|---:|---|
| 10% | **14.0×** | 4.7% | "~200B actual tokens train like ~2.80T effective" |
| 20% | **10.4×** | 4.7% | "~2.09T effective" |
| **40% (V4 production)** | **6.0×** | 4.7% | "~200B actual tokens train like ~1.20T effective" |
| 60% | **4.3×** | 4.7% | "~868B effective" |
| 90% | **3.1×** | 4.7% | "~627B effective" |

Slider caption: "The share of candidate batches OPUS keeps each iteration. **Lower = more selective
= higher effective-token multiplier.**"

### "THE DESIGN DECISION · ALWAYS-ON LANE" — `Force 8% Indic + agentic`

The headline experiment of the widget. "TRAINED THIS ITERATION" meters, at keep 40%:

| Proxy | Always-On | Indic trained | Agentic trained |
|---|---|---:|---:|
| English-heavy | **OFF** | **0.0%** | **0.0%** |
| English-heavy | **ON** | **13.0%** | **16.3%** |
| Balanced | **OFF** | **19.7%** | **0.0%** |
| Balanced | **ON** | **15.8%** | **19.7%** |

OFF caption: "the English-heavy proxy rejects almost every Indic and agentic batch. The two meters
below fall toward zero." ON caption: "a fixed 8% of every iteration is forced to be Indic +
agentic, shown violet, **bypassing OPUS entirely**. The scarce meters recover."

**The finding that matters most, and it is not the one the prose tells you**: switching the proxy
from English-heavy to Balanced *does* rescue Indic (0.0% → 19.7% with no floor at all), but
**agentic stays at 0.0%**. Fixing the proxy is not a substitute for the floor. Indic is starved
because the proxy is English-shaped; agentic is starved because agentic traces genuinely look
low-utility to *any* benchmark-derived proxy. Only the floor saves the agentic lane.
(Per-iteration meters are stochastic; these are single-iteration reads at keep 40%.)

### "V4 REALITY" callout (verbatim)

"V4 ran production OPUS keeping **~40%** of candidates for **~6× effective tokens (~200B → ~1.2T)**.
But its proxy was English-heavy (**cosine 0.876** with the English web band), so it under-valued
Indic. That is why Always-On injects **8% of every batch invisible to OPUS**."

---

## Widget 8: "Curriculum: Difficulty Bands and Staged Pretraining" (Section 9)

Standalone source: `/widgets/widget_7_curriculum_stages.html`. Badge: `Interactive · the order it
learns in`.

Header (verbatim): "A **120B model** does not read the whole corpus in one undifferentiated pile.
The run is staged: it starts on general web, then leans into reasoning and code, then stretches to
long context, and finishes on a low-LR cooldown fed the best data held back for exactly that
moment. Drag the run marker and watch the diet rebalance deliberately, stage by stage."

(Note: **120B** is the only place in the whole session that names a V5 parameter count.)

Timeline: `Seed → General → Reasoning → Long-context → Anneal`, marker 0–100%, ends labelled
"0% · run start" and "100% · cooldown end". Stage sub-labels: 1 Seed *warm start* · 2 General
*broad base* · 3 Reasoning *code + logic* · 4 Long-context *stretch ctx* · 5 Anneal *low-LR cool*.

### THE PER-STAGE MIXTURE TABLE — this is the table the transcript lost to truncation

Read directly off the widget by driving the marker (values are live linear interpolation between
per-stage profiles):

| Marker | Stage label | General web | Code | Reasoning | Long-context | Indic | STEM |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0% | Seed | **55%** | 15% | 3% | 2% | 15% | 10% |
| 10% | Seed → General | 51% | 17% | 4% | 2% | 16% | 10% |
| 20% | Seed → General | 47% | 19% | 5% | 3% | 16% | 10% |
| 25% | General | 45% | 20% | 6% | 3% | 16% | 10% |
| 40% | General → Reasoning | 33% | 25% | 13% | 4% | 15% | 10% |
| 50% | Reasoning | 25% | 28% | 18% | 5% | 14% | 10% |
| 60% | Reasoning → Long-context | 22% | 29% | 17% | 10% | 13% | 9% |
| 75% | Long-context | 18% | 30% | 16% | 18% | 12% | 6% |
| 80% | Long-context → Anneal | 16% | 28% | 17% | 16% | 16% | 7% |
| 90% | Long-context → Anneal | 12% | 25% | 18% | 13% | 23% | 9% |
| 100% | cooldown end | **8%** | 22% | 20% | 10% | **30%** | 10% |

**Cross-check**: the 75% row (web 18 / code 30 / reasoning 16 / long-ctx 18 / Indic 12 / STEM 6) is
*exactly* the mixture the instructor recited in the live class at 02:07:45, which the transcript
recorded in garbled form. The widget confirms the transcript reading.

Stage narration strings (verbatim):
- 0%: "At the Seed stage. General web dominates so the model learns basic language before anything hard."
- 40%: "Transitioning Seed → General (40% across). The mixture is rebalancing: general web falls while code, reasoning, long-context climb."
- 80%: "**Anneal reserve engaged.** The low-LR cooldown is now spending the held-back **Tier-A Indic, agentic and long-reasoning** data. The violet outline marks the reserve appearing exactly where it was planned to."

Panel captions: "FADES AS THE RUN ADVANCES — General web starts dominant and thins out. Early
tokens teach basic language; the model does not need 55% web forever." / "CLIMBS AS THE RUN
ADVANCES — Code, reasoning and long-context rise together. Harder capability is trained once the
base is solid, not from token zero."

### DIFFICULTY BANDS · B0 NURSERY TO B5 PHD — the six bands, as shipped

| Band | Name | Description as rendered |
|---|---|---|
| **B0** | Nursery | simple children's sentences |
| **B1** | Grade-school | basic grade-school text |
| **B2** | High-school | high-school level material |
| **B3** | Undergraduate | undergraduate coursework |
| **B4** | Graduate | graduate-level texts |
| **B5** | Research / PhD | research and PhD-grade papers |

**REASONING-LENGTH BAND**: `short` · `medium` · `long` · `ultra`, with the note (verbatim):
"Reasoning traces are graded by length alongside difficulty: short, medium, long and ultra traces
are separate rungs. **A hard problem with a short trace and an easy one with an ultra trace are not
interchangeable, so trace length is scheduled too.**"

So the curriculum is a **6 × 4 grid** (difficulty band × reasoning-length band), not two
independent ladders.

### "THE DESIGN DECISION · ANNEAL RESERVE" (verbatim)

"The best **Tier-A Indic, agentic and long-reasoning** data is held back from the main run and
spent in the final low-LR cooldown, where a small reserve gives a large benchmark lift.

Reserving it is **decided here, at composition time, not discovered at the end**. That is why the
violet outline appears on the Indic and reasoning bands only once the marker reaches Anneal."

Reference chips: `OLMo 2 anneal (approx / verify)` · `Qwen3 staged pretrain (approx / verify)`

### "V4 REALITY" (verbatim)

"V4 used **warmup bands at every growth seam**: a roughly **3B-token 60/40 blend** dropped in to
absorb the distribution shift when the mixture changed, so the model never hit a hard mixture edge."

Footnote: "Stage profiles are an illustrative V5 first-pass. The mixture rebalancing is real linear
interpolation between per-stage profiles computed live in JS. Staged-pretraining and anneal
patterns follow modern open models (Qwen3, OLMo 2), being verified."

---

## Widget 9: "Mixture-shift Gradient Explosion" (Section 10)

Standalone source: `/widgets/widget_8_mixture_shift_spike.html`

"Gradient norm over training steps (log scale) · every point is simulated live from the current
controls". Run length 640 steps. Trace legend: `healthy gradient-norm trace` / `unstable spike` /
`controlled / recovered` / `mixture-shift seam` / `warmup band`.

Controls:
- **Mixture shift sharpness** (0.1–1.0, default 0.70) — "How much of the Indic share is cut in a
  single seam. Gentle rebalance on the left, hard cliff on the right."
- **Frozen embeddings** toggle — "Embeddings cannot adapt to the new mixture, so the same shift
  lands far harder."
- **Warmup band width** (0–5B tokens, default 0.0B) — "Ramp the mixture change over this many
  tokens instead of at a single step. **Wider band, flatter spike.**"
- `Apply shift now` button; readout **PEAK GRADIENT NORM**, "baseline is 1.0x".

### Measured spike multipliers — driven directly

| Sharpness | Warmup band | Frozen embeddings | PEAK GRADIENT NORM | Verdict string |
|---:|---:|---|---:|---|
| 1.0 | 0 B | **ON** | **151×** | "above the 3x threshold · the run is unstable" |
| 1.0 | 0 B | OFF | **8.0×** | "above the 3x threshold · the run is unstable" |
| 1.0 | 3 B | ON | **6.0×** | "above the 3x threshold · the run is unstable" |
| 1.0 | 5 B | ON | **3.4×** | "above the 3x threshold · the run is unstable" |
| 0.4 | 3 B | OFF | **1.5×** | "under the 3x threshold · **controlled and recovered**" |

Two things fall out of this that the prose never states:

1. **The stability threshold is 3×.** The transcript records a student asking whether the 3×
   instability threshold was intuition or observation and never getting an answer; the widget
   simply hard-codes 3× as the pass/fail line.
2. **Frozen embeddings, not shift size, are the dominant term.** The same maximal shift costs
   **151×** with frozen embeddings and **8.0×** without — a ~19× difference from the embedding
   state alone. Widening the warmup band helps a lot but not enough on its own: at maximal
   sharpness with frozen embeddings, 0 B → 151×, 3 B → 6.0×, 5 B → 3.4×, still above the line.
   The only configuration measured that lands **under** the 3× threshold is **moderate sharpness
   (0.4) + a 3 B warmup band + trainable embeddings → 1.5×**. Warmup alone does not rescue a
   maximal shift, and no band width rescues frozen embeddings.

### "V4 REALITY · The ~150x spike that shaped the calendar" (verbatim)

"In V4 a sharp Hindi mixture cut met **frozen Kronecker embeddings** and the gradient norm jumped
**~150x**. The fix was a rolling warmup band, a **~3B-token 60/40 blend at every seam**, and it is
one reason the mixture is frozen at a fixed point in the calendar."

Footnote: "The gradient-norm trace is a live illustrative simulation. **The ~150x V4 spike is from
our own run.**"

---

# Extraction summary

All **9 interactive widgets** captured. The last three were read by loading their standalone
sources directly (`https://axiom.theschoolofai.in/widgets/widget_N_*.html`) and driving their
controls from JavaScript, which is far more reliable than screenshotting them inside the lesson
page's iframes — the lesson page wedges the renderer after prolonged scrolling. Widget file names
do not match their display order:

| Session § | Widget title | Standalone file |
|---|---|---|
| 2 | Mixture Composer | `widget_1_mixture_composer.html` |
| 3 | Benchmark Explainer | `widget_2_benchmark_explainer.html` |
| 4 | SOTA Dataset Inventory | `widget_9_dataset_inventory.html` |
| 5 | Training Stages Lifecycle | `widget_4_training_stages.html` |
| 6 | Agentic Trajectory Walk-through | `widget_5_agentic_trajectory.html` |
| 7 | Reasoning-effort tiers | `widget_6_reasoning_effort.html` |
| 8 | OPUS Dynamic Per-Iteration Selection | `widget_3_opus_selection.html` |
| 9 | Curriculum: Difficulty Bands and Staged Pretraining | `widget_7_curriculum_stages.html` |
| 10 | Mixture-shift Gradient Explosion | `widget_8_mixture_shift_spike.html` |

## Gaps, stated rather than papered over

- **Benchmark explainer**: 3 of 18 benchmarks opened in detail; the other 15 captured by name,
  group and one-line description only. The right-hand list does not respond to the mouse wheel.
- **Dataset inventory**: the `Tokens view` toggle was never exercised — but both the samples and
  the tokens column are recorded for all 32 rows, so it would change ordering only.
- **Training lifecycle**: only the anneal stage was expanded (one opens at a time).
- **Reasoning tiers**: ULTRA's chain-of-thought steps 9–14 not transcribed (internal scroll);
  totals are recorded.
- **Mixture composer**: sliders read at preset values rather than dragged.

## Numbers that contradict each other and should not be silently reconciled

1. **General web supply**: the composer's supply check says **4.5T**, its own MAIN PRETRAINING
   callout says "roughly **4.8T** tokens", and the inventory's running total says **4.8T**. The
   4.5T figure appears to exclude the STEM rows.
2. **STEM supply**: the composer claims **250B** for STEM/math, but the inventory has no STEM
   grouping — the plausible STEM rows (D4 STEM 49B + peS2o 42B + proof-pile-2 55B) total **146B**.
3. **V4 web start**: session prose says "roughly 70%", the composer says **72**.
4. **Anneal size**: the lifecycle widget's stage block says **~2% of tokens** while its own
   expanded TYPICAL SCALE field says **~1 to 5% of total tokens**.
