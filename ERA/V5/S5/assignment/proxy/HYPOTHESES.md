# Proxy ablation — hypotheses and decision rules

Written and committed **before** the runs were launched. The point of fixing the rules first
is that a mixture plan can be made to look good after the fact by choosing which lane to
quote; committing the comparisons in advance means a refutation has to be reported as one.

## Setup

Seven runs. Everything except the data mixture is held fixed: architecture (40.3M params,
14.2M non-embedding, d_model 384 / 8 layers / 6 heads / seq 1024), optimiser (AdamW, lr 6e-4,
cosine to 10%, 2% warmup, grad clip 1.0), seed, sequence length, and trained-token count
(75M per full arm).

| Arm | Mixture | Question |
|---|---|---|
| A | the proposed V5 main-pretraining mixture | the hypothesis |
| B | V4's web-heavy start: web 72 / code 13 / STEM 7 / Indic 8 | does composing backward from benchmarks beat the default? |
| C | A with the protected floor removed (OPUS floor-off: Indic 0%, agentic 0%) | does the floor earn its cost? |
| D | A with Indic collapsed to verified-only at equal Indic share | does the four-tier split beat verified-only-plus-repetition? |
| A(seed2) | A again, different seed | how big is the noise band? |
| E1/E2/E3 | quarter-length, mixture switches main→anneal at the midpoint | does the transition spike reproduce, and what dominates it? |

Each lane carries a **unique-token cap** set so the proxy repeats that lane at the same epoch
count the full-scale ledger implies (capped at the plan's own 4.0 ceiling). Without it the
proxy would hand every lane fresh tokens and would not be testing the repetition pressure the
plan is mostly about.

## Metric

Per-domain **bits-per-byte** on held-out documents that no arm trained on (validation
documents are selected by content hash and excluded from every training pool by the same
hash). For the agentic set only assistant tokens are scored; system prompts, user turns and
tool observations are context, exactly as in training.

bpb is used instead of per-token cross-entropy because cross-entropy rewards a tokenizer for
splitting text into more pieces, which would flatter the Indic lane for the wrong reason.

**Absolute bpb is not comparable across lanes.** A Devanagari character costs ~3 UTF-8 bytes
against 1 for ASCII, so Indic bpb is numerically lower than English bpb at equal model skill.
Only *within-lane, across-arm* differences are read.

## Decision rules, fixed in advance

Let σ(lane) = |bpb_A(lane) − bpb_A,seed2(lane)|, the seed-noise band for that lane. A
difference counts only if it exceeds σ for the lane in question.

1. **The mixture is confirmed** if A beats B on both `indic_A_verified` and `agentic` bpb by
   more than σ, while giving up no more than **2% relative** on `general_web` bpb.
   *Refuted* if B matches or beats A on the Indic or agentic lanes, or if A's general-web
   cost exceeds 2%.

2. **The protected floor earns its cost** if removing it (C) degrades `indic_A_verified` and
   `agentic` bpb by more than σ. The floor's price is whatever C gains on `general_web`,
   `code` and `stem`; that price is reported whether or not it is small.

3. **The four-tier Indic split is justified** if A beats D on `indic_A_verified` bpb by more
   than σ — i.e. mixing unverified, translated and synthetic tiers in beats spending the
   whole Indic budget on repeated verified text. *Refuted* if D wins: that would say the
   plan should buy fewer, cleaner Indic tokens and repeat them.

4. **The transition claim reproduces** if the post-switch peak grad-norm ratio exceeds the
   widget's 3× threshold for E1 (hard step, frozen embeddings) and falls under it for E3
   (warmup band, trainable embeddings). The widget's stronger claim — that *frozen
   embeddings*, not the size of the shift, dominate — reproduces if E1 ≫ E2 at identical
   shift size.

5. **Any lane where all arms land within σ of each other is reported as "no signal at this
   scale"**, not as support for the plan.

## What this scale does and does not license

The GPU available was a single T4 (16 GB, sm_75, no native bf16). At 40.3M parameters and 75M
tokens per arm this is the cheap rung *below* the assignment's 1B/3B suggestion, and it is
reported as such. Ranking mixtures by per-domain bpb is the kind of question small proxies
answer comparatively well; absolute bpb, emergent capability, and anything depending on model
scale are not testable here. The FLOP cost of doing this properly is stated in RESULTS.md.
