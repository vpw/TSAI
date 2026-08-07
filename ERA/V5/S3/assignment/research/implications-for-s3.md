# From the References to the S3 Assignment

How the V4 artifacts constrain/inform the 40B design, plus a longer plan for properly
absorbing the references.

## Direct carry-overs into the S3 answer

| V4 artifact / practice | S3 design consequence |
|---|---|
| Always-on 8% tier (Indic exempt from selector) | Keep and extend: reserve a fixed batch share for Indic **and** India-perspective English; English-tuned quality filters demonstrably erase Indic text, so protection is structural, not optional. |
| Golden proxy tier (test material only steers, never trains) | The India-first eval must be built *before* collection, as an immutable held-out rail with canary GUIDs. |
| OPUS dynamic selection (~6x effective tokens) | Data plan can assume an effective-token multiplier ≥2–3x on the general pools, but never on Indic (OPUS was not run on always-on data in V4 either). |
| Staged growth (2B→5B→9B→120B) with per-stage mixtures | The 40B mixture must be a **curriculum** (web-heavy early → code/STEM/Indic-heavy late), and the anneal reserve must be set aside on day one. |
| BrahmicTokenizer LP retrofit | Vocab size is derived from per-domain fertility targets via slot allocation — this is literally the method for assignment question 4. |
| Kronecker embeddings (input side ≈ free) | A larger-than-131K vocab costs only softmax-head params (~5K params/slot at d≈5120), changing the vocab-size trade-off materially. |
| Repetition ceiling (~4 epochs useful, 16 = wasted) | Scarce verified Indic text (~hundreds of billions of tokens, not trillions) can be stretched at most ~3–4x; the rest must come from synthetic + translation, capped near 30–50%. |
| Per-snapshot dedup (global removal cost ~58% and hurt quality) | Adopt per-snapshot MinHash + exact dedup; validate scope by ablation, not dogma. |
| Proxy-model ablations (140M-scale recipe ranking) | Every mixture/filter/dedup/synthetic decision in the report should be framed as "decided by proxy ablation", with the ablation ladder specified. |

## Longer study plan for the references (post-assignment)

Ordered so each item unlocks the next; ~4 weeks at a few hours/week.

1. **Week 1 — BrahmicTokenizer paper (2605.29379), deeply.** Reproduce fertility
   measurements on FLORES-200 with the released HF artifact; compare against the S2
   assignment tokenizer. Understand the LP slot-allocation formulation — it is the
   assignment's question 4 in miniature.
2. **Week 2 — Kronecker embeddings (2605.29459) + reference repo.** Run the nanoGPT-scale
   reproduction; verify the 2.5% val-loss claim; study how the byte codec composes with a
   *changed* vocabulary (needed if V5 grows the vocab).
3. **Weeks 3–4 — Reversible Foundations (2606.07404) + The-School-of-AI/LLM repo.**
   Read in three passes: (a) data tiers + OPUS wrapper (relevant now), (b) growth
   transforms and the failure catalog, (c) TQP/reversibility (deferred to the systems
   sessions). The "4-day read" the instructor mentions is this paper + repo docs.
4. **Ongoing.** OPUS paper (efficient principled data selection, 2025/26), DeepSeekMath
   mining loop, FineWeb-Edu / DCLM / Nemotron-CC filtering recipes, Sangraha/IndicCorp
   licensing terms — each is one evening; all feed the V5 capstone data work.
