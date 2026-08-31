# LightningLM 0.1V — "Reversible Foundations"

**Sources:** https://lightninglm.theschoolofai.in/ · arxiv 2606.07404 ("Reversible
Foundations: Training a 120B Sparse MoE through State-Preserving Scaling", Rohan Shravan,
The School of AI, Bengaluru) · github.com/The-School-of-AI/LLM · Session-3 transcript.

## What it is

A four-stage family — 2B dense seed → 5B MoE → 9B MoE → 120B sparse MoE — trained on a
**single eight-GPU H200 node** in ~67 days for roughly $200k (spot pricing), on a ~1.118T-token
corpus (33,353 shards). The transcript rounds this to "108B/120B" loosely; the paper's exact
figure is **118.67B stored / 5.93B active** parameters.

## Architecture (constant backbone, grown in place)

- Residual width **4096** at every stage; vocabulary **131,072** (BrahmicTokenizer).
- Layer motif `DDDGDDDG`: D = gated **DeltaNet** linear attention (token-level recurrence),
  G = learnable sparse attention. Three recurrence scales: token (DeltaNet), layer
  (Manifold-Constrained Hyper-Connections mixing 4 streams via Sinkhorn), chunk
  (a Memory Stream: one 4096-d summary vector per chunk).
- **Reversible midpoint stack** for 5B/9B/120B: activation memory stays flat as depth,
  experts, and context grow — the key enabler of single-node training.

| Stage | Layers | Experts | Active | Stored |
|---|---|---|---|---|
| 2B | 8 | dense | 1.78B | 1.78B |
| 5B | 8 | 20 routed + 1 shared, top-2 | 2.24B | 4.96B |
| 9B | 20 | 20 + 1, top-2 | 3.92B | 9.36B |
| 120B | 20 | 460 + 1, top-12 | 5.93B | 118.67B |

## Training methodology

1. **State-preserving growth**: each stage initializes from the previous checkpoint
   (dense→MoE conversion, depth scaling, expert proliferation), claimed ~10x cheaper than
   from-scratch. The paper documents a *catalog of silent growth failures* — checkpoints
   that look plausible but violate hidden invariants.
2. **TQP (TurboQuant-PreTraining)** at 120B: expert weights held in 8-bit quantized form
   with trainable rank-16 low-rank adapters → optimizer state falls from 100B+ to ~2.26B
   parameters on the expert path (~45x).
3. **Loss-free expert balancing** (no auxiliary loss).
4. **Data**: OPUS dynamic selection (retain ~40% of candidate batches; effective-token
   multiplier ~6x for the stage it ran on), **8% always-on tier** for Indic + benchmark-train
   splits exempted from the selector, and a **golden proxy** tier (benchmark test material
   that only steers the selector, never trains). Curriculum by stage: 1B ≈ 42% web
   foundation / 30% web diverse / 13% code + STEM, shifting to code/STEM-heavy at 120B.
5. Context 4K in early stages, 8K from 9B; batch size 7/GPU at 120B (~99 GB/GPU with their
   optimizations vs ~367 GB naive). Final training loss **1.78** at 8K context
   (transcript quotes "1.6–1.8" territory; Chinchilla-optimal for the budget would have
   needed 2.4T tokens — they consumed ~200B real, ~1.2T effective via OPUS).

## Evaluation posture

No leaderboard chasing: "sustained loss reduction on held-out data of increasing
difficulty," per-domain held-out shards (D1…D4, Hindi, Bengali, code, math) run at every
checkpoint. Claims are graded by evidence class (artifact > log > session record) — an
unusually honest reproducibility posture.

## Why it matters for V5 / S3

- Proves the *pipeline*; V5's stated gap is **data** (1.1T vs Llama-4's 30T+, Qwen3's 36T).
- The always-on tier and golden proxy are direct answers to two S3 questions
  (protecting low-resource data; evaluation integrity) — validated at 120B scale.
- The growth strategy means V5's data plan must be a **curriculum keyed to model stages**,
  not one static mixture.
