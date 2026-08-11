# The five Kronecker-V2 problems: analysis, interdependencies, and why we picked #4

Written 2026-08-10 as the decision record for the S7 submission. It covers all five problems the
instructor posed, what already exists in the literature for each, the hard theoretical obstacles,
and the measurements we ran ourselves before choosing.

**Conclusion up front: attack Problem 4 (the Fourier alternative) first**, then 5 → 1 → 2.
Problem 3 is deliberately last-ranked and is not worth a submission of its own.

**Citation provenance.** Entries marked ✔ were fetched and verified directly while writing this.
Unmarked entries come from a literature sweep and are reliable enough to reason from but **should
be re-verified before appearing in a paper**. Everything under "Our own measurements" was run
locally and is reproducible from this repo.

---

## 0. The object being extended

The seed paper is **arXiv 2605.29459, "Kronecker Embeddings: Byte-Level Structured Token
Representations for Parameter-Efficient Language Models," Rohan Shravan** ✔ (fetched), reference
implementation `github.com/theschoolofai/kronecker-embeddings` ✔.

For a token with UTF-8 bytes `b_0..b_{L-1}`:

```
kappa(b) = (1/sqrt(L)) * vec( sum_p  c[byte_p] (x) p[position_p] )     then z-normalised
L = min(len(byte_seq), pos_dim)                                        # the hard crop
```

`c` is one-hot over `char_dim = 256` byte values, `p` is one-hot over `pos_dim = 32` positions, so
the code is a **fixed, never-trained, sparse 8,192-dim grid**. A single shared
`Linear(8192, d_model, bias=False)` is the *only* trainable parameter in the input path — and the
vocabulary size appears nowhere in it. That is the whole point of the scheme.

Reported results ✔: 91–94% of input-side trainable parameters eliminated; **2.5 ± 0.2% lower
validation loss** than a BPE-tied baseline (gap 0.083 ± 0.007 nats, ~9% lower perplexity);
baseline quality in **~1.43x fewer steps**; spelling robustness — top-1 prediction preserved on
**55.5%** of 110 clean/typo pairs vs **47.3%** for BPE; 0.01–0.24% step-time overhead; 4.5 MB
buffer vs a 2.15 GB table. Stated limitation ✔: "byte-similar but semantically distant pairs
(compute/commute, nation/notion) cluster together, shifting disambiguation to early attention
layers."

**Critical scoping fact:** the seed paper's own §8.5 Future Work already lists **"Multi-modality"**
and **"Output-side Kronecker and unbounded effective vocabulary"** — i.e. *problems 2 and 5 are
already on the author's roadmap.* Problems **1 and 4 do not appear in the paper**; they are his
newer, personal ideas. This matters for where a student contribution is actually additive.

---

## 1. What the instructor himself said

From the live-class transcript (`../resources/s7-transcript.md`). This is the single most
decision-relevant evidence, and it is worth quoting rather than summarising.

**Asked point-blank which problem matters most** (lines 702–705):

> "So problem number four solves 1 2 and three. Problem number five solves completely different
> problem." … "These are the hardest…" … "but unless we solve one two and three I don't think we
> can solve four and five. So there's a sequence also."

**Fourier is his abandoned original idea** (684–685):

> "Now the [Kronecker] is actually a sub idea. I'll actually use the word **substandard idea**. **My
> first idea was a Fourier. I love Fourier.** … **that is what I started with I couldn't solve that
> so I ended up with Kronecker** and that's why everything is random."

> "is it possible to treat every single character as some sort of Fourier wave and then add all of
> them and actually say this is happening." … "neural networks understand Fourier magically for some
> reason."

**His stated blocker on reversibility** (686–689):

> "**my [forward] embedding is not reversible. And I've been banging my head on how do I make it
> reversible.**" … "any token can convert into this deterministic 8096. Can I take 8096 and convert
> it back into the tokens? Answer is yes. **But neural network does not predict exactly those 8096
> numbers we want.**" … "**how do we make it invertible? Because if you can make it invertible then
> we can force a neural network to predict a 32 character span at once.**"

He names one unexplored direction for it — VAE-style KL divergence, "we do not predict a point, we
predict x plus some Gaussian distribution" — and explicitly *rejects* a student's cosine-similarity
nearest-neighbour suggestion, on the grounds that at random initialisation nothing is close to
anything (692–694).

**What counts as proof** (691):

> "I need a good read me… a **boring read me** which tells exactly what you're doing is good enough.
> And then you need to get your agent to take that train a small network and prove that the whole
> thing works. So you're submitting code and you're [submitting a] read me."

**On collaboration** (679): *"if you can then we both will write a paper."*

---

## 2. Our own measurements

Run locally against the real **68,096-token sarvam1 tokenizer**
(`../../S4/assignment/models/tokenizer-sarvam1.json`), the actual V5-lineage vocabulary from S2/S4.
CPU only, no GPU, seconds of compute. These are the numbers that actually drove the decision.

### 2.1 The collision risk the lesson emphasises is negligible

The session calls `pos_dim = 32` collisions "the sovereign risk in the technique." Measured:

| quantity | value |
|---|---|
| colliding groups at `pos_dim=32` | **11** |
| tokens involved | **22 of 68,096 = 0.032%** |
| script of every single collision | Indic (Devanagari, Malayalam, Tamil, Gujarati) |
| tokens longer than 32 bytes | 559 (0.82%) |
| token byte length | mean 13.04, median 12, max 36 |

Example colliding pairs: `▁ऑस्ट्रेलिया / ▁ऑस्ट्रेलियन`, `▁வெற்றிகரமாக / ▁வெற்றிகரமான`,
`▁ઓસ્ટ્રેલિયન / ▁ઓસ્ટ્રેલિયા`, `ിക്കുന്നതിന / ിക്കുന്നതിന്`.

The *direction* of the lesson's claim is right — collisions are exclusively Indic, never Latin —
but the magnitude is 0.03%, not a crisis. **This is a finding worth reporting honestly**, because
it means Problem 3, sold in the lesson as the urgent one, is attacking a defect that barely exists
on a real vocabulary at 68k. (It would grow at 131k with heavier Indic coverage, but not by orders
of magnitude — max observed token length is 36 bytes.)

### 2.2 The real defect is waste — three-quarters of the grid is unreachable

**Correction to an earlier draft of this document.** An initial version of this section claimed
6,193 "dead rows" of the projection, on the reasoning that a code coordinate which is always zero
can never receive gradient. That reasoning is right about the *unnormalised* code and wrong about
the shipped one: the released codec **z-normalises**, which maps every never-activated cell to
`-mean/std` rather than to zero. Testing for a zero coordinate therefore finds nothing at all — as
the ablation harness immediately reported ("dead rows 0 of 8,192"), which is what caught the error.

The structural fact survives; the right way to state it does not involve the word "dead". Two
measurements are reported below: the cell census (a property of the codec and the vocabulary,
before normalisation) and the **effective rank of the finished code**, which is normalisation-proof
and is the number that actually bounds what the projection can distinguish.


Because UTF-8 is highly structured, most (value, position) grid cells are unreachable: position 0
of a Devanagari token is *always* `0xE0`, and so on.

| quantity | value |
|---|---|
| mean occupancy across the 32 columns | 0.407 → **59.3% of the 8,192-dim code is structurally always zero** |
| non-zeros per code | 13.03 of 8,192 = **0.159% density** |
| distinct (value, position) cells ever activated | **1,999 of 8,192 = 24.4%** |
| **unreachable cells** | **6,193 of 8,192 = 75.6%** |

Column occupancy decays fast: col 0 = 1.000, col 8 = 0.714, col 16 = 0.265, col 24 = 0.071,
col 30 = 0.020.

**Effective rank** — the normalisation-proof measure, over 10,000 randomly sampled tokens:

| codec | code dim | rank | 99%-energy rank | rank/dim |
|---|---:|---:|---:|---:|
| `kronecker_32` | 8,192 | 1,545 | 998 | 0.189 |
| `kronecker_48` | 12,288 | 1,570 | 1,004 | 0.128 |
| `fourier_2048` | 2,048 | 1,542 | 839 | **0.753** |
| `fourier_8192` | 8,192 | 1,571 | 971 | 0.192 |

**This was not the predicted result.** We expected the grid to be low-rank and the phase code
full-rank. Instead all four land on ~1,550 — that number is a property of **the data** (the byte
content of this vocabulary intrinsically spans about 1,550 dimensions) and both schemes saturate
it. The difference is not what they carry but what they charge to carry it: Kronecker spends 8,192
coordinates on 1,545 directions (19% efficient); the phase code spends 2,048 on 1,542 (75%). Same
information, a quarter of the projection.

It also prices the session's own proposed remedy: widening `pos_dim` from 32 to 48 costs 4,096
extra coordinates (+33M parameters at `d_model = 8096`) and buys **+25 rank**.

### 2.3 A dense Fourier phase code is strictly better on every axis we tested

Construction (FHRR-style binding of value to position by phase, superposed over the token's bytes):

```
phi(t) = (1/sqrt(L)) * sum_p exp( i * ( w_v * byte_p + w_p * p ) )
```

decoded by *unbinding*: multiply by the conjugate position atom, then argmax the correlation over
the 256 value atoms. No length cap — `L` is the token's true byte length.

**Exact recovery of every byte of a token** (3,000-token sample, lengths up to 64 bytes):

| real dim | byte accuracy | token-exact |
|---|---|---|
| 128 | 0.4962 | 0.2413 |
| 256 | 0.8514 | 0.5547 |
| 512 | 0.9824 | 0.8827 |
| 1024 | 0.9995 | 0.9947 |
| **2048** | **1.0000** | **1.0000** |
| 4096 | 1.0000 | 1.0000 |

**Noise robustness** — exact-token decode rate vs additive complex Gaussian noise at σ relative to
the code's own RMS:

| dim | σ=0 | σ=0.25 | σ=0.5 | σ=1.0 | σ=2.0 | σ=4.0 |
|---|---|---|---|---|---|---|
| 1024 | 0.997 | 0.992 | 0.963 | 0.709 | 0.215 | 0.023 |
| 2048 | 1.000 | 1.000 | 1.000 | **0.953** | 0.477 | 0.070 |
| 4096 | 1.000 | 1.000 | 1.000 | 1.000 | 0.815 | 0.223 |
| **8192** | 1.000 | 1.000 | 1.000 | 1.000 | **0.989** | 0.532 |

Read against the one-hot Kronecker baseline, which needs **8,192** dims and *still crops at 32
bytes*:

- **4x more compact** — exact invertibility at 2,048 dims instead of 8,192.
- **No length cap** — tokens of any length encode without cropping, so the collision class
  disappears by construction rather than by widening a window.
- **Graceful degradation** — at Kronecker's own 8,192 dims the Fourier code still decodes 98.9% of
  tokens exactly under noise *twice the size of the signal*. This is an error-correcting property
  the one-hot grid does not have, and it speaks directly to the instructor's stated blocker on
  Problem 5 ("the neural network does not predict exactly those 8096 numbers we want").

These are preliminary (numpy, random frequencies, no training). They are enough to justify the
choice; the submission's job is to confirm them inside a trained model.

---

## 3. Problem-by-problem analysis

### Problem 1 — Embeddings that carry mathematical structure

*"What if the embedding of 9 actually had somewhere physically nine mentioned there… and if I do
9 + 9 the sum of the two embeddings actually is 18."*

**Prior art.** Addition is, for practical purposes, solved several times over:
- **Abacus Embeddings** (2405.17399, "Transformers Can Do Arithmetic with the Right Embeddings") —
  per-digit positional embeddings encoding significance; up to 99% on 100-digit addition. A
  positional fix, not a homomorphism.
- **xVal** (2310.02989) — encodes a real number as a fixed direction scaled by magnitude; linear,
  no multiplicative closure.
- **FoNE, Fourier Number Embedding** (2502.09741) — sin/cos pairs at multiple base-10 periods so
  digits are recoverable mod 10^i; 64x less data to reach 99% on 6-digit addition. An *input
  encoding for sample efficiency*, not an operation-homomorphic algebra.
- Mechanistic interpretability shows models discover this themselves: **2406.03445** ("Pre-trained
  LLMs Use Fourier Features to Compute Addition", NeurIPS 2024) and **2502.00873** ("Language Models
  Use Trigonometry to Do Addition") — the latter reverse-engineers GPT-J, Pythia and Llama-3.1 and
  finds numbers laid out on a literal **helix**, with addition implemented as a rotation-and-
  translation "Clock algorithm", causally validated by activation patching.
- **Neural Isomorphic Fields** (2601.12095, Sadeghi, Momtazi & Safabakhsh, 17 Jan 2026) ✔ — the
  closest direct attempt: a fixed-length number embedding preserving addition, multiplication and
  comparison over the rationals. Verified result: **addition >95%** on identity/closure/
  associativity, **multiplication 53–73%**.

**The hard obstacle, stated cleanly.** There is no no-go theorem, but there is an elementary
argument worth putting in the writeup: if a single embedding satisfied both
`phi(a) + phi(b) = phi(a+b)` and `phi(a) + phi(b) = phi(ab)`, then `a + b = ab` for all `a,b` —
false. So **the two operations cannot share one additive structure**; they need either different
operations or different subspaces. Concretely:
- `phi(n) = n·v` makes addition exact and multiplication impossible.
- `phi(n) = log(n)·w` makes multiplication exact and addition impossible (and breaks at 0 and on
  negatives).
- A CRT/phase block `[cos(2πn/m_k), sin(2πn/m_k)]_k` makes addition a *phase rotation* (complex
  product, not vector sum) and gives exact decoding up to `lcm(m_k)`.

The defensible design is therefore exactly what the instructor gestured at — **append separate
blocks** (`[linear | log | CRT-phase]`) rather than seek one magic space. That is a real, provable
contribution *and* it explains why 2601.12095 stalled on multiplication.

- **Novelty: partially explored; multiplication genuinely open.**
- **Tractability: easy.** Grokking-scale models (single-layer transformers, p ≤ 113) are the
  standard rig; hours on one GPU.
- **Risk:** direct head-to-head with a paper six months old.

### Problem 2 — Kronecker for images and audio

*"Is there a way in which we can use the same embedding for images and audio?"*

**Prior art.** The broad claim is already demonstrated at scale, with learned embeddings:
- **bGPT** (2402.19155) — patches raw file bytes of any modality; near-lossless music format
  conversion (<0.0011 bits/byte), >99.99% CPU-emulation accuracy.
- **ByteFormer** (2306.00238, Apple) — classification straight from file bytes with *no*
  modality-specific preprocessing: ImageNet 77.33% top-1 and Speech Commands V2 95.42% from the
  same architecture.
- **MegaByte** (2305.07185) — patch-embeds bytes, SOTA density estimation on ImageNet, models raw
  audio.
- Audio-side context from the local library: **2306.06546** (Improved RVQGAN) and **2301.02111**
  (VALL-E) establish residual vector quantisation as the dominant discrete-audio codec — VALL-E's
  "eight hierarchy quantizers with 1024 entries each" at 75 Hz is the shape a Kronecker patch code
  would have to compete with.
- **KrossFuse** (2506.08645) uses Kronecker products across modalities but for *late fusion of
  pretrained embeddings*, a different problem.

No one appears to have applied the specific value⊗position byte factorization to image/audio
patches. But the *interesting* claim ("one scheme spans three modalities") is already established;
what would remain is the narrower "and it works parameter-free too."

- **Novelty: genuinely open as a construction, but in a crowded neighbourhood** — and it is
  explicitly on the seed paper's own roadmap, so it is the most likely thing to be scooped.
- **Tractability: moderate**; needs both an image and an audio pipeline.
- **Specific risk:** the seed paper's own failure mode (byte-similar → embedding-similar)
  translates to "patches of similar brightness/loudness collide regardless of content." That must
  be tested, not assumed away.

### Problem 3 — Dynamic byte window, no cropping

*"apple and a will still take 32 places. There's a waste of space… Can it be dynamic?"*

**Prior art — the most crowded field of the five.**
- **Byte Latent Transformer (BLT)** (2412.09871, Meta) ✔ verified — encodes bytes into
  **dynamically sized patches segmented by next-byte entropy**, allocating compute by complexity.
  "The first FLOP controlled scaling study of byte-level models up to **8B parameters and 4T
  training bytes**," matching tokenizer-based LLM performance with better inference scaling. This
  is *exactly* problem 3, already solved at production scale.
- Plus **FLEXITOKENS** (learnable boundary predictor with Gumbel-Sigmoid), **MrT5** (2410.20771,
  dynamic token merging), **H-Net++**, **ByteFlow**, **CANINE** (2103.06874), **Hash Embeddings**
  (1709.03933), and ACL-2025 work on retrofitting dynamic tokenization.

**Assessment.** A toy-scale result here reads as a smaller re-implementation of a solved,
productionised idea. Worse, the mechanism (entropy-gated boundaries) is *orthogonal* to Kronecker —
you would be bolting BLT's segmenter in front of an unchanged codec, which is an engineering
combination rather than a new idea. And §2.1 shows the defect it targets affects 0.03% of a real
vocabulary.

- **Novelty: heavily explored. Weakest of the five.**
- **Tractability: moderate.**
- **Note:** the *waste* framing (§2.2, 75.6% dead parameters) is a much better attack than the
  *cropping* framing — and a dense code addresses it without a segmenter. Which is Problem 4.

### Problem 4 — A real Fourier alternative ← **CHOSEN**

*"is it possible to treat every single character as some sort of Fourier wave and then add all of
them"*

**Prior art.** The mathematics is mature and that is a feature, not a bug:
- **Holographic Reduced Representations** (Plate, 1995; book 2003) — bind two vectors by circular
  convolution, bundle by elementwise addition, retrieve by approximate inverse. **Fourier HRR
  (FHRR)** is the complex-exponential form, where binding is elementwise complex multiplication.
  This is precisely "characters as waves, summed," generalised to arbitrary symbols.
- **"Learning with Holographic Reduced Representations"** (NeurIPS 2021) — modern differentiable
  revival.
- **VSA / hyperdimensional computing surveys** (2111.06077, 2106.05268).
- **Frady, Kleyko & Sommer, "A theory of sequence indexing and working memory in recurrent neural
  networks"** (1803.00412) ✔ verified — the capacity theory for exactly this kind of superposition.
  (Note: the widely-quoted closed form `I ≈ M·log2 D` is *not* stated in the abstract we verified;
  the paper's claim is that VSA models have "universal performance properties, which are superior
  to what previous analyses predicted." **We will measure our own capacity curve rather than assert
  a formula we have not confirmed.**)
- Plate's own caution, per the survey literature: naive binding capacity does **not** scale linearly
  with dimension; only cleaned-up/error-corrected variants recover near-linear scaling. Our
  unbinding step *is* that cleanup (argmax over 256 known atoms), which is why §2.3 gets exact
  recovery.

**What is genuinely absent:** nobody has built this as a **byte-level input path for a transformer
LM**, competing head-to-head with a deployed alternative on real multilingual vocabulary. The
theory exists; the instantiation does not.

**The negative result that makes the contribution sharp.** The instructor's literal phrasing is
"just add them." Plain summation of per-character waves is **permutation-invariant**: `sum_p f(b_p)`
cannot distinguish anagrams, so `listen` and `silent` collapse to the same code. That is provable
in one line and demonstrable in one test. **Phase binding to position is exactly the minimal fix**,
and framing the submission as "the naive version provably fails, here is why, here is the repair,
here is the trained evidence" is a much stronger paper shape than a bare proposal.

- **Novelty: theory old, instantiation open.**
- **Tractability: easy–moderate** — and §2.3 is already most of the static proof.

### Problem 5 — Reversibility, and dropping the output head

*"how do we make it invertible? … then we can force a neural network to predict a 32 character span
at once."*

**Prior art — old, dense, and directly relevant.**
- **The softmax bottleneck** (1711.03953, Yang et al., ICLR 2018) — *the* load-bearing theoretical
  obstacle: a softmax head with hidden width `d` can only express log-probability matrices of
  **rank ≤ d + 1**, while natural language's true context→next-token matrix is empirically
  high-rank. Any softmax output layer with small `d` is provably expressivity-limited.
- **von Mises–Fisher loss** (1812.04616) — continuous output over pretrained embeddings, no softmax;
  2.5x faster training, handles very large vocabularies. The closest classical "no softmax head"
  prior art.
- **Adaptive / hierarchical softmax** — solves compute, not parameters, not decodability.
- **Product-key memory** (1907.05242) and **Mixture of a Million Experts** (2407.04153) —
  structured exact nearest-neighbour lookup at ~1M scale.
- **"Language Models are Injective and Hence Invertible"** (2510.15511) — proves the *input →
  hidden state* map is injective and gives an algorithm (SipIt) to reconstruct input text from
  activations. Adjacent but the opposite direction from what is wanted here; worth citing precisely
  to distinguish the claims.

**The obstacle, and why it moves.** If you keep a softmax head you inherit the rank bound. If you
truly drop it and decode by nearest-neighbour/algebraic inversion, you escape the rank argument and
inherit a *different* limit — how many distinguishable points fit in a `d`-dimensional space at a
given noise level. That is a sphere-packing/rate-distortion question, and it is **the same question
as Problem 4's capacity bound, from a different literature.** Our §2.3 noise table is a direct
measurement of it.

- **Novelty: partially explored; the specific construction open.** On the seed paper's roadmap.
- **Tractability: moderate–hard.** The subtlety is avoiding a degenerate win where nearest-neighbour
  decoding succeeds on byte-similarity rather than semantics — the same trap the seed paper already
  reports on the input side.

---

## 4. Interdependencies — the actual structure

The instructor's claim that "**4 solves 1, 2 and 3**" is not rhetoric; it is structurally true, and
it is why 4 is the right first move:

```
                    ┌─────────────────────────────────────────┐
                    │  P4: dense phase/Fourier code           │
                    │  phi(t) = (1/√L) Σ_p exp(i(w_v·b_p+w_p·p)) │
                    └──────────────┬──────────────────────────┘
                                   │
        ┌──────────────────────────┼──────────────────────────┐
        ▼                          ▼                          ▼
  P3 falls out              P1 shares machinery         P5 is enabled
  no fixed window;          CRT phase block is          unbinding IS the
  length is free;           the same construction       inverse; noise
  graceful capacity         applied to numbers          robustness is the
  decay, not a crop         instead of bytes            blocker he named
                                   │
                                   ▼
                            P2 reuses it wholesale
                            quantised pixel/sample values
                            are just another byte alphabet
```

- **4 → 3.** A dense code has no fixed column count. Token length stops being a hard window and
  becomes a smooth capacity trade — the crop disappears by construction, no segmenter required.
- **4 → 1.** A CRT/phase block over moduli is the same phase machinery pointed at integers instead
  of bytes; addition becomes phase rotation with exact decoding. Problem 1's "append new dimensions"
  is literally appending another phase block.
- **4 → 2.** The codec only needs a discrete alphabet and a position index. Quantised pixel
  intensities or μ-law audio samples substitute for byte values with no change of mechanism.
- **4 → 5.** Unbinding is already an exact inverse (§2.3), and the noise table is exactly the
  robustness curve his objection demands. His stated blocker — "the network does not predict
  exactly those numbers" — is answered by an error-correcting code, which is what a redundant phase
  code is.

**Problem 5 remains genuinely separate** in the sense he meant: even with a perfect invertible
codec, replacing the output head raises its own questions (training signal, decoding constraints to
valid tokens, the softmax-bottleneck comparison). It is the natural *second* submission because it
reuses this codebase entirely.

---

## 5. Ranking

| # | Problem | Novelty | Tractability | Instructor's own weight | Verdict |
|---|---|---|---|---|---|
| **4** | **Fourier alternative** | Theory old, LM instantiation open | Easy–moderate | **"solves 1, 2 and 3"**; his abandoned original idea | **1st — submit this** |
| 5 | Reversibility | Narrow gap in a dense field | Moderate–hard | His personal obsession; "completely different problem" | 2nd |
| 1 | Math structure | Multiplication genuinely open | Easy | Not in the seed paper | 3rd |
| 2 | Image/audio | Broad claim already solved | Moderate | On the seed paper's roadmap | 4th |
| 3 | Dynamic window | Solved at 8B scale by BLT | Moderate | Defect measures 0.03% on a real vocab | 5th — skip |

A literature-only ranking would put Problem 1 first on novelty×tractability. We rank 4 first anyway
because three things the pure-literature view misses all point the same way: the instructor states
4 subsumes three others, 4 is the idea he personally abandoned and most wants solved, and **we
already have preliminary evidence that our construction works** (§2.3) — which converts 4 from the
"hardest" problem into the best-evidenced one.

---

## 6. What the submission will claim

**Problem 4 only.** The assignment says "each are separate, don't try and mix them," so the claim
is the Fourier codec. That it also removes the length cap (Problem 3) is reported as a *corollary
of the construction*, not as a second solved problem. Reversibility results (§2.3) are reported as
*properties of the codec*, with the head-replacement question explicitly deferred to a follow-up
submission on Problem 5.

Claims, and how they came out:

1. The one-hot grid leaves **75.6%** of its cells unreachable, and delivers only **1,545
   independent directions for 8,192 coordinates** where the phase code delivers 1,542 for 2,048.
   ✅ *measured, §2.2 — though the framing had to be corrected twice (see above).*
2. Naive "just add the waves" is permutation-invariant and **cannot distinguish anagrams**.
   ✅ *proved, tested, and confirmed in training: its code has rank 150 vs 1,402, and it trains
   1.83% worse.*
3. Phase binding yields **exact invertibility at 2,048 dims** vs Kronecker's 8,192, with **no
   length cap**. ✅ *measured, §2.3.*
4. The code degrades gracefully under noise, unlike a hard crop. ✅ *measured, §2.3.*
5. In a trained transformer the Fourier arm is **at least competitive** with the Kronecker arm,
   and **strictly better on long tokens**. ⚠️ *Half confirmed.* Competitive: yes — 1.2999 vs
   1.3015 macro bpb at a quarter of the input-path parameters. Better on long tokens: **no**,
   the >32-byte slice is a tie within a wide error bar. Reported as a failed prediction.

Two further predictions were **wrong** and are reported as such in the README: `fourier_8192`
was expected to be the strongest structured arm and is the weakest (+3.43%), and the dense table
still beats every structured arm by 4%, so the seed paper's "Kronecker beats BPE-tied" result is
**not** reproduced at this scale and setup.

---

## 7. References

Verified directly (✔) while writing this document:

- **2605.29459** — Shravan, *Kronecker Embeddings: Byte-Level Structured Token Representations for Parameter-Efficient Language Models*. ✔ Seed paper. Code: `github.com/theschoolofai/kronecker-embeddings` ✔
- **2412.09871** — Pagnoni et al., *Byte Latent Transformer: Patches Scale Better Than Tokens* (Meta). ✔ Entropy-based dynamic patching, 8B params / 4T bytes.
- **2601.12095** — Sadeghi, Momtazi & Safabakhsh, *Neural Isomorphic Fields: A Transformer-based Algebraic Numerical Embedding* (17 Jan 2026). ✔ Addition >95%, multiplication 53–73%.
- **1803.00412** — Frady, Kleyko & Sommer, *A theory of sequence indexing and working memory in recurrent neural networks*. ✔ VSA capacity theory. (Exact `M·log2 D` form **not** confirmed — do not quote it as theirs.)

From the literature sweep — reliable to reason from, re-verify before citing in a paper:

- 1711.03953 — Yang et al., *Breaking the Softmax Bottleneck* (rank ≤ d+1).
- 1812.04616 — Kumar & Tsvetkov, von Mises–Fisher loss / continuous output.
- 1907.05242 — Lample et al., *Large Memory Layers with Product Keys*; 2407.04153 — *Mixture of a Million Experts*.
- 2510.15511 — *Language Models are Injective and Hence Invertible* (SipIt).
- 2405.17399 — *Transformers Can Do Arithmetic with the Right Embeddings* (Abacus).
- 2310.02989 — xVal. 2502.09741 — FoNE (Fourier Number Embedding).
- 2406.03445 — *Pre-trained LLMs Use Fourier Features to Compute Addition* (NeurIPS 2024).
- 2502.00873 — *Language Models Use Trigonometry to Do Addition* (helix / Clock algorithm).
- 2301.05217 — Nanda et al., *Progress Measures for Grokking via Mechanistic Interpretability*.
- 2402.19155 — bGPT. 2306.00238 — ByteFormer. 2305.07185 — MegaByte. 2103.06874 — CANINE.
- 2410.20771 — MrT5. 1709.03933 — Hash Embeddings. 2506.08645 — KrossFuse.
- 2306.06546 — Improved RVQGAN. 2301.02111 — VALL-E.
- 2111.06077, 2106.05268 — VSA / hyperdimensional computing surveys.
- Plate, T. (1995) *Holographic Reduced Representations*, IEEE TNN; and (2003) book.
- *Learning with Holographic Reduced Representations*, NeurIPS 2021.

Local project context:
- `../resources/s7-session.md`, `../resources/s7-transcript.md`, `../resources/s7-widget-data.md`
- `../../resources/kronecker-embeddings.md`, `../../resources/lightninglm.md` (LightningLM
  0.1V / arXiv 2606.07404 — note it uses residual width **4096**, so the deployed Kronecker
  projection is 8192x4096 ≈ 33.6M params, not the 66.3M the lesson quotes at d_model 8096).
