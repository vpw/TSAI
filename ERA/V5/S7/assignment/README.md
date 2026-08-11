# A Real Fourier Alternative to Kronecker Embeddings

**ERA V5, Session 7 — Problem 4.**

> *"What is a REAL Fourier alternative of Kronecker? Why can't I represent each character like
> a fourier wave, and just add them to make a word!!"*

**Short answer: you can't just add them — and the reason you can't is half the result.**
Plain summation of per-character waves is permutation-invariant, so `listen` and `silent` receive
byte-identical embeddings. Binding each byte's wave to its *position* by a phase rotation before
superposing is the minimal repair.

The other half is what the repaired code buys. It **matches** the shipped Kronecker grid in a
trained model — 1.2999 against 1.3015 bits-per-byte, which is parity — while using **a quarter of
the input-path parameters**, and it is **exactly invertible with no length cap** and robust to
noise as large as the signal. The claim is *same quality, four times cheaper*, not *better*.

Everything below is measured on the real 68,096-token `sarvam1` vocabulary this course has carried
since Session 2, and on a trained transformer where the embedding module is the only thing that
differs between arms. Two of our predictions failed; both are reported in §4.

📊 **[Visual summary of the results](site/index.html)** — the same findings as charts, if you would
rather look than read.

---

## 1. What is being replaced

The baseline is the byte-level Kronecker embedding of
[arXiv 2605.29459](https://arxiv.org/abs/2605.29459) (Rohan Shravan), which the lesson released
and which this assignment asks us to extend. For a token with UTF-8 bytes `b_0 … b_{L-1}`:

```
kappa(b) = (1/sqrt(L)) * vec( sum_p  c[byte_p] ⊗ p[position_p] )     then z-normalised
L = min(len(byte_seq), pos_dim)                                      # <-- the crop
```

`c` is one-hot over 256 byte values and `p` is one-hot over `pos_dim = 32` positions, so each
byte marks exactly one cell of a 256 × 32 grid. Flattened, that grid is a fixed, never-trained
**8,192-dimensional** code, and a single shared `Linear(8192, d_model)` is the only learned
thing in the input path. The vocabulary size appears nowhere in the parameter count — which is
the property that makes the scheme worth having, and which we preserve.

It works: the paper reports 91–94% of input-side parameters removed, 2.5 ± 0.2% *lower*
validation loss than a BPE-tied baseline, and ~1.43× faster convergence.

## 2. What is actually wrong with it

Not what the lesson emphasises.

### The crop collisions are real but rare

The session calls the 32-byte window "the sovereign risk in the technique." Measured over the
whole vocabulary:

| `pos_dim` | colliding groups | tokens involved | share of vocab | scripts |
|---:|---:|---:|---:|---|
| 16 | 3,517 | 14,033 | **20.61%** | 71% Indic |
| **32 (shipped)** | **11** | **22** | **0.03%** | **100% Indic** |
| 48 | 0 | 0 | 0.00% | — |
| 64 | 0 | 0 | 0.00% | — |

So the direction of the lesson's claim holds exactly — every single collision is Indic, none is
Latin — but the magnitude is three hundredths of one percent, and widening the window to 48
removes them entirely. Real colliding pairs include `▁ऑस्ट्रेलिया / ▁ऑस्ट्रेलियन` and
`ப்பட்டுள்ளது / ப்பட்டுள்ளன`. Notably `pos_dim = 32` sits just past a cliff: at 16 the collision
rate is 20.6%.

### The waste is the real problem

UTF-8 is highly structured. Position 0 of a Devanagari token is *always* `0xE0`; most byte
values never occur at most positions. So most of the grid is unreachable:

| quantity | value |
|---|---|
| mean column occupancy over the 32 columns | 0.407 |
| non-zeros per code | 13.03 of 8,192 (**0.159% dense**) |
| grid cells any token ever activates | **1,999 of 8,192 (24.4%)** |
| unreachable cells | **6,193 (75.6%)** |

The measurement that actually settles it is the **effective rank** of the finished code — how
many independent directions it spans, which bounds what the projection can possibly
distinguish. Over 10,000 randomly sampled tokens:

| codec | code dim | numerical rank | 99%-energy rank | **rank / dim** |
|---|---:|---:|---:|---:|
| `kronecker_32` (shipped) | 8,192 | 1,545 | 998 | **0.189** |
| `kronecker_48` | 12,288 | 1,570 | 1,004 | **0.128** |
| **`fourier_2048`** | **2,048** | **1,542** | 839 | **0.753** |
| `fourier_8192` | 8,192 | 1,571 | 971 | 0.192 |

**This is not the result we expected, and it is more interesting than the one we did.** We
predicted the grid would be low-rank and the phase code full-rank. Instead *all four codecs land
on essentially the same rank, ~1,550.* That number is a property of **the data**, not of the
codec: the byte-level content of this vocabulary intrinsically spans about 1,550 dimensions, and
both schemes saturate it.

So the difference between them is not how much they carry. It is **how much space they charge
you to carry it**:

- Kronecker spends 8,192 coordinates on ~1,545 directions — 19% efficient.
- The phase code spends **2,048** coordinates on ~1,542 directions — **75% efficient**.
- Same information, one quarter of the projection parameters. At the lesson's reference shape
  that is 66.3M → **16.6M**.

And it prices the session's own proposed remedy. Widening the window from 32 to 48 costs 4,096
extra coordinates (+33M parameters at `d_model = 8096`) and buys **+25 rank**. That is a very
poor trade, and it is visible only once you measure rank rather than count cells.


**A correction worth stating**, because it is the kind of error this repo is meant to catch: an
earlier version of this analysis called those 6,193 cells "dead projection rows" that could
never receive gradient. That is true of the *unnormalised* code and false of the shipped one —
the codec z-normalises, which maps every never-activated cell to `−mean/std` rather than to
zero. The ablation harness printed `dead rows 0 of 8,192` and caught it. The structural fact
survives; the right measure is the code's **effective rank**, which normalisation cannot
inflate.

## 3. The construction

### Why "just add them" cannot work

Assign each byte value a wave and sum over the token's bytes:

```
phi_naive(b) = (1/sqrt(L)) * sum_p exp( i * w_value * byte_p )
```

A sum does not depend on the order of its terms. Any two tokens that are anagrams therefore get
**identical** codes, and no projection on top can separate them, because it is never shown a
difference. This is not an implementation detail — it is a proof that the literal reading of the
problem statement fails. Measured: on 8,000 real vocabulary tokens, naive summation merges 197
of them; the phase-bound code merges 0.

### The repair: bind value to position by phase

```
phi(b) = (1/sqrt(L)) * sum_p exp( i * ( w_value * byte_p + w_pos * p ) )
```

Multiplying two unit-modulus phasors adds their phases, so `exp(i·w_v·v) · exp(i·w_p·p)` binds a
byte value to a byte position exactly as the Kronecker product of two one-hot vectors does — but
*densely*, in `n_freq` complex dimensions instead of `256 × pos_dim` sparse ones. The frequencies
are drawn once from a fixed seed and frozen forever, for the same reason the Kronecker byte
buffer is frozen: change them and every code changes, so a projection trained against the old
ones is meaningless.

This is Fourier Holographic Reduced Representation binding (Plate's HRR in its complex form),
applied to bytes. That the machinery is thirty years old is a feature — it comes with a capacity
theory, so the claims below are predictions being tested rather than hopes.

### Decoding: unbind, then clean up

Multiply by the conjugate position atom to unbind position `p`, then correlate against all 256
value atoms and take the argmax. The argmax over a known, finite alphabet is the "cleanup" step
that VSA theory says is required to lift approximate retrieval to exact recovery — and it is
what makes the code invertible rather than merely injective.

## 4. Results

### It is exactly invertible, at a quarter of the size

Exact recovery of every byte of a token, 3,000 tokens sampled from the real vocabulary, lengths
up to 64 bytes:

| code dim | byte accuracy | token-exact |
|---:|---:|---:|
| 128 | 0.5146 | 0.2720 |
| 256 | 0.7973 | 0.4740 |
| 512 | 0.9746 | 0.8673 |
| 1024 | 0.9997 | 0.9967 |
| **2048** | **1.0000** | **1.0000** |
| 4096 | 1.0000 | 1.0000 |

The Kronecker grid needs **8,192** dimensions and is still exact only for tokens of ≤ 32 bytes;
past that it is not approximately right, it is *silently identical to a different token*. The
phase code is exact at **2,048** with no length cap.

### It degrades gracefully instead of failing at a threshold

This is the property the instructor names as his blocker on reversibility — *"the neural network
does not predict exactly those 8096 numbers we want."* Exact-token decode rate against additive
complex Gaussian noise at σ relative to the code's own RMS:

| code dim | σ=0 | σ=0.25 | σ=0.5 | σ=1.0 | σ=2.0 | σ=4.0 |
|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 0.997 | 0.993 | 0.950 | 0.695 | 0.215 | 0.021 |
| 2048 | 1.000 | 1.000 | 1.000 | **0.969** | 0.484 | 0.081 |
| 4096 | 1.000 | 1.000 | 1.000 | 1.000 | 0.799 | 0.236 |
| 8192 | 1.000 | 1.000 | 1.000 | 1.000 | **0.987** | 0.510 |

A redundant phase code is an error-correcting code. At the grid's own 8,192 dimensions it
decodes 98.7% of tokens exactly with noise *twice the size of the signal*. A hard crop has no
analogous margin: it is correct up to 32 bytes and catastrophically, silently wrong after.

### It has a capacity limit, and it is a slope rather than a cliff

"No length cap" does not mean unlimited capacity — it means the failure is graded instead of
abrupt. Exact-decode rate against token length, on **uniform-random bytes** (the hardest case;
real UTF-8 is structured and does better):

| code dim | L=4 | L=8 | L=16 | L=32 | L=64 | L=128 |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 1.000 | 1.000 | 0.965 | 0.007 | 0.000 | 0.000 |
| 1024 | 1.000 | 1.000 | 1.000 | 0.850 | 0.000 | 0.000 |
| 2048 | 1.000 | 1.000 | 1.000 | 1.000 | 0.797 | 0.000 |
| 4096 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.568 |

This is the qualitative behaviour VSA capacity theory predicts for superposition, and it is the
honest statement of the trade: you buy length with dimension, continuously. The grid instead buys
length in one discrete jump — and, per the rank table, buys it very badly.

### In a trained model

Six arms, 512-wide 8-layer decoder, **40M tokens each**, identical architecture, optimiser,
schedule, seed and token stream. The embedding module is the only difference. Output head untied
in every arm (tying a structured input path to a dense head is a confound, and it is what V5
itself decided). Metric is bits-per-byte, which is comparable across languages where per-token
loss is not.

| arm | input-path params | code dim | rank | **macro bpb** | vs `kronecker_32` |
|---|---:|---:|---:|---:|---:|
| `dense` (full table) | 34,865,152 | — | — | **1.2494** | −4.01% |
| **`fourier_2048`** | **1,048,576** | 2,048 | 1,402 | **1.2999** | **−0.12%** |
| `kronecker_32` (baseline) | 4,194,304 | 8,192 | 1,385 | **1.3015** | — |
| `kronecker_48` (wider window) | 6,291,456 | 12,288 | 1,400 | **1.3092** | +0.59% |
| `naive_2048` (order-blind) | 1,048,576 | 2,048 | **150** | **1.3254** | +1.83% |
| `fourier_8192` | 4,194,304 | 8,192 | 1,404 | **1.3461** | +3.43% |

**The main claim holds.** `fourier_2048` matches the shipped Kronecker grid — 1.2999 vs 1.3015,
a 0.12% difference that is parity, not a win — while using **a quarter of the input-path
parameters** (1.05M vs 4.19M). The honest headline is *same quality, four times cheaper*, and it
is exactly what the rank table predicted.

**The negative control behaves as the theory says it must**, and this is the cleanest result in
the table. `naive_2048` — the assignment's literal "just add them" — is 1.83% worse, and its code
has **rank 150 against the phase code's 1,402**. Order-blindness is not a small degradation; it
collapses the representation by an order of magnitude, because a sum over bytes can only encode a
bag-of-bytes histogram. The static anagram test and the trained result agree.

**And widening the window makes things worse, not better.** `kronecker_48` eliminates every
collision in the vocabulary (§2) and it is **0.59% worse** than `kronecker_32`, for 50% more
input-path parameters. It is even marginally worse on the >32-byte tokens the wider window exists
to serve. That is the session's own proposed remedy, and on this evidence it is not one: the
collisions it fixes are too rare to matter, while the coordinates it adds are real and have to be
trained. Two independent measurements agree — the rank table (+4,096 dimensions buys +25 rank) and
this run.

### Two results that contradicted our hypotheses

Reported because they are what the runs produced.

1. **`fourier_8192` is the *worst* structured arm (+3.43%), not the best.** We expected the
   matched-dimension control to be at least as good as `fourier_2048`. It is substantially worse.
   The most likely reading: at 8,192 dimensions carrying ~1,400 directions the code is heavily
   redundant, so the projection has 4× the parameters to fit from the same 40M tokens and is
   simply undertrained. If that is right it is an argument *for* the compact code, not merely a
   curiosity — but we have not run the seed-sweep that would establish it.

2. **No measurable advantage on long tokens.** We predicted the >32-byte slice would be where a
   crop structurally cannot compete. It isn't, at this scale: `fourier_2048` scores 0.3758 /
   0.2913 / 0.2854 on the three Indic lanes against `kronecker_32`'s 0.3731 / 0.2912 / 0.2862 —
   a tie. Support is small (382 / 149 / 157 scored positions), so the error bars are wide, but
   nothing here supports the claim we expected to make. The 559 over-long tokens are 0.8% of the
   vocabulary and evidently too rare to move a corpus-level metric.

**And the dense table still wins**, by 4.01% over the best structured arm. We do **not** reproduce
the seed paper's finding that Kronecker *beats* a BPE-tied baseline. That is not evidence against
it — our setup differs in ways that plausibly matter (untied head, a different corpus, 40M tokens
against their nanoGPT-scale run, no hyper-parameter search per arm) — but we should not claim a
result we did not observe. What this ablation supports is the *relative* ordering among structured
input paths, which is what it was designed to isolate.


## 5. What this does and does not license

- The proxy is **small** — a 512-wide, 8-layer decoder on 40M tokens. It is a proxy, in exactly
  the sense Session 5 established: enough to rank arms that differ in one component, not enough
  to license a claim at 120B.
- The Fourier frequencies are **random**. No attempt was made to learn or optimise them; a
  learned frequency schedule is an obvious next experiment and might change the picture.
- The capacity results use **uniform-random bytes** for the length sweep, which is the hardest
  case. Real UTF-8 tokens are structured and decode better at the same length.
- The rank comparison samples 10,000 tokens; a rank equal to the sample size is reported as
  sample-limited rather than as a measurement.
- **Nothing here tests generation quality.** The claim is about the input representation.

## 6. Reproducing

Everything in §2–§4 that does not involve training runs on a CPU in about 25 minutes, with no
GPU and no network:

```bash
python3 -m venv .venv
.venv/bin/pip install numpy pytest
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu

.venv/bin/python -m pytest tests/ -q            # 23 invariant tests, ~13 s
.venv/bin/python proofs/run_static_proofs.py    # every static claim -> proofs/results/
```

The trained ablation additionally needs a GPU and a tokenized corpus. This repo reuses the one
built in Session 5; `prep_subset.py` cuts a 331 MB slice from it (the slice itself is not
committed, but `data/corpus/manifest.json` describes it exactly, so it is reproducible):

```bash
.venv/bin/python scripts/prep_subset.py --tokens-per-arm 40000000
.venv/bin/python train_arm.py --arm fourier_2048 --tokens 40000000 \
    --d-model 512 --n-layers 8 --n-heads 8 --seq-len 512
.venv/bin/python scripts/make_report.py         # runs/*.json -> RESULTS.md
```

The tokenizer (`tokenizer-sarvam1.json`, 68,096 entries) is the frozen Session 2 artefact; every
claim here is computed against it, and changing it would invalidate every code and every
checkpoint trained on one.

**A note on the T4.** `kronecker_48` OOMs at `--micro-seqs 16` on a 16 GB card: its 12,288-dim
code table is 1.67 GB in fp16 before any activation memory. That is not a bug in the harness, it
is a real cost of the widen-the-window remedy, and it is why that arm runs at `--micro-seqs 8`.

## 7. Layout

| path | what |
|---|---|
| `kv2/codecs.py` | the three codecs: Kronecker baseline, naive sum (negative control), Fourier |
| `kv2/embedding.py` | frozen codec + one shared projection, `nn.Embedding`'s interface |
| `kv2/model.py` | Llama-shaped decoder with a swappable input path (adapted from S5) |
| `proofs/` | every static claim, and the JSON it produced |
| `tests/` | the invariants, as assertions |
| `train_arm.py` | one ablation arm; the embedding is the only variable |
| `analysis/FIVE-PROBLEMS.md` | why Problem 4 over the other four, with citations |
| `resources/` | session writeup, transcript, extracted widget data |

## 8. Where this goes next

The instructor's own ranking was that *"problem number four solves 1, 2 and 3"*, and the
construction bears that out — though this submission claims **only Problem 4**, as the assignment
requires:

- **Problem 3 (dynamic window)** falls out of the construction rather than being solved
  separately: there is no fixed column count, so length is a smooth capacity trade instead of a
  crop. Reported here as a property, not as a second claim.
- **Problem 5 (reversibility)** is the natural next submission — §4 already shows the code is
  exactly invertible and noise-tolerant, which is the blocker he names; what remains is
  replacing the output head and comparing against the softmax-bottleneck baselines.
- **Problem 1 (math structure)** reuses the same phase machinery via a CRT/residue block.
- **Problem 2 (image/audio)** needs only a discrete alphabet and a position index, so quantised
  patches substitute for bytes with no change of mechanism.

## References

- **2605.29459** — Shravan, *Kronecker Embeddings: Byte-Level Structured Token Representations
  for Parameter-Efficient Language Models*. The scheme being extended.
- **Plate, T. (1995/2003)** — *Holographic Reduced Representations*. Binding by circular
  convolution; FHRR is its complex/Fourier form, which is what this code is.
- **1803.00412** — Frady, Kleyko & Sommer, *A theory of sequence indexing and working memory in
  recurrent neural networks*. Capacity theory for superposition. (The often-quoted closed form
  `I ≈ M·log₂D` is **not** stated in the abstract we verified, so we measure our own curve rather
  than assert theirs.)
- **2412.09871** — Pagnoni et al., *Byte Latent Transformer*. Entropy-based dynamic byte patching
  at 8B/4T scale — the state of the art for Problem 3, and the reason we did not pick it.
- **1711.03953** — Yang et al., *Breaking the Softmax Bottleneck*. The rank ≤ d+1 argument that
  any Problem 5 follow-up has to engage with.

Full related-work discussion, including the other four problems, is in
[`analysis/FIVE-PROBLEMS.md`](analysis/FIVE-PROBLEMS.md).
