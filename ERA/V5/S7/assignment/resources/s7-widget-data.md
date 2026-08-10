# S7 widget data — Embeddings and Model Internals

Extracted 2026-08-10 from the live session page via browser automation
(`extract-widget-data` skill). All 13 widgets are standalone pages embedded as iframes at
`https://axiom.theschoolofai.in/widgets/s7_widget_N_<name>.html`; unusually for this course the
**file numbering matched display/section order exactly**. Every control responded on first try —
no gaps to flag except one internal inconsistency in widget 10, noted in place.

Numbers below are the widgets' own rendered values, not paraphrases.

---

## Widget 1 — "The Seam" (§1)

Static field selector, 10 rows. Left column "WHAT SESSION 6 HANDS OVER" (token ids, position ids,
loss mask, mixture lane and stage, ledger tags); right column "WHAT SESSION 7 BUILDS" (embedding
table, position policy, output head, adaptation boundary, `embedding_policy_id`). Clicking a left
field shows the consuming object plus "What it carries" / "What breaks if it is wrong" / "The rule
this enforces".

**Token ids → Embedding table** — flow `token id (integer) -> gather (row lookup) -> token vector [B,T,D]`
- Carries: "An address and nothing more. The tokenizer has finished; the integer says which row, not what the row contains."
- Breaks: "A tokenizer version change silently reassigns ids, so every row of the embedding now means a different token than it did. This is why the tokenizer hash travels in the shard manifest and in the embedding policy record."
- Rule: "the embedding module takes integers and never token strings. A design that needs the string at lookup time has reached back across the seam, which is a defensible thing to do and a bad thing to do by accident."

**Position ids → Position policy** — `position id (integer) -> stored or computed (policy) -> order signal into attention`
- Breaks: "A position beyond the trained maximum has no learned row, so a stored table returns its initialisation and injects noise at exactly the positions you extended the model to handle."
- Rule: "position is a policy with a version, not a constant."

**Loss mask → Output head** — `hidden state [B,T,D] -> output projection D x V -> logits [B,T,V] -> masked loss (scalar)`
- Breaks: "A mask that leaks loss onto context positions teaches the model to invent tool results instead of calling tools. A mask that suppresses too much silently discards training signal you paid full compute to generate."

**Mixture lane and stage → Adaptation boundary** — `lane tag -> mixture shift -> embedding (must adapt)`
- Breaks: "When the mixture moves, the statistics arriving at the embedding move with it. A frozen input path cannot follow, so the adjustment is pushed into the layers above and appears as a gradient spike. This is the V4 incident."
- Rule: "the embedding is where the data distribution enters the model, so every mixture decision from Session 5 is also an embedding decision."

**Ledger tags → embedding_policy_id**
- Breaks: "Two runs compared without recording what their input paths were doing produce a result nobody can attribute. If one had a trainable embedding and the other did not, the comparison measured the freeze decision and called it something else."
- Rule: "The embedding type, factor shape, rank, tokenizer hash, freeze state, unfreeze schedule, identifier layout and tying decision all travel with the checkpoint or the checkpoint cannot be compared."

---

## Widget 2 — Gather and Scatter-Add (§2)

Illustrative vocab of 24, table 8 wide. Sliders: batch tokens (4–40, default 12), Zipf skew
(0–2.00, default 1.00). Buttons: Run 200 steps / +1000 more / Reset.

**After 200 steps at skew 1.00, batch 12** — per-row update counts:
`0:605, 1:322, 2:210, 3:169, 4:136, 5:88, 6:92, 7:95, 8:63, 9:59, 10:54, 11:44, 12:56, 13:52, 14:49, 15:42, 16:33, 17:29, 18:47, 19:43, 20:31, 21:27, 22:29, 23:25`
→ **ratio 24x** between the most and least updated row.

| state | top row | bottom row | ratio |
|---|---|---|---|
| 200 steps, skew 1.00 | 605 | 25 | **24x** |
| 1,200 steps, skew 1.00 | 3.8K | 161 | **24x** (constant in steps) |
| 200 steps, skew 2.00 | 1.5K | 2 | **758x** |

Skew, not step count, drives the imbalance.

> "The optimizer applied the same learning rate to all of them; the difference is entirely in how
> many terms the scatter-add summed into each. This is why the embedding table is better thought of
> as 24 small tensors training at 24 different rates than as one tensor training at one rate…
> thinning a language in the mixture lowers the update rate of that language's rows, and a row near
> its initialisation contributes noise to every sequence it appears in."

**"At V5 scale" callout, verbatim:**
> "With a vocabulary of 131,072 and a global batch of 2,097,152 token positions, almost every row is
> read on almost every step, so coverage is not the problem. The spread is. Under a Zipf corpus the
> most frequent token contributes on the order of 170,000 gradient terms to its row in a single step
> while a token deep in the tail contributes 1, or less than one on average. Five orders of
> magnitude, and no part of the optimizer knows about it."

---

## Widget 3 — Parameter and Memory Budget (§3)

Sliders: vocabulary (8,192–262,144, default 131,072), model width (512–12,288, default 8,096),
layers (4–96, default 48), FFN multiplier (2–8x, default 4x), card size (24–192 GB, default 80).
Memory regime select: AdamW mixed 16 B/param (default) · SGD mixed 8 B/param · inference bf16
2 B/param · inference fp32 4 B/param. Buttons: Untied (default) / Tied.

**Default, untied, AdamW:**

| block | params | share | training memory |
|---|---|---|---|
| input embedding | 1.06B | 2.7% | 16.98 GB |
| output head | 1.06B | 2.7% | 16.98 GB |
| transformer stack | 37.75B | 94.7% | 604.08 GB |
| **total** | **39.88B** | | **638.03 GB** |

Token-facing matrices against one card: **33.96 GB of 80 GB = 42.4%**.

| variant | total params | total memory | token-facing share of 80 GB |
|---|---|---|---|
| untied, AdamW (default) | 39.88B | 638.03 GB | 42.4% |
| **tied**, AdamW | 38.82B | 621.06 GB | **21.2%** |
| untied, inference fp32 | — | 159.51 GB | 10.6% |

> "The two matrices that face the vocabulary are consuming 42.4% of one accelerator before a single
> attention head has been placed. They are 5.3% of the parameters and they do the least interesting
> work in the model."

> "Weights in bf16, gradients in bf16, an fp32 master copy and both AdamW moments: 2 + 2 + 4 + 4 + 4
> = 16 bytes for every parameter being trained."

---

## Widget 4 — Fertility and Embedding Cost (§4)

Presets: V5 mix (default) · English-only · Indic-heavy · Code-heavy. Sliders: six language shares,
vocabulary, model width, fertility sensitivity (default 0.22), price of memory (default 0.030).

**Per-language fertility — identical across all presets, only the shares change:**

| language | fertility | tokens/1M words | cost vs English |
|---|---|---|---|
| English | 1.22 | 1.2M | 1.00x |
| Code | 1.41 | 1.4M | 1.15x |
| Hindi | 2.03 | 2.0M | 1.66x |
| Bengali | 2.55 | 2.5M | 2.09x |
| Telugu | 2.92 | 2.9M | 2.39x |
| Tamil | 2.99 | 3.0M | 2.45x |

**Cost-minimising vocabulary per mix — confirms and sharpens the prose claims:**

| preset | mix | minimum at | mean fertility |
|---|---|---|---|
| V5 mix | En 34 / Code 20 / Hi 13 / Te 12 / Bn 11 / Ta 10 | **V = 101.1K** | 1.89 |
| English-only | En 85 / Code 15 | **V = 53.5K** | 1.25 |
| Indic-heavy | En 18 / Code 10 / Hi 19 / Te 18 / Bn 18 / Ta 17 | **V = 113.4K** | 2.24 |
| Code-heavy | En 28 / Code 52 / Hi 5 / Te 5 / Bn 5 / Ta 5 | **V = 87.5K** | 1.60 |

At V=131.1K on the V5 mix: token-facing matrices cost 2.12B params; "Tamil still costs 2.45 times
what English costs for the same meaning, which is 2.45 times the inference bill and 41% of the
effective context window."

Framing: "English begins at 1.30 and has only 0.30 to gain; Tamil begins at 3.70 and has 2.70."
"every doubling of the vocabulary removes 14% of whatever fertility headroom a language has left."

> "The absolute figure depends on the memory price you chose and should not be quoted on its own.
> The direction it moves when the mix changes is the result." And: "Read the fertility numbers as
> assumptions, not measurements… Replace all of it with the fertilities you measured in the Session 2
> tokenizer assignment before you use this to defend a number."

---

## Widget 5 — Weight Tying Board (§5)

**Default (untied, V5 class d=8096, L=88):** input 1.06B + output 1.06B, total model **71.34B**,
embedding share **3.0%**.

**Embedding share vs model scale (tied):**

| scale | shape | total params (tied) | saved by tying | embedding share |
|---|---|---|---|---|
| 0.5B | d=1280, L=24 | 640M | 168M (2.7 GB) | **26.2%** |
| 1B | d=2048, L=16 | 1.34B | 268M (4.3 GB) | **25.0%** |
| 3B | d=2560, L=32 | 2.85B | 336M (5.4 GB) | 11.8% |
| 7B | d=4096, L=32 | 6.98B | 537M (8.6 GB) | 7.7% |
| 13B | d=5120, L=40 | 13.25B | 671M (10.7 GB) | 5.1% |
| 30B | d=6656, L=60 | 32.77B | 872M (14.0 GB) | 2.7% |
| 70B | d=8192, L=80 | 65.50B | 1.07B (17.2 GB) | 1.6% |
| **V5 class** | d=8096, L=88 | 70.28B | 1.06B (17.0 GB) | **1.5%** |

Chart annotation: "below 5%: tying stops paying".

> "At this scale the token-facing matrices are only 3.0% of the model untied, so the saving from
> tying is close to a rounding error while the constraint is not… This is why Llama-2 at 7B untied,
> and it is why V5 unties. The argument is not that tying is wrong, it is that the exchange rate has
> moved."

---

## Widget 6 — Factorized Embedding Builder (§6)

Sliders: vocabulary (default 131,072), width (default 8,096), bottleneck rank (16–4096 step 16,
default 512), spectrum decay (1.0–12.0, default 5.5).

```
E_small = nn.Embedding(131.1K, 512)
P       = nn.Linear(512, 8.1K, bias=False)
x = P(E_small(token_ids))
# [B,T] -> [B,T,512] -> [B,T,8.1K]
```

**Default (rank 512, decay 5.5):** V·r = 67.1M, r·D = 4.1M, total **71.3M** vs dense 1.06B →
**93.3% reduction**, **50.1% retained energy**. Break-even rank 7.6K; max useful rank 8.1K.

**Retained energy vs rank (decay 5.5):**

| rank | 16 | 64 | 128 | 256 | 512 | 1024 | 2048 | 4096 |
|---|---|---|---|---|---|---|---|---|
| retained energy | 2.2% | 8.3% | 16.0% | 29.4% | **50.1%** | 75.1% | 93.8% | 99.6% |

At rank 128: 17.8M params, **98.3% reduction**, only 16.0% retained energy.

**Spectrum decay sensitivity (rank 512):**

| decay | half-content directions (of 8,096) | retained energy |
|---|---|---|
| 1.0 | 2,292 | 13.7% |
| 5.5 (default) | 510 | 50.1% |
| 12.0 | 234 | 78.1% |

> "93.3% of the parameters are gone and 49.9% of the capacity went with them… The layer has 131.1K
> rows and at most 512 directions." / "Nobody knows V5's value yet, which is exactly what a proxy run
> is for."

---

## Widget 7 — Kronecker Microscope (§7)

Formula as rendered: `kappa(b) = (1/sqrt(L)) * vec( sum over p of c[byte_p] (x) p[pos_p] )`

Fixed config, constant for every token: `char_dim=256`, `pos_dim=32`, `D = 256 x 32 = 8.2K`,
trainable `= D x d_model = 66.3M`. Params card: **66.3M trainable vs 1.06B dense → 16.0x smaller**,
and this is invariant to the vocabulary slider — the point of the widget.

| token | chars | UTF-8 bytes | bytes/char | fits 32-byte window |
|---|---|---|---|---|
| training | 8 | 8 | 1.0 | yes |
| trainer | 7 | 7 | 1.0 | yes |
| the | 3 | 3 | 1.0 | yes |
| def fn( | 7 | 7 | 1.0 | yes |
| भारत | 4 | 12 | 3.0 | yes |
| शक्ति | 5 | 15 | 3.0 | yes |
| தமிழ் | 5 | 15 | 3.0 | yes |
| తెలుగు | 6 | 18 | 3.0 | yes |

`training` byte sequence: 116,114,97,105,110,105,110,103. `भारत` first bytes:
224,164,173, 224,164,190, 224,164,176, 224,164,164 (three-byte Devanagari runs).

---

## Widget 8 — Released `KroneckerEmbedding` module walkthrough (§7)

**The highest-value widget in the session** — it renders the actual module, line by line. Config
panel: `pos_dim=32`, `d_model=8,096`, `vocab_size=131,072`; code size `D = 256 x 32 = 8.2K`;
trainable 66.3M; dense table 1.06B; 16.0x smaller; **fixed buffers = 0 params**; projection = 66.3M.

Source, verbatim as rendered (20 numbered lines):

```python
emb = KroneckerEmbedding(
    vocab_size = 131072,
    d_model    = 8096,
    tokenizer  = tok,
    char_dim   = 256,   # byte alphabet
    pos_dim    = 32,    # byte window
    mode       = "cached",
)

# fixed buffers, never trained
_byte_buffer    (131.1K, 32)   uint8
_length_buffer  (131.1K,)      int16
_codec_table    (131.1K, 8.2K)  cached only

# the only trainable thing in the whole path
projection = nn.Linear(8.2K, 8.1K, bias=False)

def forward(self, input_ids):        # [B, T]
    codec_out = self._codec_lookup(input_ids)   # [B, T, 8.2K]
    return self.projection(codec_out)          # [B, T, 8.1K]
```

Per-line narration, the load-bearing ones verbatim:
- `vocab_size` — "Used only to size the byte lookup buffers, one entry per token. **It does not
  appear in the parameter count. That is the whole point.**"
- `tokenizer=tok` — "At construction the module walks the vocabulary once, encodes every token to
  UTF-8, and stores the bytes."
- `char_dim=256` — "Always 256, because a byte has 256 possible values. This is the height of the
  grid and **it is not really a choice**."
- `pos_dim=32` — "The width of the grid, and the maximum number of bytes any token can contribute.
  **Bytes past this are dropped.** At 32 this is 32 English characters but only 10 Indic ones."
- `mode="cached"` — "'cached' precomputes every token code once and stores the table. 'dynamic'
  rebuilds codes on the fly from the byte buffers. Same maths, a memory-for-compute trade."
- `# fixed buffers` — "Everything below is registered as a **buffer, not a Parameter**. Buffers move
  with the model to the GPU and get saved, but no optimizer ever touches them."
- `_byte_buffer` — "The first 32 UTF-8 bytes of every token. 4.2M bytes of storage… zero trainable parameters."
- `_length_buffer` — "How many bytes each token actually has, so the codec knows what to divide by
  for the `1/sqrt(L)` scaling."
- `_codec_table` — "The precomputed code for every token. Built once under `no_grad`. Still a buffer, still not trained."
- `projection` — "66.3M parameters. **Note what is not in that product: the vocabulary.** Slide
  `vocab_size` and this number does not move at all."
- `_codec_lookup` — "**no gradient flows past here, because there is nothing behind it to train.**"
- `return self.projection(...)` — "One matrix multiply and we are done. The output shape matches what
  a dense `nn.Embedding` would have returned, so attention, the feedforward blocks and the loss are
  all unchanged."

Contract callout, shown throughout:
> "Integers in, vectors out, exactly like `nn.Embedding`. Nothing downstream can tell the difference,
> which is what lets you swap it in, measure it, and swap it back out."

---

## Widget 9 — The Byte Budget (§9 in page order, §8 in the writeup)

One slider: `pos_dim` (8–64, step 8, default 32). At 32: D = 8.2K, projection 66.3M, **16.0x smaller**.

**Word survival at pos_dim = 32:**

| word | note | vs window |
|---|---|---|
| internationalisation | English, long | 20B, fits |
| tokenizer | English, ordinary | 9B, fits |
| शक्तिशाली | Hindi, with conjuncts | 27B, fits |
| తెలుగుభాష | Telugu | 27B, fits |
| தமிழ்மொழி | Tamil | 27B, fits |
| বাংলাভাষা | Bengali | 27B, fits |
| **अंतर्राष्ट्रीयकरण** | Hindi, Devanagari | **51B total, 19B lost** |

Fixed capacity stats at 32: "32 English characters that fit" / "10 Indic characters that fit" /
"3.2x English advantage".

**The collision demo, verbatim:**
- `अंतर्राष्ट्रीयकरण` — 17 chars, 51 bytes. First 32 bytes:
  `224 164 133 224 164 130 224 164 164 224 164 176 224 165 141 224 164 176 224 164 190 224 164 183 224 165 141 224 164 159 224 165`
- `अंतर्राष्ट्रीयता` — 16 chars, 48 bytes. First 32 bytes: **identical**.
- Verdict at 32: "Identical codes. These two words are the same token to the model."

**Sweeping `pos_dim` — the separating value is 48:**

| pos_dim | verdict |
|---|---|
| 32 | identical codes |
| 40 | identical codes |
| **48** | **"Codes differ. At this window the model can still tell them apart."** |

At 48: window holds 48 English / 16 Indic chars; D = 12.3K; projection 99.5M; 10.7x smaller.
`अंतर्राष्ट्रीयकरण` now loses only 3B (51−48).

> "Two different words whose first 32 bytes agree produce the same code, so they get the same vector
> for the whole of training. Nothing errors. The model simply cannot tell them apart, ever."

> "Whether 32 is enough is not an argument, it is a count, and the assignment asks you to run it over
> the real V5 vocabulary and report it per script."

---

## Widget 10 — Frozen Input Path Lab (§9)

Sliders: mixture shift (10–90, default 70 → "+70pp"), warmup steps (0–200, default 0), learning
rate (default 0.40), init seed (default 5). Real in-browser training.

```
vocab      48  (2 domains x 24)
width      16
model      E[48x16] -> W[48x16] + b
objective  softmax cross-entropy
optimizer  SGD, lr 0.40
steps      460  batch 24
domain B   4% -> 74%
shift at   step 200 in one step
freeze at  step 200 (red run only)
both runs identical before step 200
```

| setting | grad norm trainable | grad norm frozen | ratio | loss gap |
|---|---|---|---|---|
| warmup 0 (default), seed 5 | 0.056 | 0.164 | **2.91x** | +0.728 |
| warmup 50 | 0.065 | 0.172 | 2.66x | +0.800 |
| warmup 200 | 0.129 | 0.203 | **1.57x** | +1.044 |
| warmup 0, seed 20 | 0.056 | 0.166 | 2.95x | +0.687 |

> "The frozen run is still paying for the transition when the run ends. Its output projection is
> carrying 2.91 times the gradient of the trainable run long after the shift, and its loss has
> settled 0.728 nats worse."

**Discrepancy worth flagging:** the widget's own text says widening warmup makes "both numbers
fall", but across the three warmup settings sampled the *ratio* falls (2.91 → 2.66 → 1.57) while the
absolute grad norms and the loss gap **rise**. Recorded as measured rather than smoothed over.

---

## Widget 11 — The Wall at max_position (§11)

```
pos_emb = nn.Embedding(16, 12)
head    = Linear(12, 16)
logits  = head(pos_emb(t))
target  = (t*5 + 3) % 16
train t in [0, 8)
eval  t in [0, 16)
rows 8..15 never gathered -> never scatter-added -> still at initialisation
```

**Default (trained 0–7, 400 steps, lr 0.50, seed 4):** accuracy inside range **100%**, past the
boundary **0%**, chance **6.3%** (1/16). Per-position update counts inside range:
`0:791, 1:793, 2:824, 3:749, 4:822, 5:813, 6:810, 7:798`; positions 8–15 all read **"never"**.

Moving the boundary to 12 and to 4 reproduces **100% / 0% / 6.3%** exactly — the cliff tracks the
boundary.

> "Training for longer moves the first number and cannot move the second, because the gradient that
> would have to reach those rows is never generated."
> "There is no parameter anywhere in this model that connects row 7 to row 8. They are independent
> rows in a lookup table."

---

## Widget 12 — Position Family Map (§12)

| family | made of | parameters | past trained length | status |
|---|---|---|---|---|
| Absolute learned | stored params, one row per position | max_position x d_model | **hard wall**; rows past trained length hold init values | BUILT AND MEASURED TODAY |
| Sinusoidal | deterministic function of t | none, computed each forward | defined everywhere; usability past range untested | NAMED TODAY, NOT BUILT |
| Rotary (RoPE) | computed rotation on Q/K | none stored; frequency schedule is a design choice | degrades rather than breaks | BUILT IN SESSION 8 |
| Attention bias (ALiBi) | fixed distance-dependent bias on scores | one slope per head | extrapolates gracefully by design | BUILT IN SESSION 8 |

Per-family trade lines:
- Absolute: "Maximally expressive inside the trained range because every position gets its own free parameters, and useless outside it for exactly the same reason."
- Sinusoidal: "Generalises by construction and constrains what can be represented, because the model no longer gets to choose what each position means."
- Rotary: "Relative by construction, which is usually what you wanted, at the cost of the position signal being entangled with the attention computation."
- ALiBi: "The strongest structural prior on this list and the fewest parameters, which is the same bargain as the strongest compression in the first half of this session."

> "This is the same exchange the first half of this session made twice, once when a dense table
> became a factorized one and once when a factorized table became a Kronecker one. Substituting
> structure for stored parameters buys generalisation and costs expressiveness, every time."

---

## Widget 13 — V5 Embedding Design Board (§13)

Sliders: vocabulary (16,384–262,144, default 131,072), width (1,024–12,288, default 8,096),
`pos_dim`/rank (8–1,024, default 32). Selects: input path (**kron** / fact / dense), tying
(**untied** / tied), freeze policy (**trainable** / scheduled / frozen), identifier layout
(**frequency_sorted** / script_blocked / shuffled), position policy
(**deferred_session_8** / absolute_learned).

**Default:** 66.3M input path, 1.06B output head, **18.04 GB training state**, 506.0 stored
parameters per token, **"No gates failed, 1 flagged."**

The one flagged gate at default:
> **! Byte window for Indic scripts** — "At pos_dim = 32 the window holds 32 English characters but
> only 10 Indic ones, because UTF-8 spends three bytes per character there. The shipped default of 32
> lets real word pairs collide, which the model can never undo. **Set this from a collision count on
> the actual vocabulary, not from the default.**"

**Default `embedding_policy_id` record, verbatim:**

```json
{
  "embedding_policy_id": "v5-emb-kron-r32",
  "vocab_size": 131072,
  "d_model": 8096,
  "input_path": "kron",
  "pos_dim": 32,
  "char_dim": 256,
  "codec": "byte_kronecker_fixed",
  "code_dim": 8192,
  "params_input": 66322432,
  "output_head": "untied",
  "params_head": 1061158912,
  "trainable_state": "trainable",
  "unfreeze_schedule": null,
  "identifier_layout": "frequency_sorted",
  "position_policy": "deferred_session_8",
  "tokenizer_hash": "REQUIRED",
  "cleaning_pipeline_hash": "REQUIRED",
  "training_state_bytes": 18039701504
}
```

**States exercised:**

| change | params_input | training state | stored params/token | gates |
|---|---|---|---|---|
| default (kron, r32) | 66.3M | 18.04 GB | 506.0 | 0 failed, 1 flagged |
| input path → `fact` (rank 32) | 4,453,376 | 17.05 GB | 34.0 | 0 failed |
| input path → `dense` | 1,061,158,912 | 33.96 GB | 8096.0 | byte-window gate becomes N/A |
| output head → `tied` | 66.3M | **1.06 GB** (`params_head: 0`) | 506.0 | 0 failed, **2 flagged** |
| freeze → `frozen` | 66.3M | 18.04 GB | 506.0 | **1 FAILED**, 1 flagged |
| freeze → `scheduled` | 66.3M | 18.04 GB | 506.0 | OK |
| vocab 262,144 + width 12,288, kron | 100.7M | 53.15 GB (66.4%) | 384.0 | flagged |
| vocab 262,144 + width 12,288, dense | 3.22B | **103.08 GB (128.8%)** | 12,288.0 | **FAILED** |

The hard-fail gate on freezing, verbatim:
> **X Freezing a compressed input path** — "This is the exact combination that produced the V4
> incident. A compressed path has fewer degrees of freedom to absorb a distribution change than a
> dense one, so freezing it removes adaptation at precisely the boundary where mixture shifts arrive."

Tying a structured input path, flagged:
> **!** "A Kronecker or factorized input path and a dense output head are not naturally the same
> object, so tying them constrains the output head to whatever structure the input path has."

Gate thresholds inferred from the sweep: training state OK below ~65% of the card, flagged 65–100%,
hard fail above 100%.

Closing callout:
> "Passing every gate does not make this design correct, it makes it defensible on the arguments this
> session made. Every number in it is a hypothesis until a proxy run has tested it, which is the
> standard Session 5 set for mixture decisions and there is no reason an architecture decision should
> be held to a lower one."
