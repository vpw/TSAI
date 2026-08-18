# Session 8: Modern Attention Variants

Full lesson writeup, pulled verbatim from the Axiom lesson page and restructured under the
session's own numbered headings. Widget descriptions are summarized inline at the end of each
section (see the "Widget:" line) — `get_page_text` captures the surrounding prose and the widget's
own narrated caption, not its live rendered state.

## 1. What this session is

At the end of Session 7, we had reached this point:

```
token id
  ↓
fixed byte codec
  ↓
code
  ↓
trainable projection
  ↓
[B, T, D]
```

So every token has now become a vector of width D. The obvious next question is: what consumes
those vectors? That is what this session is about.

Attention is the first place where tokens start interacting with one another. Until now, token 17
could become a beautiful 4,096-dimensional vector and still know absolutely nothing about token 16
or token 18. Attention gives each token a way to look at the other tokens in its context and bring
back the information that matters.

We will build the ordinary version first. Only after that will we ask what becomes difficult when
the context gets long: comparing every token with every other token becomes expensive, and
remembering the useful parts of every earlier token takes memory. Those are two different problems.
The rest of the session is a tour of the different mechanisms people have proposed to reduce one or
both of them.

## 2. Attention

We have used the word attention since Session 1, but deliberately not built it yet. Now we will.

Assume one token enters the attention layer as a vector `x`. The layer makes three different
projections of that same vector:

```
        ┌── Wq ──> query
x ──────┼── Wk ──> key
        └── Wv ──> value
```

Why three? Because looking for information and giving information are different jobs.

- `query` = what am I looking for?
- `key` = what kind of information do I contain?
- `value` = what information should I give you if you choose me?

Suppose the sequence is "The cat sat on the mat". When the model is processing `sat`, its query may
strongly match the key for `cat`, because knowing who sat matters. If that match is strong, more of
`cat`'s value flows into the representation for `sat`. That is attention in one sentence: each token
asks a question, every other token advertises what it contains, and the best matches contribute the
most information.

The standard scaled dot-product attention equation says exactly the same thing:

```
Attention(Q, K, V) = softmax( QKᵀ / sqrt(d_k) ) V
```

Read it from the inside out:

- `QKᵀ` compares every query with every key and produces the score matrix.
- Dividing by `sqrt(d_k)` keeps those scores from growing too large as the query/key width
  increases. `d_k` is the number of components in each query and key for one head.
- `softmax` is applied across each query's row of scores, turning that row into positive weights
  that add to one.
- Multiplying by `V` uses those weights to combine the value vectors.

Mechanically, standard scaled dot-product attention is only four steps: **Score** (dot product of
current query with every key), **Scale** (divide by `sqrt(d_k)`), **Softmax** (convert to positive
weights summing to one), **Weighted sum** (multiply every value by its weight and add).

For a decoder model there is one extra rule: the **causal mask**.

```
CausalAttention(Q, K, V) = softmax( QKᵀ / sqrt(d_k) + M ) V
```

`M` is 0 where attention is allowed and `-∞` for a future token, added before softmax, so every
forbidden position receives weight zero. If generating token 4, it may read tokens 0-3 but not
token 5. During training we process the whole sequence in parallel and set every forbidden future
score to `-∞` before softmax (`softmax(-∞) = 0`), which lets us train on the whole sequence at once
while preserving left-to-right generation.

**Two facts from standard attention matter for the rest of the session:**

- **Fact 1:** every token is compared with every other token. 6 tokens → 36 scores. 600 tokens →
  360,000. 10,000 tokens → ~100 million scores. This exact all-to-all comparison is both the
  strength and the cost of attention.
- **Fact 2:** attention itself does not know token order. If token 0 and token 4 contain the same
  word and enter with the same vector, their Q/K/V projections are also identical. Something else
  has to tell the model which came earlier. Fixed in Section 8.

**Widget:** a live small transformer (32 dims, 4 heads, 2 blocks) running in-page — follow one word,
watch its embedding split into Q/K/V, see the ribbons and the dot-grid of attention weights, and the
next-word bars on the right; a temperature slider sharpens/spreads the prediction. A second widget
walks 6 tokens (4 numbers each) through Q/K/V projection, the 36-score matrix, scaling+softmax, the
causal mask, and the weighted sum — turning the mask off visibly leaks attention onto future tokens.

## 3. The two bills attention sends

The first problem: as context grows, every new token has more other tokens to compare with.
`T=1,000 → 1,000,000` query-key scores; `T=10,000 → 100,000,000`; `T=100,000 → 10,000,000,000`.
That is the first bill: **compute**, growing ~`T²` — double the context, ~4x the work.

The second bill belongs to generation. When a model writes one token at a time, it keeps the
already-computed keys/values for earlier tokens rather than recomputing them:

```
after "The"          → K1, V1
after "The cat"       → K1,V1, K2,V2
after "The cat sat"   → K1,V1, K2,V2, K3,V3
```

This growing list is the **KV cache** — the model's saved attention history for one active
conversation. The two bills grow differently:

| BILL | WHAT GROWS | SIMPLE QUESTION |
|---|---|---|
| Compute | token-to-token comparisons, ~T² | Can we afford to calculate all those comparisons? |
| KV-cache memory | one saved K/V per earlier token, ~T | Can we afford to keep all that conversation history? |

Model weights are mostly fixed — load once, share across users. A KV cache is *not* shared: each
user's conversation gets its own cache. So a longer context creates two separate engineering
questions: does attention become too expensive to *compute*, or does the saved history become too
large to *store*?

**Widget:** a context-length slider showing attention compute growing quadratically against the KV
cache growing linearly — introduces the two growth patterns; the full cache formula arrives in
Section 10.

## 4. What happens if you remove the softmax

Everything through Section 7 builds on standard attention. Start with: what if we remove the
softmax?

**Without softmax**, given query `q=2` and three key-value pairs `(k1=0.5,v1=10)`, `(k2=1.0,v2=20)`,
`(k3=1.5,v3=30)`:

```
output = (q·k1)v1 + (q·k2)v2 + (q·k3)v3 = 1·10 + 2·20 + 3·30 = 140
```

Regroup by factoring out `q`: `output = q × (k1v1 + k2v2 + k3v3) = q × S`, where
`S = k1v1+k2v2+k3v3 = 70`, and `q × S = 140` — same answer. `S` is a **running state** that can be
precomputed as old tokens arrive, updated as `S = S + key×value`, instead of keeping a growing list
of individual `(k,v)` pairs. With real vectors: `S = Σ vⱼkⱼᵀ` (a matrix), `y = Sq`. Its shape (e.g.
4×4) does not grow with the number of tokens — after 10 tokens or 1,000,000 tokens, still one 4×4
matrix.

**With softmax back**: raw scores `[1,2,3]` → `exp` → `[2.72, 7.39, 20.09]`, total ≈ 30.20 → weights
`≈[0.09, 0.24, 0.67]` → output ≈ 25.8. The key point is the *shared denominator*
`Σ exp(score)`: to weight key 1 you need the scores of keys 2 and 3 too — softmax ties the scores
together, so it needs the individual old keys to remain available when each new query arrives.
Without softmax, terms are independent and can be pre-combined; with softmax, each new query must
revisit the individual old keys.

This is what opens the door to **linear attention**: replace a growing exact history with a
fixed-size state, making sequence work grow linearly instead of quadratically.

**What softmax was doing:** made every weight positive; made weights sum to one (bounded output
scale); created competition between keys; gave every query a fresh distribution over the exact old
keys. Removing it gives up all of that — raw scores can be negative/unbounded, the accumulated
state's magnitude can grow, and different memories compressed into one fixed-size object can
interfere with each other.

| EXACT SOFTMAX ATTENTION | SIMPLE NO-SOFTMAX STATE |
|---|---|
| Keeps individual keys and values | Compresses them into one state |
| Positive weights summing to one | Raw, unnormalised scores |
| Fresh distribution per query | Same accumulated state, read differently |
| More exact access to old tokens | Cheaper but lossy memory |
| Growing KV cache, expensive all-pairs work | Fixed-size state, linear sequence work |

Practical linear-attention systems rarely just delete softmax — they add back useful behavior via
feature maps, explicit normalization, gates, decay, and better write rules. Hybrids also keep
occasional softmax/sparse layers for direct access to old tokens. Sets up: §5 (naive writes
interfere), §6 (delta rule), §7 (sparse attention keeps softmax but reads fewer keys), §13 (hybrid
schedules).

**Widget:** side-by-side "visit every old key" vs. "read one pre-built state" — with softmax off,
both paths match exactly; turn softmax on and a Replay control shows where they diverge.

## 5. A new state can still carry the old contribution

The fixed-size state from §4 is a matrix — think of it as a scratchpad with two jobs: **WRITE**
(connect a clue/key to an answer) and **READ** (use the clue to recover the answer).

First write: `key A → old answer`. Later, the answer changes: `key A → new answer`. The new answer
should *replace* the old one — but the update rule is `new state = old state + new write`, so the
old numeric contribution is still baked in. Example: key A currently returns 40, should now return
55. An add-only write gives `new read for A = 40 + 55 = 95` — **not** 55. There is only one current
matrix after each write, and its fixed size isn't the problem; the problem is the add-only rule
always carries the old contribution forward with no correction term.

**Widget:** two sequential writes to key A (40, then 55) show the resulting read is 95, not 55 —
because there are never two separate state matrices, only one running sum.

## 6. The delta rule: write only what needs to change

If key A currently returns 40 but should now return 55, adding the complete new answer repeats §5's
mistake (`40+55=95`). The delta rule first measures the gap:

```
current answer: 40
wanted answer:  55
correction (delta): 55 - 40 = 15
write the correction: 40 + 15 = 55
```

Read what memory says, find the difference, write only that difference. This matters for attention
because removing softmax gave a compact state, but an add-only state can't revise an old
association — the delta rule turns it into an *updateable* memory. (Modern variants also learn how
strongly to write, or when old information fades — later refinements, not required here.)

**Widget:** one correction walkthrough — read 40, compute +15, reach 55 — contrasted with the
add-the-whole-answer path that reaches 95.

## 7. The other lever: do not look at everything

Linear attention changes the *kind* of memory. Sparse attention keeps normal softmax attention but
lets each query use only a small number of keys. Full attention uses all `T` keys; **top-k**
attention keeps only the `k` highest-scoring keys and drops the rest before forming the output:
`compare query with candidates → keep best k → softmax over those k → combine those k values`.

This works when useful attention concentrates on a few tokens — an empirical property, not a
guarantee. The catch: "keep the best k" still requires scoring every candidate first
(`score all T keys → discover top k → use only k values`) — that reduces work *after* selection but
doesn't remove the cost of scoring everyone. Practical sparse systems need a cheaper *proposal* step
(a local window, a learned router, a compressed index) so exact attention runs only on the
candidates it suggests. Real trade-off: cheaper candidate search vs. risk of missing a useful key.

**Architecture note:** LightningLM V4 used sparse-attention G-layers alongside DeltaNet layers, with
its max budget reduced from 1024 to 256 due to backward-kernel contention on that hardware/software
stack — 256 is an implementation constraint from that run, not a universal law. V4 also varied the
budget by token (a later design choice).

**Widget:** move `k` through a 12-key example — every key gets scored, only the top `k` survive,
only those enter the weighted sum; bars show value-work falling with `k` while naive selection cost
stays flat at "score all twelve."

## 8. Position, and the part that is usually skipped

The causal mask handles *future* tokens, but among tokens a query is allowed to see, the dot product
still needs a clue about *distance* — attention itself is otherwise blind to order (§2, Fact 2). If
"bank" appears at position 2 and again at position 20 with identical vectors, content-only
comparison can't tell which occurrence is nearby.

**RoPE (Rotary Position Embedding)** handles this via rotation. Take two dimensions from a query or
key as one 2D arrow `(x0, x1)`. Choose a small rotation angle `θ` per sequence step; rotate the
arrow by `position × θ`. A query at position `i` is rotated by `iθ`, a key at position `j` by `jθ`.
Their dot product depends only on the *angle between* the two arrows — both absolute rotations
cancel, leaving only the difference:

```
R(iθ)q · R(jθ)k = q · R((j−i)θ)k
```

So the positional part of the score depends on `i − j`, the distance. Move both tokens forward
together (positions 2,8 → gap 6; positions 12,18 → gap 6) and the angle between them — hence the
positional relationship — stays the same. RoPE turns relative token distance into relative rotation
inside the attention score, which generalizes better than an absolute "I am token 8" label.

**Implementation detail people miss:** RoPE groups dimensions into 2D pairs and rotates each pair,
possibly at different rates, letting a head represent distance at multiple scales. Not every
dimension has to be rotated — DeepSeek-V4 applies RoPE only to the last 64 dimensions (an
implementation choice, not a change to the mechanism). RoPE's rotation is computed from a function,
so it has no finite lookup-table wall and *can* be evaluated beyond the training length — but that
doesn't prove the model behaves well there (§9 handles that separately).

**Widget:** toggle RoPE off/on — off, identical content vectors reveal no distance; on, changing the
distance changes the relative angle and score; moving both tokens together keeps distance/angle/
score fixed even as absolute positions change.

## 9. DroPE, and how eight thousand became two hundred and fifty-six thousand

RoPE replaced a finite lookup table with a rotation computable at any position — but "calculable" ≠
"works well." A model trained only up to 8K tokens can still be asked for the rotation at position
256K; the formula answers, but that only proves the positional rule is *defined* at 256K, not that
the model can *use* a 256K context reliably (a claim about the whole trained model, since every
layer learned within the 8K range it actually saw).

The V4 record: `trained context: 8K`, `reported context: 256K`, `extension: 32×`, with
`positional recalibration: DroPE, applied before annealing`. "Before annealing" matters — it means
this happened *during* training, with learning rate and steps still available to adapt, not a
magic inference-time switch.

**Evidence boundary:** what the record establishes — V4 trained at 8K, reportedly reached 256K,
using a step called DroPE before annealing. What it does *not* establish — the exact DroPE algorithm
or which rotary dimensions it changes; any more detailed mechanism is a hypothesis until checked
against the reference implementation. Main lesson: a position function can exist beyond training
length; model capability there must still be earned and demonstrated. `256K ÷ 8K = 32×`.

**Widget:** switch between plain RoPE at 256K (rotation still computes, says nothing about quality)
and the reported V4+DroPE result (8K→256K, 32×, recalibration before annealing) — the widget
deliberately does not simulate an unverified DroPE mechanism.

## 10. The cache bill, done properly

Building the KV-cache formula factor by factor. One token in one layer stores one key + one value
vector (`2 ×`). Each vector has `head_dim` numbers, one pair per KV head
(`2 × kv_heads × head_dim`). Every layer makes its own K/V (`2 × layers × kv_heads × head_dim` per
token). Over `T` tokens: `2 × layers × kv_heads × head_dim × T` numbers per user. In bf16 (2
bytes/number):

```
cache bytes per user = 2 × layers × kv_heads × head_dim × T × bytes_per_number
```

Multiple concurrent users don't share history — with `B` active users:

```
total cache bytes = 2 × layers × kv_heads × head_dim × T × batch × bytes_per_number
```

**Worked example** (this lesson's yardstick: 48 layers, 8 KV heads, head_dim 128, bf16, T=32,768):
one user ≈ **6.44 GB**, eight users ≈ **51.54 GB**. Double the context and both numbers double;
double active users at fixed context and total also doubles. This formula counts only the raw K/V
tensors — a real server also needs memory for weights, activations, attention workspaces, and
allocator headroom.

**Widget:** a progressive cache-formula builder — "Next multiplier" reveals each factor in turn;
sliders for context length (per-user cache) and active users (batch multiplier), architecture held
fixed at the 48-layer/8-KV-head/128-head_dim/bf16 yardstick.

## 11. Grouped-query attention, and why it is the baseline rather than the answer

`cache size ∝ kv_heads × context length`. **GQA** reduces the `kv_heads` term: query heads stay
separate, but several share one K/V head (e.g. query heads 1-4 → shared K/V head A, 5-8 → shared
K/V head B). Compare 8 query heads: **MHA** = 8 query heads/8 KV heads; **GQA** = 8/2 (4× smaller KV
cache than MHA here); **MQA** (multi-query attention) = 8/1, the one-KV-head extreme, saving even
more cache at greater risk to quality (widget compares memory scaling only, not the quality
trade-off).

Why is GQA a *baseline*, not the answer? It changes how much is stored **per token**, but still
stores *something* for every token: MHA cache `∝ 8×T`, GQA `∝ 2×T`, MQA `∝ 1×T` — GQA lowers the
slope but the line still grows linearly; at a million tokens it still holds a million positions'
worth of entries. GQA is a strong practical baseline (large cache saving, useful quality trade-off)
but does not solve long-context memory on its own.

**Widget:** switch MHA/GQA/MQA to see 8 query heads reuse fewer stored KV heads; increase context
length — sharing flattens the slope but every line still climbs.

## 12. Compressing the sequence itself

Even with GQA, the cache still keeps one entry per token position. This section reduces the *other*
term: what if several nearby positions shared one stored entry? Combine `m` tokens into one block
summary: stored positions fall from `T` to `T/m`. GQA stores fewer heads per position; sequence
compression stores fewer *positions*.

**DeepSeek-V4's Compressed Sparse Attention** adds a second saving on top: after block summaries,
it doesn't run expensive attention over every summary — it selects only the top-k most relevant
summaries (`all tokens → compress → fewer block summaries → select top-k → fewer expensive attention
reads`). Compression reduces entries *stored*; top-k reduces summaries *read*. But selecting the
best blocks cheaply is itself a problem — if ranking requires full attention over every block, little
is saved. DeepSeek uses a small low-rank **indexer** to rank block summaries cheaply; the indexer
doesn't produce the final output, only chooses which blocks deserve the expensive read.

Trade-off: compression can lose token-level detail (one summary speaks for several tokens);
approximate top-k selection can miss a useful block. The reported DeepSeek-V4 architecture also
interleaves a heavily compressed dense form with the top-k sparse form — old history doesn't have to
keep one equally expensive representation per original token.

**Widget:** two-stage visual — increasing tokens-per-block shrinks stored positions; changing top-k
limits how many summaries expensive attention reads; a note on the low-rank indexer's cheaper
ranking role.

## 13. Schedules across depth

A deep model doesn't have to pick standard/linear/sparse attention once for the whole network —
**different layers can use different memory systems**. A fixed-state layer is cheap to serve (memory
doesn't grow with context) but loses token-by-token access to the old sequence. A sparse-attention
layer keeps exact keys/values and restores direct access, but adds a KV cache that grows with
context. This creates a division of labor: many fixed-state layers process/combine information
cheaply; occasional sparse-attention layers revisit selected earlier tokens directly. The layer
ordering is the **depth schedule**.

LightningLM V4 used a repeating 8-layer motif `D D D G D D D G` (D = DeltaNet fixed-state layer, G =
sparse-attention layer) — 6 fixed-state, 2 sparse per motif. Going from 1-in-8 to 8-in-8 sparse
layers makes KV state **8.0× larger** while estimated mixing compute rises only **~1.41×** — for
this configuration, the price of more G layers is mostly serving memory, not FLOPs. Too much D
everywhere means the model never gets a direct look at exact old token representations.

V4 kept `DDDGDDDG` from the 1.78B seed model to the 120B run — meaningful evidence the mixture works
across scale, but *not* evidence 6:2 is the optimal ratio (neighboring schedules were never cleanly
ablated). **`DDDGDDDG` is a successful design choice, not a proven optimum.** Broader lesson: the
schedule itself is part of the architecture.

**Widget:** compare three depth schedules — D blocks compress history into fixed state, G blocks
give direct access to earlier tokens; the V4 preset is marked as reported evidence, alternatives as
illustrations only.

## 14. State that outlives the window

Everything so far describes memory *inside* the current sequence window. Split a long document into
chunks: chunk 1 ends, its internal attention state is discarded — what crosses the boundary?
Without an explicit mechanism: nothing.

**V4's Memory Stream:** at the end of a chunk, take its final hidden state and use it as one summary
vector nudging the next chunk (`chunk 1 → one memory vector → chunk 2`). The vector has the model's
width and does not grow with document length — fixed-size, O(1) cross-chunk state, and a severe
compression (one vector can't preserve every sentence, only a useful summary signal).

Two rules make it practical:

1. **Stop-gradient write:** `memory = stop_gradient(final hidden state)` — the next chunk reads it
   on the forward pass, but training loss cannot backpropagate through the boundary into all
   previous chunks (avoids one enormous cross-document training graph).
2. **Learned per-token gate:** `updated token = current token + scale × gate × memory`, gate ∈
   [0,1] — near 0 means the token mostly ignores the old summary, large means the summary can
   dominate. The gate is learned because different tokens need different amounts of old
   information.

Reported V4 injection scale starts near **0.078**, reaches about **0.391** by the 2B stage; average
injected memory is only about **6%** of the current embedding magnitude — a small nudge from the
previous chunk, not a copy of it and not a replacement for the current token.

**Widget:** cross a chunk boundary and ask a question in chunk 2 — stream off, nothing survives;
stream on, adjust the learned gate (closed = ignore memory, default ≈ 6% nudge); a red crossed arrow
marks the stop-gradient rule (information flows forward, gradients don't flow back).

## 15. Long context is a system, not a number

"Our model supports 256K context" sounds like one specification — it isn't. Six things must work
together:

1. **Position must still make sense** — trained natively at that length, or extended successfully
   (DroPE, NTK-aware scaling, YaRN).
2. **The cache must fit** — a theoretical 1M-token model is useless if one sequence exhausts
   accelerator memory (GQA, MQA, sequence compression matter here).
3. **Compute must be affordable** — even if the cache fits, full quadratic attention over the whole
   context may still be too expensive (sparse attention, top-k, linear attention matter here).
4. **State outside the active window must be handled** — chunking/discarding at boundaries may need
   a mechanism to carry information forward (V4's Memory Stream is one answer).
5. **Training must expose the model to the behavior expected** — "architecture is mathematically
   defined at 256K" ≠ "model has learned to use 256K well."
6. **Evaluation has to test understanding, not only retrieval** — a model can find one planted
   sentence in a huge context while failing to combine evidence or follow long dependencies; a
   passing needle-in-a-haystack test is useful but not a complete long-context evaluation.

**Context length is not one number — it's the point where position, memory, compute, training and
evaluation all still work together.**

One more dimension specific to the model being built: context windows are advertised in tokens, but
users have documents in languages. Session 3 showed fertility differences (English ~1x tokens,
Telugu ~3x tokens for the same meaning) — a fixed 256K-token window holds roughly 1/3 as much
comparable Telugu content as English. The accelerator charges per token regardless. Tokenizer
efficiency determines how fast a language consumes the window; embedding/architecture choices
determine model cost; attention determines what happens when the window gets long. For an
India-first model, long context also helps recover effective document length lost to higher-fertility
tokenization.

**Widget:** a context-readiness board — choose a target context and model configuration, the board
checks all six conditions and names which constraint fails first and by how much; alongside it, the
same document rendered under different tokenizer-fertility assumptions shows the cost as lost
document capacity rather than an abstract ratio.

## 16. What V5 has to decide

V4 is finished and gives evidence — it does not automatically dictate the V5 configuration. The
frontier moved, hardware and kernels changed, and some V4 choices were never ablated. The useful
question isn't "what did V4 use" but "which V4 decisions are now evidence, which are merely
defaults, and what must be re-tested for V5?"

**What current evidence already settles:**

- A stored absolute position table is out (Session 7's hard length wall).
- GQA alone is not enough for the target — it reduces KV-cache growth by a constant factor but
  doesn't remove linear growth with context.
- Some stronger long-context mechanism is required — every one of the three reference architectures
  uses at least one of linear state, sparsity, or sequence compression; none uses none.

**Open questions, more interesting:**

| OPEN QUESTION | NEEDS |
|---|---|
| Extend or build native | V4 stretched 32× with DroPE; DeepSeek built for a million natively. Extension is cheaper but has a ceiling; native long training costs much more. → An extension-factor study at V5's target, plus a real compute/memory estimate for training natively at that length. |
| The schedule and its ratio | `DDDGDDDG` worked but was never varied systematically. → A schedule ablation at a scale large enough to likely transfer. |
| Linear state vs. sequence compression as the primary lever | V4/Qwen3.6 chose the first family, DeepSeek chose the second. → A matched-budget comparison at genuinely long context, not a short-context benchmark. |
| The sparsity budget | V4's cap of 256 came partly from kernel constraints on the previous hardware/software stack. → Re-measurement on current kernels and hardware. |
| Whether the Memory Stream still earns its place once the window is already very long | An ablation at the actual target context length. |
| Every head count and head dimension | Derived around a width of 4,096; V5 hasn't committed to that width. → Decide model width first, then derive attention geometry from it. |

Known design pressures: position must extrapolate or be trained long; cache cannot remain huge;
compute cannot remain quadratic everywhere; exact memory is still useful somewhere; fixed state is
cheap but lossy; compression is cheap but destroys detail unless repaired. What remains is an
experimental question of which mixture and ratios give the best trade at V5's actual target.

**Widget:** a V5 attention decision board — build a candidate from layer schedule, position scheme,
extension policy, sparse/compressed attention, head layout, cache precision, cross-chunk state, and
target context; reports cache/compute/reach/training-cost estimates using the session's formulas,
naming failed constraints numerically.

## 17. Where this leaves you: two roads

**Road 1 — train short, then stretch.** Train at an affordable context length, then modify/
recalibrate the positional system to operate much farther beyond it
(`affordable training length → positional extension/recalibration → much longer serving context`).
The cheaper road — this is V4's path with DroPE (8K → 256K). Limitation: an extension method has a
practical ceiling; 32× working doesn't prove 320× or 3,200× will.

**Road 2 — build long, then train long.** Design the architecture so long sequences are affordable
natively, reducing compute/memory enough to actually train on long sequences rather than
extrapolating position after training
(`long-context architecture → train on long sequences → native long-context competence`). The
expensive road — DeepSeek's compression/sparsity approach is an example. No positional-extension
ceiling, since the model trains in the long regime itself, at the cost of training complexity and
cost.

Neither road is automatically correct — it depends on target context, training budget, serving
budget, expected workloads, and how much quality is lost when stretching. Several mechanisms serve
either road: GQA reduces cache in both; sparse attention can reduce long-context compute in both;
better positional schemes help both. Before the assignment: put an actual target on the problem
(32K? 256K? 1M? 10M?) — only then ask which road is realistic.

**Widget:** the fork behind the whole session — pick a target context and trace both roads through
training cost, KV cache, inference compute, and failure mode; "long context" is a system design
choice, not one technique.

## 18. The assignment

(Full text also in `S8-assignment.md`.) This one is for the whole cohort — the instructor wants to
keep the best submission and put it in front of next year's batch. Build a web app (with the AI
agent) hosted on Netlify/Vercel/equivalent that visually explains every attention mechanism covered
this session, starting from standard scaled dot-product attention (Q×K → scores → scale → mask →
softmax → weighted sum of V) before anything that modifies it (GQA, RoPE, DeltaNet, etc.).

**The part the instructor cares about most:** arrange the mechanisms **chronologically by actual
launch date**, not in teaching order and not grouped into a taxonomy — because the point is to show
the field's priorities shifting over time (exact global attention → cheaper decoding memory →
better position handling → longer contexts → recurrent state returning → sparsity returning →
compression becoming more aggressive). A chronological story shows *why* an idea appeared; a list
only shows that it exists.

For every mechanism: honestly written pros and cons — what it buys, what it gives up, when you'd
actually choose it (a mechanism can be right for a 2K chatbot and wrong for a 1M-token agent without
being a bad mechanism).

**Minimum coverage:** standard attention, absolute learned positions, sinusoidal, RoPE, ALiBi, MQA,
GQA, sliding window, attention sinks, NTK-aware scaling, YaRN, linear attention, the delta rule and
Gated DeltaNet, MLA, sparse and top-k attention, DeepSeek's compressed sparse attention, and DroPE.
Finding and adding a relevant mechanism not covered in class counts in the submitter's favor, held
to the same standard (original date, motivation, mechanism, advantage, cost, timeline placement).

**Submit:** the live link, the GitHub repository, and a README listing sources for the chronology —
dates are exactly where an agent can sound confident while being wrong (the instructor cites getting
a mechanism's date wrong in Session 7 and having to correct it live at the start of Session 8).
