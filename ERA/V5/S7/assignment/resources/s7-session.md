# Session 7: Embeddings and Model Internals

Source: https://axiom.theschoolofai.in/courses/cmq97i5kn032208o8xu5dab4q/sessions/cmsipt1x20w6608nhlglutv9x/lesson
(lesson prose, extracted 2026-08-10 via `get_page_text`; widget captions are summarized inline
below since they are described in prose rather than separately linked pages this session)

## 1. What this session is

Four sessions ago this course turned away from the model and went to work on the data, and it
stayed there. Session 3 decided what to collect, Session 4 built the pipeline that turns
collected bytes into admissible text, Session 5 arranged the clean pools into a mixture and a
curriculum, and Session 6 compiled that recipe into an executable stream with a ledger behind
it. That arc is finished. What comes out of Session 6 is a packed batch of integer token
identifiers, and this session is where those integers finally meet the model.

The place we are standing was named precisely once before. Session 2 ended on the sentence that
the tokenizer's job is complete the moment text has become a sequence of integers, and that the
question of what vector each integer turns into belongs to a different component on the far
side of a clean seam. This is that component, and this is that class. Session 2 also left an
unpaid debt here in writing, which is the full treatment of Kronecker factorization, deferred at
the time with the note that it is an embedding-side trick and belongs to the class on embeddings
and model internals. That debt is the centre of today.

It is worth saying plainly that Session 6 closed by pointing forward at the distributed training
loop, gradient accumulation and parallelism. That pointer was wrong against the calendar. Those
belong to Sessions 10 through 13, and there is a reason the model's front and back doors come
first. The two matrices that face the token vocabulary are the largest single tensors in the
model, they are decided before any training system is designed around them, and at the
vocabulary and width V5 is heading toward they are large enough that the decision changes what
the rest of the architecture can afford.

Here is the number that makes the session necessary. The V4 BrahmicTokenizer carries a
vocabulary of 131,072 tokens, and the model width that vocabulary was paired with is 8,096. A
dense input embedding table at that shape holds

`131,072 x 8,096 = 1,061,158,912 parameters`

which is a little over a billion parameters, occupying 2.12 GB in bf16, before a single
attention head or feedforward block exists. If the output projection is untied it is a second
matrix of the same size. This is the storage problem Session 2 set up and handed forward, and
the whole of today is spent first making it visible and then designing a way out of it that we
can defend.

A view of the seam itself, with Session 6's packed batch on one side and the model's first
objects on the other. Each field the dataloader carries, the token ids, the loss mask, the
position ids and the ledger tags, is traced to the model object that consumes it and to what
breaks when it is wrong. The point to leave with is that the tokenizer's contract ends at an
integer, and everything in this session lives on the far side of that boundary.

## 2. The lookup is a gather, not a matrix multiply

Almost every explanation of an embedding layer describes it as a matrix multiplication against a
one-hot vector, and that description is mathematically true and operationally misleading. Nobody
builds a 131,072-element one-hot vector and multiplies it by a billion-parameter matrix in order
to retrieve one row. The operation that actually runs is a gather: the integer is an offset, the
hardware reads the row at that offset, and the cost is a memory read rather than an arithmetic
one. Holding the gather picture rather than the matrix-multiply picture is what makes the rest of
this session's behaviour predictable instead of surprising.

The consequence that matters lives in the backward pass. When gradient flows back into an
embedding layer it does not update the table. It updates the rows that appeared in this batch,
and it leaves every other row untouched. The operation is a scatter-add, which accumulates one
gradient contribution per occurrence of a token into that token's row and writes nothing anywhere
else. A row that did not appear in this batch receives exactly zero gradient this step, and under
a plain optimizer it does not move at all.

Now put a real batch against a real vocabulary. A global batch of 256 sequences at 8,192 tokens
is 2,097,152 token positions, and it is tempting to conclude that a batch that large leaves most
of the table untouched. It does not, and it is worth being precise here rather than reaching for
the dramatic version. At that batch size nearly every row is gathered on nearly every step,
because two million draws against a hundred and thirty-one thousand rows will find even quite
rare tokens. Coverage is not the problem.

The asymmetry is in how much gradient each row receives, and there the numbers are severe.
Natural language is Zipfian, so the most frequent token contributes on the order of a hundred
thousand gradient terms to its own row in that single step, while a token far down the tail
contributes one, or less than one on average. The scatter-add sums those contributions, so the
effective step size applied to a row scales with how often its token appeared. Across a realistic
vocabulary that spread runs to five or six orders of magnitude.

The embedding table is not one object that trains at one rate. It is a hundred and thirty-one
thousand small objects whose effective learning rates are set by the corpus rather than by the
optimizer, and the slow end of that range is where the low-resource languages live.

The picture does change at smaller batch sizes, and it is worth seeing both. At the microbatch a
single GPU actually processes, coverage genuinely is sparse and the tail rows go many steps
between visits, which is why gradient accumulation across a large global batch is doing more work
for the embedding layer than it is for any other tensor in the model.

This is the mechanism underneath a claim Session 3 made from the data side and could not yet
explain from the model side. When an English-tuned filter thins the Indic pools, the effect is
not only that the model sees less Indic text. It is that the Indic rows of the embedding table
receive gradient less often, so they stay closer to their random initialisation for longer, and a
row near initialisation contributes noise to every sequence it appears in. Protection at the
mixture level, which Session 5 built as the always-on lane, is protection at the gradient level
for these specific rows.

**Widget — gather/scatter-add lab.** Left panel walks one optimizer step through three stages:
batch of ids arriving, forward gather reading one row per token, backward scatter-add writing
into exactly those rows (a token appearing 4 times contributes 4 terms summed into one update,
not four separate ones). Rows the batch didn't name receive zero and don't move. Right panel runs
the same sampling for hundreds of steps and plots total accumulated gradient per row: rows end up
trained to wildly different degrees, not because the optimizer treated them differently — it
treated them identically; the corpus did not.

## 3. What the table costs, and closing Session 2's loop

Session 2 ended its treatment of vocabulary size by reporting that the embedding table at the
131K setting is already larger than many complete models, and then handed the question forward
rather than answering it. The answer is arithmetic, and it is worth doing carefully once, because
the number that hurts is not the one most people quote.

The parameter count is the easy part. The input table is V x D. The output projection, which
turns the final hidden state into a score for every token in the vocabulary, is another D x V. At
V5's reference shape each of those is 1.06 billion parameters, and untied they are 2.12 billion
together. Against a 120B mixture-of-experts model that is a small share of the total; against a
7B dense model it would be a third of everything. That is exactly why the same tokenizer decision
reads as trivial at one scale and catastrophic at another.

The number that actually constrains the run is memory during training, and it is much larger than
the parameter count suggests. A parameter being trained under AdamW in mixed precision is not two
bytes. It is two bytes of bf16 weight, two bytes of bf16 gradient, four bytes of fp32 master copy,
and four bytes each for the two optimizer moments: sixteen bytes per parameter. The input table
alone therefore occupies

`1,061,158,912 x 16 bytes = 16.98 GB`

of training state, and an untied pair occupies just under 34 GB. On an 80 GB accelerator that is
two-fifths of the card consumed by the two matrices that do the least interesting work in the
model: one is a lookup, the other a dot product against every row of that lookup.

This is the real reason the embedding decision is an architecture decision and not a detail. It
is the pressure that makes every technique in the rest of this session worth the complexity it
adds.

**Widget — parameter and memory budget.** Set vocabulary, width, depth, feedforward multiplier,
tying and precision; computes input table, output head and transformer stack as competing shares
of the total, then converts each to training memory with full AdamW accounting. Sweeping width
from 2,048 to 8,096 shows the embedding share collapsing while its absolute memory climbs.

## 4. The tokenizer and the embedding are one design surface

Before we start compressing the table it is worth being clear that its size is not handed to us
from outside. It is the product of a decision the cohort made in Session 2, and the two decisions
have to be made together or they will fight.

The dial pulls in both directions at once. A smaller vocabulary means fewer rows and a smaller
table, but it raises fertility, so the same document becomes more tokens, and every one of those
extra tokens is paid for in attention compute at every layer for the whole of training and the
whole of inference. A larger vocabulary lowers fertility and shortens sequences, but it grows the
table and the output head in a straight line. Neither end is free and there is no neutral
setting.

For an English-only model this is a mild optimisation. For an India-first model it is the whole
problem, because fertility is not uniform across the languages we care about. Session 3 measured
this directly: a tokenizer that splits Telugu into roughly three times as many tokens as it uses
for the same meaning in English hands Telugu a third of the effective context and three times the
inference bill, and no amount of extra vocabulary fixes that unless the extra slots are spent on
the scripts that need them. The vocabulary is a budget, and its allocation across scripts is a
sovereign decision, which is precisely what the BrahmicTokenizer retrofit was about.

The tokenizer decides how much attention compute each language costs. The embedding system
decides how much parameter memory that vocabulary costs. Change either and you have changed the
other's bill.

**Widget — fertility and cost lab.** Set a corpus mix and a vocabulary; attention-side cost falls
as vocabulary grows (fertility falls) while parameter-side cost rises linearly. Their sum has a
minimum that moves with the corpus mix: English-only puts it near 53K; the V5 mix moves it to
about 101K; an Indic-heavy mix pushes it past 113K, because fertility decays toward a floor of
one token per word and English has almost no reducible headroom while Tamil has a great deal. A
memory-price control changes where the minimum sits but not which way it moves with the mix.

## 5. Weight tying

The cheapest available saving is to notice that we have two matrices of identical shape facing
the same vocabulary, and to use one matrix for both jobs. Weight tying sets the output projection
to the transpose of the input embedding, which removes an entire billion-parameter tensor and
about 17 GB of training state at V5's reference shape. It has a respectable argument behind it
beyond the saving: both matrices live in the same space, and tying them forces the geometry the
model reads tokens with to be the geometry it predicts tokens with.

The argument against is that the two jobs are not the same job.

- At the input, the row has to be a good starting representation for a token about to be
  processed by the whole network, which rewards encoding what the token means.
- At the output, the row has to produce a score that separates this token from every other token
  that could plausibly come next, which rewards encoding what distinguishes it from its
  competitors.

Those two objectives agree for most of the vocabulary and disagree exactly where it matters: on
the frequent function words whose meaning is thin but whose prediction is constant.

The empirical position is stable enough to state as a rule. Tying pays when the token-facing
matrices are a large share of the model, because the saving is real and the regularisation helps
a small model that would otherwise overfit them. It stops paying as the model grows, because the
saving becomes a rounding error while the constraint on the geometry does not. GPT-2 tied and
Gemma ties; Llama-2 at 7B does not. The threshold is not a magic number, it is the point where the
embedding share of total parameters falls into the low single digits, and V5 is well past it.

**Widget — weight tying board.** Shows the two token-facing matrices side by side and collapses
them into one when tying is enabled, with parameter/training-memory saving computed live. A scale
sweep runs the same tokenizer against model sizes from 500M to 120B and plots embedding share of
total parameters, marking the band where the saving stops being worth the constraint; V5's
reference configuration sits well to the right of that band.

## 6. Factorized embeddings, the on-ramp

Tying halves the problem. Factorization attacks what is left, and the simplest form of it is
worth meeting first because it makes the trade explicit before we reach for anything cleverer.

The observation is that the row a token is looked up as does not have to be as wide as the model.
We can look the token up in a narrow table and then project the narrow vector up to the model
width with a matrix shared by every token. The narrow table is V x r and the projection is r x D,
so the parameter count becomes `V*r + V*D` — sorry, `V*r + r*D` — in place of `V*D`. At V5's shape
with a rank of 512 that is 67.1 million plus 4.1 million: about 71 million parameters in place of
1.06 billion, a reduction of roughly 93 percent. This is the scheme ALBERT used, and the
arithmetic is as good as it looks.

The cost is a rank bottleneck, and it is not a soft cost. Whatever the projection does, the set of
vectors the embedding layer can produce is confined to an r-dimensional subspace of the
D-dimensional space, because every output is the same r x D matrix applied to some narrow vector.
The layer has V rows but at most r directions. If r is generous the constraint is invisible; if r
is aggressive the model is being asked to represent a hundred and thirty thousand distinct tokens
inside a few hundred directions, and tokens start having to share.

This is the honest shape of every compression technique in this session. You are not getting the
same table for fewer parameters. You are getting a structured table, and the question is always
whether the structure you imposed happens to match the structure the data has.

**Widget — factorized embedding builder.** Parameter arithmetic on one side, bottleneck on the
other. A reference table with a known spectrum makes the reconstruction error of a rank-r
approximation exact rather than estimated; plots retained energy against rank alongside parameter
saving. Shows the knee where saving is still large and error still small, and the region past it
where each further parameter saved costs real capacity.

## 7. Kronecker factorization

Now the technique Session 2 promised, and the one the lab released.

A dense table keeps one row per token. Kronecker embeddings do not keep rows at all. They build
each token's row out of the token's own bytes. The whole idea is three steps, and none of them is
complicated.

**Step one: read the bytes.** Every token is a piece of text, and every piece of text is a
sequence of UTF-8 bytes. The token `the` is three bytes. Nothing is learned here.

**Step two: mark a grid.** Picture a grid with 256 rows, one for each possible byte value, and 32
columns, one for each byte position. For every byte in the token, mark the cell at (its value,
its position). A three-byte token marks three cells. Flatten that grid and you have a vector of
`256 x 32 = 8,192` numbers, almost all of them zero. That vector is the token's code. It is fixed,
it is never trained, and it is identical every time.

Marking the cell for one value and one position is exactly a Kronecker product of two one-hot
vectors, which is where the name comes from:

`kappa(b) = (1/sqrt(L)) * vec( sum over positions p of c[byte_p] (x) p[position_p] )`

The `1/sqrt(L)` divides by the square root of the token's byte length so short and long tokens
come out at a comparable scale, and the result is then z-normalised.

**Step three: project.** One shared `Linear(8192, d_model)` turns that fixed code into the vector
the model actually uses. That projection is the only thing in the entire input path that is
learned.

Now the parameter count, which is the part that matters:

`8,192 x 8,096 = 66,322,432 parameters`

against a dense table's 1,061,158,912. That is a 93.75% reduction. Now notice what is missing
from that multiplication: the vocabulary size. There is no V in it anywhere. The input path costs
the same whether the vocabulary holds thirty thousand tokens or five hundred thousand, because
the grid has 256 rows and 32 columns no matter how many tokens you point at it.

**What you gain**
- Cost stops depending on vocabulary. Adding tokens is free on the input side, which turns
  Section 4's whole trade-off on its head.
- Unseen tokens still work. A token the model never trained on still has bytes, so it still gets
  a sensible code rather than an untrained random row.
- Similar spellings start out similar. `train`, `training` and `trainer` share most of their
  bytes, so they begin near each other instead of unrelated.

**What you pay**
- The layer needs the token text. It takes a tokenizer and keeps a byte buffer per token, so it
  reaches back across the seam Session 2 drew. That is a deliberate trade, not an accident, and
  it is why the constructor carries both `vocab_size` and `tokenizer`.
- The code is frozen by construction. The codec never learns anything. If two tokens produce the
  same code, no amount of training can pull them apart (see Section 8).

One clarification, because the same name is used for two different things. There is a separate
family that factorises the table as `A ⊗ B`, two learned matrices whose Kronecker product has the
table's shape. That is a real technique in the literature — it is **not** this one. Here the
Kronecker product is taken between a byte value and a byte position, both factors are one-hot
rather than learned, and the full table is never reconstructed at all.

**Widget — Kronecker microscope.** Type any token, in any script, and watch the three steps run:
UTF-8 bytes read, each byte marks one cell in a 256x32 grid at its value and position, flattened
grid goes through the single shared projection. Parameter count updates live, and sliding the
vocabulary does nothing to it. Try `training`, then `भारत`, then `తెలుగు`, and watch the byte
count per character change from one to three.

**Widget — walkthrough of the released module.** Stepped line by line: which lines create a
buffer (fixed, never trained) vs. which line creates the single `Parameter`. Byte buffer and
codec table are sized by the vocabulary but hold no trainable weights; the projection holds all
of them and is not sized by the vocabulary at any point. Forward pass is two lines.

The architectural point worth keeping: the module's entire visible behaviour is that it takes
`[B, T]` integers and returns `[B, T, D]` floats. Everything about how the row was manufactured is
private to it — which is why a compression decision this aggressive can be made, measured, and
reversed without touching attention, the feedforward blocks, the loss, or the dataloader.

## 8. The thirty-two byte budget

Look at the grid again. It has 32 columns, which means only the first 32 bytes of a token are
ever seen. The code says so plainly:

`L = min(len(byte_seq), pos_dim)`

Bytes past position 32 are dropped. For English that limit is generous, because ASCII spends one
byte per character, so 32 bytes is 32 characters and hardly any token comes close.

For Indic scripts it is not generous at all. Devanagari, Telugu, Tamil, Bengali and their
neighbours sit in a region of Unicode that UTF-8 encodes in three bytes per character. The same
window now holds ten characters, not thirty-two.

It gets tighter still. A conjunct such as `क्ष` is not one character to Unicode. It is three code
points: `क`, the halant, and `ष`. At three bytes each, that single visual character costs nine of
the thirty-two bytes. A word carrying three or four conjuncts has spent the entire budget before
it has finished.

And the failure is silent. Two tokens that agree on their first 32 bytes produce exactly the same
code, and therefore exactly the same embedding vector, permanently. The projection cannot
separate them because it is never shown a difference. Nothing raises an error and nothing warns;
the model simply cannot tell those two tokens apart, and it never will.

This is the sovereign risk in the technique, and unlike most such risks it is a number rather than
an argument.

So the question for V5 is not whether Kronecker embeddings save memory. They plainly do. It is
whether `pos_dim = 32` is the right window for the scripts we care about, and that is answered by
measurement: take the V5 vocabulary, encode every token, and count the collisions per script.
Raising `pos_dim` to 64 doubles D and doubles the projection, to roughly 133M parameters, which is
still eight times smaller than a dense table. If the collision count says buy it, buy it. **The
assignment asks you to produce that count** (note: the *published* Session 7 assignment text —
see `../S7-assignment.md` — actually redirects this into the five open embedding-design problems
below; the collision-count exercise described here is the session's own internal framing of why
`pos_dim` matters, kept for context).

**Widget — byte budget lab.** Encoding real words in-browser. Each bar shows how much of a word
survives the window vs. is thrown away, limit drawn as a black line. Two genuinely different
Hindi words, `अंतर्राष्ट्रीयकरण` and `अंतर्राष्ट्रीयता`, collide at the shipped `pos_dim = 32`:
identical codes, identical vectors, indistinguishable to the model forever. Widen the window to 48
and they separate.

## 9. The V4 scar: a frozen input path meets a mixture shift

Session 5 reported an incident from the inside of the previous run: a sudden increase in the
Hindi share, arriving against embeddings that had been frozen, drove the gradient norm up by
roughly one hundred and fifty times over a short stretch. Session 5 read that as a lesson about
changing the blend too abruptly, and it is. Read from this side of the seam it is a lesson about
something more specific, and the more specific version is the one that generalises.

The embedding layer is the only place in the model where the token distribution meets continuous
computation. Every distributional fact about the corpus — which tokens are common, which scripts
are present, what the mixture currently is — enters the model through that one adapter. When the
mixture changes, the statistics arriving at that adapter change, and the natural response is for
the adapter to move. If it is frozen it cannot, and the adjustment has to happen somewhere. It
happens in the layers above, which now have to absorb a shifted input distribution using
parameters tuned for the old one, and that shows up as large updates that propagate.

This is not a learning-rate problem and it does not have a learning-rate fix. It is an
adaptation-boundary problem: we removed the degrees of freedom at the point where the change
enters, and were surprised when the change came out somewhere else.

Kronecker embeddings sharpen this considerably, which is the part V4 did not anticipate. A dense
table has a billion degrees of freedom with which to absorb a distributional change, and it can
move the rows that need to move and leave the rest alone. A Kronecker input path has the
projection and nothing else. The codec is fixed by construction and holds no parameters at all,
so every token in the vocabulary adapts through one shared matrix or does not adapt. There is no
way to adjust how Hindi is represented without touching every other token at the same time. A
compressed input path is a less capable adapter by construction, which makes the case for keeping
the projection trainable stronger, not weaker, than it is for a dense table.

Compressing the embedding and freezing it are two decisions that each look locally reasonable and
are jointly dangerous.

The operational conclusions V5 adopts:
- The projection and factors stay trainable. Freezing, if it happens at all, is scheduled and
  logged rather than left on from an experiment.
- Mixture transitions get the warmup band Session 5 already mandated, and the gradient norms of
  the layers immediately above the embedding are monitored as a leading indicator rather than the
  global norm, which averages the signal away.
- The ledger Session 6 built acquires one more field: `embedding_policy_id` — behind which sits
  the embedding type, the `char_dim` and `pos_dim` that fixed the codec, the tokenizer hash the
  byte buffers were built from, the trainable/frozen state of the projection, the unfreeze
  schedule, the position policy and the tying decision. The tokenizer hash matters more here than
  anywhere else in the course: change the tokenizer and every token's bytes change, so every code
  changes, and the projection is now trained against a codec that no longer exists.

**Widget — frozen input path lab.** Trains a real small model in-browser (genuine forward passes,
real cross-entropy, hand-written gradient descent) over a two-domain vocabulary, on a stream whose
mixture shifts partway through the run. Freeze the embedding and loss + gradient norm of the layer
above both jump at the transition; leave it trainable and the same transition is absorbed. A
warmup control spreads the shift and shrinks the spike.

## 10. Position: the model reads a set

Everything so far has answered the first of the two questions the model asks about an incoming
token: what the token is. The second question is where it is, and the reason it needs a separate
answer is structural rather than incidental.

Attention computes a score between every pair of positions from their query and key vectors and
mixes values according to those scores. Nothing in that computation refers to the order of the
sequence. Permute the input tokens and every pairwise score permutes with them, and the output
permutes correspondingly: the mechanism is equivariant to permutation, which is a precise way of
saying it cannot tell "dog bites man" from "man bites dog".

The transformer does not read a sequence. It reads a set, and order has to be supplied to it
deliberately. (Session 2's minor topic demonstrated this: a token-only model was pinned at chance
on a task rigged so every sequence appeared once with each label under a swap of two tokens,
because the two cases were literally identical inputs to it; a token-plus-position model learned
the rule.)

## 11. The absolute table, and the wall at max_position

The simplest answer mirrors the token table: keep a second table, indexed by position instead of
by identifier, and add its row to the token's row before the stack sees either.

```
token_embedding = nn.Embedding(vocab_size, d_model)
position_embedding = nn.Embedding(max_position, d_model)
x = token_embedding(token_ids) + position_embedding(torch.arange(T))
```

This is what GPT-2 did and it works. As a parameter block it's small enough to ignore next to the
vocabulary table (max_position is thousands, not hundreds of thousands). The objection isn't cost.

The objection is that it inherits the exact property from Section 2 we learned to be careful
about. A position table is gathered and scatter-added exactly like a token table, so row `t`
learns only from steps in which some sequence had a token at position `t`. Train with max position
4,096: rows 0-4,095 are trained, row 7,000 does not exist. Extend context at inference and there
is no row to fetch; allocate rows in advance and never train them, they sit at random
initialisation and inject noise at exactly the positions you extended to handle.

The table cannot extrapolate — there is no signal connecting row 4,095 to row 4,096, because they
were only ever independent rows in a lookup table. That is a hard wall, and it is the wall the
entire modern positional-encoding literature exists to get past: stop storing position and start
computing it, so position enters through a function of `t` defined for every `t` rather than a row
that exists only for values seen in training. Sinusoidal encoding was the first version of that
idea; what the field converged on afterwards (rotary, etc.) is next class's subject.

**Widget — absolute position table lab.** Trains live on a task whose answer depends on position,
on positions 0-7, near-perfect accuracy inside that range. Evaluated at positions 8-15: accuracy
falls to chance, because those rows never appeared in a batch — the scatter-add never wrote to
them and they hold their initialisation. The cliff at the trained boundary is measured, not drawn.

## 12. The families, and what Session 8 builds

Four broad places a position signal can be injected:
1. **Stored and added at the input** — the absolute learned table just shown failing.
2. **Computed and added at the input** — sinusoidal, extrapolates in the weak sense the function
   is defined everywhere (though a model trained only on short sequences has never had to use the
   far part of it).
3. **Applied inside attention to queries/keys**, so the score depends on offset rather than
   absolute index — where the rotary family lives, where the field has settled.
4. **Applied to attention scores directly as a distance-dependent bias** — the ALiBi family.

The trade across all four: how much the model is told vs. how much it has to infer. Stored is
maximally expressive within its trained range and useless outside it; computed generalises by
construction but constrains what can be represented. Same trade this session made twice already
(dense → factorized → Kronecker): structure substituted for stored parameters.

Session 8 builds rotary position embeddings and their scaling/extension schemes, the attention-side
variants that change which pairs get computed at all, and treats long-context extension as its own
engineering problem. Today's job was to hand it a model whose input representation is decided and
whose positional wall is felt rather than described.

**Widget — position family map.** Shows where each scheme injects its signal, what it costs in
parameters/compute, and how it behaves when context is extended past trained length. The three
families Session 8 builds are marked as such, not explained here — a table of contents for next
week.

## 13. The V5 embedding decision

- **Vocabulary** stays in the 131K class inherited from the BrahmicTokenizer work, chosen against
  fertility across target scripts and against Section 3's parameter arithmetic together, not
  either alone.
- **Input path** is the released Kronecker byte codec plus a trainable projection, with a dense
  table as the control arm. The `pos_dim` window is chosen by measurement, starting from the
  shipped value of 32.
- **The projection is trainable throughout.** The codec is already frozen by construction, so
  freezing the projection too leaves the input path with no adaptive capacity at all. Freezing is
  a scheduled and logged decision, never a residue of an experiment.
- **Output head is untied** — at V5's scale the saving from tying is small and the constraint on
  geometry is not, and a structured input path and a dense output head aren't naturally the same
  object anyway.
- **The byte window is an architectural parameter, not a default to inherit.** 32 bytes is 32
  English characters but only 10 Indic ones, so `pos_dim` is set by a collision count measured on
  the real V5 vocabulary, per script.
- **Position policy is deferred to Session 8 by design**, with the absolute table ruled out for
  the long-context target on the evidence in Section 11.

All of it is written into the ledger under `embedding_policy_id`, so any checkpoint can answer
what its input path was doing when it was written.

Every number above is a hypothesis. Session 5 established the standard that a proportion is an
opinion until a proxy run has tested it; there's no reason an architecture decision should be held
to a lower standard than a mixture decision. The byte window, projection width and tying decision
all get proxy runs at the one-billion and three-billion scale before being trusted at full scale.

**Widget — V5 embedding design board.** Choosing vocabulary, width, input path, rank, tying and
freeze policy produces parameter count, training memory (full optimizer accounting), compression
vs. dense, and degrees of freedom per token the chosen structure leaves. Gates that fail are named
with the number that failed them. Emits the `embedding_policy_id` record as JSON — the object the
ledger stores and the assignment asks you to defend.

## 14. The assignment

See `../S7-assignment.md` for the verbatim text. In short: pick **one** of five open embedding-
design problems (math-structure-preserving embeddings, a Kronecker extension to images/audio, a
dynamic/non-cropped byte window, a real Fourier-wave character embedding, or an invertible/
reversible Kronecker that could remove the output head), state which one, build a small
transformer to prove the idea works, and write it up (README + code, optionally a demo webapp).
This is explicitly framed by the instructor as candidate material for a "Kronecker Embedding V2"
paper he's planning to write.
