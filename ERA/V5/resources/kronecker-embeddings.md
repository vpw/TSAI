# Kronecker Embeddings

**Source:** arxiv **2605.29459** (note: `URLS.md` mislists this as 2605.29379, which is the
tokenizer paper) — "Kronecker Embeddings: Byte-Level Structured Token Representations for
Parameter-Efficient Language Models" (Rohan Shravan). Reference implementation:
github.com/theschoolofai/kronecker-embeddings.

## Idea

Replace the `|V| × d_model` embedding table with a **deterministic byte-level
character-position factorization**: a fixed (non-learned) encoder built from each token's
byte content and character positions (a 256×32 factored basis in LightningLM), followed by a
**single learned projection** to d_model. Embeddings are reconstructed on the fly.

## Numbers

- At V=131,072, d=4096: **~33.6M trainable params vs ~537M** for a full table
  (91–94% of input-side parameters eliminated). Runtime buffer **4.5 MB vs 2.15 GB**,
  step-time overhead 0.01–0.24%.
- On nanoGPT/GPT-2-124M: **2.5 ± 0.2% *lower* validation loss** than the BPE-tied baseline
  and ~1.43x faster convergence — parameter savings that *helped* rather than cost.
- Spelling robustness: predictions preserved on 55.5% of clean/typo pairs vs 47.3% (BPE).
- Probing insight: trained embedding tables cluster *typographic* variants more than
  *morphological* relatives; the byte-level construction sidesteps this failure.
- Limitation: byte-similar but semantically distant pairs (compute/commute, nation/notion)
  start close; disambiguation shifts to early attention layers.

## Why it exists (transcript context)

Weight tying (sharing embedding + LM head) works only for small models; at 120B they
couldn't tie. A 131K vocab would have cost ~0.5B params for the table + ~0.5B for the head
in a "1B" model — the 1B was really 1.8B. Kronecker cut the embedding side to ~16–34M.

## Takeaway for S3 — the key unlock for the vocab-size question

The classic argument *against* a large vocabulary is embedding-table cost: at 40B scale,
going 131K → 262K with d≈5120 would add ~1.3B parameters (table + head). With Kronecker
embeddings the input side is **effectively decoupled from |V|** — only the softmax head
still scales with vocab. That halves the cost of a bigger vocabulary and shifts the real
constraints to: (a) rare-token undertraining (tokens seen too few times to learn), and
(b) softmax head compute. So S3's vocab answer can be more aggressive than a naive
parameter-count analysis would allow — but not unbounded, because of (a).
