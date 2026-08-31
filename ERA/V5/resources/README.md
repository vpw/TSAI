# ERA V4 Reference Material — Research Notes

Analyses of the material referenced in `../resources/URLS.md`, gathered 2026-07-17.

> **Correction found while researching:** `URLS.md` points *both* the Brahmic tokenizer and
> Kronecker embeddings at arxiv `2605.29379`. The tokenizer is indeed `2605.29379`
> ("BrahmicTokenizer-131K"), but Kronecker embeddings is a **separate paper: arxiv
> `2605.29459`** ("Kronecker Embeddings: Byte-Level Structured Token Representations for
> Parameter-Efficient Language Models"). Reference implementation:
> github.com/theschoolofai/kronecker-embeddings; training pipeline:
> github.com/The-School-of-AI/LLM.

## Files

| File | Covers |
|---|---|
| [lightninglm.md](lightninglm.md) | LightningLM 0.1V family + "Reversible Foundations" paper (arxiv 2606.07404) |
| [brahmic-tokenizer.md](brahmic-tokenizer.md) | BrahmicTokenizer-131K (arxiv 2605.29379) |
| [kronecker-embeddings.md](kronecker-embeddings.md) | Kronecker Embeddings (arxiv 2605.29459) |
| [implications-for-s3.md](implications-for-s3.md) | What all of this means for the S3 assignment, + a longer study plan |

## One-paragraph synthesis

The three artifacts are one system, and that is the main insight. LightningLM's training
economics (120B-stored/5.93B-active MoE on a single 8-GPU node, ~67 days) only close because
the *input side* was made nearly free: BrahmicTokenizer-131K halves the vocabulary of
o200k_base (200,019 → 131,072) while *improving* Indic fertility, and Kronecker embeddings
shrink the embedding table for that 131K vocab from ~537M trained parameters to ~33.6M
(a 4.5 MB byte buffer at runtime instead of a 2.15 GB table). The data side mirrors this:
OPUS dynamic selection stretched ~200B consumed tokens to an effective ~1.2T, and the 8%
always-on Indic channel protected exactly the data that an English-tuned quality selector
undervalues. Every choice trades stored parameters or raw tokens for *engineering* —
which is exactly the mindset the S3 assignment asks us to apply to data.
