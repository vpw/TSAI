# BrahmicTokenizer-131K

**Source:** arxiv 2605.29379 — "BrahmicTokenizer-131K: An Indic-Capable Drop-In Replacement
for o200k_base" (Rohan Shravan). Apache-2.0, on Hugging Face. (The transcript calls it
"Krona/chrono tokenizer" — same artifact.)

## Idea

Do not train an Indic tokenizer from scratch — **retrofit** OpenAI's o200k_base:

1. **Prune** nine out-of-scope writing systems: 200,019 → **131,072** tokens.
   (131,072 = 2^17 — GPU-friendly, and exactly the LightningLM vocab.)
2. **Retrofit** 2,372 freed slots across nine Brahmic Unicode blocks, allocated by
   **linear programming** against measured compression deficits (e.g. Odia got 725 tokens).

## Results

- **26.7% fewer tokens** than Mistral-Nemo Tekken / Sarvam-m on 2.84B Indic words.
- Per-language savings vary hugely: Tamil 15.8% … **Odia 76.8%** (4.31x compression) —
  the LP allocation targets the worst deficits first.
- **English fertility 1.235 tokens/word** vs o200k_base's 1.232 — i.e. *no English
  regression* despite a 34% smaller vocab.
- Code/math: *beats* Tekken/Sarvam-m by 4.0–14.2% on HumanEval/MBPP/GSM8K token-cost
  metrics.

## Transcript context (what fertility still costs)

Even with a good tokenizer, V4-era fertility for Indic was punishing: with the GPT-4-class
tokenizer shown in session, a 15-token English sentence cost Hindi 56 (3.7x), Bengali 66,
Tamil 107, Telugu 123 tokens. So "100B Hindi tokens" was only ~27B words of Hindi. The
whole-corpus average fertility was ~1.33. Word counts are standardized on **FLORES-200**.

## Takeaways for S3

1. **Vocab size is a budget allocation problem, not a size problem.** 131K beat 200K for
   the target languages because slots were *allocated* (by LP against measured deficits),
   not accumulated. The S3 "what vocab size" question should be answered the same way:
   set per-domain fertility targets first, then size the vocab to meet them.
2. **Fertility is the real data budget.** A tokenizer with fertility 3.7 turns a 500B-token
   Hindi corpus into 135B words; fertility 1.5 turns the *same corpus* into 333B words.
   Tokenizer quality multiplies the effective corpus before a single document is collected.
3. **A retrofit preserves English/code quality for free** — important because the 40B model
   must also beat Gemma-class benchmarks in English and code.
