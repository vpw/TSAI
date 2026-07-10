"""Train BPE tokenizer variants (byte-level, char-level, word-level) over the
English/Hindi/Telugu/Marathi India-Wikipedia corpora, all sharing the same
weighted-mixture strategy from common.py so results are comparable.

Variants (all still BPE — only the pre-tokenization differs):
  byte: GPT-2-style byte-level pre-tokenization, merges over byte pairs.
        Immune to unknown tokens, but Devanagari/Telugu characters (3 UTF-8
        bytes each) can fall back to raw bytes if too few merges are spent
        on them.
  char: pre-tokenizes into individual Unicode codepoints, merges over
        character pairs. No byte-fallback risk, but combining marks
        (matras) are treated as separate base units from their base
        consonant until a merge joins them.
  word: classic whitespace/punctuation pre-tokenization (Sennrich-style),
        merges only within a word's characters — never across word
        boundaries.
"""

import argparse

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from common import DEFAULT_ALPHA, DEFAULT_VOCAB_SIZE, TOKENIZER_DIR, compute_repeat_counts, load_corpora, weighted_lines

VARIANTS = ["byte", "char", "word"]


def build_pre_tokenizer(variant: str):
    if variant == "byte":
        return pre_tokenizers.ByteLevel(add_prefix_space=False)
    if variant == "char":
        # No boundary restriction: merges may freely span whitespace/word
        # boundaries. The base alphabet is still individual Unicode
        # characters (BPE's own default symbol-splitting), not bytes.
        return None
    if variant == "word":
        return pre_tokenizers.Whitespace()
    raise ValueError(variant)


def build_decoder(variant: str):
    if variant == "byte":
        return decoders.ByteLevel()
    return None


def train_variant(
    variant: str, texts: dict[str, str], repeat_counts: dict[str, int], vocab_size: int, min_frequency: int = 2
) -> Tokenizer:
    tokenizer = Tokenizer(models.BPE(unk_token=None))
    pre_tokenizer = build_pre_tokenizer(variant)
    if pre_tokenizer is not None:
        tokenizer.pre_tokenizer = pre_tokenizer
    decoder = build_decoder(variant)
    if decoder is not None:
        tokenizer.decoder = decoder

    trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=min_frequency, special_tokens=[], show_progress=False)
    tokenizer.train_from_iterator(weighted_lines(texts, repeat_counts), trainer=trainer)
    return tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE)
    parser.add_argument("--variants", nargs="+", default=VARIANTS, choices=VARIANTS)
    args = parser.parse_args()

    texts = load_corpora()
    repeat_counts = compute_repeat_counts(texts, args.alpha)
    print(f"alpha={args.alpha} repeat_counts={repeat_counts}")

    for variant in args.variants:
        tokenizer = train_variant(variant, texts, repeat_counts, args.vocab_size)
        out_dir = TOKENIZER_DIR / variant
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "tokenizer.json"
        tokenizer.save(str(out_path))
        print(f"[{variant}] saved -> {out_path} (vocab_size={tokenizer.get_vocab_size()})")


if __name__ == "__main__":
    main()
