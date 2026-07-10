"""Train a SentencePiece BPE tokenizer (the family used by XLM-R, though
XLM-R itself uses SentencePiece's unigram mode rather than BPE) over the same
weighted English/Hindi/Telugu/Marathi mixture as the other variants, for a
like-for-like comparison.

SentencePiece operates directly on raw Unicode text (whitespace becomes a
meta symbol, "_"), so it needs no separate word-segmentation step -- which is
exactly why it's the standard choice for multilingual/Indic-script models.
"""

import argparse
import pathlib

import sentencepiece as spm

from common import DEFAULT_ALPHA, DEFAULT_VOCAB_SIZE, TOKENIZER_DIR, compute_repeat_counts, load_corpora, weighted_lines


def train_sentencepiece_variant(
    texts: dict[str, str], repeat_counts: dict[str, int], vocab_size: int, out_dir: pathlib.Path
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    input_path = out_dir / "spm_input.txt"
    with input_path.open("w", encoding="utf-8") as f:
        for text in weighted_lines(texts, repeat_counts):
            f.write(text + "\n")

    model_prefix = str(out_dir / "spm")
    spm.SentencePieceTrainer.Train(
        input=str(input_path),
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=1.0,
        input_sentence_size=0,
        shuffle_input_sentence=True,
        max_sentence_length=20000,
        unk_id=0,
        bos_id=-1,
        eos_id=-1,
        pad_id=-1,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE)
    args = parser.parse_args()

    texts = load_corpora()
    repeat_counts = compute_repeat_counts(texts, args.alpha)
    print(f"alpha={args.alpha} repeat_counts={repeat_counts}")

    out_dir = TOKENIZER_DIR / "sentencepiece"
    train_sentencepiece_variant(texts, repeat_counts, args.vocab_size, out_dir)
    print(f"[sentencepiece] saved -> {out_dir}/spm.model / {out_dir}/spm.vocab")


if __name__ == "__main__":
    main()
