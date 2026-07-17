"""Train the shared 10k-vocab BPE tokenizer.

Architecture (phase 2 -- replaces the phase 1 byte/char/word-Whitespace
variants, none of which can pass a faithful round-trip check):

  - normalizer: NFKC only. No destructive character replacement -- every
    visible character the trainer sees survives into the vocabulary.
  - pre_tokenizer: Metaspace(replacement="▁", prepend_scheme="always").
    Unlike Whitespace, Metaspace never discards a separator: it marks word
    boundaries by prefixing "▁" rather than splitting text into disjoint
    matched groups, so punctuation stays attached to its pretoken instead
    of being thrown away.
  - decoder: the matching Metaspace decoder, so decode() reverses exactly
    what the pre-tokenizer did (strip "▁" back to spaces) instead of the
    phase 1 bug of leaving decode() unconfigured and joining pretokens with
    a single space, which permanently destroyed every original separator
    character (":", "/", ".", "#", "-").
  - min_frequency=1: without this the trainer keeps reusing a handful of
    highly-shared merges and starves English's own long tail of
    distinctive words (see phase 1 findings) -- still true here.

This mirrors the architecture verified to round-trip correctly in the
"R2" reference implementation named in MODIFY.md (see PLAN_PHASE2.md for
the comparison against the instructor's own reference, which -- despite
being the score-formula source of truth -- uses a Whitespace pretokenizer
and a destructive normalizer and does NOT pass round-trip fidelity).
"""

import argparse
import json

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, normalizers, trainers

from common import DEFAULT_ALPHA, DEFAULT_VOCAB_SIZE, TOKENIZER_DIR, compute_repeat_counts, langset, load_corpora, weighted_lines

METASPACE_REPLACEMENT = "▁"


def build_tokenizer() -> Tokenizer:
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(replacement=METASPACE_REPLACEMENT, prepend_scheme="always")
    tokenizer.decoder = decoders.Metaspace(replacement=METASPACE_REPLACEMENT, prepend_scheme="always")
    return tokenizer


def train(
    texts: dict[str, str],
    repeat_counts: dict[str, int],
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    min_frequency: int = 1,
) -> Tokenizer:
    tokenizer = build_tokenizer()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=["[UNK]"],
        show_progress=False,
    )
    tokenizer.train_from_iterator(weighted_lines(texts, repeat_counts), trainer=trainer)
    return tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fourth", default="mr", choices=["mr", "bn"])
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE)
    parser.add_argument("--min-frequency", type=int, default=1)
    parser.add_argument("--repeat-counts", type=str, default=None, help='JSON dict, e.g. \'{"en":1,"hi":1,"te":8,"mr":1}\'')
    args = parser.parse_args()

    langs = langset(args.fourth)
    texts = load_corpora(langs)
    repeat_counts = json.loads(args.repeat_counts) if args.repeat_counts else compute_repeat_counts(texts, args.alpha)
    print(f"langs={langs} repeat_counts={repeat_counts}")

    tokenizer = train(texts, repeat_counts, args.vocab_size, args.min_frequency)

    out_dir = TOKENIZER_DIR / args.fourth
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "tokenizer.json"
    tokenizer.save(str(out_path))
    print(f"saved -> {out_path} (vocab_size={tokenizer.get_vocab_size()})")


if __name__ == "__main__":
    main()
