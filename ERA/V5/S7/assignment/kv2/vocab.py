"""Loading the real V5-lineage vocabulary and turning its tokens back into bytes.

Every claim in this submission is measured against a real tokenizer rather than a synthetic
one, because the defect being attacked (dead grid cells) is a property of how UTF-8 lays out
real scripts, and a uniformly-random byte vocabulary would not exhibit it at all.

The tokenizer is `tokenizer-sarvam1.json`, 68,096 entries, inherited through S2/S4 -- the same
frozen artefact S6 hashes into its shard manifests.
"""

from __future__ import annotations

import json
import os
import unicodedata

# SentencePiece writes a leading word boundary as U+2581 LOWER ONE EIGHTH BLOCK.
SP_SPACE = "▁"

_DEFAULT_TOKENIZER = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "..", "..", "S4", "assignment", "models", "tokenizer-sarvam1.json",
)

SCRIPT_PREFIXES = (
    "DEVANAGARI", "TELUGU", "TAMIL", "BENGALI", "KANNADA",
    "MALAYALAM", "GUJARATI", "GURMUKHI", "ORIYA", "LATIN", "ARABIC", "CJK",
)

INDIC_SCRIPTS = frozenset(
    {"DEVANAGARI", "TELUGU", "TAMIL", "BENGALI", "KANNADA",
     "MALAYALAM", "GUJARATI", "GURMUKHI", "ORIYA"}
)


def tokenizer_path(path: str | None = None) -> str:
    return os.path.normpath(path or _DEFAULT_TOKENIZER)


def load_vocab(path: str | None = None) -> list[str]:
    """Return the vocabulary as a list of token strings, ordered by token id."""
    with open(tokenizer_path(path), encoding="utf-8") as fh:
        blob = json.load(fh)
    vocab = blob["model"]["vocab"]
    # vocab maps token -> id; restore id order so indices mean what the shards mean.
    ordered = sorted(vocab.items(), key=lambda kv: kv[1])
    return [tok for tok, _ in ordered]


def token_bytes(token: str) -> bytes:
    """The UTF-8 bytes the codec actually sees for a token.

    The SentencePiece space marker is restored to a real space first: the released Kronecker
    module encodes the token's *text*, and treating U+2581 as content would inflate every
    word-initial token by two bytes and distort the byte-length statistics.
    """
    return token.replace(SP_SPACE, " ").encode("utf-8", "surrogatepass")


def vocab_bytes(path: str | None = None) -> list[bytes]:
    return [token_bytes(t) for t in load_vocab(path)]


def script_of(token: str) -> str:
    """Best-effort script label for a token, used to report collisions per script."""
    stripped = token.replace(SP_SPACE, "").strip()
    for ch in stripped:
        if not ch.isalpha():
            continue
        try:
            name = unicodedata.name(ch)
        except ValueError:
            return "OTHER"
        for prefix in SCRIPT_PREFIXES:
            if name.startswith(prefix):
                return prefix
        return "OTHER"
    return "NONALPHA"
