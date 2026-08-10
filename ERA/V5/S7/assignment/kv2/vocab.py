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

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Local copy first (this is what ships to the GPU box), then S4's canonical artefact when
# running on a workstation that has the whole course checked out.
_TOKENIZER_CANDIDATES = (
    os.path.join(_ROOT, "models", "tokenizer-sarvam1.json"),
    os.path.join(_ROOT, "..", "..", "S4", "assignment", "models", "tokenizer-sarvam1.json"),
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
    """Resolve the frozen tokenizer: explicit arg, then `$KV2_TOKENIZER`, then known locations.

    The tokenizer is a Session 2 contract -- every codec table and every token id in the
    corpus is meaningful only under this exact artefact -- so a missing one is a hard error
    rather than a silent fallback to something else.
    """
    if path:
        return os.path.normpath(path)
    env = os.environ.get("KV2_TOKENIZER")
    if env:
        return os.path.normpath(env)
    for cand in _TOKENIZER_CANDIDATES:
        if os.path.exists(cand):
            return os.path.normpath(cand)
    raise FileNotFoundError(
        "tokenizer-sarvam1.json not found. Looked in: "
        + ", ".join(os.path.normpath(c) for c in _TOKENIZER_CANDIDATES)
        + ". Set KV2_TOKENIZER to point at it."
    )


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
