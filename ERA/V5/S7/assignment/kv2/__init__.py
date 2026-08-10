"""Kronecker V2 — a Fourier alternative to byte-level Kronecker embeddings.

Session 7 of ERA V5. The assignment poses five open problems extending the Kronecker
embedding of arXiv 2605.29459; this package answers **Problem 4**: what a real Fourier
alternative looks like, and whether it works.

Three codecs live here, all producing a fixed (never-trained) code per token that a single
shared projection maps to d_model:

  KroneckerCodec  the shipped baseline: a one-hot (byte value) x (byte position) grid,
                  char_dim * pos_dim dimensions, hard-cropped at pos_dim bytes.
  NaiveSumCodec   the assignment's literal phrasing, "represent each character like a
                  fourier wave, and just add them". Permutation-invariant, therefore
                  provably unable to separate anagrams. This is the negative control.
  FourierCodec    the repair: bind each byte value to its position by phase before
                  superposing, which restores order sensitivity and exact invertibility
                  while keeping the code dense and length-unbounded.

All three expose the same interface, so an ablation arm differs by one constructor call.
"""

from .codecs import ByteCodec, FourierCodec, KroneckerCodec, NaiveSumCodec
from .embedding import CodecEmbedding

__all__ = [
    "ByteCodec",
    "KroneckerCodec",
    "NaiveSumCodec",
    "FourierCodec",
    "CodecEmbedding",
]
