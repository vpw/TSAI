"""The torch module that swaps in for `nn.Embedding`.

Mirrors the contract the released Kronecker module states: integers in, vectors out, nothing
downstream can tell the difference. That contract is what makes the ablation clean -- the arms
differ in which codec built the frozen table, and in nothing else.

    codec table   (V, code_dim)   registered buffer, never trained
    projection    Linear(code_dim, d_model, bias=False)   the ONLY trainable parameter
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .codecs import ByteCodec


class CodecEmbedding(nn.Module):
    """A frozen byte codec plus one shared learned projection.

    The vocabulary size sizes the buffer and never the parameter count -- slide `vocab_size`
    and `n_trainable` does not move. That is the property inherited from the Kronecker paper
    and preserved here.
    """

    def __init__(
        self,
        codec: ByteCodec,
        token_byte_seqs: list[bytes],
        d_model: int,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.codec = codec
        self.d_model = d_model
        self.vocab_size = len(token_byte_seqs)

        # Built in the storage dtype directly: at 68k tokens and an 8,192-dim code the table
        # is ~1.1 GB in fp16, so materialising an fp32 copy first would be wasteful. The
        # table is frozen and O(1)-valued after z-normalisation, so fp16 costs nothing real.
        np_dtype = {torch.float16: np.float16, torch.float32: np.float32}.get(dtype, np.float32)
        table = codec.encode_many(token_byte_seqs, dtype=np_dtype)  # (V, code_dim)
        self.register_buffer("codec_table", torch.from_numpy(table))
        # Kept for the same reason the released module keeps `_length_buffer`: decoding needs
        # to know how many positions to read back, and the code itself does not carry length.
        lengths = np.array([len(b) for b in token_byte_seqs], dtype=np.int32)
        self.register_buffer("length_buffer", torch.from_numpy(lengths))

        self.projection = nn.Linear(codec.code_dim, d_model, bias=False)
        nn.init.normal_(self.projection.weight, std=0.02)

    @property
    def n_trainable(self) -> int:
        return self.projection.weight.numel()

    @property
    def n_dense_equivalent(self) -> int:
        """Parameters a plain `nn.Embedding(V, d_model)` would have cost."""
        return self.vocab_size * self.d_model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """[B, T] int -> [B, T, d_model] float."""
        codes = self.codec_table[input_ids].to(self.projection.weight.dtype)
        return self.projection(codes)

    def constant_input_rows(self, rel_tol: float = 1e-6) -> torch.Tensor:
        """Code coordinates that carry no information: (near-)zero variance across the vocab.

        Note this is deliberately *not* "the coordinate is zero". The released codec
        z-normalises, which shifts never-activated cells to a non-zero value -- so testing
        for zero finds nothing even when three quarters of the grid is unreachable. What
        actually matters is whether a coordinate varies across tokens; one that does not
        contributes a constant, and its column of the projection is only ever a bias.

        `KroneckerCodec.unreachable_cells()` measures the same waste structurally, before
        normalisation, and `effective_rank()` measures what survives it.
        """
        var = self.codec_table.float().var(dim=0)
        return var <= rel_tol * float(var.median().clamp(min=1e-12))

    def effective_rank(self, n_sample: int = 6000, seed: int = 0) -> dict:
        """How many independent directions the code actually spans.

        This is the fair way to compare codes of different widths: a codec that spends 8,192
        coordinates to deliver 2,000 directions is buying nothing with the other 6,192,
        whatever their individual variances look like.
        """
        g = torch.Generator().manual_seed(seed)
        n = min(n_sample, self.vocab_size)
        idx = torch.randperm(self.vocab_size, generator=g)[:n]
        m = self.codec_table[idx].float()
        m = m - m.mean(dim=0, keepdim=True)
        s = torch.linalg.svdvals(m)
        tol = s.max() * max(m.shape) * torch.finfo(torch.float32).eps
        rank = int((s > tol).sum())
        # 99% energy rank: how many directions hold essentially all the signal.
        energy = torch.cumsum(s**2, 0) / (s**2).sum()
        rank99 = int((energy < 0.99).sum()) + 1
        return {
            "code_dim": self.codec.code_dim,
            "sampled": n,
            "numerical_rank": rank,
            "rank_99pct_energy": rank99,
            "rank_per_dim": round(rank / self.codec.code_dim, 4),
        }

    def extra_repr(self) -> str:
        return (
            f"codec={type(self.codec).__name__}, code_dim={self.codec.code_dim}, "
            f"d_model={self.d_model}, vocab_size={self.vocab_size}, "
            f"trainable={self.n_trainable:,}"
        )
