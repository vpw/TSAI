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

    def dead_input_rows(self) -> torch.Tensor:
        """Which projection input rows never receive signal from any token in the vocabulary.

        A row of `projection.weight` is dead if its corresponding code dimension is zero for
        every token: no forward pass can ever activate it and no backward pass can ever reach
        it, so its parameters are structurally untrainable. This is the census that shows the
        one-hot grid wasting three-quarters of what it allocates.
        """
        return (self.codec_table.abs().max(dim=0).values == 0)

    def extra_repr(self) -> str:
        return (
            f"codec={type(self.codec).__name__}, code_dim={self.codec.code_dim}, "
            f"d_model={self.d_model}, vocab_size={self.vocab_size}, "
            f"trainable={self.n_trainable:,}"
        )
