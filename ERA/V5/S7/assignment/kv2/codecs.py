"""Byte codecs: token bytes -> a fixed vector the model's projection consumes.

A codec is *not* trained. It turns the UTF-8 bytes of a token into a deterministic vector,
and one shared `Linear(code_dim, d_model)` on top of it is the only learned thing in the
input path. That split is what makes the input-side parameter count independent of the
vocabulary, which is the property the Kronecker paper is built around and which we preserve.

Everything here is numpy so the static proofs run without a deep-learning framework. The
torch wrapper in `embedding.py` consumes the precomputed table.
"""

from __future__ import annotations

import numpy as np

# UTF-8 never produces these two bytes, so no real token can ever activate their rows.
# Kept as a named constant because it is load-bearing in the dead-parameter census.
UTF8_IMPOSSIBLE_BYTES = (0xC0, 0xC1)


class ByteCodec:
    """Base class. Subclasses implement `encode_one` and declare `code_dim`."""

    code_dim: int
    #: Longest token, in bytes, the codec can represent without loss. `None` means unbounded.
    max_bytes: int | None = None

    def encode_one(self, bs: bytes) -> np.ndarray:
        raise NotImplementedError

    def encode_many(self, seqs: list[bytes], dtype=np.float32, chunk: int = 4096) -> np.ndarray:
        """Encode a whole vocabulary into a (V, code_dim) table.

        Subclasses override `_encode_chunk` to do this vectorised; the default falls back to
        a per-token loop. Chunking keeps peak memory bounded -- at 68k tokens and an
        8,192-dim code the full table is already ~1.1 GB in fp16, and building it via a list
        of per-token arrays would transiently double that.
        """
        out = np.zeros((len(seqs), self.code_dim), dtype=dtype)
        for s in range(0, len(seqs), chunk):
            block = seqs[s:s + chunk]
            out[s:s + len(block)] = self._encode_chunk(block).astype(dtype, copy=False)
        return out

    def _encode_chunk(self, seqs: list[bytes]) -> np.ndarray:
        return np.stack([self.encode_one(bs) for bs in seqs])

    def truncates(self, bs: bytes) -> bool:
        """Does this codec silently drop bytes from this token?"""
        return self.max_bytes is not None and len(bs) > self.max_bytes

    def describe(self) -> dict:
        return {
            "codec": type(self).__name__,
            "code_dim": self.code_dim,
            "max_bytes": self.max_bytes,
        }


def _znorm(v: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-variance, matching the released module's final normalisation."""
    v = v - v.mean()
    s = v.std()
    return v / s if s > 1e-8 else v


def _znorm_rows(m: np.ndarray) -> np.ndarray:
    """Row-wise `_znorm` for a (N, code_dim) block."""
    m = m - m.mean(axis=1, keepdims=True)
    s = m.std(axis=1, keepdims=True)
    np.maximum(s, 1e-8, out=s)
    return m / s


class KroneckerCodec(ByteCodec):
    """The shipped baseline (arXiv 2605.29459).

        kappa(b) = (1/sqrt(L)) * vec( sum_p  c[byte_p] (x) p[position_p] ),  L = min(len(b), pos_dim)

    `c` is one-hot over 256 byte values, `p` one-hot over `pos_dim` positions, so the
    Kronecker product of the two one-hots is a single marked cell in a (char_dim, pos_dim)
    grid. Flattened, that grid is the code: `char_dim * pos_dim` numbers, of which at most
    `pos_dim` are non-zero.

    The `L = min(...)` is the crop this whole submission is about: two tokens agreeing on
    their first `pos_dim` bytes produce byte-identical codes and are therefore the same token
    to the model, permanently and silently.
    """

    def __init__(self, char_dim: int = 256, pos_dim: int = 32):
        self.char_dim = char_dim
        self.pos_dim = pos_dim
        self.code_dim = char_dim * pos_dim
        self.max_bytes = pos_dim

    def encode_one(self, bs: bytes) -> np.ndarray:
        grid = np.zeros((self.char_dim, self.pos_dim), dtype=np.float32)
        L = min(len(bs), self.pos_dim)
        if L == 0:
            return grid.reshape(-1)
        for p in range(L):
            grid[bs[p], p] = 1.0
        return _znorm(grid.reshape(-1) / np.sqrt(L))

    def _encode_chunk(self, seqs: list[bytes]) -> np.ndarray:
        n = len(seqs)
        out = np.zeros((n, self.code_dim), dtype=np.float32)
        rows, cols = [], []
        scales = np.ones(n, dtype=np.float32)
        for i, bs in enumerate(seqs):
            L = min(len(bs), self.pos_dim)
            if L == 0:
                continue
            scales[i] = 1.0 / np.sqrt(L)
            for p in range(L):
                rows.append(i)
                # flat index of grid cell (byte value, position) in a (char_dim, pos_dim) grid
                cols.append(bs[p] * self.pos_dim + p)
        if rows:
            out[np.array(rows), np.array(cols)] = 1.0
            out *= scales[:, None]
        return _znorm_rows(out)

    def unreachable_cells(self, byte_seqs: list[bytes]) -> dict:
        """Grid cells no token in this vocabulary can ever activate.

        UTF-8 is highly structured -- position 0 of a Devanagari token is always 0xE0, most
        byte values never appear at most positions -- so the (256 x pos_dim) grid is far
        larger than the set of (value, position) pairs that actually occur. Every unreachable
        cell is a row of the projection matrix that receives no signal in the forward pass and
        no gradient in the backward pass, before z-normalisation smears a constant over it.
        """
        used = set()
        occ = np.zeros(self.pos_dim)
        for bs in byte_seqs:
            L = min(len(bs), self.pos_dim)
            occ[:L] += 1
            for p in range(L):
                used.add((bs[p], p))
        unreachable = self.code_dim - len(used)
        return {
            "code_dim": self.code_dim,
            "cells_used": len(used),
            "cells_unreachable": unreachable,
            "unreachable_pct": round(100.0 * unreachable / self.code_dim, 2),
            "mean_column_occupancy": float(occ.mean() / max(len(byte_seqs), 1)),
        }

    def describe(self) -> dict:
        d = super().describe()
        d.update(char_dim=self.char_dim, pos_dim=self.pos_dim)
        return d


def _pad_batch(seqs: list[bytes]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pack a list of byte strings into a padded (N, Lmax) uint8 array plus a validity mask.

    Returned as (values, mask, lengths). Padding is masked out of the superposition, so the
    padded positions contribute nothing -- this is a batching device, not a crop.
    """
    n = len(seqs)
    lengths = np.array([len(b) for b in seqs], dtype=np.int64)
    lmax = int(lengths.max()) if n and lengths.max() > 0 else 1
    vals = np.zeros((n, lmax), dtype=np.uint8)
    mask = np.zeros((n, lmax), dtype=bool)
    for i, bs in enumerate(seqs):
        if bs:
            vals[i, :len(bs)] = np.frombuffer(bs, dtype=np.uint8)
            mask[i, :len(bs)] = True
    return vals, mask, lengths


class _PhaseAtoms:
    """Shared frequency draw for the two wave codecs.

    Each of the `n_freq` complex dimensions gets one frequency for byte VALUE and one for
    byte POSITION, drawn once from a fixed seed and then frozen forever. Freezing matters
    for the same reason the Kronecker byte buffer is frozen: change the atoms and every
    token's code changes, so a projection trained against the old atoms is meaningless.
    """

    def __init__(self, n_freq: int, max_pos: int, seed: int = 0, char_dim: int = 256):
        rng = np.random.default_rng(seed)
        self.n_freq = n_freq
        self.char_dim = char_dim
        self.max_pos = max_pos
        self.w_value = rng.uniform(0.0, 2.0 * np.pi, n_freq)
        self.w_pos = rng.uniform(0.0, 2.0 * np.pi, n_freq)
        # value_atoms[v] = exp(i * w_value * v);  pos_atoms[p] = exp(i * w_pos * p)
        self.value_atoms = np.exp(1j * np.outer(np.arange(char_dim), self.w_value))
        self.pos_atoms = np.exp(1j * np.outer(np.arange(max_pos), self.w_pos))


class NaiveSumCodec(ByteCodec):
    """The assignment's literal phrasing, kept honest as a negative control.

    "Why can't I represent each character like a fourier wave, and just add them to make a
    word!!"  -- so: no position information at all, just superpose one wave per byte.

        phi(b) = (1/sqrt(L)) * sum_p exp(i * w_value * byte_p)

    This is **permutation-invariant by construction**: the sum does not depend on the order
    of its terms, so any two tokens that are anagrams of each other receive byte-identical
    codes. No projection on top can separate them, because it is never shown a difference.

    That is not a bug in the implementation, it is the reason the naive reading of Problem 4
    cannot work, and `proofs/` demonstrates it on real vocabulary rather than asserting it.
    """

    def __init__(self, n_freq: int = 1024, seed: int = 0, char_dim: int = 256):
        self.atoms = _PhaseAtoms(n_freq, max_pos=1, seed=seed, char_dim=char_dim)
        self.n_freq = n_freq
        self.code_dim = 2 * n_freq
        self.max_bytes = None  # no crop, but also no order

    def encode_one(self, bs: bytes) -> np.ndarray:
        if len(bs) == 0:
            return np.zeros(self.code_dim, dtype=np.float32)
        vals = np.frombuffer(bs, dtype=np.uint8)
        acc = self.atoms.value_atoms[vals].sum(axis=0) / np.sqrt(len(vals))
        return _znorm(np.concatenate([acc.real, acc.imag]).astype(np.float32))

    def _encode_chunk(self, seqs: list[bytes]) -> np.ndarray:
        vals, mask, lengths = _pad_batch(seqs)
        acc = np.zeros((len(seqs), self.n_freq), dtype=np.complex64)
        # Accumulate position by position: materialising the full (N, Lmax, K) product would
        # cost gigabytes per chunk for no benefit, since it is summed away immediately.
        for p in range(vals.shape[1]):
            acc += self.atoms.value_atoms[vals[:, p]] * mask[:, p][:, None]
        acc /= np.sqrt(np.maximum(lengths, 1))[:, None]
        return _znorm_rows(np.concatenate([acc.real, acc.imag], axis=1).astype(np.float32))


class FourierCodec(ByteCodec):
    """The repair: bind value to position by phase, then superpose.

        phi(b) = (1/sqrt(L)) * sum_p exp( i * ( w_value * byte_p + w_pos * p ) )

    Multiplying two unit-modulus phasors adds their phases, so `exp(i*w_v*v) * exp(i*w_p*p)`
    binds a byte value to a byte position exactly as the Kronecker product of two one-hots
    does -- but *densely*, in `n_freq` complex dimensions instead of `char_dim * pos_dim`
    sparse ones. This is FHRR binding (Plate's Holographic Reduced Representations in their
    complex/Fourier form) applied to bytes.

    Three consequences, all of which `proofs/` measures:

    * **No length cap.** `L` is the token's true byte length. Nothing is cropped, so the
      collision class the Kronecker crop creates does not exist here. Capacity degrades
      smoothly with length instead of failing abruptly at a threshold.
    * **Invertible.** Unbind position `p` by multiplying by the conjugate position atom, then
      correlate against all 256 value atoms and take the argmax. The cleanup step (argmax
      over a known, finite alphabet) is what lifts recovery to exact.
    * **Dense.** Every dimension carries signal for every token, so there are no structurally
      dead rows in the projection -- which is precisely the 75.6% the one-hot grid wastes.
    """

    def __init__(self, n_freq: int = 1024, max_pos: int = 64, seed: int = 0, char_dim: int = 256):
        self.atoms = _PhaseAtoms(n_freq, max_pos=max_pos, seed=seed, char_dim=char_dim)
        self.n_freq = n_freq
        self.max_pos = max_pos
        self.code_dim = 2 * n_freq
        # Positions beyond max_pos wrap rather than truncate, so nothing is ever dropped.
        self.max_bytes = None

    def _complex_code(self, bs: bytes) -> np.ndarray:
        vals = np.frombuffer(bs, dtype=np.uint8)
        L = len(vals)
        pos = np.arange(L) % self.max_pos
        bound = self.atoms.value_atoms[vals] * self.atoms.pos_atoms[pos]
        return bound.sum(axis=0) / np.sqrt(L)

    def encode_one(self, bs: bytes) -> np.ndarray:
        if len(bs) == 0:
            return np.zeros(self.code_dim, dtype=np.float32)
        acc = self._complex_code(bs)
        return _znorm(np.concatenate([acc.real, acc.imag]).astype(np.float32))

    def _encode_chunk(self, seqs: list[bytes]) -> np.ndarray:
        vals, mask, lengths = _pad_batch(seqs)
        acc = np.zeros((len(seqs), self.n_freq), dtype=np.complex64)
        for p in range(vals.shape[1]):
            atom_p = self.atoms.pos_atoms[p % self.max_pos]
            acc += self.atoms.value_atoms[vals[:, p]] * atom_p[None, :] * mask[:, p][:, None]
        acc /= np.sqrt(np.maximum(lengths, 1))[:, None]
        return _znorm_rows(np.concatenate([acc.real, acc.imag], axis=1).astype(np.float32))

    # -- inversion -------------------------------------------------------------------

    def decode(self, code: np.ndarray, n_bytes: int) -> bytes:
        """Recover the token's bytes from its code.

        `code` may be the real 2*n_freq vector or the raw complex vector. `n_bytes` is how
        many positions to read back; the codec does not store length, exactly as the
        Kronecker module keeps a separate `_length_buffer`.
        """
        z = self._as_complex(code)
        out = bytearray()
        for p in range(n_bytes):
            q = z * np.conj(self.atoms.pos_atoms[p % self.max_pos])
            scores = (self.atoms.value_atoms.conj() * q).sum(axis=1).real
            out.append(int(np.argmax(scores)))
        return bytes(out)

    def decode_batch(self, codes: np.ndarray, lengths: np.ndarray) -> list[bytes]:
        """Vectorised `decode` over many codes at once.

        Same maths, but the per-position correlation becomes one (N, K) x (K, 256) matmul
        instead of N separate ones, which is what makes sweeping dimensions and noise levels
        over thousands of tokens tractable.
        """
        z = self._as_complex(codes)                      # (N, K)
        n = z.shape[0]
        lmax = int(lengths.max()) if n else 0
        atoms_h = self.atoms.value_atoms.conj().T        # (K, 256)
        out = [bytearray() for _ in range(n)]
        for p in range(lmax):
            q = z * np.conj(self.atoms.pos_atoms[p % self.max_pos])[None, :]
            pred = np.argmax((q @ atoms_h).real, axis=1)  # (N,)
            live = lengths > p
            for i in np.nonzero(live)[0]:
                out[i].append(int(pred[i]))
        return [bytes(b) for b in out]

    def complex_codes(self, seqs: list[bytes]) -> np.ndarray:
        """Raw (pre-normalisation) complex codes for a batch, for decoding experiments."""
        vals, mask, lengths = _pad_batch(seqs)
        acc = np.zeros((len(seqs), self.n_freq), dtype=np.complex64)
        for p in range(vals.shape[1]):
            atom_p = self.atoms.pos_atoms[p % self.max_pos]
            acc += self.atoms.value_atoms[vals[:, p]] * atom_p[None, :] * mask[:, p][:, None]
        return acc / np.sqrt(np.maximum(lengths, 1))[:, None]

    def _as_complex(self, code: np.ndarray) -> np.ndarray:
        if np.iscomplexobj(code):
            return code
        half = code.shape[-1] // 2
        return code[..., :half] + 1j * code[..., half:]

    def describe(self) -> dict:
        d = super().describe()
        d.update(n_freq=self.n_freq, max_pos=self.max_pos)
        return d
