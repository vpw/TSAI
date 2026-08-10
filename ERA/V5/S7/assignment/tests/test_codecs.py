"""Invariants the three codecs must satisfy.

Run:  .venv/bin/python -m pytest tests/ -q

These are the claims the README makes, expressed as assertions so that a change which breaks
one of them fails loudly instead of quietly producing a different result.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kv2.codecs import FourierCodec, KroneckerCodec, NaiveSumCodec
from kv2.vocab import token_bytes, vocab_bytes

SEED = 20260810


@pytest.fixture(scope="module")
def vocab():
    return vocab_bytes()


# ---------------------------------------------------------------- determinism / contract

@pytest.mark.parametrize("codec", [
    KroneckerCodec(pos_dim=32),
    FourierCodec(n_freq=512, seed=SEED),
    NaiveSumCodec(n_freq=512, seed=SEED),
])
def test_codec_is_deterministic(codec):
    """The same token must always produce the same code -- it is a frozen buffer, not a
    parameter, and a run that reproduced a different code would invalidate every checkpoint
    trained against the old one."""
    a = codec.encode_one(b"training")
    b = codec.encode_one(b"training")
    assert np.array_equal(a, b)


@pytest.mark.parametrize("codec", [
    KroneckerCodec(pos_dim=32),
    FourierCodec(n_freq=512, seed=SEED),
    NaiveSumCodec(n_freq=512, seed=SEED),
])
def test_vectorised_matches_reference(codec, vocab):
    """The batched path used to build the table must equal the per-token reference."""
    sample = vocab[500:800]
    ref = np.stack([codec.encode_one(b) for b in sample])
    vec = codec._encode_chunk(sample)
    assert np.abs(vec - ref).max() < 1e-4


def test_code_is_normalised():
    """z-normalisation: zero mean, unit variance, as the released module specifies."""
    c = FourierCodec(n_freq=512, seed=SEED)
    v = c.encode_one("अंतर्राष्ट्रीयकरण".encode())
    assert abs(float(v.mean())) < 1e-5
    assert abs(float(v.std()) - 1.0) < 1e-3


# ---------------------------------------------------------------- the negative control

def test_naive_sum_cannot_separate_anagrams():
    """The assignment's literal reading -- 'just add them' -- is permutation-invariant.

    This is the failure that motivates phase binding, so it is asserted rather than assumed.
    """
    c = NaiveSumCodec(n_freq=512, seed=SEED)
    assert np.allclose(c.encode_one(b"listen"), c.encode_one(b"silent"), atol=1e-5)
    assert np.allclose(c.encode_one(b"abc"), c.encode_one(b"cba"), atol=1e-5)


def test_phase_binding_separates_anagrams():
    c = FourierCodec(n_freq=512, seed=SEED)
    assert not np.allclose(c.encode_one(b"listen"), c.encode_one(b"silent"), atol=1e-3)
    assert not np.allclose(c.encode_one(b"abc"), c.encode_one(b"cba"), atol=1e-3)


# ---------------------------------------------------------------- the crop

def test_kronecker_crops_at_pos_dim():
    """Two tokens agreeing on their first pos_dim bytes get byte-identical codes.

    Uses a real pair from the sarvam1 vocabulary, not a synthetic one.
    """
    a = "अंतर्राष्ट्रीयकरण".encode()   # 51 bytes
    b = "अंतर्राष्ट्रीयता".encode()     # 48 bytes
    assert a[:32] == b[:32] and a != b

    k32 = KroneckerCodec(pos_dim=32)
    assert np.allclose(k32.encode_one(a), k32.encode_one(b))

    # Widening the window to 48 separates them -- matching the session's own byte-budget lab.
    k48 = KroneckerCodec(pos_dim=48)
    assert not np.allclose(k48.encode_one(a), k48.encode_one(b))


def test_fourier_has_no_crop():
    a = "अंतर्राष्ट्रीयकरण".encode()
    b = "अंतर्राष्ट्रीयता".encode()
    c = FourierCodec(n_freq=1024, seed=SEED)
    assert not np.allclose(c.encode_one(a), c.encode_one(b), atol=1e-3)
    assert c.max_bytes is None


# ---------------------------------------------------------------- invertibility

@pytest.mark.parametrize("text", [
    "training", "a", "the", "भारत", "తెలుగు", "தமிழ்",
    "अंतर्राष्ट्रीयकरण",              # 51 bytes: past Kronecker's window
    "internationalisation",
])
def test_phase_code_is_exactly_invertible(text):
    """Unbinding recovers every byte, including for tokens the 32-byte window would crop."""
    c = FourierCodec(n_freq=1024, seed=SEED)
    bs = text.encode("utf-8")
    assert c.decode(c._complex_code(bs), len(bs)) == bs


def test_batched_decode_matches_scalar():
    c = FourierCodec(n_freq=512, seed=SEED)
    seqs = [b"training", b"the", "भारत".encode(), b"x"]
    lens = np.array([len(s) for s in seqs])
    batched = c.decode_batch(c.complex_codes(seqs), lens)
    scalar = [c.decode(c._complex_code(s), len(s)) for s in seqs]
    assert batched == scalar


def test_invertibility_needs_enough_dimensions():
    """Capacity is real: too narrow a code cannot be inverted, and it should fail rather
    than silently appear to work."""
    narrow = FourierCodec(n_freq=32, seed=SEED)   # 64-dim code
    wide = FourierCodec(n_freq=1024, seed=SEED)   # 2048-dim
    bs = "अंतर्राष्ट्रीयकरण".encode()
    assert narrow.decode(narrow._complex_code(bs), len(bs)) != bs
    assert wide.decode(wide._complex_code(bs), len(bs)) == bs


# ---------------------------------------------------------------- parameter accounting

def test_input_path_cost_is_independent_of_vocabulary():
    """The property inherited from the Kronecker paper: no |V| in the parameter count."""
    torch = pytest.importorskip("torch")
    from kv2.embedding import CodecEmbedding

    codec = FourierCodec(n_freq=256, seed=SEED)
    small = CodecEmbedding(codec, [b"a", b"bb", b"ccc"], d_model=64)
    large = CodecEmbedding(codec, [bytes([i % 256]) * (1 + i % 7) for i in range(5000)],
                           d_model=64)
    assert small.n_trainable == large.n_trainable == codec.code_dim * 64
    # ... whereas a dense table would have grown by exactly the vocabulary ratio.
    assert large.n_dense_equivalent > small.n_dense_equivalent


def test_fourier_code_is_denser_than_kronecker(vocab):
    """The core structural claim: the grid leaves most of its cells unreachable."""
    k = KroneckerCodec(pos_dim=32)
    stats = k.unreachable_cells(vocab[:20000])
    assert stats["cells_unreachable"] > 0.5 * stats["code_dim"]

    f = FourierCodec(n_freq=1024, seed=SEED)
    tbl = f.encode_many(vocab[:2000], dtype=np.float32)
    # every Fourier coordinate varies across tokens; none is structurally inert
    assert float(tbl.var(axis=0).min()) > 0
