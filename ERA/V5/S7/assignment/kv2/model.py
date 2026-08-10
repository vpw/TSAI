"""A small Llama-shaped decoder whose input path is swappable.

Adapted from `S5/assignment/proxy/scripts/model.py` (the mixture-ablation model that ran 8
arms on a T4) with exactly one change: the embedding is injected rather than hard-coded, so
an arm is defined by which codec built it.

Two deliberate protocol choices:

* **The output head is untied in every arm.** S5's model tied head to embedding, which is
  impossible for a structured input path and would make the comparison measure tying rather
  than the codec. It is also what V5 itself decided (§13 of the session), and the design
  board flags tying a structured input path as a hazard in its own right.
* **Nothing else varies.** Depth, width, heads, sequence length, optimiser, schedule, seed
  and token budget are identical across arms, so a difference in bits-per-byte is
  attributable to the input path and to nothing else.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int = 68096
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    seq_len: int = 512
    ffn_mult: float = 2.6667
    rope_theta: float = 10000.0
    init_std: float = 0.02

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads

    @property
    def d_ffn(self) -> int:
        return int(round(self.d_model * self.ffn_mult / 64) * 64)


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dt)


def build_rope(seq_len, d_head, theta, device):
    inv = 1.0 / (theta ** (torch.arange(0, d_head, 2, device=device).float() / d_head))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x, cos, sin):
    x1, x2 = x[..., ::2], x[..., 1::2]
    c = cos[None, None, : x.shape[2], :]
    s = sin[None, None, : x.shape[2], :]
    o1 = x1 * c - x2 * s
    o2 = x1 * s + x2 * c
    return torch.stack((o1, o2), dim=-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        H, D = self.cfg.n_heads, self.cfg.d_head
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, H, D).transpose(1, 2)
        k = k.view(B, T, H, D).transpose(1, 2)
        v = v.view(B, T, H, D).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).contiguous().view(B, T, C))


class MLP(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        h = cfg.d_ffn
        self.gate = nn.Linear(cfg.d_model, h, bias=False)
        self.up = nn.Linear(cfg.d_model, h, bias=False)
        self.down = nn.Linear(h, cfg.d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.n1 = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.n2 = RMSNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.n1(x), cos, sin)
        return x + self.mlp(self.n2(x))


class Transformer(nn.Module):
    """`embedding` is any module mapping [B,T] int -> [B,T,d_model] float.

    Pass `nn.Embedding(V, d_model)` for the dense control arm, or a `CodecEmbedding` for a
    structured arm. Everything above the input path is identical either way, which is the
    whole point of the released module's "integers in, vectors out" contract.
    """

    def __init__(self, cfg: Config, embedding: nn.Module):
        super().__init__()
        self.cfg = cfg
        self.embed = embedding
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Initialise only what we own: the codec table is a buffer and must not be touched,
        # and a CodecEmbedding's projection was already initialised by its constructor.
        self.blocks.apply(self._init)
        self._init(self.head)
        if isinstance(self.embed, nn.Embedding):
            nn.init.normal_(self.embed.weight, std=cfg.init_std)

        for n, p in self.named_parameters():
            if n.endswith("proj.weight") or n.endswith("down.weight"):
                nn.init.normal_(p, std=cfg.init_std / math.sqrt(2 * cfg.n_layers))
        self._rope = None

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=self.cfg.init_std)

    def rope(self, device, dtype):
        if self._rope is None or self._rope[0].device != device:
            cos, sin = build_rope(self.cfg.seq_len, self.cfg.d_head, self.cfg.rope_theta, device)
            self._rope = (cos.to(dtype), sin.to(dtype))
        return self._rope

    def forward(self, idx):
        x = self.embed(idx)
        cos, sin = self.rope(idx.device, x.dtype)
        for b in self.blocks:
            x = b(x, cos, sin)
        return self.head(self.norm(x))

    # ------------------------------------------------------------------ accounting
    def param_counts(self) -> dict:
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        embed = sum(p.numel() for p in self.embed.parameters() if p.requires_grad)
        head = self.head.weight.numel()
        return {
            "total_trainable": total,
            "input_path": embed,
            "output_head": head,
            "backbone": total - embed - head,
        }
