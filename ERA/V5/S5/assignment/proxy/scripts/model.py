#!/usr/bin/env python3
"""A small Llama-shaped decoder for the mixture ablation.

Deliberately plain: RMSNorm, rotary embeddings, SwiGLU, tied input/output embeddings, no
biases. Nothing here varies between arms — architecture, optimiser, seed and token count are
held fixed so that the only thing separating two runs is the data mixture.
"""
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
    seq_len: int = 1024
    ffn_mult: float = 2.6667      # SwiGLU hidden = round(d_model * ffn_mult) to a multiple of 64
    rope_theta: float = 10000.0
    init_std: float = 0.02

    @property
    def d_head(self):
        return self.d_model // self.n_heads

    @property
    def d_ffn(self):
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
    # x: (B, H, T, D)
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
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.embed.weight          # tied
        self.apply(self._init)
        # scaled init on residual projections, as in GPT-2/Llama practice
        for n, p in self.named_parameters():
            if n.endswith("proj.weight") or n.endswith("down.weight"):
                nn.init.normal_(p, std=cfg.init_std / math.sqrt(2 * cfg.n_layers))
        self._rope = None

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=self.cfg.init_std)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=self.cfg.init_std)

    def rope(self, device, dtype):
        if self._rope is None or self._rope[0].device != device:
            cos, sin = build_rope(self.cfg.seq_len, self.cfg.d_head,
                                  self.cfg.rope_theta, device)
            self._rope = (cos.to(dtype), sin.to(dtype))
        return self._rope

    def forward(self, idx):
        x = self.embed(idx)
        cos, sin = self.rope(idx.device, x.dtype)
        for b in self.blocks:
            x = b(x, cos, sin)
        return self.head(self.norm(x))

    # ------------------------------------------------------------------ accounting
    def param_counts(self):
        total = sum(p.numel() for p in self.parameters())
        embed = self.embed.weight.numel()
        return {"total": total, "embedding": embed, "non_embedding": total - embed}

    def flops_per_token(self):
        """Forward+backward FLOPs per token, counting the tied output head (which at this
        vocabulary is the single largest term) and the attention score matmuls."""
        c = self.cfg
        pc = self.param_counts()
        # 2 FLOPs per parameter per token in the forward pass, x3 for fwd+bwd
        dense = 6 * pc["total"]
        # attention scores + values: 2 * 2 * T * d_model per token, x3
        attn = 6 * 2 * c.seq_len * c.d_model
        return dense + attn
