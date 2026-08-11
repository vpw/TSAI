# Ablation results — input path, everything else pinned

Model: d_model 512, 8 layers, 8 heads, seq_len 512, vocab 68,096.
Budget: 39,993,344 tokens per arm, seed 20260810, device cuda.
Mixture (declared): {'general_web': 34, 'code': 16, 'stem': 10, 'indic_A_verified': 20, 'indic_B_unverified': 10, 'indic_C_translated': 10}
Mixture (realised, first arm): {'code': 15.86, 'general_web': 33.99, 'indic_A_verified': 19.81, 'indic_B_unverified': 10.0, 'indic_C_translated': 10.03, 'stem': 10.3}

Every arm shares architecture, optimiser, schedule, seed, sequence length, token budget and token stream. The embedding module is the only difference.

## Parameters

| arm | what it is | code dim | input path | dead rows | total trainable |
|---|---|---|---|---|---|
| `dense` | full V x d_model table (control, upper bound) | - | 34,865,152 | - | 94,642,688 |
| `kronecker_32` | shipped byte-Kronecker grid, 32-byte window | 8192 | 4,194,304 | - | 63,971,840 |
| `fourier_2048` | phase-bound Fourier code, 4x smaller than the grid | 2048 | 1,048,576 | - | 60,826,112 |
| `fourier_8192` | phase-bound Fourier code at matched dimension | 8192 | 4,194,304 | - | 63,971,840 |
| `naive_2048` | unbound wave sum (negative control, order-blind) | 2048 | 1,048,576 | - | 60,826,112 |

## Bits per byte (lower is better)

| arm | general_web | code | stem | indic_A_verified | indic_B_unverified | indic_C_translated | **macro** |
|---|---|---|---|---|---|---|---|
| `dense` | 1.5958 | 1.4027 | 1.7745 | 1.0856 | 0.6794 | 0.9582 | **1.2494** |
| `kronecker_32` | 1.6332 | 1.5367 | 1.8586 | 1.1142 | 0.6902 | 0.9762 | **1.3015** |
| `fourier_2048` | 1.6277 | 1.5324 | 1.8516 | 1.1216 | 0.6911 | 0.9750 | **1.2999** |
| `fourier_8192` | 1.6765 | 1.6279 | 1.9225 | 1.1364 | 0.7081 | 1.0054 | **1.3461** |
| `naive_2048` | 1.6684 | 1.5674 | 1.8835 | 1.1216 | 0.7081 | 1.0031 | **1.3254** |

## Bits per byte on long targets (>32 UTF-8 bytes)

These are exactly the tokens the 32-byte window truncates.

| arm | general_web | code | stem | indic_A_verified | indic_B_unverified | indic_C_translated |
|---|---|---|---|---|---|---|
| `dense` | - | - | - | 0.3706 | 0.2869 | 0.2740 |
| `kronecker_32` | - | - | - | 0.3731 | 0.2912 | 0.2862 |
| `fourier_2048` | - | - | - | 0.3758 | 0.2913 | 0.2854 |
| `fourier_8192` | - | - | - | 0.3808 | 0.3002 | 0.2976 |
| `naive_2048` | - | - | - | 0.3818 | 0.3012 | 0.2976 |

Support (scored positions per lane): {'general_web': None, 'code': None, 'stem': None, 'indic_A_verified': 382, 'indic_B_unverified': 149, 'indic_C_translated': 157}

*A small support means a wide error bar. Read this slice as directional unless the counts are large.*

## Change vs `kronecker_32` (macro bpb, negative = better)

- `dense`: -0.0521 bpb (-4.01%)
- `fourier_2048`: -0.0016 bpb (-0.12%)
- `fourier_8192`: +0.0446 bpb (+3.43%)
- `naive_2048`: +0.0239 bpb (+1.83%)

## Wall clock

- `dense`: 30.92 min
- `kronecker_32`: 29.42 min
- `fourier_2048`: 28.81 min
- `fourier_8192`: 29.51 min
- `naive_2048`: 28.8 min
