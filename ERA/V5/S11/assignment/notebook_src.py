# %% [markdown]
"""
# Session 11 — Optimizers and Learning-Rate Schedules

Model: reusing Session 10's nanoGPT (`../S10/assignment/notebook_src_nanogpt.py`'s
`GPTConfig`/`GPT` classes, a from-scratch char-level GPT), per the instructor's own
line in this session's transcript — *"take basically a small model, the model that you
have taken last assignment."* CPU throughout (D1); no item this session needs a measured
hardware peak the way S10's MFU item did.

Built incrementally, one item at a time. Currently: **Item 1 only.**

1. Reproduce Adam by hand.
2. (not yet) Disable bias correction, plot first 20 steps both ways.
3. (not yet) Log the update-to-weight ratio through warmup.
4. (not yet) Cosine vs WSD, 300 steps, compared at step 200.
5. (not yet) LR sweep at widths 256/512/1,024.
"""

# %%
## 0. Environment
import json
import pathlib

import torch

torch.manual_seed(0)
RESULTS = {}

# %% [markdown]
"""
## Item 1 — Reproduce Adam by hand

Section 6 of the lesson (`resources/s11-session.md`) gives a fully worked example: one
weight starting at `w0 = 1.0`, five gradients `0.50, 0.40, 0.60, 0.45, 0.55`, learning
rate `eta = 0.001`, and the standard betas/epsilon (`beta1 = 0.9`, `beta2 = 0.999`,
`eps = 1e-8`). We reuse that exact example rather than a fresh one, since it gives a
third thing to check against (the lesson's own printed numbers) in addition to PyTorch.

The update rule, exactly as given in the lesson:

```
m <- beta1*m + (1-beta1)*g
v <- beta2*v + (1-beta2)*g^2
m_hat = m / (1 - beta1^t)
v_hat = v / (1 - beta2^t)
w <- w - eta * m_hat / (sqrt(v_hat) + eps)
```
"""

# %%
### 1a. By-hand computation, pure Python — no torch, no shortcuts
GRADS = [0.50, 0.40, 0.60, 0.45, 0.55]
ETA = 0.001
BETA1 = 0.9
BETA2 = 0.999
EPS = 1e-8
W0 = 1.0


def adam_by_hand(w0, grads, eta, beta1, beta2, eps):
    m, v, w = 0.0, 0.0, w0
    rows = []
    for t, g in enumerate(grads, start=1):
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g ** 2
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        step = -eta * m_hat / (v_hat ** 0.5 + eps)
        w = w + step
        rows.append(dict(t=t, g=g, m=m, v=v, m_hat=m_hat, v_hat=v_hat, step=step, w=w))
    return rows


hand_rows = adam_by_hand(W0, GRADS, ETA, BETA1, BETA2, EPS)
print(f"{'t':>2} {'g':>6} {'m':>10} {'v':>12} {'m_hat':>10} {'v_hat':>10} {'step':>12} {'w':>10}")
for r in hand_rows:
    print(f"{r['t']:>2} {r['g']:>6.2f} {r['m']:>10.6f} {r['v']:>12.8f} "
          f"{r['m_hat']:>10.6f} {r['v_hat']:>10.8f} {r['step']:>12.6f} {r['w']:>10.6f}")

# %%
### 1b. Cross-check against the lesson's own printed table (Section 6)
# The lesson states w after each of the 5 steps as:
#   0.999000, 0.998012, 0.997018, 0.996028, 0.995031
LESSON_W = [0.999000, 0.998012, 0.997018, 0.996028, 0.995031]

print(f"{'t':>2} {'hand w':>12} {'lesson w':>12} {'|diff|':>10}")
diffs_lesson = []
for r, lesson_w in zip(hand_rows, LESSON_W):
    diff = abs(r["w"] - lesson_w)
    diffs_lesson.append(diff)
    print(f"{r['t']:>2} {r['w']:>12.6f} {lesson_w:>12.6f} {diff:>10.2e}")

max_diff_lesson = max(diffs_lesson)
print(f"\nmax |hand - lesson| over 5 steps = {max_diff_lesson:.2e}")
assert max_diff_lesson < 1e-5, "by-hand computation disagrees with the lesson's own table"

# %%
### 1c. Verify against torch.optim.Adam
# Real tensor, real optimizer (plain Adam, not AdamW — Section 6 has no weight decay),
# float64 throughout so we're checking the formula, not float32 rounding.
w = torch.nn.Parameter(torch.tensor(W0, dtype=torch.float64))
opt = torch.optim.Adam([w], lr=ETA, betas=(BETA1, BETA2), eps=EPS)

torch_ws = []
for g in GRADS:
    opt.zero_grad()
    w.grad = torch.tensor(g, dtype=torch.float64)
    opt.step()
    torch_ws.append(w.item())

print(f"{'t':>2} {'hand w':>14} {'torch w':>14} {'|diff|':>10}")
diffs_torch = []
for r, tw in zip(hand_rows, torch_ws):
    diff = abs(r["w"] - tw)
    diffs_torch.append(diff)
    print(f"{r['t']:>2} {r['w']:>14.10f} {tw:>14.10f} {diff:>10.2e}")

max_diff_torch = max(diffs_torch)
print(f"\nmax |hand - torch| over 5 steps = {max_diff_torch:.2e}")
assert max_diff_torch < 1e-9, "by-hand computation disagrees with torch.optim.Adam"

# %%
### 1d. Save results
RESULTS["item1"] = {
    "w0": W0,
    "eta": ETA,
    "beta1": BETA1,
    "beta2": BETA2,
    "eps": EPS,
    "grads": GRADS,
    "hand": hand_rows,
    "torch_w": torch_ws,
    "lesson_w": LESSON_W,
    "max_abs_diff_hand_vs_lesson": max_diff_lesson,
    "max_abs_diff_hand_vs_torch": max_diff_torch,
}

pathlib.Path("results.json").write_text(json.dumps(RESULTS, indent=2))
print("wrote results.json")
