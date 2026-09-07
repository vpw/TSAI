# %% [markdown]
"""
# Session 11 — Optimizers and Learning-Rate Schedules

Model: reusing Session 10's nanoGPT (`../S10/assignment/notebook_src_nanogpt.py`'s
`GPTConfig`/`GPT` classes, a from-scratch char-level GPT), per the instructor's own
line in this session's transcript — *"take basically a small model, the model that you
have taken last assignment."* CPU throughout (D1); no item this session needs a measured
hardware peak the way S10's MFU item did.

Built incrementally, one item at a time. Currently: **Items 1-2.**

1. Reproduce Adam by hand.
2. Disable bias correction, plot first 20 steps both ways.
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
### 1d. Assemble item 1 results
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

# %% [markdown]
"""
## Item 2 — Disable bias correction, first 20 steps

Extends item 1's setup to 20 steps, running the *same* gradient sequence through Adam
twice — once with bias correction (as in item 1) and once with it switched off
(`m_hat = m`, `v_hat = v`) — so the only thing that differs between the two runs is the
correction itself, not the data.

Section 6 already gives the closed-form relationship between the two: since correction
divides `m` by `(1 - beta1^t)` and `v` by `(1 - beta2^t)`, the ratio of the two runs'
step sizes at any step `t` is exactly

    corrected_step / uncorrected_step = sqrt(1 - beta2^t) / (1 - beta1^t)

independent of the actual gradients. That closed form — not just eyeballing two curves
— is what pins down exactly when the difference "stops mattering," and lets us check
the two plotted trajectories against an independent calculation.
"""

# %%
### 2a. Extend item 1's gradient sequence to 20 steps
# First 5 steps are identical to item 1 (same numbers); 15 more are a seeded, mildly
# noisy continuation around the same mean, so the sequence stays realistic (same-ish
# sign, per Section 9's "correlated gradients" regime) rather than hand-picked.
import numpy as np

rng = np.random.default_rng(11)
extra_grads = np.clip(rng.normal(0.5, 0.07, size=15), 0.05, None).tolist()
GRADS_20 = GRADS + [round(float(g), 4) for g in extra_grads]
print(f"20-step gradient sequence: {GRADS_20}")

# %%
### 2b. Adam step with a bias-correction on/off switch
def adam_by_hand_switch(w0, grads, eta, beta1, beta2, eps, bias_correction):
    m, v, w = 0.0, 0.0, w0
    rows = []
    for t, g in enumerate(grads, start=1):
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g ** 2
        if bias_correction:
            m_hat, v_hat = m / (1 - beta1 ** t), v / (1 - beta2 ** t)
        else:
            m_hat, v_hat = m, v
        step = -eta * m_hat / (v_hat ** 0.5 + eps)
        w = w + step
        rows.append(dict(t=t, g=g, step=step, w=w))
    return rows


rows_corrected = adam_by_hand_switch(W0, GRADS_20, ETA, BETA1, BETA2, EPS, True)
rows_uncorrected = adam_by_hand_switch(W0, GRADS_20, ETA, BETA1, BETA2, EPS, False)

# sanity: the corrected run's first 5 steps must exactly match item 1's own numbers
for a, b in zip(rows_corrected[:5], hand_rows):
    assert abs(a["w"] - b["w"]) < 1e-12, "item 2's corrected run diverged from item 1"

print(f"{'t':>2} {'w corrected':>14} {'w uncorrected':>14} {'|diff|':>10}")
for rc, ru in zip(rows_corrected, rows_uncorrected):
    print(f"{rc['t']:>2} {rc['w']:>14.8f} {ru['w']:>14.8f} {abs(rc['w'] - ru['w']):>10.2e}")

# %%
### 2c. The exact closed-form step-size ratio, corrected/uncorrected
def correction_ratio(t, beta1, beta2):
    return ((1 - beta2 ** t) ** 0.5) / (1 - beta1 ** t)


ratios_20 = [correction_ratio(t, BETA1, BETA2) for t in range(1, 21)]
print(f"{'t':>2} {'corrected/uncorrected step ratio':>34}")
for t, r in zip(range(1, 21), ratios_20):
    print(f"{t:>2} {r:>34.4f}")

# First step (searching well past the 20-step window) where the ratio is within a
# tolerance of 1 — i.e. where bias correction stops materially changing the step size.
TOL = 0.05
t_converge = next(
    t for t in range(1, 20_000) if abs(correction_ratio(t, BETA1, BETA2) - 1) <= TOL
)
print(f"\nfirst step where |ratio - 1| <= {TOL:.0%}: t = {t_converge} "
      f"(ratio = {correction_ratio(t_converge, BETA1, BETA2):.4f})")

# %%
### 2d. Plots
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ASSETS = pathlib.Path("assets")
ASSETS.mkdir(exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

ts = [r["t"] for r in rows_corrected]
axes[0].plot(ts, [r["w"] for r in rows_corrected], marker="o", label="bias-corrected")
axes[0].plot(ts, [r["w"] for r in rows_uncorrected], marker="s", label="uncorrected")
axes[0].set_xlabel("step")
axes[0].set_ylabel("w")
axes[0].set_title("Weight trajectory, first 20 steps")
axes[0].legend()
axes[0].grid(alpha=0.3)

t_long = list(range(1, t_converge + 200))
r_long = [correction_ratio(t, BETA1, BETA2) for t in t_long]
axes[1].plot(t_long, r_long)
axes[1].axhline(1.0, color="gray", linestyle="--", linewidth=1)
axes[1].axvline(t_converge, color="red", linestyle=":", label=f"t={t_converge}")
axes[1].set_xlabel("step")
axes[1].set_ylabel("corrected / uncorrected step ratio")
axes[1].set_title("Bias-correction ratio (closed form)")
axes[1].legend()
axes[1].grid(alpha=0.3)

fig.tight_layout()
fig.savefig(ASSETS / "item2_bias_correction.png", dpi=130)
plt.close(fig)
print(f"saved {ASSETS / 'item2_bias_correction.png'}")

# %%
### 2e. Assemble item 2 results
RESULTS["item2"] = {
    "grads_20": GRADS_20,
    "rows_corrected": rows_corrected,
    "rows_uncorrected": rows_uncorrected,
    "ratio_first_20": ratios_20,
    "tolerance": TOL,
    "t_converge_within_tolerance": t_converge,
    "plot": "assets/item2_bias_correction.png",
}

# %%
## Save results (all items so far)
pathlib.Path("results.json").write_text(json.dumps(RESULTS, indent=2))
print("wrote results.json")
