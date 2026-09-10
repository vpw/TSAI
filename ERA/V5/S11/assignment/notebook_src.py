# %% [markdown]
"""
# Session 11 — Optimizers and Learning-Rate Schedules

Model: reusing Session 10's nanoGPT (`../S10/assignment/notebook_src_nanogpt.py`'s
`GPTConfig`/`GPT` classes, a from-scratch char-level GPT), per the instructor's own
line in this session's transcript — *"take basically a small model, the model that you
have taken last assignment."* CPU throughout (D1); no item this session needs a measured
hardware peak the way S10's MFU item did.

Built incrementally, one item at a time. All five items below.

1. Reproduce Adam by hand.
2. Disable bias correction, plot first 20 steps both ways.
3. Log the update-to-weight ratio through warmup.
4. Cosine vs WSD, 300 steps, compared at step 200.
5. LR sweep at widths 256/512/1,024, extrapolate to 4,096.
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

# %% [markdown]
"""
## Item 3 — Update-to-weight ratio through warmup, nanoGPT

From here on, items reuse Session 10's nanoGPT model rather than items 1-2's single-
weight toy — per D2 (`TODO.md`), and per the instructor's own line in this session's
transcript: *"take basically a small model, the model that you have taken last
assignment."* The model classes below are copied verbatim from
`../S10/assignment/notebook_src_nanogpt.py` so this notebook stays self-contained (it
will eventually be split into its own repo — see `TODO.md`'s push-destination
convention). Baseline config: `n_embd=128, n_layer=4, n_head=4, seq_len=128,
batch_size=8`, char-level tinyshakespeare. CPU throughout (D1).

Section 14 states the V5 decision directly: *log the update-to-weight ratio per layer
from step one.* Section 9 names the exact quantity — the size of the update divided by
the size of the weight — and its own widget reports the largest ratio any layer sees
across a run falling from **19.2e-3 (no warmup) to 2.83e-3 (with warmup)**. We reproduce
that same before/after contrast at our own (much smaller) scale, then use the
with-warmup run to answer the assignment's actual question: at which step does warmup
stop changing the ratio?
"""

# %%
### 3a. Data — tinyshakespeare-char (same source as S10's nanoGPT)
import urllib.request

DATA_PATH = ASSETS / "tinyshakespeare.txt"
if not DATA_PATH.exists():
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        DATA_PATH,
    )

text3 = DATA_PATH.read_text(encoding="utf-8")
chars3 = sorted(set(text3))
stoi3 = {c: i for i, c in enumerate(chars3)}
STREAM3 = torch.tensor([stoi3[c] for c in text3], dtype=torch.long)
print(f"corpus: {len(text3):,} chars, vocab V={len(chars3)}")

# %%
### 3b. Model — nanoGPT (copied from S10's notebook_src_nanogpt.py, unchanged)
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

DEVICE = torch.device("cpu")  # D1: CPU throughout this session


@dataclass
class GPTConfig:
    vocab_size: int = 65
    n_embd: int = 128
    n_layer: int = 4
    n_head: int = 4
    seq_len: int = 128
    batch_size: int = 8
    dropout: float = 0.0


def get_batch3(cfg, stream, B=None, T=None, generator=None):
    B, T = B or cfg.batch_size, T or cfg.seq_len
    ix = torch.randint(len(stream) - T - 1, (B,), generator=generator)
    return torch.stack([stream[i:i + T] for i in ix]).to(DEVICE)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_head, self.head_dim = cfg.n_head, cfg.n_embd // cfg.n_head
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, self.head_dim).transpose(1, 2) for t in (q, k, v))
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.c_proj(out.transpose(1, 2).reshape(B, T, C))


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=False)

    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.seq_len, cfg.n_embd)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight  # weight tying, nanoGPT's own convention
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, tokens):
        B, T = tokens.shape
        pos = torch.arange(T, device=tokens.device)
        x = self.wte(tokens) + self.wpe(pos)[None]
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        return self.lm_head(x)


def build_model(cfg, seed):
    torch.manual_seed(seed)
    return GPT(cfg).to(DEVICE)


CFG3 = GPTConfig(vocab_size=len(chars3))  # baseline n_embd=128, per D2
model3 = build_model(CFG3, seed=1337)
n_params3 = sum(p.numel() for p in model3.parameters())
param_names3 = [n for n, _ in model3.named_parameters()]
weight_names3 = [n for n, p in model3.named_parameters() if p.dim() >= 2]
print(f"config: n_embd={CFG3.n_embd} n_layer={CFG3.n_layer} n_head={CFG3.n_head}  "
      f"params={n_params3:,}  tensors={len(param_names3)} (tied wte/lm_head counted once), "
      f"of which {len(weight_names3)} are weight matrices (dim>=2, tracked for item 3's ratio)")

# %%
### 3c. Training loop with per-layer update-to-weight ratio logging
# ratio_t(layer) = ||update||_2 / ||weight before the update||_2 — exactly Section 9's
# "size of the update divided by the size of the weight," computed from the optimizer's
# *actual* step (AdamW, with decoupled decay per Section 14), not just the raw gradient.
#
# Tracked only for weight matrices (dim >= 2) — the same tensors Section 7 excludes from
# decay (norm scales, biases). This isn't just consistency: LayerNorm's bias is
# initialized at exactly 0, so its weight-norm denominator starts at 0 and the ratio
# blows up to ~1e9 on step 1 (confirmed empirically — training itself stayed healthy,
# loss fell from ~4.2 to ~2.5 with no NaN/inf, so the blow-up was purely this metric
# dividing by a near-zero norm, not a real training problem).
PEAK_LR = 3e-4
TOTAL_STEPS = 300
WARMUP_STEPS = 60  # 20% of the budget — enough to see the ramp complete and plateau


def lr_at_step(step, warmup_steps, peak_lr):
    if warmup_steps <= 0:
        return peak_lr
    return peak_lr * min(1.0, (step + 1) / warmup_steps)


def train_and_log_ratios(cfg, stream, seed, warmup_steps, total_steps, peak_lr=PEAK_LR):
    model = build_model(cfg, seed)
    # decoupled decay excludes 1D tensors (norm scales/biases, per Section 7)
    decay_params = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay_params = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay_params, "weight_decay": 0.1},
         {"params": no_decay_params, "weight_decay": 0.0}],
        lr=peak_lr,
    )
    gen = torch.Generator().manual_seed(seed + 1)
    tracked = [n for n, p in model.named_parameters() if p.dim() >= 2]
    ratio_log = {n: [] for n in tracked}
    max_ratio_log, loss_log, lr_log = [], [], []
    for step in range(total_steps):
        lr = lr_at_step(step, warmup_steps, peak_lr)
        for g in opt.param_groups:
            g["lr"] = lr
        before = {n: p.detach().clone() for n, p in model.named_parameters() if p.dim() >= 2}
        batch = get_batch3(cfg, stream, generator=gen)
        opt.zero_grad(set_to_none=True)
        logits = model(batch)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, cfg.vocab_size), batch[:, 1:].reshape(-1))
        loss.backward()
        opt.step()
        step_max = 0.0
        for n, p in model.named_parameters():
            if p.dim() < 2:
                continue
            delta = (p.detach() - before[n]).norm().item()
            wnorm = before[n].norm().item()
            ratio = delta / (wnorm + 1e-12)
            ratio_log[n].append(ratio)
            step_max = max(step_max, ratio)
        max_ratio_log.append(step_max)
        loss_log.append(loss.item())
        lr_log.append(lr)
    return dict(ratio_log=ratio_log, max_ratio_log=max_ratio_log, loss_log=loss_log,
                lr_log=lr_log, names=tracked)


run_warmup = train_and_log_ratios(CFG3, STREAM3, seed=11, warmup_steps=WARMUP_STEPS, total_steps=TOTAL_STEPS)
run_nowarmup = train_and_log_ratios(CFG3, STREAM3, seed=11, warmup_steps=0, total_steps=TOTAL_STEPS)

peak_nowarmup = max(run_nowarmup["max_ratio_log"])
peak_warmup = max(run_warmup["max_ratio_log"])
print(f"largest update/weight ratio ever seen — no warmup:   {peak_nowarmup:.4e}")
print(f"largest update/weight ratio ever seen — with warmup: {peak_warmup:.4e}")
print(f"(lesson's own widget, at its scale: 19.2e-3 -> 2.83e-3)")

# %%
### 3d. When does warmup stop changing the ratio?
# Detect the step after which the *max-over-layers* ratio settles into (and stays in) a
# band around its own steady state, rather than eyeballing the curve. Steady state is
# estimated from the final 30% of the run; "settled" means a step-count-window rolling
# mean stays within `tol` of that steady state for the rest of the run.
def find_stabilization_step(series, tail_frac=0.3, window=10, tol=0.20):
    n = len(series)
    tail = series[int(n * (1 - tail_frac)):]
    target = sum(tail) / len(tail)
    roll = [
        sum(series[max(0, i - window + 1):i + 1]) / len(series[max(0, i - window + 1):i + 1])
        for i in range(n)
    ]
    for i in range(n):
        if all(abs(roll[j] - target) <= tol * target for j in range(i, n)):
            return i, target
    return n - 1, target


stabilize_step, steady_state = find_stabilization_step(run_warmup["max_ratio_log"])
print(f"warmup schedule itself completes ramping at step {WARMUP_STEPS - 1} (by construction)")
print(f"measured: max-ratio curve settles within 20% of its steady state "
      f"({steady_state:.4e}) from step {stabilize_step} onward")

# %%
### 3e. Plots
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

axes[0].plot(run_nowarmup["max_ratio_log"], label="no warmup", alpha=0.8)
axes[0].plot(run_warmup["max_ratio_log"], label="with warmup", alpha=0.8)
axes[0].axvline(WARMUP_STEPS - 1, color="gray", linestyle="--", linewidth=1, label="warmup ends")
axes[0].axvline(stabilize_step, color="red", linestyle=":", linewidth=1.5, label=f"settles (step {stabilize_step})")
axes[0].set_yscale("log")
axes[0].set_xlabel("step")
axes[0].set_ylabel("max update/weight ratio (log scale)")
axes[0].set_title("Largest per-layer ratio, warmup vs none")
axes[0].legend(fontsize=8)
axes[0].grid(alpha=0.3)

TRACKED_LAYERS = [
    "wte.weight", "wpe.weight", "blocks.0.attn.c_attn.weight",
    "blocks.0.mlp.c_fc.weight", "blocks.3.mlp.c_fc.weight",
]
for layer in TRACKED_LAYERS:
    axes[1].plot(run_warmup["ratio_log"][layer], label=layer, alpha=0.8)
axes[1].axvline(WARMUP_STEPS - 1, color="gray", linestyle="--", linewidth=1)
axes[1].set_yscale("log")
axes[1].set_xlabel("step")
axes[1].set_ylabel("update/weight ratio (log scale)")
axes[1].set_title("Per-layer ratio, with-warmup run (representative layers)")
axes[1].legend(fontsize=7)
axes[1].grid(alpha=0.3)

fig.tight_layout()
fig.savefig(ASSETS / "item3_warmup_ratio.png", dpi=130)
plt.close(fig)
print(f"saved {ASSETS / 'item3_warmup_ratio.png'}")

# %%
### 3f. Assemble item 3 results
RESULTS["item3"] = {
    "config": {"n_embd": CFG3.n_embd, "n_layer": CFG3.n_layer, "n_head": CFG3.n_head,
               "seq_len": CFG3.seq_len, "batch_size": CFG3.batch_size, "n_params": n_params3},
    "peak_lr": PEAK_LR,
    "total_steps": TOTAL_STEPS,
    "warmup_steps": WARMUP_STEPS,
    "peak_ratio_no_warmup": peak_nowarmup,
    "peak_ratio_with_warmup": peak_warmup,
    "stabilize_step": stabilize_step,
    "stabilize_steady_state": steady_state,
    "tracked_layers_plotted": TRACKED_LAYERS,
    "max_ratio_log_warmup": run_warmup["max_ratio_log"],
    "max_ratio_log_nowarmup": run_nowarmup["max_ratio_log"],
    "ratio_log_all_layers_warmup": run_warmup["ratio_log"],
    "loss_log_warmup": run_warmup["loss_log"],
    "loss_log_nowarmup": run_nowarmup["loss_log"],
    "plot": "assets/item3_warmup_ratio.png",
}

# %% [markdown]
"""
## Item 4 — Cosine vs WSD, 300-step budget, stopped at step 200

Same nanoGPT config and warmup length as item 3 (`n_embd=128, n_layer=4, n_head=4`,
warmup=60 steps). Both schedules are nominally planned for a 300-step run, and both
runs are stopped early, at step 200 — exactly the scenario Section 10 names as cosine's
structural weakness and WSD's structural advantage:

* **Cosine** commits to its decay curve across the full 300 steps from the first step.
  Interrupting it at step 200 catches the learning rate mid-decay, at whatever value the
  pre-committed curve happens to be passing through — not a value chosen because it's
  good for stopping *here*.
* **WSD** never commits to a total length. It holds flat until *we* decide to stop, and
  only then decays — *"we can save the weights at any point... and decay separately from
  there."* Stopping a WSD run at step 200 means treating the final ~10% of those 200
  steps (180-200) as the decay window, so the checkpoint at 200 is a genuinely finished,
  fully-annealed model at exactly the length we chose — the thing cosine cannot produce
  without having planned for length 200 from the very start.

Both runs share the same weight init and — matching item 2's principle of isolating one
variable — the exact same sequence of training batches, so the schedule is the only
thing that differs.
"""

# %%
### 4a. Schedules
import math

WARMUP4 = 60
STOP_STEP4 = 200
PLANNED_TOTAL4 = 300  # cosine's pre-committed horizon
DECAY_FRAC4 = 0.10    # WSD's own convention: decay over the final ~10%
MIN_LR_FRAC4 = 0.05   # floor as a fraction of peak, avoids a literal-zero LR


def cosine_lr(step, peak_lr, warmup, planned_total, min_lr_frac=MIN_LR_FRAC4):
    if step < warmup:
        return peak_lr * (step + 1) / warmup
    min_lr = peak_lr * min_lr_frac
    frac = min(1.0, (step - warmup) / max(1, planned_total - warmup))
    return min_lr + 0.5 * (peak_lr - min_lr) * (1 + math.cos(math.pi * frac))


def wsd_lr(step, peak_lr, warmup, stop_step, decay_frac=DECAY_FRAC4, min_lr_frac=MIN_LR_FRAC4):
    if step < warmup:
        return peak_lr * (step + 1) / warmup
    decay_start = stop_step - int(decay_frac * stop_step)
    if step < decay_start:
        return peak_lr
    min_lr = peak_lr * min_lr_frac
    frac = min(1.0, (step - decay_start) / max(1, stop_step - decay_start))
    return min_lr + 0.5 * (peak_lr - min_lr) * (1 + math.cos(math.pi * frac))


cosine_curve4 = [cosine_lr(t, PEAK_LR, WARMUP4, PLANNED_TOTAL4) for t in range(STOP_STEP4)]
wsd_curve4 = [wsd_lr(t, PEAK_LR, WARMUP4, STOP_STEP4) for t in range(STOP_STEP4)]
print(f"LR at step {STOP_STEP4 - 1}: cosine={cosine_curve4[-1]:.2e}  "
      f"wsd={wsd_curve4[-1]:.2e}  (peak={PEAK_LR:.2e})")
print(f"cosine's own decay would reach its floor at step {PLANNED_TOTAL4 - 1}, "
      f"{PLANNED_TOTAL4 - STOP_STEP4} steps past where we stop it")

# %%
### 4b. Train both, identical init + identical batch sequence
def train_with_schedule(cfg, stream, seed, lr_fn, steps):
    model = build_model(cfg, seed)
    decay_params = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay_params = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay_params, "weight_decay": 0.1},
         {"params": no_decay_params, "weight_decay": 0.0}],
        lr=PEAK_LR,
    )
    gen = torch.Generator().manual_seed(seed + 1)
    loss_log, lr_log = [], []
    for step in range(steps):
        lr = lr_fn(step)
        for g in opt.param_groups:
            g["lr"] = lr
        batch = get_batch3(cfg, stream, generator=gen)
        opt.zero_grad(set_to_none=True)
        logits = model(batch)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, cfg.vocab_size), batch[:, 1:].reshape(-1))
        loss.backward()
        opt.step()
        loss_log.append(loss.item())
        lr_log.append(lr)
    return model, loss_log, lr_log


SEED4 = 7
_, cos_loss4, cos_lr4 = train_with_schedule(
    CFG3, STREAM3, SEED4, lambda t: cosine_lr(t, PEAK_LR, WARMUP4, PLANNED_TOTAL4), STOP_STEP4
)
_, wsd_loss4, wsd_lr4 = train_with_schedule(
    CFG3, STREAM3, SEED4, lambda t: wsd_lr(t, PEAK_LR, WARMUP4, STOP_STEP4), STOP_STEP4
)

final_cos4, final_wsd4 = cos_loss4[-1], wsd_loss4[-1]
avg_cos4 = sum(cos_loss4[-10:]) / 10
avg_wsd4 = sum(wsd_loss4[-10:]) / 10
print(f"loss @ step {STOP_STEP4}:              cosine={final_cos4:.4f}  wsd={final_wsd4:.4f}")
print(f"loss, mean of last 10 steps:  cosine={avg_cos4:.4f}  wsd={avg_wsd4:.4f}")

verdict4 = "WSD" if avg_wsd4 < avg_cos4 else "cosine"
print(f"\nverdict: keep the {verdict4} checkpoint "
      f"(lower mean loss over the last 10 steps: wsd={avg_wsd4:.4f} vs cosine={avg_cos4:.4f})")

# %%
### 4c. Plots
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

axes[0].plot(cos_lr4, label="cosine (planned for 300, stopped at 200)")
axes[0].plot(wsd_lr4, label="WSD (decays into its own stop at 200)")
axes[0].axvline(WARMUP4 - 1, color="gray", linestyle="--", linewidth=1)
axes[0].set_xlabel("step")
axes[0].set_ylabel("learning rate")
axes[0].set_title("Schedules")
axes[0].legend(fontsize=8)
axes[0].grid(alpha=0.3)

axes[1].plot(cos_loss4, label=f"cosine (final {final_cos4:.3f})", alpha=0.8)
axes[1].plot(wsd_loss4, label=f"WSD (final {final_wsd4:.3f})", alpha=0.8)
axes[1].set_xlabel("step")
axes[1].set_ylabel("loss")
axes[1].set_title("Loss, same init + same batches")
axes[1].legend(fontsize=8)
axes[1].grid(alpha=0.3)

fig.tight_layout()
fig.savefig(ASSETS / "item4_cosine_vs_wsd.png", dpi=130)
plt.close(fig)
print(f"saved {ASSETS / 'item4_cosine_vs_wsd.png'}")

# %%
### 4d. Assemble item 4 results
RESULTS["item4"] = {
    "warmup": WARMUP4,
    "stop_step": STOP_STEP4,
    "planned_total_cosine": PLANNED_TOTAL4,
    "decay_frac_wsd": DECAY_FRAC4,
    "min_lr_frac": MIN_LR_FRAC4,
    "peak_lr": PEAK_LR,
    "seed": SEED4,
    "cosine_lr_curve": cosine_curve4,
    "wsd_lr_curve": wsd_curve4,
    "cosine_loss_curve": cos_loss4,
    "wsd_loss_curve": wsd_loss4,
    "final_loss_cosine": final_cos4,
    "final_loss_wsd": final_wsd4,
    "mean_last10_loss_cosine": avg_cos4,
    "mean_last10_loss_wsd": avg_wsd4,
    "verdict": verdict4,
    "plot": "assets/item4_cosine_vs_wsd.png",
}

# %% [markdown]
"""
## Item 5 — Width sweep at 256/512/1,024, LR transfer toward 4,096

Same nanoGPT config as items 3-4, varying only `n_embd` in {256, 512, 1024} (D2:
`n_head=4` divides all three evenly — head_dim 64/128/256, `n_layer=4`, `seq_len=128`,
`batch_size=8` held fixed). Section 12's own width -> η table (standard parameterization,
*not* muP) says the best learning rate roughly halves each time width doubles: `256:
3.0e-3, 512: 1.5e-3, 1024: 7.5e-4, 2048: 3.8e-4, 4096: 1.9e-4`. This item sweeps our own
LR grid at each width, marks each width's own loss-minimizing LR, checks whether the
three minima follow the same halving pattern, and extrapolates to width 4,096.

Compute note (D1, CPU-only): width=1,024 alone measured ~4.8s/optimizer-step on this
machine — the most expensive run this session by far. Budget: 40 steps per (width, LR)
run (enough to separate a diverging LR from a slow one from a good one — not enough for
full convergence, since the question is *where* the minimum sits, not a fully trained
checkpoint at each point). Same init and same batch sequence within one width's sweep,
so LR is the only thing that varies (same isolate-one-variable principle as items 2 and
4) — but init necessarily differs *across* widths, since a different `n_embd` is a
different parameter count.

**Grid re-centering (a finding worth stating up front):** a first pass used a 6-point
grid spanning `2e-4` to `2e-2` — centered on the lesson's own table, on the assumption
our setup's optimal LRs would land near 3.0e-3/1.5e-3/7.5e-4. Every width's loss-vs-LR
curve came back monotonically *decreasing* toward that grid's lower edge — not an
interior minimum, a boundary artifact. A quick preliminary probe (reduced step count,
not the graded run) at lower LRs found genuine interior minima for all three widths, but
roughly an order of magnitude below the lesson's table: ~2e-4 (256), ~6e-5 (512), ~2e-5
(1,024). This makes sense in hindsight — this toy setup (tiny char-level model,
`batch_size=8`, `seq_len=128`, plain standard-parameterization init) has no reason to
share the lesson's own absolute LR scale, only the *qualitative* muP claim (optimal LR
shrinks as width grows) is testable here. The grid below is re-centered on that finding:
`1e-5` to `5e-4`, still standard parameterization, still exactly the run graded below.
"""

# %%
### 5a. Sweep setup
WIDTHS5 = [256, 512, 1024]
LR_GRID5 = np.geomspace(1e-5, 5e-4, 9).tolist()
STEPS5 = 40
SEED5 = 42
LESSON_ETA_TABLE = {256: 3.0e-3, 512: 1.5e-3, 1024: 7.5e-4, 2048: 3.8e-4, 4096: 1.9e-4}

print(f"widths: {WIDTHS5}")
print("LR grid (" + str(len(LR_GRID5)) + " points): " + ", ".join(f"{lr:.2e}" for lr in LR_GRID5))
print(f"steps per (width, LR) run: {STEPS5}")

# %%
### 5b. Run the sweep — same init + same batch sequence per width, LR is the only variable
def train_at_lr(cfg, stream, seed, lr, steps):
    model = build_model(cfg, seed)
    decay_params = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay_params = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay_params, "weight_decay": 0.1},
         {"params": no_decay_params, "weight_decay": 0.0}],
        lr=lr,
    )
    gen = torch.Generator().manual_seed(seed + 1)
    loss_log = []
    for _ in range(steps):
        batch = get_batch3(cfg, stream, generator=gen)
        opt.zero_grad(set_to_none=True)
        logits = model(batch)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, cfg.vocab_size), batch[:, 1:].reshape(-1))
        if not torch.isfinite(loss):
            loss_log.append(float("nan"))
            break
        loss.backward()
        opt.step()
        loss_log.append(loss.item())
    while len(loss_log) < steps:
        loss_log.append(float("nan"))  # diverged early — pad so every curve is the same length
    return loss_log


sweep5 = {}
for width in WIDTHS5:
    cfg5 = GPTConfig(vocab_size=len(chars3), n_embd=width)
    n_params5 = sum(p.numel() for p in build_model(cfg5, SEED5).parameters())
    per_lr = {}
    for lr in LR_GRID5:
        loss_curve = train_at_lr(cfg5, STREAM3, SEED5, lr, STEPS5)
        tail = [l for l in loss_curve[-10:] if math.isfinite(l)]
        mean_last10 = sum(tail) / len(tail) if tail else float("inf")
        per_lr[lr] = {"loss_curve": loss_curve, "mean_last10": mean_last10}
        print(f"  n_embd={width:>4}  lr={lr:.2e}  mean_last10={mean_last10:.4f}"
              + ("  (diverged)" if not tail else ""))
    sweep5[width] = {"n_params": n_params5, "per_lr": per_lr}

# %%
### 5c. Best LR per width (grid-argmin), compared against the lesson's own table
best5 = {}
for width in WIDTHS5:
    per_lr = sweep5[width]["per_lr"]
    best_lr = min(per_lr, key=lambda lr: per_lr[lr]["mean_last10"])
    best5[width] = {"best_lr": best_lr, "best_loss": per_lr[best_lr]["mean_last10"]}
    print(f"n_embd={width:>4}  measured best lr={best_lr:.2e} (loss={best5[width]['best_loss']:.4f})"
          f"   lesson's table: {LESSON_ETA_TABLE[width]:.2e}")

# %%
### 5d. Fit log(lr) vs log(width) on our own 3 points, extrapolate to width=4,096
log2_widths5 = [math.log2(w) for w in WIDTHS5]
log2_lrs5 = [math.log2(best5[w]["best_lr"]) for w in WIDTHS5]
slope5, intercept5 = np.polyfit(log2_widths5, log2_lrs5, 1)
predicted_4096_fit = float(2 ** (slope5 * math.log2(4096) + intercept5))

# same halving-per-doubling heuristic read directly off the lesson's own table, applied
# to our own measured width=1,024 point (two doublings away from 4,096)
predicted_4096_halving = best5[1024]["best_lr"] / 4

print(f"our fitted slope (log2 lr vs log2 width): {slope5:.3f}  "
      f"(lesson's table implies -1.0, i.e. exact halving)")
print(f"fitted extrapolation to width=4,096:      {predicted_4096_fit:.2e}")
print(f"halving heuristic from our width=1,024:   {predicted_4096_halving:.2e}")
print(f"lesson's own table at width=4,096:        {LESSON_ETA_TABLE[4096]:.2e}")

CONFIDENCE_NOTE5 = (
    "Extrapolation, not a fourth measurement. Width=4,096 sits two doublings past our "
    "widest measured point (1,024), and the sweep itself used a 9-point LR grid with "
    "only 40 steps/run on a tiny char-level model and dataset unrelated to the lesson's "
    "own model, so both the grid-argmin resolution and short-run noise carry real "
    "uncertainty into the fitted slope. Moderate confidence that the sign and rough "
    "magnitude of the trend hold (a several-times-smaller LR at 4,096 than at 1,024); "
    "low confidence in matching the lesson's precise 1.9e-4 value, since the exact scale "
    "depends on architecture/data details that muP's own reparameterization is designed "
    "to make irrelevant — only under muP itself would a number transfer exactly."
)
print("\n" + CONFIDENCE_NOTE5)

# %%
### 5e. Plots
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

colors5 = {256: "tab:blue", 512: "tab:orange", 1024: "tab:green"}
for width in WIDTHS5:
    per_lr = sweep5[width]["per_lr"]
    lrs_sorted = sorted(per_lr)
    losses = [per_lr[lr]["mean_last10"] for lr in lrs_sorted]
    axes[0].plot(lrs_sorted, losses, "o-", color=colors5[width], label=f"n_embd={width}")
    axes[0].scatter([best5[width]["best_lr"]], [best5[width]["best_loss"]],
                     color=colors5[width], marker="*", s=200, zorder=5, edgecolor="black")
axes[0].set_xscale("log")
axes[0].set_xlabel("learning rate")
axes[0].set_ylabel("loss (mean, last 10 steps)")
axes[0].set_title("Loss vs LR, per width (star = grid minimum)")
axes[0].legend(fontsize=8)
axes[0].grid(alpha=0.3, which="both")

widths_all5 = WIDTHS5 + [2048, 4096]
lesson_lrs_all5 = [LESSON_ETA_TABLE[w] for w in widths_all5]
axes[1].plot(widths_all5, lesson_lrs_all5, "s--", color="gray", label="lesson's table (§12)")
axes[1].plot(WIDTHS5, [best5[w]["best_lr"] for w in WIDTHS5], "o-", color="tab:red", label="measured (this run)")
axes[1].scatter([4096], [predicted_4096_fit], color="tab:red", marker="*", s=200, zorder=5,
                 edgecolor="black", label=f"our extrapolation ({predicted_4096_fit:.2e})")
axes[1].set_xscale("log", base=2)
axes[1].set_yscale("log")
axes[1].set_xlabel("width (n_embd)")
axes[1].set_ylabel("best learning rate")
axes[1].set_title("LR transfer across width")
axes[1].legend(fontsize=8)
axes[1].grid(alpha=0.3, which="both")

fig.tight_layout()
fig.savefig(ASSETS / "item5_width_sweep.png", dpi=130)
plt.close(fig)
print(f"saved {ASSETS / 'item5_width_sweep.png'}")

# %%
### 5f. Assemble item 5 results
RESULTS["item5"] = {
    "widths": WIDTHS5,
    "lr_grid": LR_GRID5,
    "steps_per_run": STEPS5,
    "seed": SEED5,
    "config_fixed": {"n_layer": 4, "n_head": 4, "seq_len": 128, "batch_size": 8},
    "n_params_by_width": {w: sweep5[w]["n_params"] for w in WIDTHS5},
    "mean_last10_by_width_lr": {
        w: {lr: sweep5[w]["per_lr"][lr]["mean_last10"] for lr in LR_GRID5} for w in WIDTHS5
    },
    "best_lr_by_width": {w: best5[w]["best_lr"] for w in WIDTHS5},
    "best_loss_by_width": {w: best5[w]["best_loss"] for w in WIDTHS5},
    "lesson_eta_table": LESSON_ETA_TABLE,
    "fitted_slope_log2": float(slope5),
    "predicted_lr_4096_fit": predicted_4096_fit,
    "predicted_lr_4096_halving_heuristic": predicted_4096_halving,
    "lesson_lr_4096": LESSON_ETA_TABLE[4096],
    "confidence_note": CONFIDENCE_NOTE5,
    "plot": "assets/item5_width_sweep.png",
}

# %%
## Save results (all items so far)
pathlib.Path("results.json").write_text(json.dumps(RESULTS, indent=2))
print("wrote results.json")
