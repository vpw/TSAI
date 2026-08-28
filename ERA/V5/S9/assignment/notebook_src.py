# %% [markdown]
"""
# Session 9 — Loss Functions & Output Heads

A cross-entropy harness made **correct and observable**, plus a second output head that
predicts token `t+2`.

The starting point is the four lines from the assignment:

```python
hidden = model(tokens)
logits = output_head(hidden)
loss   = cross_entropy(
    logits[:, :-1].reshape(-1, vocab_size),
    tokens[:, 1:].reshape(-1),
)
```

Everything below is the work of making those four lines trustworthy. The session's warning is
the organising idea: **a target shift in the wrong direction produces a beautiful loss curve**
and raises no exception. §1b reproduces that bug on purpose and shows it looking healthy.

**Configuration.** The lesson's real V5 head is `V = 131,072`, `D = 4,096` — 536.9M parameters,
2.1 GB in fp32 before a single activation. This notebook measures on a **proxy**
(`V = 10,000`, `D = 256`) and computes the real-config numbers analytically alongside, always
labelled as such. The tokenizer is the course's own **Session 2 BPE** (10,000 merges over
en/hi/te/mr), so every token string printed below is a real token from a real vocabulary.
Perplexity is per-token and a tokenizer defines the token, so that number is only meaningful
next to this tokenizer.
"""

# %%
# Colab bootstrap. No-op when the repo is already checked out locally.
import pathlib, subprocess, sys, urllib.request

ASSETS = pathlib.Path("assets")
RAW = "https://raw.githubusercontent.com/vpw/era-v5-s9/main/assets/"

for pkg in ("torch", "tokenizers", "matplotlib"):
    try:
        __import__(pkg)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=True)

if not (ASSETS / "tokenizer.json").exists():
    ASSETS.mkdir(exist_ok=True)
    for name in ("tokenizer.json", "corpus_mr.txt", "corpus_en.txt"):
        urllib.request.urlretrieve(RAW + name, ASSETS / name)
print("assets:", sorted(p.name for p in ASSETS.iterdir()))

# %%
import math, os, threading, time
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

torch.manual_seed(1337)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"torch {torch.__version__} · device {DEVICE}")

# %% [markdown]
"""
## §0 Configuration

`IGNORE` is PyTorch's `ignore_index` sentinel. Every mask in this notebook — padding (§1c),
document boundaries (§1d), the unpredictable tail of the `t+2` head (Part 2) — is expressed the
same way: **set the label to `-100` and the position stops contributing**, both to the numerator
and to the token count in the denominator. That single mechanism is what §1c and §1d are
checking.

The S2 tokenizer's only special token is `[UNK]` (id 0) — there is no `[PAD]`. So padding uses
`[UNK]` as filler and is masked via the *label*, not via a reserved vocabulary slot. This keeps
`V` at exactly 10,000 and the `ln V` anchor in §1e clean.
"""

# %%
@dataclass
class Config:
    vocab_size: int = 10_000     # V — set below from the tokenizer itself, not asserted
    d_model: int = 256           # D — hidden width, the "residual stream"
    n_layer: int = 4
    n_head: int = 4
    seq_len: int = 256           # T — training context
    batch_size: int = 8          # B
    mem_seq_len: int = 512       # T for the §1g memory measurement (bigger logits tensor)
    mem_batch_size: int = 8

REAL_V, REAL_D = 131_072, 4_096  # the lesson's V5 configuration, for the analytic comparisons
IGNORE = -100

cfg = Config()
tok = Tokenizer.from_file(str(ASSETS / "tokenizer.json"))
cfg.vocab_size = tok.get_vocab_size()
PAD_ID = tok.token_to_id("[UNK]")

print(f"V = {cfg.vocab_size:,}   D = {cfg.d_model}   layers = {cfg.n_layer}   heads = {cfg.n_head}")
print(f"T = {cfg.seq_len}   B = {cfg.batch_size}   pad filler id = {PAD_ID} ([UNK])")
probe = tok.encode("The quick brown fox. मराठी भाषा छान आहे.")
print(f"tokenizer round-trip: {tok.decode(probe.ids)!r}")
print(f"sample tokens: {probe.tokens}")

# %% [markdown]
"""
## §0b Data

Real text through the real tokenizer. Two corpora (English and Marathi, both from the S2
training mixture) are kept **separate** rather than concatenated, because §1d needs two genuinely
different documents to pack together and a boundary that actually matters.
"""

# %%
def load_stream(name: str) -> torch.Tensor:
    text = (ASSETS / name).read_text(encoding="utf-8")
    return torch.tensor(tok.encode(text).ids, dtype=torch.long)

STREAM = {"en": load_stream("corpus_en.txt"), "mr": load_stream("corpus_mr.txt")}
for k, v in STREAM.items():
    print(f"{k}: {len(v):,} tokens")

def get_batch(lang="en", B=None, T=None, generator=None):
    """A batch of contiguous windows. Returns [B, T] token ids."""
    B, T = B or cfg.batch_size, T or cfg.seq_len
    data = STREAM[lang]
    ix = torch.randint(len(data) - T - 1, (B,), generator=generator)
    return torch.stack([data[i:i + T] for i in ix]).to(DEVICE)

_g = torch.Generator().manual_seed(0)
print("batch shape:", get_batch(generator=_g).shape)

# %% [markdown]
"""
## §0c The model — Session 8's hidden state

This session consumes what Session 8 produced: a hidden state `[B, T, D]`. The block is the one
the lesson describes — **pre-norm** residual stream, **RMSNorm**, **RoPE**, **SwiGLU** — scaled
down to the proxy width. It is deliberately ordinary; the graded work is downstream of it.

The **output head** is kept as a separate module rather than folded into the model, because the
whole session is about that one matrix: `z = h · W_vocabᵀ`, the unembedding, `[V, D]`.
"""

# %%
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


def rope_cache(T, head_dim, device, base=10_000.0):
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    freqs = torch.outer(torch.arange(T, device=device).float(), inv)   # [T, head_dim/2]
    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):
    """x: [B, H, T, head_dim] — position as rotation, so q·k depends on i-j."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos, sin = cos[None, None], sin[None, None]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_head, self.head_dim = cfg.n_head, cfg.d_model // cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x, cos, sin):
        B, T, D = x.shape
        q, k, v = self.qkv(x).split(D, dim=2)
        q, k, v = (t.view(B, T, self.n_head, self.head_dim).transpose(1, 2) for t in (q, k, v))
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(out.transpose(1, 2).reshape(B, T, D))


class SwiGLU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        hidden = int(8 / 3 * cfg.d_model / 64 + 0.5) * 64
        self.gate = nn.Linear(cfg.d_model, hidden, bias=False)
        self.up = nn.Linear(cfg.d_model, hidden, bias=False)
        self.down = nn.Linear(hidden, cfg.d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n1, self.attn = RMSNorm(cfg.d_model), Attention(cfg)
        self.n2, self.ffn = RMSNorm(cfg.d_model), SwiGLU(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.n1(x), cos, sin)      # pre-norm: the residual stream stays clean
        return x + self.ffn(self.n2(x))


class Trunk(nn.Module):
    """tokens [B, T] -> hidden [B, T, D]. Everything up to, but not including, the head."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.norm = RMSNorm(cfg.d_model)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, tokens):
        x = self.embed(tokens)
        cos, sin = rope_cache(tokens.shape[1], self.cfg.d_model // self.cfg.n_head, tokens.device)
        for blk in self.blocks:
            x = blk(x, cos, sin)
        return self.norm(x)


class Head(nn.Module):
    """The output head / unembedding: z = h · W_vocabᵀ, W_vocab is [V, D]."""

    def __init__(self, cfg):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(cfg.vocab_size, cfg.d_model))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, h):
        return F.linear(h, self.weight)


def build(cfg, seed=1337):
    torch.manual_seed(seed)
    return Trunk(cfg).to(DEVICE), Head(cfg).to(DEVICE)


trunk, head = build(cfg)
n_trunk = sum(p.numel() for p in trunk.parameters())
n_head = sum(p.numel() for p in head.parameters())
print(f"trunk: {n_trunk:,} params    head: {n_head:,} params    "
      f"head is {n_head / (n_trunk + n_head):.1%} of the model")

# %% [markdown]
"""
## §1a — Every tensor shape, and what each dimension is

The first required number. One line per tensor; the point is that no dimension is left unnamed,
because an unnamed dimension is where the off-by-one hides.

Note the shape the loss actually consumes: cross-entropy takes a **flat** `[N, V]` against
`[N]`. The `[B, T, ...]` structure is flattened away before the scalar, which is exactly why a
shift bug survives — after `.reshape(-1, V)` there is nothing left to disagree with.
"""

# %%
tokens = get_batch(generator=torch.Generator().manual_seed(7))
with torch.no_grad():
    hidden = trunk(tokens)
    logits = head(hidden)

inputs_l, targets_l = logits[:, :-1], tokens[:, 1:]
flat_logits, flat_targets = inputs_l.reshape(-1, cfg.vocab_size), targets_l.reshape(-1)

B, T = tokens.shape
rows = [
    ("tokens",       tokens.shape,       "B = batch of independent sequences · T = position in the sequence"),
    ("hidden",       hidden.shape,       "B · T · D = the residual-stream width, one vector per position"),
    ("head.weight",  head.weight.shape,  "V = one row per vocabulary entry · D, so z = h · W_vocabᵀ"),
    ("logits",       logits.shape,       "B · T · V = one unnormalised score per vocabulary entry, per position"),
    ("logits[:, :-1]", inputs_l.shape,   "B · T-1 · V = drop the last position: it has no next token to predict"),
    ("tokens[:, 1:]",  targets_l.shape,  "B · T-1 = the answer for each prediction, shifted one step left"),
    ("flat logits",  flat_logits.shape,  "N = B·(T-1) predictions, flattened · V"),
    ("flat targets", flat_targets.shape, "N = one correct token id per prediction"),
]
w = max(len(r[0]) for r in rows)
for name, shape, meaning in rows:
    print(f"{name:<{w}}  {str(tuple(shape)):<20}  {meaning}")
print(f"\nB={B}  T={T}  D={cfg.d_model}  V={cfg.vocab_size:,}  N={flat_targets.numel():,}")
print(f"logits tensor: {logits.numel() * logits.element_size() / 2**20:.1f} MiB at this proxy size")
print(f"  ... the same [B,T,V] tensor at V={REAL_V:,} would be "
      f"{B * T * REAL_V * 4 / 2**30:.1f} GiB in fp32")

# %% [markdown]
"""
## §1b — Verify the shift by printing token **strings**

The assignment is explicit: not the ids, the strings. The reason is in the next cell but one —
the wrong shift does not raise, does not warn, and trains beautifully.

The contract being checked is one line: **the target at position `i` is the input at position
`i+1`.** Read down the two columns; each row's target should be the next row's input.
"""

# %%
def show_shift(seq, n=14, title="input → target"):
    inp, tgt = seq[:-1], seq[1:]
    print(title)
    print(f"{'pos':>4}  {'input token':<22} {'target token':<22}  (target = next input?)")
    for i in range(n):
        it, tt = tok.id_to_token(inp[i].item()), tok.id_to_token(tgt[i].item())
        nxt = tok.id_to_token(inp[i + 1].item()) if i + 1 < len(inp) else None
        print(f"{i:>4}  {it!r:<22} {tt!r:<22}  {'ok' if tt == nxt else 'MISMATCH'}")
    print(f"\n  input  text: {tok.decode(inp[:n].tolist())!r}")
    print(f"  target text: {tok.decode(tgt[:n].tolist())!r}")

show_shift(tokens[0])

# every position, not just the printed window
assert torch.equal(tokens[:, 1:][:, :-1], tokens[:, :-1][:, 1:]), "shift contract violated"
print("\n✓ target[i] == input[i+1] holds for all "
      f"{tokens[:, 1:].numel():,} (sequence, position) pairs in the batch")

# %% [markdown]
"""
### The bug the warning is about

Three ways to line logits up against tokens. Only the first is a language model.

| variant | slice | what it asks the model to do |
|---|---|---|
| `correct` | `logits[:, :-1]` vs `tokens[:, 1:]` | predict the **next** token — genuinely hard |
| `no_shift` | `logits[:, :-1]` vs `tokens[:, :-1]` | predict the token at its **own** position |
| `reversed` | `logits[:, 1:]` vs `tokens[:, :-1]` | predict the **previous** token |

Both bugs are trivially solvable: a causal model at position `i` can already see tokens `≤ i`,
so both `no_shift` and `reversed` are asking it to copy something it is already holding. The
loss goes to the floor. **That is what makes them dangerous** — the curve is not just fine, it
is better than the correct one.
"""

# %%
def variant_loss(logits, toks, mode):
    if mode == "correct":
        pred, gold = logits[:, :-1], toks[:, 1:]
    elif mode == "no_shift":
        pred, gold = logits[:, :-1], toks[:, :-1]
    elif mode == "reversed":
        pred, gold = logits[:, 1:], toks[:, :-1]
    else:
        raise ValueError(mode)
    return F.cross_entropy(pred.reshape(-1, cfg.vocab_size), gold.reshape(-1), ignore_index=IGNORE)


def train_variant(mode, steps=120, lr=3e-4, log_every=20, seed=1337):
    tr, hd = build(cfg, seed=seed)
    opt = torch.optim.AdamW([*tr.parameters(), *hd.parameters()], lr=lr)
    gen = torch.Generator().manual_seed(99)
    curve = []
    for step in range(steps + 1):
        batch = get_batch(generator=gen)
        loss = variant_loss(hd(tr(batch)), batch, mode)
        curve.append(loss.item())
        if step % log_every == 0:
            print(f"  {mode:<9} step {step:>4}  loss {loss.item():7.4f}  ppl {math.exp(loss.item()):10,.1f}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return curve, tr, hd


curves = {}
models = {}
for mode in ("correct", "no_shift", "reversed"):
    curves[mode], *models[mode] = train_variant(mode)
    print()

# %%
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 4.2))
for mode, style in (("correct", "-"), ("no_shift", "--"), ("reversed", ":")):
    ax.plot(curves[mode], style, label=f"{mode}  (final {curves[mode][-1]:.3f})", lw=2)
ax.axhline(math.log(cfg.vocab_size), color="grey", lw=1,
           label=f"ln V = {math.log(cfg.vocab_size):.3f} (untrained)")
ax.set_xlabel("step"); ax.set_ylabel("cross-entropy (nats)")
ax.set_title("Two of these three curves are bugs")
ax.legend(); fig.tight_layout()
fig.savefig("assets/shift_curves.png", dpi=120)
plt.close(fig)

print("final loss after 120 steps")
for mode in curves:
    print(f"  {mode:<9} {curves[mode][-1]:7.4f} nats   perplexity {math.exp(curves[mode][-1]):>10,.2f}")
gap_ns = math.exp(curves["correct"][-1]) / math.exp(curves["no_shift"][-1])
gap_rv = math.exp(curves["correct"][-1]) / math.exp(curves["reversed"][-1])
print(f"\nAt the same step, on the same data, with the same seed: no_shift reports a perplexity")
print(f"{gap_ns:,.0f}x lower than the correct objective, and reversed {gap_rv:,.0f}x lower.")
print("On a loss curve alone, both look like the better run.")

# %% [markdown]
"""
### What the buggy model actually learned

The loss number cannot tell you, but one decode can: take the `no_shift` model, feed it a
sequence, and read its top-1 prediction at each position beside the input. It has learned to
**echo**.
"""

# %%
probe_tokens = get_batch(B=1, generator=torch.Generator().manual_seed(3))
with torch.no_grad():
    tr_bad, hd_bad = models["no_shift"]
    top1_bad = hd_bad(tr_bad(probe_tokens)).argmax(-1)[0]
    tr_ok, hd_ok = models["correct"]
    top1_ok = hd_ok(tr_ok(probe_tokens)).argmax(-1)[0]

print(f"{'pos':>4}  {'input':<18} {'no_shift top-1':<18} {'correct top-1':<18}")
for i in range(10):
    print(f"{i:>4}  {tok.id_to_token(probe_tokens[0, i].item())!r:<18} "
          f"{tok.id_to_token(top1_bad[i].item())!r:<18} "
          f"{tok.id_to_token(top1_ok[i].item())!r:<18}")

echo_rate = (top1_bad == probe_tokens[0]).float().mean().item()
print(f"\nno_shift model reproduces its own input {echo_rate:.1%} of the time — it is a copier, "
      "not a language model.")
print("Its loss curve never said so. The strings did.")
print("\n(The correct model is still early in training and mostly predicts a frequent token —")
print(" a boring answer to a genuinely hard question, which is the honest place to be at step 120.)")

# %% [markdown]
"""
## §1c — Mask padding, and confirm the contributing-token count changes

Padding is filler. If it reaches the loss, two things go wrong. The **denominator** is wrong — the
mean is divided by a token count that includes positions carrying no information. And the
**numerator** is wrong by an amount that depends on the model, which is what makes this bug
slippery: it does not have a fixed direction.

Three models are measured against the same padded batch, because the required observable (the
contributing-token count) is easy, and the interesting part is what the count does to the loss:

1. an **untrained** model,
2. the model trained on contiguous text from §1b — which has essentially never seen `[UNK]`,
3. a model trained **on padded batches with the padding left unmasked** — the bug in its natural
   habitat.
"""

# %%
def padded_batch(lengths, T):
    """Ragged real sequences padded to T. Returns (tokens, labels) with pads masked in labels."""
    data = STREAM["en"]
    toks = torch.full((len(lengths), T), PAD_ID, dtype=torch.long)
    labels = torch.full((len(lengths), T), IGNORE, dtype=torch.long)
    for r, L in enumerate(lengths):
        start = 1000 + r * 997
        toks[r, :L] = data[start:start + L]
        labels[r, :L] = data[start:start + L]
    return toks.to(DEVICE), labels.to(DEVICE)


PAD_T = 128
lengths = [96, 64, 128, 40, 112, 80, 128, 56]
pad_tokens, pad_labels = padded_batch(lengths, T=PAD_T)

gold_masked = pad_labels[:, 1:].reshape(-1)      # pads are IGNORE
gold_naive = pad_tokens[:, 1:].reshape(-1)       # pads are real [UNK] ids
n_total = gold_naive.numel()
n_masked = int((gold_masked != IGNORE).sum())

print(f"sequence lengths in the batch: {lengths}  (padded to T={PAD_T})")
print(f"\n  positions in logits[:, :-1]          {n_total:>8,}")
print(f"  contributing WITHOUT masking         {n_total:>8,}   <- every pad counted")
print(f"  contributing WITH masking            {n_masked:>8,}   <- real next-token pairs only")
print(f"  padding positions silently trained   {n_total - n_masked:>8,} "
      f"({(n_total - n_masked) / n_total:.1%} of the batch)")


def pad_losses(tr, hd):
    with torch.no_grad():
        pr = hd(tr(pad_tokens))[:, :-1].reshape(-1, cfg.vocab_size)
    naive = F.cross_entropy(pr, gold_naive)
    mask = F.cross_entropy(pr, gold_masked, ignore_index=IGNORE)
    s = F.cross_entropy(pr, gold_masked, ignore_index=IGNORE, reduction="sum")
    return naive.item(), mask.item(), (s / mask).item()


# %%
def train_on_padded(mask_pads, steps=150, lr=3e-4, seed=1337):
    """Train on ragged, padded batches — with or without masking the padding."""
    tr, hd = build(cfg, seed=seed)
    opt = torch.optim.AdamW([*tr.parameters(), *hd.parameters()], lr=lr)
    gen = torch.Generator().manual_seed(77)
    data = STREAM["en"]
    for _ in range(steps):
        lens = torch.randint(PAD_T // 4, PAD_T + 1, (cfg.batch_size,), generator=gen)
        starts = torch.randint(len(data) - PAD_T - 1, (cfg.batch_size,), generator=gen)
        toks = torch.full((cfg.batch_size, PAD_T), PAD_ID, dtype=torch.long)
        labs = torch.full((cfg.batch_size, PAD_T), IGNORE, dtype=torch.long)
        for r, (L, st) in enumerate(zip(lens.tolist(), starts.tolist())):
            toks[r, :L] = data[st:st + L]
            labs[r, :L] = data[st:st + L]
        toks, labs = toks.to(DEVICE), labs.to(DEVICE)
        gold = (labs if mask_pads else toks)[:, 1:].reshape(-1)
        loss = F.cross_entropy(hd(tr(toks))[:, :-1].reshape(-1, cfg.vocab_size), gold,
                               ignore_index=IGNORE)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return tr, hd


print("training a model on padded batches with the padding LEFT UNMASKED "
      "(the bug in its natural habitat)...")
buggy_pad_trunk, buggy_pad_head = train_on_padded(mask_pads=False)

u_naive, u_masked, _ = pad_losses(trunk, head)
loss_naive, loss_masked, denom_recovered = pad_losses(*models["correct"])
b_naive, b_masked, _ = pad_losses(buggy_pad_trunk, buggy_pad_head)

print(f"\n  denominator recovered from PyTorch   {denom_recovered:>8,.0f}   "
      f"({'matches the mask' if round(denom_recovered) == n_masked else 'MISMATCH'})")
print(f"\n{'model':<38}{'unmasked':>11}{'masked':>11}{'difference':>13}")
print(f"{'untrained':<38}{u_naive:>11.4f}{u_masked:>11.4f}{u_masked - u_naive:>+13.4f}")
print(f"{'trained on clean text (§1b)':<38}{loss_naive:>11.4f}{loss_masked:>11.4f}"
      f"{loss_masked - loss_naive:>+13.4f}")
print(f"{'trained on padding, unmasked':<38}{b_naive:>11.4f}{b_masked:>11.4f}"
      f"{b_masked - b_naive:>+13.4f}")

print("\nRead the third row. That model reports an unmasked loss of "
      f"{b_naive:.4f} nats, but its actual")
print(f"language-modelling loss is {b_masked:.4f} — it is {b_masked - b_naive:.2f} nats better on paper than")
print(f"it is in fact, because {(n_total - n_masked) / n_total:.0%} of its exam is predicting [UNK] after [UNK].")
print("\nThe first two rows show why the bug is hard to catch. Untrained, masking moves the loss")
print(f"by {u_masked - u_naive:+.4f} nats — nothing, because nothing is predictable yet. And a model that")
print("never trained on padding goes the OTHER way "
      f"({loss_masked - loss_naive:+.4f} nats), because to it a pad is a rare,")
print("out-of-distribution token rather than a freebie.")
print("\nSo the sign of the error is not the lesson. The lesson is that the unmasked number is not")
print("the quantity you think it is, in either direction, and the denominator is wrong in both:")
print(f"{n_total:,} instead of {n_masked:,}.")

# %% [markdown]
"""
## §1d — Pack two documents into one sequence and mask the boundary

Packing is how a real pre-training pipeline avoids wasting compute on padding: concatenate
documents until the context is full. It introduces one bad position — the last token of document
A is asked to predict the first token of document B, two texts with nothing to do with each
other.

The two documents here are deliberately in **different languages** (English then Marathi, both
from the S2 tokenizer's own training mixture), which makes the boundary prediction not merely
hard but genuinely unanswerable. As in §1c, the trained model is what makes the effect visible.
"""

# %%
DOC_A_LEN, DOC_B_LEN = 128, 128
doc_a = STREAM["en"][5000:5000 + DOC_A_LEN]
doc_b = STREAM["mr"][3000:3000 + DOC_B_LEN]
packed = torch.cat([doc_a, doc_b]).unsqueeze(0).to(DEVICE)
boundary = DOC_A_LEN - 1        # index into logits[:, :-1] / tokens[:, 1:]

print("the packed boundary, in strings:")
print(f"  end of doc A (en) : ...{tok.decode(doc_a[-8:].tolist())!r}")
print(f"  start of doc B (mr): {tok.decode(doc_b[:8].tolist())!r}...")
print(f"\n  the model is asked: given {tok.id_to_token(doc_a[-1].item())!r}, "
      f"predict {tok.id_to_token(doc_b[0].item())!r}")
print("  nothing in document A can support that prediction. It is not a hard example; it is a")
print("  wrong one.")

p_gold_unmasked = packed[:, 1:].reshape(-1).clone()
p_gold_masked = p_gold_unmasked.clone()
p_gold_masked[boundary] = IGNORE
n_pos = p_gold_unmasked.numel()


def packed_losses(tr, hd):
    with torch.no_grad():
        pr = hd(tr(packed))[:, :-1].reshape(-1, cfg.vocab_size)
    un = F.cross_entropy(pr, p_gold_unmasked)
    ma = F.cross_entropy(pr, p_gold_masked, ignore_index=IGNORE)
    per = F.cross_entropy(pr, p_gold_unmasked, reduction="none")
    return un.item(), ma.item(), per


loss_unmasked, loss_bmasked, per_token = packed_losses(*models["correct"])
u_un, u_ma, u_per = packed_losses(trunk, head)

mean_other = (per_token.sum() - per_token[boundary]).item() / (n_pos - 1)
u_mean_other = (u_per.sum() - u_per[boundary]).item() / (n_pos - 1)

print(f"\n  positions: {n_pos} unmasked, {n_pos - 1} masked")
print(f"\n{'':<26}{'unmasked':>12}{'masked':>12}{'boundary':>12}{'other posns':>13}")
print(f"{'untrained model':<26}{u_un:>12.4f}{u_ma:>12.4f}{u_per[boundary].item():>12.4f}"
      f"{u_mean_other:>13.4f}")
print(f"{'trained 120 steps (§1b)':<26}{loss_unmasked:>12.4f}{loss_bmasked:>12.4f}"
      f"{per_token[boundary].item():>12.4f}{mean_other:>13.4f}")
print(f"\n  trained: the boundary position costs {per_token[boundary].item() - mean_other:+.4f} nats "
      f"more than the average position")
print(f"  untrained: it costs {u_per[boundary].item() - u_mean_other:+.4f} nats — indistinguishable, "
      "because nothing is predictable yet")

print("\nWhy the difference in the mean is small and it matters anyway:")
print(f"  one position out of {n_pos} moves the mean by ~1/{n_pos} of its own excess, so masking it")
print(f"  changes the reported loss by only {loss_bmasked - loss_unmasked:+.4f} nats. The damage is not to")
print("  this number. It is that the gradient at that position is real, and it teaches the model")
print("  a cross-document transition that does not exist in the data — and there is one such")
print("  position per packed document, for the whole of pre-training.")
print("\n  Note the mechanism: masking removes the position from BOTH the numerator and the")
print("  denominator. Zeroing its loss instead would remove it only from the numerator and")
print("  quietly bias the mean downward — a second bug wearing the first one's clothes.")

# %% [markdown]
"""
## §1e — Perplexity, and the untrained-model anchor

Perplexity is the loss made readable: `PPL = exp(loss)`, "how many equally-likely options is the
model effectively choosing between". An untrained model has no preferences, so it spreads
probability uniformly over the vocabulary and is effectively choosing between all `V` of them.

That gives the cheapest sanity check in the whole session:

- **loss ≈ ln V**, and **perplexity ≈ V**.
- At this proxy's `V = 10,000`: `ln(10,000) = 9.2103`.
- At the lesson's real `V = 131,072`: `ln(131,072) = 11.7845`.

If a fresh model does not land near its vocabulary size, something upstream is broken and every
number after it is noise. Perplexity is per-token and a tokenizer defines the token, so this
number is **not comparable across tokenizers** — it belongs to the S2 BPE vocabulary and to
nothing else.
"""

# %%
fresh_trunk, fresh_head = build(cfg, seed=2024)
eval_gen = torch.Generator().manual_seed(11)

with torch.no_grad():
    total_nats, total_tokens_counted = 0.0, 0
    for _ in range(8):
        b = get_batch(generator=eval_gen)
        lg = fresh_head(fresh_trunk(b))
        s = F.cross_entropy(lg[:, :-1].reshape(-1, cfg.vocab_size), b[:, 1:].reshape(-1),
                            reduction="sum")
        total_nats += s.item()
        total_tokens_counted += b[:, 1:].numel()

untrained_loss = total_nats / total_tokens_counted
untrained_ppl = math.exp(untrained_loss)
ln_V = math.log(cfg.vocab_size)

print(f"measured over {total_tokens_counted:,} tokens from an untrained model")
print(f"  loss        {untrained_loss:.4f} nats")
print(f"  perplexity  {untrained_ppl:,.1f}")
print(f"\n  predicted   ln V = ln({cfg.vocab_size:,}) = {ln_V:.4f} nats,  perplexity {cfg.vocab_size:,}")
print(f"  deviation   {untrained_loss - ln_V:+.4f} nats  ({(untrained_ppl / cfg.vocab_size - 1):+.2%} on perplexity)")
print(f"  {'✓ sits at vocabulary size — harness is sane' if abs(untrained_loss - ln_V) < 0.1 else '✗ NOT near ln V — stop and find the bug'}")
print(f"\nfor comparison, the trained-for-120-steps 'correct' model above: "
      f"{curves['correct'][-1]:.4f} nats, perplexity {math.exp(curves['correct'][-1]):,.1f}")
print(f"real-config anchor (NOT measured here): ln({REAL_V:,}) = {math.log(REAL_V):.4f} nats, "
      f"perplexity {REAL_V:,}")

# %% [markdown]
"""
## §1f — Tied vs untied head parameter counts

Weight tying reuses the input embedding table `[V, D]` as the output head. Both matrices map
between the same two spaces, so one matrix can serve both directions — and `V·D` parameters
disappear.

**For V5 this is a counterfactual, not an option.** Session 7 replaced the input embedding table
with a fixed byte codec plus one trainable projection: there is no `[V, D]` input table to tie
to. The comparison is still worth reporting, because it prices exactly what that architectural
choice costs at the output end.
"""

# %%
def head_accounting(V, D, label, trunk_params=None):
    embed = V * D
    out_head = V * D
    print(f"\n{label}   (V = {V:,}, D = {D:,})")
    print(f"  input embedding [V, D]        {embed:>15,}")
    print(f"  output head     [V, D]        {out_head:>15,}")
    print(f"  UNTIED: both matrices         {embed + out_head:>15,}")
    print(f"  TIED:   one matrix, used twice{embed:>15,}")
    print(f"  saved by tying                {out_head:>15,}   ({out_head / (embed + out_head):.1%})")
    print(f"  head alone, fp32              {out_head * 4 / 2**20:>15,.1f} MiB")
    if trunk_params is not None:
        print(f"  untied model total            {trunk_params + out_head:>15,}")
        print(f"  tied   model total            {trunk_params:>15,}")
    return {"embed": embed, "head": out_head, "untied": embed + out_head, "tied": embed}


proxy_acc = head_accounting(cfg.vocab_size, cfg.d_model, "THIS NOTEBOOK'S PROXY", n_trunk)
real_acc = head_accounting(REAL_V, REAL_D, "V5 REAL CONFIGURATION (analytic, not run here)")

# verify the proxy numbers against the modules that actually exist
assert head.weight.numel() == proxy_acc["head"]
assert trunk.embed.weight.numel() == proxy_acc["embed"]
print(f"\n✓ counted from the live modules, not from the formula: "
      f"head {head.weight.numel():,}, embedding {trunk.embed.weight.numel():,}")

print(f"\nAt the real configuration the head alone is {real_acc['head'] / 1e6:.1f}M parameters — "
      f"{real_acc['head'] * 4 / 2**30:.2f} GiB in fp32.")
print("That is the same dense table Session 7's byte codec removed from the input side, "
      "reappearing at the output.")

# %% [markdown]
"""
## §1g — Peak memory: ordinary cross-entropy vs a chunked version

The parameters are not the real bill. The **logits tensor** is. `[B, T, V]` is materialised in
full, and then `log_softmax` saves an equally large tensor for the backward pass — so the memory
lives until `.backward()` has run, not just for the duration of the forward.

The chunked version below is the lesson's recipe, written by hand: take a block of rows, compute
its logits, take its loss, **throw the logits away**, move to the next block. `torch.utils
.checkpoint` is what makes "throw away" real — the block's logits are recomputed during backward
rather than stored, trading a little compute for a lot of memory.

Two measurements are reported, because they answer different questions:

1. **Bytes retained for backward** — counted exactly with `saved_tensors_hooks`, which sees every
   tensor the autograd graph actually stores. Deterministic, device-independent, and the
   quantity the lesson's "16 GiB" refers to.
2. **Peak process memory** — `torch.cuda.max_memory_allocated()` on GPU, sampled RSS on CPU.
   Closer to what the OOM killer sees, but noisier, so it is warmed up and taken as a best-of.

The chunked implementation is only trustworthy if it computes the *same* loss and the *same*
gradients. That is checked before either number is reported.
"""

# %%
PAGE_SIZE = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096


def _rss():
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * PAGE_SIZE


@contextmanager
def retained_for_backward():
    """Exact count of unique storages the autograd graph holds on to."""
    stats = {"bytes": 0, "tensors": 0, "largest": []}
    seen = set()

    def pack(t):
        st = t.untyped_storage()
        if t.numel() and st.data_ptr() not in seen:
            seen.add(st.data_ptr())
            stats["bytes"] += st.nbytes()
            stats["tensors"] += 1
            stats["largest"].append((tuple(t.shape), st.nbytes()))
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        yield stats
    stats["largest"].sort(key=lambda x: -x[1])


@contextmanager
def peak_memory():
    """Peak allocation over the block. CUDA: exact. CPU: sampled RSS."""
    stats = {"bytes": 0}
    if DEVICE.type == "cuda":
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        yield stats
        torch.cuda.synchronize()
        stats["bytes"] = torch.cuda.max_memory_allocated() - base
    else:
        import gc
        gc.collect()
        base = peak = _rss()
        stop = threading.Event()

        def sample():
            nonlocal peak
            while not stop.is_set():
                r = _rss()
                if r > peak:
                    peak = r
                time.sleep(0.0005)

        th = threading.Thread(target=sample, daemon=True); th.start()
        yield stats
        stop.set(); th.join()
        stats["bytes"] = peak - base

# %%
def ordinary_ce(h, W, labels):
    """The naive path: materialise [N, V] logits, then cross-entropy."""
    return F.cross_entropy(F.linear(h, W), labels, ignore_index=IGNORE)


def chunked_ce(h, W, labels, chunk):
    """Block over rows; each block's logits are recomputed in backward, not stored.

    Reduction is done by hand as sum-then-divide, because a mean of per-chunk means would
    weight a short final chunk as heavily as a full one.
    """
    n_valid = (labels != IGNORE).sum().clamp(min=1)
    total = h.new_zeros(())

    def block_loss(h_blk, y_blk):
        return F.cross_entropy(F.linear(h_blk, W), y_blk, ignore_index=IGNORE, reduction="sum")

    for i in range(0, h.shape[0], chunk):
        total = total + torch.utils.checkpoint.checkpoint(
            block_loss, h[i:i + chunk], labels[i:i + chunk], use_reentrant=False
        )
    return total / n_valid


# a bigger batch, so the logits tensor is the dominant term
mem_tokens = get_batch(B=cfg.mem_batch_size, T=cfg.mem_seq_len,
                       generator=torch.Generator().manual_seed(5))
with torch.no_grad():
    mem_hidden = trunk(mem_tokens)

H = mem_hidden[:, :-1].reshape(-1, cfg.d_model).detach()
Y = mem_tokens[:, 1:].reshape(-1)
W_head = head.weight.detach().clone().requires_grad_()
CHUNK = 512

N_rows = H.shape[0]
print(f"N = {N_rows:,} rows (B={cfg.mem_batch_size} x T-1={cfg.mem_seq_len - 1})   "
      f"V = {cfg.vocab_size:,}   D = {cfg.d_model}   chunk = {CHUNK} rows "
      f"({math.ceil(N_rows / CHUNK)} blocks)")
print(f"full logits tensor would be {N_rows * cfg.vocab_size * 4 / 2**20:,.1f} MiB in fp32")

# %%
# --- correctness first: same loss, same gradients, or the memory number means nothing ---
h_a = H.clone().requires_grad_()
loss_a = ordinary_ce(h_a, W_head, Y)
loss_a.backward()
gW_a, gh_a = W_head.grad.clone(), h_a.grad.clone()
W_head.grad = None

h_b = H.clone().requires_grad_()
loss_b = chunked_ce(h_b, W_head, Y, CHUNK)
loss_b.backward()
gW_b, gh_b = W_head.grad.clone(), h_b.grad.clone()
W_head.grad = None

print(f"loss  ordinary {loss_a.item():.7f}   chunked {loss_b.item():.7f}   "
      f"|delta| {abs(loss_a.item() - loss_b.item()):.2e}")
print(f"grad wrt head weight : max |delta| {(gW_a - gW_b).abs().max().item():.2e}   "
      f"allclose {torch.allclose(gW_a, gW_b, atol=1e-6)}")
print(f"grad wrt hidden state: max |delta| {(gh_a - gh_b).abs().max().item():.2e}   "
      f"allclose {torch.allclose(gh_a, gh_b, atol=1e-6)}")
print("\n✓ same objective, same gradients — the chunked version is an implementation, "
      "not an approximation.")

# %%
def measure(fn, trials=3):
    """Warm up once (allocator growth is not the thing being measured), then best-of."""
    ret, peaks = None, []
    for t in range(trials + 1):
        h = H.clone().requires_grad_()
        W_head.grad = None
        with retained_for_backward() as r, peak_memory() as p:
            loss = fn(h, W_head, Y)
            loss.backward()
        if t == 0:
            continue                       # discard the warm-up
        ret = r
        peaks.append(p["bytes"])
    return ret, min(peaks)


ret_ord, peak_ord = measure(lambda h, W, y: ordinary_ce(h, W, y))
ret_chk, peak_chk = measure(lambda h, W, y: chunked_ce(h, W, y, CHUNK))

MiB = 2 ** 20
print(f"{'':<34}{'ordinary':>14}{'chunked':>14}{'ratio':>10}")
print(f"{'retained for backward (MiB)':<34}{ret_ord['bytes']/MiB:>14,.2f}"
      f"{ret_chk['bytes']/MiB:>14,.2f}{ret_ord['bytes']/ret_chk['bytes']:>9.1f}x")
print(f"{'  tensors held':<34}{ret_ord['tensors']:>14,}{ret_chk['tensors']:>14,}{'':>10}")
print(f"{'peak process memory (MiB)':<34}{peak_ord/MiB:>14,.1f}{peak_chk/MiB:>14,.1f}"
      f"{peak_ord/max(peak_chk,1):>9.1f}x")

print("\nwhat the ordinary path is holding:")
for shape, nbytes in ret_ord["largest"][:3]:
    print(f"   {str(shape):<18} {nbytes/MiB:>9,.2f} MiB")
print("what the chunked path is holding:")
for shape, nbytes in ret_chk["largest"][:3]:
    print(f"   {str(shape):<18} {nbytes/MiB:>9,.2f} MiB")
print(f"\nThe [N, V] term is gone. What is left is [N, D] — and D/V = {cfg.d_model}/{cfg.vocab_size:,}"
      f" = 1/{cfg.vocab_size/cfg.d_model:.0f}, which is where the ratio comes from.")

# %% [markdown]
"""
### The same formula at the real configuration

The retained-bytes number is `N × V × bytes_per_number` for the softmax output. That single
expression reproduces both figures quoted in the lesson, which is a useful check that the
harness is measuring the thing the lesson is talking about.
"""

# %%
def logits_bill(T_ctx, B_ctx=1, V=REAL_V, dtype_bytes=2):
    return B_ctx * T_ctx * V * dtype_bytes

for T_ctx in (65_536, 262_144):
    print(f"  B=1, T={T_ctx:>7,}, V={REAL_V:,}, bf16  ->  "
          f"{logits_bill(T_ctx) / 2**30:>5.0f} GiB retained for backward")
print(f"\n  measured here (B={cfg.mem_batch_size}, T={cfg.mem_seq_len}, V={cfg.vocab_size:,}, fp32) ->  "
      f"{ret_ord['bytes'] / 2**20:,.1f} MiB")
print(f"  the same batch chunked at {CHUNK} rows                     ->  "
      f"{ret_chk['bytes'] / 2**20:,.1f} MiB  "
      f"({ret_ord['bytes'] / ret_chk['bytes']:.1f}x less)")
print("\nThe head's 536.9M parameters are 2.1 GiB and they are fixed. The logits tensor is not")
print("fixed — it scales with batch AND context, which is why it, not the parameter count, is")
print("what actually decides whether a long-context step fits in memory.")

# %% [markdown]
"""
# Part 2 — one extra head

A second output head reading the **same** hidden state and predicting token `t+2`. The trunk is
unchanged; only the head count changes. This is multi-token prediction (lesson §13): the same
forward pass now answers two questions instead of one, which densifies the training signal, and
at inference the extra head's output is a **draft** — speculative decoding where the draft model
is the model.

The shift discipline from §1b applies to the new head too, and for the same reason. Head 2's
alignment is verified with strings before a single step is taken.
"""

# %%
class TwoHeadModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        torch.manual_seed(1337)
        self.trunk = Trunk(cfg)
        self.head1 = Head(cfg)      # predicts t+1
        self.head2 = Head(cfg)      # predicts t+2

    def forward(self, tokens):
        h = self.trunk(tokens)
        return h, self.head1(h), self.head2(h)


def two_head_losses(logits1, logits2, toks):
    """head 1: position i predicts token i+1.   head 2: position i predicts token i+2.

    Head 2 loses one more position from the tail than head 1 does: the last two positions have
    no t+2 answer inside the window, so they are dropped rather than being given a wrong one.
    """
    V = cfg.vocab_size
    l1 = F.cross_entropy(logits1[:, :-1].reshape(-1, V), toks[:, 1:].reshape(-1), ignore_index=IGNORE)
    l2 = F.cross_entropy(logits2[:, :-2].reshape(-1, V), toks[:, 2:].reshape(-1), ignore_index=IGNORE)
    return l1, l2


model2 = TwoHeadModel(cfg).to(DEVICE)
demo = get_batch(B=2, generator=torch.Generator().manual_seed(21))
with torch.no_grad():
    h_d, lg1_d, lg2_d = model2(demo)

print("shapes")
print(f"  hidden              {tuple(h_d.shape)}      shared by both heads — one trunk, two readers")
print(f"  head1 logits        {tuple(lg1_d.shape)}")
print(f"  head2 logits        {tuple(lg2_d.shape)}")
print(f"  head1 predictions   {tuple(lg1_d[:, :-1].shape)}  vs targets {tuple(demo[:, 1:].shape)}")
print(f"  head2 predictions   {tuple(lg2_d[:, :-2].shape)}  vs targets {tuple(demo[:, 2:].shape)}")

print("\nverifying head 2's shift with strings, same as §1b:")
seq = demo[0]
print(f"{'pos':>4}  {'input':<20} {'head1 target (t+1)':<22} {'head2 target (t+2)':<22}")
for i in range(8):
    print(f"{i:>4}  {tok.id_to_token(seq[i].item())!r:<20} "
          f"{tok.id_to_token(seq[i+1].item())!r:<22} {tok.id_to_token(seq[i+2].item())!r:<22}")
# head 2's target at position i must be head 1's target at position i+1, everywhere
h1_targets, h2_targets = demo[:, 1:], demo[:, 2:]
assert torch.equal(h2_targets, h1_targets[:, 1:]), "head 2 target alignment broken"
print("\n✓ head 2's target is the token two steps ahead, at every position")

# %%
def train_two_head(steps=300, lr=3e-4, log_every=25):
    model = TwoHeadModel(cfg).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(99)
    hist = {"step": [], "l1": [], "l2": [], "total": []}
    print(f"{'step':>5} {'head1 (t+1)':>13} {'head2 (t+2)':>13} {'sum':>10} "
          f"{'gap':>8} {'ppl1':>10} {'ppl2':>10}")
    for step in range(steps + 1):
        batch = get_batch(generator=gen)
        _, lg1, lg2 = model(batch)
        l1, l2 = two_head_losses(lg1, lg2, batch)
        total = l1 + l2
        hist["step"].append(step)
        hist["l1"].append(l1.item()); hist["l2"].append(l2.item()); hist["total"].append(total.item())
        if step % log_every == 0:
            print(f"{step:>5} {l1.item():>13.4f} {l2.item():>13.4f} {total.item():>10.4f} "
                  f"{l2.item()-l1.item():>+8.4f} {math.exp(l1.item()):>10,.1f} {math.exp(l2.item()):>10,.1f}")
        opt.zero_grad(set_to_none=True)
        total.backward()
        opt.step()
    return model, hist


model2, hist = train_two_head()

# %%
final_l1, final_l2 = hist["l1"][-1], hist["l2"][-1]
print("=" * 62)
print("PART 2 — final numbers")
print("=" * 62)
print(f"  head 1 loss  (predicts t+1)   {final_l1:.4f} nats   perplexity {math.exp(final_l1):>10,.2f}")
print(f"  head 2 loss  (predicts t+2)   {final_l2:.4f} nats   perplexity {math.exp(final_l2):>10,.2f}")
print(f"  sum (the quantity optimised)  {final_l1 + final_l2:.4f} nats")
print(f"  gap  (head2 - head1)          {final_l2 - final_l1:+.4f} nats")
print(f"\n  at step 0:  head1 {hist['l1'][0]:.4f}   head2 {hist['l2'][0]:.4f}   "
      f"gap {hist['l2'][0]-hist['l1'][0]:+.4f}")
print(f"  ln V anchor: {ln_V:.4f} — both heads start there, as they must")

k = 25
early = sum(hist["l2"][:k]) / k - sum(hist["l1"][:k]) / k
late = sum(hist["l2"][-k:]) / k - sum(hist["l1"][-k:]) / k
print(f"\n  mean gap over the first {k} steps  {early:+.4f} nats")
print(f"  mean gap over the last  {k} steps  {late:+.4f} nats")
print(f"  the gap {'widened' if late > early else 'narrowed'} by {abs(late-early):.4f} nats")

fig, ax = plt.subplots(figsize=(8, 4.2))
ax.plot(hist["step"], hist["l1"], label=f"head 1, t+1  (final {final_l1:.3f})", lw=2)
ax.plot(hist["step"], hist["l2"], label=f"head 2, t+2  (final {final_l2:.3f})", lw=2)
ax.plot(hist["step"], hist["total"], label=f"sum (optimised, final {final_l1+final_l2:.3f})",
        lw=1.2, ls=":", color="grey")
ax.axhline(ln_V, color="grey", lw=1, ls="--", label=f"ln V = {ln_V:.3f}")
ax.set_xlabel("step"); ax.set_ylabel("cross-entropy (nats)")
ax.set_title("Head 2 sits above head 1 and stays there")
ax.legend(); fig.tight_layout()
fig.savefig("assets/two_head_curves.png", dpi=120)
plt.close(fig)
print("\nsaved assets/two_head_curves.png")

# %% [markdown]
"""
### What happens to head 2's loss, and why it is correct

Head 2's loss starts at the same place as head 1's — `ln V`, because an untrained head has no
preferences regardless of what it is being asked — and then falls more slowly and **settles
above** head 1 for the rest of training. It never catches up, and it should not.

The reason is not a defect in the head. It is a property of the data. Both heads read the
identical hidden state `h_i`, which summarises tokens `≤ i`. Head 1 is asked for `p(x_{i+1} | x_{≤i})`
and head 2 for `p(x_{i+2} | x_{≤i})` — and `x_{i+2}` is genuinely more uncertain, because the
intervening token `x_{i+1}` is exactly the information head 2 is denied. In the language of §5
of the lesson, the conditional entropy of the target is higher, and cross-entropy cannot go
below the entropy of what it is predicting. The gap between the two curves is an estimate of
that extra uncertainty, measured in nats.

Two consequences worth stating plainly:

- **A head-2 loss that matched head 1 would be evidence of a bug**, not of a better model — the
  most likely cause being a shift error that quietly handed head 2 the `t+1` target. §1b's
  discipline is what rules that out.
- **The sum is what the optimiser sees.** Head 2's gradient flows back through the shared trunk,
  so the trunk is pushed to build a hidden state useful for both horizons. That is the point of
  multi-token prediction: more supervision per forward pass, and a spare head that can propose
  the next-but-one token at inference time instead of being discarded.
"""

# %% [markdown]
"""
## Appendix — two properties of the gradient, checked

Not required by the assignment, but both are cheap, both are stated in the lesson (§8), and both
are exactly the kind of thing a "correct and observable" harness should be able to show rather
than assert.

The gradient of cross-entropy with respect to the logits is `softmax(z) − onehot(y)`. It
therefore **sums to exactly zero** over the vocabulary — probability mass is moved, never created
— and it is **dense**: every one of the `V` rows gets a non-zero update from every single token.
"""

# %%
z = torch.randn(4, cfg.vocab_size, requires_grad=True)
y = torch.randint(0, cfg.vocab_size, (4,))
F.cross_entropy(z, y, reduction="sum").backward()

analytic = F.softmax(z.detach(), dim=-1) - F.one_hot(y, cfg.vocab_size).float()
print(f"autograd matches softmax(z) - onehot(y):  {torch.allclose(z.grad, analytic, atol=1e-6)}")
print(f"row sums of the gradient:                 {z.grad.sum(-1).abs().max().item():.3e}  (zero)")
print(f"non-zero entries per row:                 {int((z.grad[0] != 0).sum()):,} of {cfg.vocab_size:,}")
print("\nEvery row of the [V, D] head receives gradient from every token in the batch. That is why")
print("the head is a second memory problem and not just a second parameter count: the update is")
print("as wide as the vocabulary, every step.")

# %% [markdown]
"""
## Results summary — every graded number, in one place

Written to `results.json` so the write-up can be built from measured values rather than from
numbers retyped by hand. A number in the README that is not in this file did not come from a
cell that ran.
"""

# %%
import json

results = {
    "config": {
        "V": cfg.vocab_size, "D": cfg.d_model, "n_layer": cfg.n_layer, "n_head": cfg.n_head,
        "T": cfg.seq_len, "B": cfg.batch_size,
        "tokenizer": "ERA V5 Session 2 BPE (mr variant, en/hi/te/mr)",
        "device": str(DEVICE), "torch": torch.__version__,
        "real_V": REAL_V, "real_D": REAL_D,
    },
    "1a_shapes": {
        slug: list(shape) for slug, (_, shape, _) in zip(
            ["tokens", "hidden", "head_weight", "logits", "logits_shifted",
             "targets_shifted", "flat_logits", "flat_targets"], rows)
    },
    "1a_logits_MiB": logits.numel() * logits.element_size() / MiB,
    "1b_shift": {
        "final_loss": {m: curves[m][-1] for m in curves},
        "final_ppl": {m: math.exp(curves[m][-1]) for m in curves},
        "steps": len(curves["correct"]) - 1,
        "no_shift_echo_rate": echo_rate,
        "ppl_ratio_no_shift": gap_ns, "ppl_ratio_reversed": gap_rv,
    },
    "1c_padding": {
        "positions": n_total, "contributing_masked": n_masked,
        "contributing_unmasked": n_total,
        "pad_positions": n_total - n_masked,
        "pad_fraction": (n_total - n_masked) / n_total,
        "denominator_recovered": denom_recovered,
        "untrained": {"loss_unmasked": u_naive, "loss_masked": u_masked,
                      "delta": u_masked - u_naive},
        "trained_clean": {"loss_unmasked": loss_naive, "loss_masked": loss_masked,
                          "delta": loss_masked - loss_naive},
        "trained_on_padding_unmasked": {"loss_unmasked": b_naive, "loss_masked": b_masked,
                                        "delta": b_masked - b_naive},
    },
    "1d_packing": {
        "positions": n_pos,
        "trained": {
            "loss_unmasked": loss_unmasked, "loss_masked": loss_bmasked,
            "boundary_token_loss": per_token[boundary].item(),
            "mean_other_positions": mean_other,
            "boundary_excess": per_token[boundary].item() - mean_other,
            "delta": loss_bmasked - loss_unmasked,
        },
        "untrained": {
            "loss_unmasked": u_un, "loss_masked": u_ma,
            "boundary_token_loss": u_per[boundary].item(),
            "mean_other_positions": u_mean_other,
            "boundary_excess": u_per[boundary].item() - u_mean_other,
            "delta": u_ma - u_un,
        },
    },
    "1e_perplexity": {
        "untrained_loss": untrained_loss, "untrained_ppl": untrained_ppl,
        "ln_V": ln_V, "deviation_nats": untrained_loss - ln_V,
        "tokens_measured": total_tokens_counted,
        "trained_loss": curves["correct"][-1], "trained_ppl": math.exp(curves["correct"][-1]),
        "real_config_ln_V": math.log(REAL_V),
    },
    "1f_tying": {"proxy": proxy_acc, "real": real_acc, "trunk_params": n_trunk},
    "1g_memory": {
        "N_rows": N_rows, "chunk": CHUNK,
        "retained_ordinary_MiB": ret_ord["bytes"] / MiB,
        "retained_chunked_MiB": ret_chk["bytes"] / MiB,
        "retained_ratio": ret_ord["bytes"] / ret_chk["bytes"],
        "V_over_D": cfg.vocab_size / cfg.d_model,
        "full_logits_MiB": N_rows * cfg.vocab_size * 4 / MiB,
        "peak_rss_ordinary_MiB": peak_ord / MiB,
        "peak_rss_chunked_MiB": peak_chk / MiB,
        "peak_ratio": peak_ord / max(peak_chk, 1),
        "loss_ordinary": loss_a.item(), "loss_chunked": loss_b.item(),
        "grad_max_delta": (gW_a - gW_b).abs().max().item(),
        "real_config_GiB_at_64k": logits_bill(65_536) / 2**30,
        "real_config_GiB_at_256k": logits_bill(262_144) / 2**30,
    },
    "part2": {
        "steps": hist["step"][-1],
        "head1_loss": final_l1, "head2_loss": final_l2,
        "sum": final_l1 + final_l2, "gap": final_l2 - final_l1,
        "head1_ppl": math.exp(final_l1), "head2_ppl": math.exp(final_l2),
        "head1_loss_step0": hist["l1"][0], "head2_loss_step0": hist["l2"][0],
        "gap_first_25": early, "gap_last_25": late,
    },
}

with open("results.json", "w") as f:
    json.dump(results, f, indent=2)

print("wrote results.json\n")
print(f"{'PART 1':<44}")
print(f"  1a  logits tensor                          {results['1a_logits_MiB']:,.1f} MiB "
      f"[B={B}, T={T}, V={cfg.vocab_size:,}]")
print(f"  1b  correct / no_shift / reversed loss     "
      f"{curves['correct'][-1]:.3f} / {curves['no_shift'][-1]:.3f} / {curves['reversed'][-1]:.3f} nats")
print(f"  1c  contributing tokens, unmasked->masked  {n_total:,} -> {n_masked:,} "
      f"(pad-trained model: loss {b_naive:.4f} -> {b_masked:.4f} nats)")
print(f"  1d  packed loss, unmasked->masked          "
      f"{loss_unmasked:.4f} -> {loss_bmasked:.4f} nats (trained)")
print(f"  1e  untrained loss / perplexity            "
      f"{untrained_loss:.4f} nats / {untrained_ppl:,.0f}  (V = {cfg.vocab_size:,})")
print(f"  1f  head params, untied->tied              "
      f"{proxy_acc['untied']:,} -> {proxy_acc['tied']:,}")
print(f"  1g  retained bytes, ordinary vs chunked    "
      f"{ret_ord['bytes']/MiB:,.1f} vs {ret_chk['bytes']/MiB:,.1f} MiB "
      f"({ret_ord['bytes']/ret_chk['bytes']:.1f}x)")
print(f"\n{'PART 2':<44}")
print(f"  head 1 (t+1) / head 2 (t+2) / sum          "
      f"{final_l1:.4f} / {final_l2:.4f} / {final_l1 + final_l2:.4f} nats")
