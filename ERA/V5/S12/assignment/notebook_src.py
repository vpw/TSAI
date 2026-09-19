# %% [markdown]
"""
# Session 12 — Distributed Training I: Data Parallel and ZeRO

**32 virtual GPUs, a real model on top of them, and ZeRO-1/2/3 implemented over that.**

This session is different from every previous one in the course. S9–S11 measured a real
model actually training and reported what happened. Here the deliverable is a *simulator*,
which means "my run produced a number" is not evidence of anything by itself — a simulator
can produce any number I want it to. So the whole notebook is organised around a single
discipline:

> **every quantity the simulator computes is checked against a number the lesson publishes
> independently, and the check is an assertion that fails the build if it breaks.**

The lesson gives four such anchors, and they are genuinely independent of each other:

| anchor | where | what it pins down |
| --- | --- | --- |
| 16 bytes per weight | §1 | the memory model itself |
| 16.00 / 5.50 / 3.75 / 2.00 bytes per weight at 8 GPUs | §6 | the per-stage sharding arithmetic |
| 447.0 / 122.2 / 68.1 / 14.0 GiB at 32 GPUs, 30B params | §7 | the same arithmetic at a *different* world size |
| 2P / 2P / 2P / 3P | §6 | the communication volume, which is a separate quantity entirely |

The assignment asks for 32 virtual GPUs, and 32 is one of the columns the lesson's §7 ladder
publishes — so the simulator is not being checked against a plausible trend, it is being
checked against four numbers that were written down before it existed.

## What is simulated and what is real

Being clear about this up front, because a simulator that blurs the line is worthless:

- **Real:** the model (nanoGPT, 813,440 parameters), its forward and backward passes, the
  gradients, the Adam updates, the loss curve. All of it is genuine PyTorch on CPU.
- **Real:** the *contents* of every collective. `reduce_scatter` really does reduce and
  scatter; the tensors that come out are the tensors that would come out on real hardware.
  Section 3 checks this against `torch.distributed`'s actual gloo backend.
- **Simulated:** the 32 GPUs. They are Python objects in one process, so nothing runs in
  parallel and no data crosses a wire.
- **Accounted, not measured:** per-GPU memory. Python threads share one address space, so
  there is no honest way to read "how much memory rank 7 is using" off the process. Memory
  is computed from the shard map instead, and every such number in this notebook is labelled
  as accounted rather than measured.
- **Modelled, not measured:** wall-clock at scale. Simulator wall-clock measures Python
  overhead. Where this notebook reports time it applies §5's stated bandwidths to measured
  byte counts, and says so.

## Contents

0. Environment
1. The 16-byte weight — why this session exists
2. The fabric — 32 virtual GPUs and four collectives
3. Are the collectives right? (including a real gloo cross-check)
4. The model and the shard map
5. Data parallelism — the baseline
6. ZeRO-1 — shard the optimizer state
7. ZeRO-2 — shard the gradients too
8. ZeRO-3 — shard the weights too
9. The comparison, against the lesson's published table
10. Do all four learn the same thing?
11. The memory wall
12. Projecting to 30B
13. Bucketing and overlap
14. Offload to CPU
15. Precision — what MXFP8 changes
16. Pros and cons, stage by stage
"""

# %%
## 0. Environment
import json
import math
import pathlib
import subprocess
import sys
import time
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cpu")
ASSETS = pathlib.Path("assets")
ASSETS.mkdir(exist_ok=True)
torch.manual_seed(0)
RESULTS = {}

# The world size the assignment asks for.
WORLD_SIZE = 32

print(f"torch {torch.__version__}  device={DEVICE}  world_size={WORLD_SIZE}")

# %% [markdown]
"""
## 1. The 16-byte weight — why this session exists

The whole session is settled by one piece of arithmetic in §1 of the lesson, and everything
after it is a consequence. A weight does not cost 2 bytes to train. It costs 16:

| what is stored for one weight | bytes |
| --- | --- |
| the weight, in the 16-bit format used for arithmetic | 2 |
| its gradient | 2 |
| a 32-bit copy of the weight, kept for accuracy | 4 |
| two running averages the optimizer keeps (fp32 each) | 8 |
| **total** | **16** |

Two of those entries are worth pausing on, because they are the ones that look redundant.

**The 32-bit copy.** The lesson's reason is that "repeatedly adding very small updates to a
16-bit number loses them to rounding." That is a checkable claim, not a slogan, so cell 1b
checks it.

**The two running averages.** These are Adam's `m` and `v` from Session 11 §6 — the same two
bars that sat under each weight in that session's widget. S11 established that they are half
of training memory; this session is what follows from that.
"""

# %%
### 1a. The 16-byte table, and what it costs at V5's scale
BYTES = {
    "weight": 2,    # bf16 parameter used for the arithmetic
    "grad": 2,      # bf16 gradient
    "master": 4,    # fp32 master copy
    "m": 4,         # Adam first moment, fp32
    "v": 4,         # Adam second moment, fp32
}
BYTES_PER_WEIGHT = sum(BYTES.values())

# The three groupings the ZeRO stages act on, which is why they are named separately.
B_WEIGHT = BYTES["weight"]                          # 2  — sharded by stage 3
B_GRAD = BYTES["grad"]                              # 2  — sharded by stage 2
B_OPT = BYTES["master"] + BYTES["m"] + BYTES["v"]   # 12 — sharded by stage 1

GiB = 1024 ** 3
V5_PARAMS = 30e9          # the lesson uses 30 billion for every calculation
CARD_BYTES = 80e9         # an 80 GB card
CARD_GiB = CARD_BYTES / GiB

for k, v in BYTES.items():
    print(f"  {k:>8}: {v:>2} bytes")
print(f"  {'total':>8}: {BYTES_PER_WEIGHT:>2} bytes per weight")
print()

v5_state_gib = V5_PARAMS * BYTES_PER_WEIGHT / GiB
print(f"30e9 weights x {BYTES_PER_WEIGHT} bytes = {V5_PARAMS * BYTES_PER_WEIGHT / 1e9:.0f} GB "
      f"= {v5_state_gib:.1f} GiB")
print(f"one card holds {CARD_GiB:.1f} GiB  ->  {v5_state_gib / CARD_GiB:.1f} cards "
      f"({math.ceil(v5_state_gib / CARD_GiB)} cards) before a single activation exists")

# ANCHOR 1: the lesson's §1 figure.
assert BYTES_PER_WEIGHT == 16, BYTES_PER_WEIGHT
assert abs(v5_state_gib - 447.0) < 0.05, v5_state_gib
print("\nmatches the lesson's §1 figure of 447.0 GiB")

# %%
### 1b. Why the fp32 master copy exists — the claim, tested
# "Repeatedly adding very small updates to a 16-bit number loses them to rounding."
# bfloat16 keeps 8 mantissa bits, so near 1.0 its resolution is 2^-8 = 0.0039. An update
# of 1e-4 is far below that, and round-to-nearest sends every single one of them to zero.
UPDATE = 1e-4
N_UPDATES = 1000

w_bf16 = torch.tensor([1.0], dtype=torch.bfloat16)
w_fp32 = torch.tensor([1.0], dtype=torch.float32)
for _ in range(N_UPDATES):
    w_bf16 += UPDATE      # 16-bit accumulation, as if no master copy existed
    w_fp32 += UPDATE      # 32-bit accumulation, which is what the master copy is for

bf16_resolution = float(torch.tensor(1.0, dtype=torch.bfloat16).float()
                        - torch.nextafter(torch.tensor(1.0, dtype=torch.bfloat16),
                                          torch.tensor(0.0, dtype=torch.bfloat16)).float())
expected = 1.0 + N_UPDATES * UPDATE

print(f"adding {UPDATE} to 1.0, {N_UPDATES:,} times")
print(f"  bf16 resolution near 1.0 : {bf16_resolution:.6f}  (update is {UPDATE}, "
      f"{bf16_resolution / UPDATE:.0f}x smaller)")
print(f"  exact answer             : {expected:.4f}")
print(f"  fp32 accumulation        : {float(w_fp32):.4f}")
print(f"  bf16 accumulation        : {float(w_bf16):.4f}   <- every update rounded away")

lost_fraction = 1.0 - (float(w_bf16) - 1.0) / (expected - 1.0)
print(f"\n{lost_fraction:.1%} of the training signal is lost without the fp32 master copy.")
print("That is what 4 of the 16 bytes buy.")
assert float(w_bf16) == 1.0, float(w_bf16)

RESULTS["section1"] = {
    "bytes": BYTES,
    "bytes_per_weight": BYTES_PER_WEIGHT,
    "b_weight": B_WEIGHT, "b_grad": B_GRAD, "b_opt": B_OPT,
    "v5_params": V5_PARAMS,
    "v5_state_gib": v5_state_gib,
    "card_gib": CARD_GiB,
    "cards_needed": math.ceil(v5_state_gib / CARD_GiB),
    "lesson_v5_state_gib": 447.0,
    "bf16_demo": {
        "update": UPDATE, "n_updates": N_UPDATES,
        "bf16_resolution_at_1": bf16_resolution,
        "exact": expected, "fp32": float(w_fp32), "bf16": float(w_bf16),
        "lost_fraction": lost_fraction,
    },
}

# %% [markdown]
"""
## 2. The fabric — 32 virtual GPUs and four collectives

A "virtual GPU" here is deliberately minimal: a rank number and a memory ledger. It holds no
data of its own, because in this simulation the data lives in ordinary Python lists indexed
by rank. Pretending otherwise would add ceremony without adding fidelity.

What the fabric *does* do carefully is two things:

**1. Compute the real result of each collective.** `reduce_scatter` genuinely reduces and
scatters. Section 3 checks the output against `torch.distributed`'s gloo backend.

**2. Count bytes under the ring cost model.** This is where a simulator earns its keep,
because the byte counts are what the assignment's "how the computation changes" actually
asks for, and they are easy to get subtly wrong.

### The ring cost model, and where it differs from the lesson

The lesson (§4) says a ring all-reduce sends "about one copy of the data during the first
phase and receives about one copy during the second, giving a total of 2P."

"About" is doing real work in that sentence. In a ring of `N` GPUs, a reduce-scatter is
`N-1` sends of one `1/N` chunk each, so each GPU sends `(N-1)/N` of a full copy — not a full
copy. The all-gather is another `N-1` sends of a chunk. So the exact per-GPU traffic is:

```
reduce-scatter : (N-1)/N · P
all-gather     : (N-1)/N · P
all-reduce     : 2(N-1)/N · P
```

At `N=32` that is `1.9375P`, not `2P`. The lesson's `2P` is the large-`N` limit, and it is
the right number to quote for a 30B model on 32 cards where the 3% gap is noise against the
other approximations. But a simulator that counts actual sends should report what it
actually counted, so this notebook reports the exact figure and shows it converging on the
lesson's. **Where the simulator is more precise than the table it is checked against, it
says so rather than rounding itself into agreement.**
"""

# %%
### 2a. The memory ledger — the only honest way to report per-GPU memory here
# Python threads share one address space. There is no meaningful `rank 7's RSS`. So memory
# is *accounted* from the shard map: every byte a rank is responsible for is recorded in a
# category, and the categories are exactly the ones the ZeRO stages act on.
class MemoryLedger:
    """Per-rank byte accounting, in the lesson's own categories."""

    CATEGORIES = ("weight", "grad", "master", "m", "v", "transient", "cpu_offload")

    def __init__(self, rank):
        self.rank = rank
        self.bytes = {c: 0 for c in self.CATEGORIES}
        self.peak_transient = 0

    def hold(self, category, n_elements, bytes_per_element):
        self.bytes[category] += n_elements * bytes_per_element

    def set_transient(self, n_bytes):
        """Stage 3 gathers a layer, uses it, discards it. Peak is what matters."""
        self.bytes["transient"] = n_bytes
        self.peak_transient = max(self.peak_transient, n_bytes)

    @property
    def resident(self):
        """Bytes on the GPU: everything except what was offloaded to system memory."""
        return sum(v for k, v in self.bytes.items() if k != "cpu_offload")

    @property
    def steady_state(self):
        """Resident bytes excluding transient gathers — the quantity the lesson tabulates."""
        return self.resident - self.bytes["transient"]


# %%
### 2b. The fabric itself
class Fabric:
    """`world_size` virtual GPUs and the collectives that run across them.

    Every collective returns the mathematically correct result and records the per-rank
    traffic it would cost on a ring interconnect.
    """

    def __init__(self, world_size, elem_bytes=2):
        self.world_size = world_size
        self.elem_bytes = elem_bytes          # bf16 on the wire, per the lesson
        self.gpus = [MemoryLedger(r) for r in range(world_size)]
        self.reset_counters()

    # -- counters ----------------------------------------------------------------
    def reset_counters(self):
        self.bytes_sent = [0] * self.world_size
        self.ops = []                          # (name, n_elements) for the audit trail

    def _charge(self, name, n_elements, copies):
        """Ring traffic: each rank sends `copies * (N-1)/N` of a full copy of the tensor."""
        N = self.world_size
        per_rank = copies * (N - 1) / N * n_elements * self.elem_bytes
        for r in range(N):
            self.bytes_sent[r] += per_rank
        self.ops.append((name, n_elements))
        return per_rank

    # -- collectives -------------------------------------------------------------
    def all_reduce(self, shards):
        """Every rank contributes a full tensor, every rank ends with the mean. Cost 2(N-1)/N·P."""
        assert len(shards) == self.world_size
        self._charge("all_reduce", shards[0].numel(), copies=2)
        mean = torch.stack(shards, dim=0).mean(dim=0)
        return [mean.clone() for _ in range(self.world_size)]

    def reduce_scatter(self, shards):
        """Every rank contributes a full tensor, rank r ends with slice r of the mean. Cost (N-1)/N·P."""
        assert len(shards) == self.world_size
        n = shards[0].numel()
        assert n % self.world_size == 0, f"{n} not divisible by {self.world_size}"
        self._charge("reduce_scatter", n, copies=1)
        # Exactly the same reduction as all_reduce, then sliced — so stage-1/2/3 results are
        # bit-identical to data parallelism's, not merely close. Section 10 relies on this.
        mean = torch.stack(shards, dim=0).mean(dim=0)
        return list(mean.chunk(self.world_size))

    def all_gather(self, slices):
        """Rank r contributes slice r, every rank ends with the whole. Cost (N-1)/N·P."""
        assert len(slices) == self.world_size
        full = torch.cat([s.reshape(-1) for s in slices])
        self._charge("all_gather", full.numel(), copies=1)
        return [full.clone() for _ in range(self.world_size)]

    def broadcast(self, tensor, src=0):
        """One rank's tensor reaches all. Cost (N-1)/N·P."""
        self._charge("broadcast", tensor.numel(), copies=1)
        return [tensor.clone() for _ in range(self.world_size)]

    # -- reporting ---------------------------------------------------------------
    def bytes_per_rank(self):
        return self.bytes_sent[0]

    def as_multiple_of_P(self, n_params):
        """Traffic expressed in the lesson's units: P = one bf16 copy of the parameters."""
        P = n_params * self.elem_bytes
        return self.bytes_per_rank() / P


fabric = Fabric(WORLD_SIZE)
print(f"fabric: {fabric.world_size} virtual GPUs, {fabric.elem_bytes} bytes/element on the wire")
print(f"ring factor (N-1)/N at N={WORLD_SIZE}: {(WORLD_SIZE - 1) / WORLD_SIZE:.4f}")
print(f"  so an all-reduce costs {2 * (WORLD_SIZE - 1) / WORLD_SIZE:.4f}P, "
      f"which the lesson rounds to 2P")

# %% [markdown]
"""
## 3. Are the collectives right?

Three checks, in increasing order of how much they would embarrass me if they failed.

**3a. The identity everything else rests on.** The lesson's §4 claim is that a
reduce-scatter followed by an all-gather produces *exactly* what an all-reduce produces.
This is not a nice-to-have: it is the entire reason ZeRO-1 and ZeRO-2 are free in
communication. If it does not hold elementwise, the rest of the notebook is built on sand.

**3b. The cost model.** The byte counters should land on `2(N-1)/N·P` and `(N-1)/N·P`, and
the all-reduce total should equal the sum of its two halves — which is the cost-side
statement of the same identity.

**3c. A real backend.** Everything above checks the simulator against itself, which proves
nothing about whether my idea of a collective matches anyone else's. So this cell spawns
four real processes running `torch.distributed` on the gloo backend and compares their
output to the fabric's, tensor by tensor. Four ranks rather than 32 because this box has two
cores and the point is agreement, not scale.
"""

# %%
### 3a. reduce-scatter + all-gather == all-reduce, elementwise
torch.manual_seed(1)
N_TEST = 4096
test_shards = [torch.randn(N_TEST) for _ in range(WORLD_SIZE)]

fabric.reset_counters()
ar = fabric.all_reduce(test_shards)[0]
ar_bytes = fabric.bytes_per_rank()

fabric.reset_counters()
rs = fabric.reduce_scatter(test_shards)
ag = fabric.all_gather(rs)[0]
rs_ag_bytes = fabric.bytes_per_rank()

max_abs_diff = float((ar - ag).abs().max())
print(f"all_reduce           -> tensor of {ar.numel()}")
print(f"reduce_scatter+gather-> tensor of {ag.numel()}")
print(f"max |difference|     : {max_abs_diff:.3e}")
print(f"bit-identical        : {torch.equal(ar, ag)}")
assert torch.equal(ar, ag), max_abs_diff
print("\n§4's identity holds exactly. ZeRO-1 and ZeRO-2 can therefore be free.")

# %%
### 3b. The cost model — counted, then compared to the closed form
N = WORLD_SIZE
ring = (N - 1) / N
elem_bytes = fabric.elem_bytes
P_test = N_TEST * elem_bytes

print(f"{'collective':<26} {'counted bytes/rank':>20} {'as multiple of P':>18} {'closed form':>14}")
for name, counted, copies in (("all_reduce", ar_bytes, 2),
                              ("reduce_scatter+all_gather", rs_ag_bytes, 2)):
    print(f"{name:<26} {counted:>20,.0f} {counted / P_test:>18.4f} {copies * ring:>14.4f}")

assert abs(ar_bytes - rs_ag_bytes) < 1e-6, (ar_bytes, rs_ag_bytes)
assert abs(ar_bytes / P_test - 2 * ring) < 1e-9
print(f"\nall-reduce costs the same as its two halves: {ar_bytes:,.0f} bytes either way.")
print(f"exact: {2 * ring:.4f}P at N={N}.  lesson's large-N figure: 2P.  "
      f"gap: {(2 - 2 * ring) / 2:.1%}")

# convergence of the exact figure on the lesson's 2P
print(f"\n{'N':>6} {'all-reduce cost':>18}")
for n in (4, 8, 32, 128, 1024):
    print(f"{n:>6} {2 * (n - 1) / n:>17.4f}P")

RESULTS["section3"] = {
    "n_test_elements": N_TEST,
    "identity_bit_identical": bool(torch.equal(ar, ag)),
    "identity_max_abs_diff": max_abs_diff,
    "all_reduce_bytes_per_rank": ar_bytes,
    "rs_ag_bytes_per_rank": rs_ag_bytes,
    "all_reduce_multiple_of_P": ar_bytes / P_test,
    "ring_factor": ring,
    "lesson_all_reduce_multiple_of_P": 2.0,
    "convergence": {n: 2 * (n - 1) / n for n in (4, 8, 32, 128, 1024)},
}

# %%
### 3c. Cross-check against a real backend — torch.distributed, gloo, 4 processes
# Everything above checks the simulator against itself. This checks it against somebody
# else's implementation. The script below is written out, run in a subprocess, and spawns
# four real gloo ranks; it prints JSON that this cell compares to the fabric's output.
GLOO_WORLD = 4
GLOO_SCRIPT = ASSETS / "gloo_check.py"
GLOO_SCRIPT.write_text('''
"""Four real gloo ranks. Run standalone; prints JSON on rank 0."""
import json, os, sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

WORLD, N = 4, 256


def worker(rank, out):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT="29517")
    dist.init_process_group("gloo", rank=rank, world_size=WORLD)

    torch.manual_seed(100 + rank)
    x = torch.randn(N)

    ar = x.clone()
    dist.all_reduce(ar, op=dist.ReduceOp.SUM)
    ar /= WORLD

    rs = torch.empty(N // WORLD)
    dist.reduce_scatter_tensor(rs, x.clone(), op=dist.ReduceOp.SUM)
    rs /= WORLD

    ag = torch.empty(N)
    dist.all_gather_into_tensor(ag, rs.contiguous())

    if rank == 0:
        out["all_reduce"] = ar.tolist()
        out["all_gather_of_reduce_scatter"] = ag.tolist()
        out["inputs"] = None
    dist.destroy_process_group()


if __name__ == "__main__":
    mgr = mp.Manager()
    out = mgr.dict()
    mp.spawn(worker, args=(out,), nprocs=WORLD, join=True)
    inputs = []
    for r in range(WORLD):
        torch.manual_seed(100 + r)
        inputs.append(torch.randn(N).tolist())
    print(json.dumps({"all_reduce": out["all_reduce"],
                      "all_gather_of_reduce_scatter": out["all_gather_of_reduce_scatter"],
                      "inputs": inputs}))
''')

proc = subprocess.run([sys.executable, str(GLOO_SCRIPT)], capture_output=True, text=True, timeout=600)
if proc.returncode != 0:
    print(proc.stdout[-2000:])
    print(proc.stderr[-2000:])
    raise RuntimeError("gloo cross-check failed to run")

gloo = json.loads(proc.stdout.strip().splitlines()[-1])
gloo_inputs = [torch.tensor(x) for x in gloo["inputs"]]

sim = Fabric(GLOO_WORLD)
sim_ar = sim.all_reduce(gloo_inputs)[0]
sim_ag = sim.all_gather(sim.reduce_scatter(gloo_inputs))[0]

real_ar = torch.tensor(gloo["all_reduce"])
real_ag = torch.tensor(gloo["all_gather_of_reduce_scatter"])

d_ar = float((sim_ar - real_ar).abs().max())
d_ag = float((sim_ag - real_ag).abs().max())

print(f"gloo ranks: {GLOO_WORLD}, tensor of {real_ar.numel()} elements\n")
print(f"{'quantity':<38} {'max |simulated - real gloo|':>28}")
print(f"{'all_reduce':<38} {d_ar:>28.3e}")
print(f"{'all_gather(reduce_scatter(.))':<38} {d_ag:>28.3e}")

assert d_ar < 1e-6, d_ar
assert d_ag < 1e-6, d_ag
print("\nThe simulated collectives agree with the real gloo backend to float32 precision.")
print("(Not bit-identical: gloo reduces in ring order, the fabric via torch.stack().mean(0).")
print(" Different summation orders, same mathematics.)")

RESULTS["section3"]["gloo"] = {
    "world_size": GLOO_WORLD,
    "n_elements": real_ar.numel(),
    "max_diff_all_reduce": d_ar,
    "max_diff_all_gather_reduce_scatter": d_ag,
}

# %% [markdown]
"""
## 4. The model and the shard map

The demo model is Session 10/11's nanoGPT, copied unchanged — a from-scratch char-level GPT
with `n_embd=128, n_layer=4, n_head=4, seq_len=128`, **813,440 parameters**. Reusing it keeps
continuity with the previous two sessions and, more usefully here, gives a realistic
*per-layer* parameter distribution: an embedding table, four transformer blocks of very
different internal shapes, and a final layer norm. A uniform toy model would make stage 3's
per-layer gather look tidier than it really is.

### How the sharding is done

ZeRO splits state into `N` equal pieces. There are two ways to choose the pieces:

- **One flat buffer for the whole model, sliced into `N`** — DeepSpeed's approach. Simple,
  exactly `1/N`, but a rank's slice spans arbitrary fragments of several layers.
- **Each layer's buffer sliced into `N`** — closer to FSDP2, which shards each parameter
  along its first dimension. A rank's slice of layer `L` is well defined, which is what makes
  stage 3's *per-layer* gather-use-discard cycle expressible.

This notebook shards **per group**, where a group is one top-level module (`wte`, `wpe`, each
of the four blocks, `ln_f`). Both give exactly `1/N` of the state per rank, so the memory
arithmetic is identical; per-group is chosen because stage 3 needs it.

One convenience worth flagging rather than hiding: every group in this model happens to
divide evenly by 32, so no padding is needed. Real implementations pad the last shard, which
costs a little memory and complicates nothing conceptually. The cell below asserts the
divisibility rather than assuming it.
"""

# %%
### 4a. The model — nanoGPT, copied unchanged from S10/S11's notebook_src.py
@dataclass
class GPTConfig:
    vocab_size: int = 65
    n_embd: int = 128
    n_layer: int = 4
    n_head: int = 4
    seq_len: int = 128
    batch_size: int = 8
    dropout: float = 0.0


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


CFG = GPTConfig()
SEED = 1337
model = build_model(CFG, SEED)
N_PARAMS = sum(p.numel() for p in model.parameters())
print(f"nanoGPT: n_embd={CFG.n_embd} n_layer={CFG.n_layer} n_head={CFG.n_head} "
      f"seq_len={CFG.seq_len}")
print(f"parameters: {N_PARAMS:,} across {len(list(model.parameters()))} tensors "
      f"(wte/lm_head tied, counted once)")

# %%
### 4b. The shard map — which rank owns what
def group_key(param_name):
    """Top-level module a parameter belongs to."""
    parts = param_name.split(".")
    return f"blocks.{parts[1]}" if parts[0] == "blocks" else parts[0]


def build_groups(model):
    """Ordered {group_name: [(param_name, shape, numel), ...]}."""
    groups = {}
    for name, p in model.named_parameters():
        groups.setdefault(group_key(name), []).append((name, tuple(p.shape), p.numel()))
    return groups


GROUPS = build_groups(model)
GROUP_NUMEL = {g: sum(n for _, _, n in members) for g, members in GROUPS.items()}

print(f"{'group':<12} {'tensors':>8} {'parameters':>12} {'% of model':>11} "
      f"{'per-rank shard':>15}")
for g, members in GROUPS.items():
    n = GROUP_NUMEL[g]
    print(f"{g:<12} {len(members):>8} {n:>12,} {n / N_PARAMS:>10.1%} "
          f"{n // WORLD_SIZE:>15,}")
print(f"{'TOTAL':<12} {sum(len(m) for m in GROUPS.values()):>8} {N_PARAMS:>12,} "
      f"{1.0:>10.1%} {N_PARAMS // WORLD_SIZE:>15,}")

# Every group divides evenly at this world size, so no shard padding is needed here.
for g, n in GROUP_NUMEL.items():
    assert n % WORLD_SIZE == 0, f"group {g} has {n} params, not divisible by {WORLD_SIZE}"
assert sum(GROUP_NUMEL.values()) == N_PARAMS
SHARD_NUMEL = N_PARAMS // WORLD_SIZE
print(f"\nevery group divides evenly by {WORLD_SIZE} — no padding needed")
print(f"each rank owns {SHARD_NUMEL:,} of {N_PARAMS:,} parameters "
      f"({1 / WORLD_SIZE:.3%})")

LARGEST_GROUP = max(GROUP_NUMEL, key=GROUP_NUMEL.get)
print(f"largest group: {LARGEST_GROUP} at {GROUP_NUMEL[LARGEST_GROUP]:,} parameters "
      f"— this sets stage 3's transient peak")

RESULTS["section4"] = {
    "config": {"n_embd": CFG.n_embd, "n_layer": CFG.n_layer, "n_head": CFG.n_head,
               "seq_len": CFG.seq_len, "vocab_size": CFG.vocab_size},
    "n_params": N_PARAMS,
    "n_tensors": len(list(model.parameters())),
    "seed": SEED,
    "groups": {g: GROUP_NUMEL[g] for g in GROUPS},
    "shard_numel": SHARD_NUMEL,
    "largest_group": LARGEST_GROUP,
    "largest_group_numel": GROUP_NUMEL[LARGEST_GROUP],
}

# %%
### 4c. Flat-state helpers — move between the module and per-group flat vectors
def model_to_groups(model):
    """Current parameters as {group: flat fp32 tensor}."""
    out = {}
    named = dict(model.named_parameters())
    for g, members in GROUPS.items():
        out[g] = torch.cat([named[n].detach().reshape(-1) for n, _, _ in members]).clone()
    return out


def groups_to_model(model, gstate):
    """Write {group: flat tensor} back into the module's parameters."""
    named = dict(model.named_parameters())
    with torch.no_grad():
        for g, members in GROUPS.items():
            off = 0
            flat = gstate[g]
            for n, shape, numel in members:
                named[n].copy_(flat[off:off + numel].view(shape))
                off += numel


def grads_to_groups(model):
    """Current .grad as {group: flat fp32 tensor}."""
    out = {}
    named = dict(model.named_parameters())
    for g, members in GROUPS.items():
        out[g] = torch.cat([named[n].grad.reshape(-1) for n, _, _ in members]).clone()
    return out


def zeros_like_groups(scale=1):
    """A per-group state dict of zeros, optionally only 1/scale of each group (a shard)."""
    return {g: torch.zeros(GROUP_NUMEL[g] // scale) for g in GROUPS}


# Round-trip check: the helpers must not perturb a single bit.
_before = model_to_groups(model)
groups_to_model(model, _before)
_after = model_to_groups(model)
assert all(torch.equal(_before[g], _after[g]) for g in GROUPS)
print("group <-> model round-trip is bit-exact")

# %% [markdown]
"""
## 5. Data parallelism — the baseline

Data parallelism is the arrangement every ZeRO stage is measured against, so it gets built
first and in full. Each of the 32 virtual GPUs holds a **complete copy** of the model and
reads a **different** micro-batch. Each produces its own gradients, which differ because each
saw different text. An all-reduce averages them, every rank applies the same average to its
own copy, and — having started identical and applied an identical update — the copies remain
identical.

The lesson's §3 widget reports exactly one number to demonstrate this: *the largest
difference between any two weight copies, which stays at 0.000 while averaging is on.* That
is the check worth reproducing, because it is the property that makes the whole arrangement
correct rather than merely parallel.

### What the run is

- 32 ranks, micro-batch 1, sequence length 128 → **global batch 32 sequences, 4,096 tokens
  per step**, which is `sequences per GPU x GPUs x accumulation steps` from §3 with
  accumulation set to 1.
- Character-level tinyshakespeare, the same corpus S10 and S11 used.
- Adam, implemented directly on flat vectors so the identical code path serves both the
  whole-model update (data parallelism) and the per-shard update (every ZeRO stage). That is
  not a shortcut — it is the reason §10's equivalence check can be exact rather than
  approximate.

### One thing being simulated, stated plainly

The arithmetic below runs in fp32. The lesson's memory model — bf16 weights and gradients
with an fp32 master copy — is *accounted* in the ledger, not executed. Running the model in
genuine bf16 would add rounding noise that varies with reduction order and would wreck §10's
exact-equivalence check for no gain, since the assignment asks how memory and communication
change, and both are counted, not inferred from the dtype. Cell 1b already demonstrated the
one thing the dtype choice actually matters for.
"""

# %%
### 5a. Data — character-level tinyshakespeare, same corpus as S10/S11
import urllib.request

DATA_PATH = ASSETS / "tinyshakespeare.txt"
if not DATA_PATH.exists():
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        DATA_PATH,
    )
text = DATA_PATH.read_text(encoding="utf-8")
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
STREAM = torch.tensor([stoi[c] for c in text], dtype=torch.long)
assert len(chars) == CFG.vocab_size, (len(chars), CFG.vocab_size)
print(f"corpus: {len(text):,} characters, vocab V={len(chars)}")

# %%
### 5b. The batch schedule — generated once, replayed by every arrangement
# Section 10 compares the four arrangements' loss curves. That comparison is only meaningful
# if they consume byte-for-byte the same data in the same order, so the schedule is built
# once here rather than drawn inside each run.
MICRO_BATCH = 1
STEPS = 40
LR = 1e-3
BETAS = (0.9, 0.999)
EPS = 1e-8

_gen = torch.Generator().manual_seed(4242)
BATCHES = []
for _ in range(STEPS):
    per_rank = []
    for _ in range(WORLD_SIZE):
        ix = torch.randint(len(STREAM) - CFG.seq_len - 1, (MICRO_BATCH,), generator=_gen)
        per_rank.append(torch.stack([STREAM[i:i + CFG.seq_len + 1] for i in ix]))
    BATCHES.append(per_rank)

global_batch = MICRO_BATCH * WORLD_SIZE
print(f"schedule: {STEPS} steps x {WORLD_SIZE} ranks x micro-batch {MICRO_BATCH}")
print(f"global batch = {MICRO_BATCH} x {WORLD_SIZE} x 1 accumulation = "
      f"{global_batch} sequences = {global_batch * CFG.seq_len:,} tokens per step")

# %%
### 5c. Adam on a flat vector — the same code path for a whole model or one shard
def adam_(master, grad, m, v, t, lr=LR, betas=BETAS, eps=EPS):
    """In-place Adam. Every operation is elementwise, which is exactly why a shard update
    and a whole-model update produce bit-identical results on the elements they share."""
    b1, b2 = betas
    m.mul_(b1).add_(grad, alpha=1 - b1)
    v.mul_(b2).addcmul_(grad, grad, value=1 - b2)
    m_hat = m / (1 - b1 ** t)
    v_hat = v / (1 - b2 ** t)
    master.addcdiv_(m_hat, v_hat.sqrt_().add_(eps), value=-lr)


# %%
### 5d. The arrangement runner — data parallelism and all three ZeRO stages
STAGES = ("dp", "zero1", "zero2", "zero3")
STAGE_LABEL = {"dp": "data parallel", "zero1": "ZeRO-1", "zero2": "ZeRO-2", "zero3": "ZeRO-3"}


def run_arrangement(stage, steps=STEPS, verbose=True):
    """Train nanoGPT across WORLD_SIZE virtual GPUs under one arrangement.

    The three booleans below are the whole difference between the four arrangements. Every
    other line of this function is shared, which is the point: ZeRO changes where state
    lives, never what is computed.
    """
    assert stage in STAGES
    shard_opt = stage in ("zero1", "zero2", "zero3")   # stage 1 shards the 12 optimizer bytes
    shard_grad = stage in ("zero2", "zero3")           # stage 2 shards the 2 gradient bytes
    shard_weight = stage == "zero3"                    # stage 3 shards the 2 weight bytes

    N = WORLD_SIZE
    fab = Fabric(N)
    net = build_model(CFG, SEED)          # identical init for every arrangement
    init = model_to_groups(net)

    def shard(g, r):
        return init[g].chunk(N)[r].clone()

    # --- per-rank state, sized exactly as the arrangement dictates -------------------
    if shard_weight:
        W = [{g: shard(g, r) for g in GROUPS} for r in range(N)]
    else:
        W = [{g: init[g].clone() for g in GROUPS} for r in range(N)]

    if shard_opt:
        MASTER = [{g: shard(g, r) for g in GROUPS} for r in range(N)]
        M = [zeros_like_groups(N) for _ in range(N)]
        V = [zeros_like_groups(N) for _ in range(N)]
    else:
        MASTER = [{g: init[g].clone() for g in GROUPS} for r in range(N)]
        M = [zeros_like_groups(1) for _ in range(N)]
        V = [zeros_like_groups(1) for _ in range(N)]

    losses, comm_per_step, wall_per_step = [], [], []
    max_replica_gap = 0.0

    for t in range(1, steps + 1):
        fab.reset_counters()
        t0 = time.perf_counter()

        # --- stage 3 gathers each layer's weights before the forward pass -------------
        if shard_weight:
            full = {}
            for g in GROUPS:
                full[g] = fab.all_gather([W[r][g] for r in range(N)])[0]

        # --- every rank runs its own micro-batch --------------------------------------
        grads, step_loss = [], 0.0
        for r in range(N):
            groups_to_model(net, full if shard_weight else W[r])
            net.zero_grad(set_to_none=False)
            tokens = BATCHES[t - 1][r]
            x, y = tokens[:, :-1], tokens[:, 1:]
            loss = F.cross_entropy(net(x).reshape(-1, CFG.vocab_size), y.reshape(-1))
            loss.backward()
            grads.append(grads_to_groups(net))
            step_loss += loss.item()
        losses.append(step_loss / N)

        # --- stage 3 gathers them again for the backward pass -------------------------
        # The forward discarded each layer as it finished with it, so the backward pass has
        # to fetch them a second time. This is the entire reason stage 3 costs 3P not 2P.
        if shard_weight:
            for g in GROUPS:
                fab.all_gather([W[r][g] for r in range(N)])

        # --- reduce the gradients and update ------------------------------------------
        if stage == "dp":
            # Every rank ends with the full averaged gradient and updates its whole copy.
            for g in GROUPS:
                avg = fab.all_reduce([grads[r][g] for r in range(N)])[0]
                for r in range(N):
                    adam_(MASTER[r][g], avg, M[r][g], V[r][g], t)
                    W[r][g] = MASTER[r][g].clone()
        else:
            # Rank r ends with slice r of the averaged gradient and updates only that slice.
            for g in GROUPS:
                sl = fab.reduce_scatter([grads[r][g] for r in range(N)])
                for r in range(N):
                    adam_(MASTER[r][g], sl[r], M[r][g], V[r][g], t)
            if shard_weight:
                # No gather here: the next forward pass gathers what it needs, when it needs it.
                for r in range(N):
                    for g in GROUPS:
                        W[r][g] = MASTER[r][g].clone()
            else:
                # Stages 1 and 2 keep full weights on every rank, so the updated slices are
                # shared out. This all-gather is the second half of the all-reduce that data
                # parallelism was already paying for — which is why stages 1 and 2 are free.
                for g in GROUPS:
                    full_w = fab.all_gather([MASTER[r][g] for r in range(N)])[0]
                    for r in range(N):
                        W[r][g] = full_w.clone()

        wall_per_step.append(time.perf_counter() - t0)
        comm_per_step.append(fab.bytes_per_rank())

        # --- §3's readout: do the copies stay identical? ------------------------------
        if not shard_weight:
            for g in GROUPS:
                gap = max(float((W[0][g] - W[r][g]).abs().max()) for r in range(1, N))
                max_replica_gap = max(max_replica_gap, gap)

    # --- memory ledger, filled from the tensors that actually exist ------------------
    led = MemoryLedger(0)
    led.hold("weight", sum(W[0][g].numel() for g in GROUPS), BYTES["weight"])
    led.hold("grad", N_PARAMS // N if shard_grad else N_PARAMS, BYTES["grad"])
    led.hold("master", sum(MASTER[0][g].numel() for g in GROUPS), BYTES["master"])
    led.hold("m", sum(M[0][g].numel() for g in GROUPS), BYTES["m"])
    led.hold("v", sum(V[0][g].numel() for g in GROUPS), BYTES["v"])
    if shard_weight:
        led.set_transient(GROUP_NUMEL[LARGEST_GROUP] * BYTES["weight"])

    P = N_PARAMS * fab.elem_bytes
    out = {
        "stage": stage,
        "label": STAGE_LABEL[stage],
        "losses": losses,
        "final_loss": losses[-1],
        "bytes_per_weight": led.steady_state / N_PARAMS,
        "ledger": dict(led.bytes),
        "steady_state_bytes": led.steady_state,
        "transient_bytes": led.bytes["transient"],
        "comm_bytes_per_step": comm_per_step[-1],
        "comm_multiple_of_P": comm_per_step[-1] / P,
        "wall_seconds_total": sum(wall_per_step),
        "wall_seconds_per_step": sum(wall_per_step) / len(wall_per_step),
        "max_replica_gap": max_replica_gap,
        # The whole model's final weights, reassembled if they were sharded, so §10 can
        # compare the four arrangements parameter by parameter.
        "weights_final": {
            g: (torch.cat([W[r][g] for r in range(N)]) if shard_weight else W[0][g].clone())
            for g in GROUPS
        },
    }
    if verbose:
        print(f"{STAGE_LABEL[stage]:<15} "
              f"loss {losses[0]:.4f} -> {losses[-1]:.4f}   "
              f"{out['bytes_per_weight']:>7.4f} bytes/weight (accounted)   "
              f"{out['comm_multiple_of_P']:.4f}P per step   "
              f"{out['wall_seconds_per_step']:.2f}s/step")
    return out


# %%
### 5e. Run data parallelism
t_start = time.perf_counter()
runs = {}
runs["dp"] = run_arrangement("dp")

print()
print(f"largest difference between any two of the {WORLD_SIZE} weight copies, over all "
      f"{STEPS} steps: {runs['dp']['max_replica_gap']:.6f}")
assert runs["dp"]["max_replica_gap"] == 0.0
print("The copies are bit-identical, which is what makes averaging the gradients correct.")
print("(§3's widget reports the same quantity and the same 0.000.)")

# %% [markdown]
"""
## 6. ZeRO-1 — shard the optimizer state

Look at what data parallelism just stored. Every one of the 32 ranks holds all 16 bytes for
every weight, and §5 just proved those copies are *bit-identical*. So 32 ranks spent 32x the
memory to store exactly one model's worth of information, and during the update all 32 did
the same arithmetic on the same numbers and got the same answer. Thirty-one of them were
redundant.

ZeRO-1 removes the redundancy from the largest block: the **optimizer state**, which is the
fp32 master copy plus Adam's `m` and `v` — **12 of the 16 bytes**. Rank `r` keeps only slice
`r`, updates only slice `r`, and the updated slices are shared out so every rank has current
weights again.

**Why this is free.** Data parallelism performs an all-reduce, and §4 established that an
all-reduce *is* a reduce-scatter followed by an all-gather. ZeRO-1 performs those same two
phases; it simply keeps the intermediate slice instead of throwing it away. Same two phases,
same bytes, one twelfth the optimizer memory. The byte counter below is the check.

Gradients stay replicated in full — a rank still materialises every gradient during the
backward pass and holds it until the reduce-scatter. That is why stage 1 is `4 + 12/N` and
not `2 + 14/N`, and it is the reason stage 1 alone is not enough for a large model.
"""

# %%
### 6a. Run ZeRO-1
runs["zero1"] = run_arrangement("zero1")

print()
print(f"{'':<22} {'data parallel':>16} {'ZeRO-1':>16}")
print(f"{'bytes/weight':<22} {runs['dp']['bytes_per_weight']:>16.4f} "
      f"{runs['zero1']['bytes_per_weight']:>16.4f}")
print(f"{'communication':<22} {runs['dp']['comm_multiple_of_P']:>15.4f}P "
      f"{runs['zero1']['comm_multiple_of_P']:>15.4f}P")
print(f"\nmemory: {runs['dp']['bytes_per_weight'] / runs['zero1']['bytes_per_weight']:.2f}x less  "
      f"communication: unchanged to "
      f"{abs(runs['dp']['comm_multiple_of_P'] - runs['zero1']['comm_multiple_of_P']):.1e}P")
assert abs(runs["dp"]["comm_multiple_of_P"] - runs["zero1"]["comm_multiple_of_P"]) < 1e-9

# %% [markdown]
"""
## 7. ZeRO-2 — shard the gradients too

Stage 1 left one duplication in place. A rank only ever updates slice `r` of the weights, so
it only ever *needs* slice `r` of the gradient — yet it is holding all of them. Stage 2
discards each gradient as soon as it has been sent where it is needed, taking the gradient
from 2 bytes per weight down to `2/N`.

**This is also free**, for the same reason: the reduce-scatter that sends each gradient slice
to its owner is already happening. Stage 2 just stops keeping what it sent.

**One honest caveat about the accounting.** The steady-state figure below is `2/N` bytes of
gradient per weight, which is what the lesson tabulates. A real implementation still produces
full-sized gradients *transiently* during the backward pass and frees them bucket by bucket
as each bucket is reduced — so the true peak sits slightly above the steady-state number, by
roughly one bucket. §13 is where bucket size gets examined; the lesson's table, and this
notebook's, report steady state.
"""

# %%
### 7a. Run ZeRO-2
runs["zero2"] = run_arrangement("zero2")

print()
print(f"{'':<22} {'ZeRO-1':>16} {'ZeRO-2':>16}")
print(f"{'gradient bytes/weight':<22} {runs['zero1']['ledger']['grad'] / N_PARAMS:>16.4f} "
      f"{runs['zero2']['ledger']['grad'] / N_PARAMS:>16.4f}")
print(f"{'total bytes/weight':<22} {runs['zero1']['bytes_per_weight']:>16.4f} "
      f"{runs['zero2']['bytes_per_weight']:>16.4f}")
print(f"{'communication':<22} {runs['zero1']['comm_multiple_of_P']:>15.4f}P "
      f"{runs['zero2']['comm_multiple_of_P']:>15.4f}P")
assert abs(runs["zero1"]["comm_multiple_of_P"] - runs["zero2"]["comm_multiple_of_P"]) < 1e-9
print("\nStill the same two phases, still the same bytes on the wire.")

# %% [markdown]
"""
## 8. ZeRO-3 — shard the weights too

The last duplicated thing is the weights themselves. Stage 3 gives rank `r` only slice `r`,
which takes every one of the 16 bytes down to `16/N` — full zero redundancy.

The cost is that a rank no longer has the weights it needs to compute with. When the forward
pass reaches a layer, the ranks **collect that layer's weights from each other, use them, and
discard them immediately**. Then the backward pass needs them again, and has to fetch them a
second time.

That is where the third `P` comes from, and the runner makes it explicit — there are three
charged phases per step rather than two:

1. `all_gather` the weights, layer by layer, for the forward pass
2. `all_gather` them again for the backward pass
3. `reduce_scatter` the gradients

and notably **no fourth phase**: stage 3 never gathers the updated weights at the end of the
step, because the next step's forward pass is going to gather them anyway.

**Transient memory.** "Discard them immediately" means peak memory is the steady-state
`16/N` plus whatever is gathered at that instant. This notebook gathers one group at a time,
so the transient peak is the largest group — one transformer block, 197,120 parameters. That
is reported separately below rather than folded into the bytes-per-weight figure, because the
lesson's table is a steady-state table and mixing the two would make the comparison
meaningless.
"""

# %%
### 8a. Run ZeRO-3
runs["zero3"] = run_arrangement("zero3")

z3 = runs["zero3"]
print()
print(f"steady state       : {z3['bytes_per_weight']:.4f} bytes/weight "
      f"= 16/{WORLD_SIZE} exactly")
print(f"transient peak     : {z3['transient_bytes']:,} bytes "
      f"({LARGEST_GROUP}, {GROUP_NUMEL[LARGEST_GROUP]:,} params x {BYTES['weight']} bytes)")
print(f"  as bytes/weight  : {z3['transient_bytes'] / N_PARAMS:.4f} on top of the steady state")
print(f"communication      : {z3['comm_multiple_of_P']:.4f}P, against "
      f"{runs['zero2']['comm_multiple_of_P']:.4f}P for stage 2")
print(f"  ratio            : {z3['comm_multiple_of_P'] / runs['zero2']['comm_multiple_of_P']:.3f}x "
      f"— the lesson's 'half as much again'")

# %% [markdown]
"""
## 9. The comparison, against the lesson's published table

This is the section the assignment is really asking for, and the one where a simulator has to
prove it is not just generating plausible-looking numbers.

Two closed forms, read off the mechanism rather than fitted to anything:

```
bytes per weight                      communication per step
  data parallel : 2 + 2 + 12            data parallel : 2P
  ZeRO-1        : 2 + 2 + 12/N          ZeRO-1        : 2P
  ZeRO-2        : 2 + (2 + 12)/N        ZeRO-2        : 2P
  ZeRO-3        : (2 + 2 + 12)/N        ZeRO-3        : 3P
```

They get checked in **three independent directions**:

1. against what the simulator actually measured at N=32 (below),
2. against the lesson's §6 table, which is published at **N=8** — a world size this notebook
   never ran, so agreement there is not something the code could have been tuned into,
3. against the lesson's §7 memory ladder at 30B parameters (§12).
"""

# %%
### 9a. The closed forms
def bytes_per_weight(stage, N):
    """Per-GPU bytes per weight, steady state, read off the mechanism."""
    if stage == "dp":
        return B_WEIGHT + B_GRAD + B_OPT
    if stage == "zero1":
        return B_WEIGHT + B_GRAD + B_OPT / N
    if stage == "zero2":
        return B_WEIGHT + (B_GRAD + B_OPT) / N
    if stage == "zero3":
        return (B_WEIGHT + B_GRAD + B_OPT) / N
    raise ValueError(stage)


def comm_multiple_of_P(stage, N, exact=True):
    """Per-GPU traffic per step, in multiples of P. `exact` includes the ring's (N-1)/N."""
    phases = 3 if stage == "zero3" else 2
    return phases * ((N - 1) / N if exact else 1.0)


# The lesson's own published numbers, typed in from resources/s12-session.md.
LESSON_BPW_AT_8 = {"dp": 16.00, "zero1": 5.50, "zero2": 3.75, "zero3": 2.00}
LESSON_COMM_P = {"dp": 2, "zero1": 2, "zero2": 2, "zero3": 3}

# %%
### 9b. Measured vs closed form, at the assignment's world size of 32
print(f"world size = {WORLD_SIZE}, model = {N_PARAMS:,} parameters\n")
print(f"{'arrangement':<15} {'measured':>10} {'formula':>10} {'diff':>10} "
      f"{'comm measured':>15} {'comm formula':>14}")
bpw_rows = {}
for s in STAGES:
    measured = runs[s]["bytes_per_weight"]
    formula = bytes_per_weight(s, WORLD_SIZE)
    cm = runs[s]["comm_multiple_of_P"]
    cf = comm_multiple_of_P(s, WORLD_SIZE)
    bpw_rows[s] = (measured, formula)
    print(f"{STAGE_LABEL[s]:<15} {measured:>10.4f} {formula:>10.4f} "
          f"{abs(measured - formula):>10.1e} {cm:>14.4f}P {cf:>13.4f}P")
    assert abs(measured - formula) < 1e-9, (s, measured, formula)
    assert abs(cm - cf) < 1e-9, (s, cm, cf)

print("\nEvery measured value matches the closed form to floating-point exactness.")

# %%
### 9c. The same closed form evaluated at N=8, against the lesson's published §6 table
# This is the check that matters most: N=8 is a world size this notebook never ran, so
# agreement here cannot have been engineered by fitting the simulator to its own output.
print(f"{'arrangement':<15} {'formula @ N=8':>14} {'lesson §6':>11} {'match':>7} "
      f"{'comm':>7} {'lesson':>8}")
for s in STAGES:
    f8 = bytes_per_weight(s, 8)
    lesson = LESSON_BPW_AT_8[s]
    c8 = comm_multiple_of_P(s, 8, exact=False)
    print(f"{STAGE_LABEL[s]:<15} {f8:>14.2f} {lesson:>11.2f} "
          f"{'yes' if abs(f8 - lesson) < 5e-3 else 'NO':>7} {c8:>6.0f}P {LESSON_COMM_P[s]:>7}P")
    assert abs(f8 - lesson) < 5e-3, (s, f8, lesson)
    assert abs(c8 - LESSON_COMM_P[s]) < 1e-9, (s, c8, LESSON_COMM_P[s])

print("\nThe mechanism reproduces the lesson's §6 table exactly, at a world size never run.")

RESULTS["section9"] = {
    "world_size": WORLD_SIZE,
    "n_params": N_PARAMS,
    "steps": STEPS,
    "micro_batch": MICRO_BATCH,
    "global_batch_sequences": global_batch,
    "global_batch_tokens": global_batch * CFG.seq_len,
    "lr": LR,
    "measured": {s: {
        "bytes_per_weight": runs[s]["bytes_per_weight"],
        "ledger": runs[s]["ledger"],
        "comm_multiple_of_P": runs[s]["comm_multiple_of_P"],
        "comm_bytes_per_step": runs[s]["comm_bytes_per_step"],
        "transient_bytes": runs[s]["transient_bytes"],
        "final_loss": runs[s]["final_loss"],
        "wall_seconds_per_step": runs[s]["wall_seconds_per_step"],
        "max_replica_gap": runs[s]["max_replica_gap"],
        "vs_dp": runs["dp"]["bytes_per_weight"] / runs[s]["bytes_per_weight"],
    } for s in STAGES},
    "formula_at_32": {s: bytes_per_weight(s, 32) for s in STAGES},
    "formula_at_8": {s: bytes_per_weight(s, 8) for s in STAGES},
    "lesson_bpw_at_8": LESSON_BPW_AT_8,
    "lesson_comm_P": LESSON_COMM_P,
}

# %% [markdown]
"""
## 10. Do all four learn the same thing?

Everything so far has compared *accounting*. This section compares the thing that actually
matters, and it is the strictest reading of the instructor's own instruction to *"make sure
that it matches what ZeRO does."*

ZeRO is a claim about **storage**, not about mathematics. Rearranging where the optimizer
state lives must not change a single number the model learns. If the four arrangements
produce different loss curves, the implementation is wrong no matter how well the byte
counts line up — a simulator that gets the memory table right and the model wrong has
demonstrated nothing.

The bar here is deliberately set at **bit-identical**, not "close". That is achievable
because of a specific design choice in §2: `reduce_scatter` computes exactly the same
reduction as `all_reduce` and then slices it, so rank `r` receives bit-for-bit the same
gradient values data parallelism would have used for those elements. Adam is elementwise, so
updating a slice and updating the whole produce identical results on the elements they share.

Anything less than bit-identical here would mean a real difference had crept in — so the
assertion is written at that level rather than at a tolerance that could hide one.
"""

# %%
### 10a. Loss curves and final weights, compared against data parallelism
base_losses = runs["dp"]["losses"]
base_weights = runs["dp"]["weights_final"]

print(f"{'arrangement':<15} {'first loss':>11} {'final loss':>11} "
      f"{'max |Δloss| vs DP':>19} {'max |Δweight| vs DP':>21}")
equiv = {}
for s in STAGES:
    dl = max(abs(a - b) for a, b in zip(base_losses, runs[s]["losses"]))
    dw = max(float((base_weights[g] - runs[s]["weights_final"][g]).abs().max()) for g in GROUPS)
    equiv[s] = {"max_loss_diff": dl, "max_weight_diff": dw}
    print(f"{STAGE_LABEL[s]:<15} {runs[s]['losses'][0]:>11.4f} {runs[s]['final_loss']:>11.4f} "
          f"{dl:>19.1e} {dw:>21.1e}")

for s in STAGES:
    assert equiv[s]["max_loss_diff"] == 0.0, (s, equiv[s])
    assert equiv[s]["max_weight_diff"] == 0.0, (s, equiv[s])

removed = 1 - runs["zero3"]["bytes_per_weight"] / runs["dp"]["bytes_per_weight"]
print(f"\nAll four arrangements are bit-identical over {STEPS} steps — same losses, same")
print(f"weights, every parameter. ZeRO-3 moved {removed:.1%} of the training state off each")
print("GPU and changed nothing about what the model learned. That is the property being")
print("claimed, verified.")

RESULTS["section10"] = {
    "steps": STEPS,
    "losses": {s: runs[s]["losses"] for s in STAGES},
    "equivalence": equiv,
    "bit_identical": all(equiv[s]["max_loss_diff"] == 0.0 and equiv[s]["max_weight_diff"] == 0.0
                         for s in STAGES),
    "state_removed_fraction": 1 - runs["zero3"]["bytes_per_weight"] / runs["dp"]["bytes_per_weight"],
}

# %%
### 10b. The four curves on one axis, and the memory each one needed
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

styles = {"dp": ("-", 3.0, 0.35), "zero1": ("--", 2.0, 1.0),
          "zero2": ("-.", 1.8, 1.0), "zero3": (":", 1.8, 1.0)}
for s in STAGES:
    ls, lw, alpha = styles[s]
    axes[0].plot(range(1, STEPS + 1), runs[s]["losses"], ls, linewidth=lw, alpha=alpha,
                 label=STAGE_LABEL[s])
axes[0].set_xlabel("step")
axes[0].set_ylabel("loss")
axes[0].set_title(f"All four arrangements, {WORLD_SIZE} virtual GPUs\n"
                  f"(curves coincide exactly — that is the result)")
axes[0].legend()
axes[0].grid(alpha=0.3)

labels = [STAGE_LABEL[s] for s in STAGES]
vals = [runs[s]["bytes_per_weight"] for s in STAGES]
bars = axes[1].bar(labels, vals, color=["#c44", "#d84", "#4a8", "#48c"])
for b, v in zip(bars, vals):
    axes[1].text(b.get_x() + b.get_width() / 2, v, f"{v:.4f}", ha="center", va="bottom")
axes[1].set_ylabel("bytes per weight, per GPU (accounted)")
axes[1].set_title(f"Training state per GPU at N={WORLD_SIZE}\n"
                  f"(identical mathematics, {vals[0] / vals[-1]:.0f}x less memory)")
axes[1].grid(alpha=0.3, axis="y")

fig.tight_layout()
fig.savefig(ASSETS / "stages_loss_and_memory.png", dpi=130)
plt.close(fig)
print(f"saved {ASSETS / 'stages_loss_and_memory.png'}")
RESULTS["section10"]["plot"] = "assets/stages_loss_and_memory.png"

# %% [markdown]
"""
## 11. The memory wall

The simulator has been checked at N=32 against the mechanism, and at N=8 against the lesson's
table. The mechanism can now be asked a question the simulator was never run for: **at what
world size does each arrangement actually fit on a card?**

The answer contains the single most useful fact in this session, and it is one that a table
of four numbers hides:

> **Data parallelism and ZeRO-1 never fit. Not at 64 GPUs, not at 1,024, not ever.**

Both leave the weights *and* the gradients replicated on every card — `2 + 2 = 4` bytes per
weight — and 4 bytes across 30 billion weights is 111.8 GiB regardless of how many cards
there are. Adding GPUs divides the sharded part and does nothing at all to the replicated
part, so those two lines approach a floor above the 74.5 GiB rule and stay there.

That floor also says exactly where the boundary sits: 4 bytes per weight fills a 74.5 GiB
card at precisely 20 billion parameters. A 20B model is on the line; V5's 30B is past it.
This is why the session concludes the choice is ZeRO-2 or ZeRO-3 and nothing below.
"""

# %%
### 11a. Where each arrangement fits, and where it cannot
WORLD_SIZES = [2 ** k for k in range(11)]      # 1 .. 1024
REPLICATED_FLOOR_BPW = B_WEIGHT + B_GRAD       # 4 bytes/weight, independent of world size

def per_gpu_gib(stage, N, params=V5_PARAMS):
    return bytes_per_weight(stage, N) * params / GiB

floor_gib = REPLICATED_FLOOR_BPW * V5_PARAMS / GiB
print(f"the replicated floor: {REPLICATED_FLOOR_BPW} bytes/weight x {V5_PARAMS:.0f} "
      f"= {floor_gib:.1f} GiB, at any world size")
print(f"a card holds        : {CARD_GiB:.1f} GiB")
print(f"so DP and ZeRO-1 miss by {floor_gib - CARD_GiB:.1f} GiB no matter how many cards\n")

boundary_params = CARD_BYTES / REPLICATED_FLOOR_BPW
print(f"{REPLICATED_FLOOR_BPW} bytes/weight fills a card exactly at "
      f"{boundary_params / 1e9:.0f}B parameters — V5 at {V5_PARAMS / 1e9:.0f}B is past it\n")
assert abs(boundary_params - 20e9) < 1e6, boundary_params

dp_max_params = CARD_BYTES / BYTES_PER_WEIGHT
weight_only_floor_gib = B_WEIGHT * V5_PARAMS / GiB
print(f"at {BYTES_PER_WEIGHT} bytes/weight a card holds a model of at most "
      f"{dp_max_params / 1e9:.1f}B parameters under data parallelism")
print(f"ZeRO-2 keeps weights replicated, so it approaches a floor of {B_WEIGHT} bytes/weight "
      f"= {weight_only_floor_gib:.1f} GiB at {V5_PARAMS / 1e9:.0f}B\n")

fits_from = {}
for s in STAGES:
    ok = [N for N in WORLD_SIZES if per_gpu_gib(s, N) <= CARD_GiB]
    fits_from[s] = min(ok) if ok else None
    where = f"from {fits_from[s]} GPUs" if ok else "never, at any world size"
    print(f"{STAGE_LABEL[s]:<15} fits {where}")

assert fits_from["dp"] is None and fits_from["zero1"] is None
assert fits_from["zero2"] == 32, fits_from["zero2"]
assert fits_from["zero3"] == 8, fits_from["zero3"]
print("\nMatches the lesson's §7 conclusion: ZeRO-2 from 32 GPUs, ZeRO-3 from 8.")

# %%
### 11b. The memory wall, drawn
fig, ax = plt.subplots(figsize=(7.5, 5))
colors = {"dp": "#c44", "zero1": "#d84", "zero2": "#4a8", "zero3": "#48c"}
for s in STAGES:
    ys = [per_gpu_gib(s, N) for N in WORLD_SIZES]
    ax.plot(WORLD_SIZES, ys, marker="o", ms=3.5, color=colors[s], label=STAGE_LABEL[s])
ax.axhline(CARD_GiB, color="black", linestyle="--", linewidth=1.3)
ax.text(WORLD_SIZES[-1], CARD_GiB * 1.06, f"one 80 GB card = {CARD_GiB:.1f} GiB",
        ha="right", va="bottom", fontsize=9)
ax.axhline(floor_gib, color="#888", linestyle=":", linewidth=1.2)
ax.text(1.2, floor_gib * 1.06, f"replicated floor, {REPLICATED_FLOOR_BPW} bytes/weight "
        f"= {floor_gib:.1f} GiB", fontsize=8.5, color="#555")
ax.set_xscale("log", base=2)
ax.set_yscale("log", base=10)
ax.set_xlabel("world size (GPUs)")
ax.set_ylabel("training state per GPU (GiB)")
ax.set_title(f"The memory wall — {V5_PARAMS / 1e9:.0f}B parameters\n"
             "two of the four lines never reach the card")
ax.legend()
ax.grid(alpha=0.3, which="both")
fig.tight_layout()
fig.savefig(ASSETS / "memory_wall.png", dpi=130)
plt.close(fig)
print(f"saved {ASSETS / 'memory_wall.png'}")

RESULTS["section11"] = {
    "world_sizes": WORLD_SIZES,
    "per_gpu_gib": {s: {N: per_gpu_gib(s, N) for N in WORLD_SIZES} for s in STAGES},
    "card_gib": CARD_GiB,
    "replicated_floor_bpw": REPLICATED_FLOOR_BPW,
    "replicated_floor_gib": floor_gib,
    "boundary_params_b": boundary_params / 1e9,
    "dp_max_params_b": dp_max_params / 1e9,
    "weight_only_floor_gib": weight_only_floor_gib,
    "fits_from": fits_from,
    "plot": "assets/memory_wall.png",
}

# %% [markdown]
"""
## 12. Projecting to 30B — against the lesson's §7 ladder

The simulator trained a 813,440-parameter model. V5 is 30 billion. Those are two different
things and this notebook keeps them apart: everything above is **measured** on the toy model;
this section is a **projection**, the same accounting function evaluated at 30e9 with nothing
measured at all.

The projection is worth making because it is falsifiable. The lesson publishes the full
ladder in §7 — four arrangements at four world sizes, sixteen numbers — and the function
being evaluated here was derived from the mechanism, checked against the simulator at N=32,
and checked against §6's table at N=8. If it also reproduces all sixteen of §7's numbers,
including the twelve at world sizes nothing in this notebook ever touched, the accounting is
right.
"""

# %%
### 12a. The full ladder, projected and compared
LADDER_N = [8, 16, 32, 64]
# The lesson's §7 table, typed in from resources/s12-session.md.
LESSON_LADDER = {
    "dp":    {8: 447.0, 16: 447.0, 32: 447.0, 64: 447.0},
    "zero1": {8: 153.7, 16: 132.7, 32: 122.2, 64: 117.0},
    "zero2": {8: 104.8, 16: 80.3,  32: 68.1,  64: 62.0},
    "zero3": {8: 55.9,  16: 27.9,  32: 14.0,  64: 7.0},
}

print(f"GiB per GPU, {V5_PARAMS / 1e9:.0f}B parameters   "
      f"(projected / lesson §7)\n")
header = f"{'arrangement':<15}" + "".join(f"{f'{N} GPUs':>20}" for N in LADDER_N)
print(header)
ladder_rows, max_ladder_err = {}, 0.0
for s in STAGES:
    cells, row = [], {}
    for N in LADDER_N:
        proj = per_gpu_gib(s, N)
        lesson = LESSON_LADDER[s][N]
        max_ladder_err = max(max_ladder_err, abs(proj - lesson))
        row[N] = proj
        cells.append(f"{proj:>9.1f} / {lesson:<8.1f}")
    ladder_rows[s] = row
    print(f"{STAGE_LABEL[s]:<15}" + "".join(f"{c:>20}" for c in cells))

print(f"\nlargest disagreement anywhere in the table: {max_ladder_err:.2f} GiB")
assert max_ladder_err < 0.1, max_ladder_err
print("All sixteen numbers reproduced, twelve of them at world sizes never simulated.")

# The assignment's own world size, called out.
print(f"\nAt the assignment's {WORLD_SIZE} virtual GPUs, a {V5_PARAMS / 1e9:.0f}B model would need:")
for s in STAGES:
    fits = "fits" if ladder_rows[s][WORLD_SIZE] <= CARD_GiB else "does NOT fit"
    print(f"  {STAGE_LABEL[s]:<15} {ladder_rows[s][WORLD_SIZE]:>8.1f} GiB per GPU   ({fits})")

RESULTS["section12"] = {
    "ladder_world_sizes": LADDER_N,
    "projected": ladder_rows,
    "lesson_ladder": LESSON_LADDER,
    "max_abs_error_gib": max_ladder_err,
    "at_assignment_world_size": {s: ladder_rows[s][WORLD_SIZE] for s in STAGES},
}

# %% [markdown]
"""
## 13. Bucketing and overlap

§5 measured communication as a share of step time and found it *rising* as cards get faster —
34% on 64xH100, 77% on 64xB200, for the same 120 GB. §10 of the lesson is the answer to that,
and it does not involve sending less data. It involves sending it **while the GPU is busy**.

The backward pass makes this possible because it runs **last layer first**. The last layer's
gradients are finished long before the pass reaches the first layer, so they can start their
journey immediately. Gradients are collected into buckets and a bucket is sent as soon as it
fills, part way through the pass.

### The model, stated before the numbers

- A step is compute-bound for `T` seconds; the backward pass occupies the last two thirds of
  it (a forward pass is roughly half the cost of a backward one).
- 12 layers, matching the lesson's widget.
- Layer gradients become ready evenly through the backward pass, last layer first.
- **One serial link.** A bucket that is ready while the link is busy waits its turn. This is
  the part that makes small buckets stop helping.
- Each transfer pays a fixed startup cost `α` — the lesson's "the fixed cost of starting a
  transfer is paid less often" — plus time proportional to its bytes.

The lesson gives one number to check against: at a bucket of two layers on an H100 step,
**83 percent** of the transfer finished before the backward pass ended.
"""

# %%
### 13a. The overlap model
H100_STEP_S = 7.10        # §5: 64 x H100, one million tokens
B200_STEP_S = 3.12        # §5: 64 x B200, same step
TRANSFER_2P_IB_S = 2.40   # §5: 2P = 120 GB over InfiniBand at ~50 GB/s
N_LAYERS = 12             # the lesson's widget groups the model into twelve blocks
BUCKET_CHOICES = [1, 2, 3, 4, 6, 12]


def simulate_overlap(compute_s, transfer_total_s, bucket_layers,
                     n_layers=N_LAYERS, latency_s=0.0):
    """One step. Returns how much of the transfer hid under the compute, and the step time."""
    bwd_start, bwd_end = compute_s / 3.0, compute_s
    bwd_dur = bwd_end - bwd_start
    n_buckets = math.ceil(n_layers / bucket_layers)
    per_bucket_s = transfer_total_s / n_buckets

    link_free, hidden_s = 0.0, 0.0
    for j in range(n_buckets):
        layers_done = min((j + 1) * bucket_layers, n_layers)
        ready = bwd_start + layers_done / n_layers * bwd_dur   # last layer first
        start = max(ready, link_free)                          # the link is serial
        end = start + latency_s + per_bucket_s
        link_free = end
        if end <= bwd_end + 1e-12:
            hidden_s += latency_s + per_bucket_s               # this bucket finished in time

    wire_total_s = n_buckets * (latency_s + per_bucket_s)
    return {
        "bucket_layers": bucket_layers,
        "n_buckets": n_buckets,
        "hidden_fraction": hidden_s / wire_total_s,
        "step_time_s": max(compute_s, link_free),
        "exposed_s": max(0.0, link_free - compute_s),
    }


# %%
### 13b. The lesson's 83% check — bucket of two layers on an H100 step
h100_b2 = simulate_overlap(H100_STEP_S, TRANSFER_2P_IB_S, bucket_layers=2)
print(f"H100 step {H100_STEP_S}s, transfer {TRANSFER_2P_IB_S}s, bucket = 2 layers "
      f"({h100_b2['n_buckets']} buckets)")
print(f"  share of transfer finished before the backward pass ended: "
      f"{h100_b2['hidden_fraction']:.1%}")
print(f"  lesson §10 states                                        : 83%")
assert abs(h100_b2["hidden_fraction"] - 0.83) < 0.01, h100_b2["hidden_fraction"]
print("\nThe model reproduces the lesson's figure. The reason is worth stating plainly:")
print("with 6 buckets, the last one cannot start until the backward pass is over, so")
print("exactly 5 of 6 hide — 83.3%. With 12 buckets it would be 11/12, which is why the")
print("lesson says the smallest bucket is still the best one on an H100 step.")

# %%
### 13c. Why that stops being true on faster hardware
# Without a startup cost, smaller is always better and the lesson's B200 claim cannot be
# reproduced. The startup cost is the missing term, so rather than assume a value, this
# sweeps it and finds where the lesson's claim becomes true.
print(f"{'card':<8} {'bucket':>7} {'buckets':>8} {'hidden':>8} {'exposed':>9} {'step time':>10}")
overlap_rows = {}
for card, step_s in (("H100", H100_STEP_S), ("B200", B200_STEP_S)):
    overlap_rows[card] = {}
    for b in BUCKET_CHOICES:
        r = simulate_overlap(step_s, TRANSFER_2P_IB_S, bucket_layers=b)
        overlap_rows[card][b] = r
        print(f"{card:<8} {b:>7} {r['n_buckets']:>8} {r['hidden_fraction']:>7.1%} "
              f"{r['exposed_s']:>8.2f}s {r['step_time_s']:>9.2f}s")
    print()

# The startup cost at which B200's best bucket stops being the smallest one.
alpha_threshold = None
for alpha_ms in [x / 2 for x in range(0, 201)]:
    best = min(BUCKET_CHOICES,
               key=lambda b: simulate_overlap(B200_STEP_S, TRANSFER_2P_IB_S, b,
                                              latency_s=alpha_ms / 1000)["step_time_s"])
    if best != 1:
        alpha_threshold = alpha_ms
        break

best_h100_zero = min(BUCKET_CHOICES,
                     key=lambda b: overlap_rows["H100"][b]["step_time_s"])
print(f"with no startup cost, the best bucket is {best_h100_zero} layer(s) on both cards —")
print(f"which reproduces the lesson's H100 claim but not its B200 one.")
if alpha_threshold is not None:
    best_at = min(BUCKET_CHOICES,
                  key=lambda b: simulate_overlap(B200_STEP_S, TRANSFER_2P_IB_S, b,
                                                 latency_s=alpha_threshold / 1000)["step_time_s"])
    print(f"\nadding a fixed per-transfer startup cost, B200's optimum moves off 1 layer at")
    print(f"alpha = {alpha_threshold:.1f} ms, landing on {best_at} layer(s) — the lesson's "
          f"stated B200 answer.")
    print(f"So the lesson's claim holds in this model for alpha >= {alpha_threshold:.1f} ms.")
    print("The lesson does not publish alpha, so this is the model reporting what it would")
    print("take to agree, rather than a startup cost reverse-fitted to force agreement.")

print(f"\ncommunication as a fraction of compute (§5's test, unhidden):")
for card, step_s in (("H100", H100_STEP_S), ("B200", B200_STEP_S)):
    print(f"  {card}: {TRANSFER_2P_IB_S / step_s:.0%}   "
          f"(lesson: {'34%' if card == 'H100' else '77%'})")

RESULTS["section13"] = {
    "h100_step_s": H100_STEP_S, "b200_step_s": B200_STEP_S,
    "transfer_2P_ib_s": TRANSFER_2P_IB_S, "n_layers": N_LAYERS,
    "h100_bucket2_hidden_fraction": h100_b2["hidden_fraction"],
    "lesson_hidden_fraction": 0.83,
    "sweep": {card: {b: overlap_rows[card][b] for b in BUCKET_CHOICES} for card in overlap_rows},
    "b200_alpha_threshold_ms": alpha_threshold,
    "comm_fraction_h100": TRANSFER_2P_IB_S / H100_STEP_S,
    "comm_fraction_b200": TRANSFER_2P_IB_S / B200_STEP_S,
}

# %% [markdown]
"""
## 14. Offload to CPU

A node has system memory, usually far more of it than the GPUs have. The optimizer state is
the natural thing to put there: it is 12 of the 16 bytes and it is touched exactly **once per
step**, which is the lowest access frequency of anything in the training state.

The lesson's framing is the one to keep: **offload converts a memory problem into a bandwidth
problem.** PCIe carries roughly 60 GB/s — below NVLink's 450 and comparable to a network
cable. So the question is never "is offload good", it is "is memory the binding constraint",
and the table below is what that trade looks like in numbers.

Modelled here is the more effective of the two variants the lesson describes: the update runs
on the CPU as well, so the optimizer state never moves. Only the gradient shard travels out
and the updated weight shard travels back.
"""

# %%
### 14a. ZeRO-2 with and without optimizer offload
PCIE_GBPS = 60.0

print(f"{'N':>5} {'GPU, no offload':>17} {'GPU, offloaded':>16} {'saved':>10} "
      f"{'to system RAM':>15} {'PCIe/step':>11} {'PCIe time':>11}")
offload_rows = {}
for N in LADDER_N:
    gpu_no = bytes_per_weight("zero2", N)
    gpu_off = B_WEIGHT + B_GRAD / N                 # optimizer state is gone from the GPU
    cpu_bpw = B_OPT / N
    pcie_bpw = B_GRAD / N + B_WEIGHT / N            # grad shard out, updated weight shard back
    pcie_bytes = pcie_bpw * V5_PARAMS
    row = {
        "gpu_gib_no_offload": gpu_no * V5_PARAMS / GiB,
        "gpu_gib_offloaded": gpu_off * V5_PARAMS / GiB,
        "cpu_gib": cpu_bpw * V5_PARAMS / GiB,
        "pcie_gb_per_step": pcie_bytes / 1e9,
        "pcie_seconds": pcie_bytes / 1e9 / PCIE_GBPS,
    }
    offload_rows[N] = row
    print(f"{N:>5} {row['gpu_gib_no_offload']:>14.1f} GiB {row['gpu_gib_offloaded']:>13.1f} GiB "
          f"{row['gpu_gib_no_offload'] - row['gpu_gib_offloaded']:>7.1f} GiB "
          f"{row['cpu_gib']:>12.1f} GiB {row['pcie_gb_per_step']:>8.2f} GB "
          f"{row['pcie_seconds']:>10.3f} s")

n8 = offload_rows[8]
print(f"\nThe case where it matters: at 8 GPUs, ZeRO-2 needs "
      f"{n8['gpu_gib_no_offload']:.1f} GiB and the card holds {CARD_GiB:.1f}.")
print(f"Offload brings it to {n8['gpu_gib_offloaded']:.1f} GiB, which "
      f"{'fits' if n8['gpu_gib_offloaded'] <= CARD_GiB else 'still does not fit'}, "
      f"for {n8['pcie_seconds']:.3f}s of PCIe per step.")
print(f"Against an H100 step of {H100_STEP_S}s that is "
      f"{n8['pcie_seconds'] / H100_STEP_S:.1%} of step time — cheap, if and only if")
print("memory was what was actually blocking the run.")

RESULTS["section14"] = {
    "pcie_gbps": PCIE_GBPS,
    "rows": offload_rows,
    "note": "optimizer state offloaded to system memory, update performed on the CPU",
}

# %% [markdown]
"""
## 15. Precision — what MXFP8 changes

Blackwell's arithmetic units multiply 8-bit floating point numbers directly. An 8-bit number
holds far too little detail to use naively, so MXFP8 stores a **shared scale factor** for each
block of 32 values — one 8-bit exponent per 32 numbers.

The interesting part is how *little* this changes the memory picture, and how much it changes
everything else. Moving weights and gradients to 8 bits removes 2 of the 16 bytes and adds
back the scale bytes. The fp32 master copy and the two running averages — 12 of the 16 — are
untouched, because the update arithmetic still needs the accuracy.

So the stored state falls by only about an eighth. The real gains are elsewhere: matrix
multiplication runs faster, activation memory falls, and the volume crossing the interconnect
falls with it. The lesson cites TorchTitan at up to **41 percent faster** pre-training on
B200 for a large mixture-of-experts model, with loss curves over 1,500 steps matching 16-bit.
"""

# %%
### 15a. The 8-bit memory model
MXFP8_BLOCK = 32
FP8_BYTES = {"weight": 1, "grad": 1, "master": 4, "m": 4, "v": 4}
scale_overhead = 2 / MXFP8_BLOCK        # one shared scale byte per 32 values, weights + grads
mxfp8_bpw = sum(FP8_BYTES.values()) + scale_overhead

print(f"{'':<34} {'bf16':>10} {'MXFP8':>10}")
for k in BYTES:
    print(f"{k:<34} {BYTES[k]:>10} {FP8_BYTES[k]:>10}")
print(f"{'shared scale bytes (1 per 32 values)':<34} {0:>10} {scale_overhead:>10.4f}")
print(f"{'total bytes per weight':<34} {BYTES_PER_WEIGHT:>10.4f} {mxfp8_bpw:>10.4f}")
print()
print(f"{'30B model, one full copy of state':<34} "
      f"{BYTES_PER_WEIGHT * V5_PARAMS / GiB:>9.1f}G {mxfp8_bpw * V5_PARAMS / GiB:>9.1f}G")

reduction = 1 - mxfp8_bpw / BYTES_PER_WEIGHT
print(f"\nreduction in stored state: {reduction:.1%}   (lesson §11 states 12.1%)")
assert abs(mxfp8_bpw - 14.0625) < 1e-9, mxfp8_bpw
assert abs(mxfp8_bpw * V5_PARAMS / GiB - 392.9) < 0.1
assert abs(reduction - 0.121) < 0.001, reduction

print("\nWhere the 12 untouched bytes go:")
print(f"  fp32 master copy + Adam's m and v = {B_OPT} of {BYTES_PER_WEIGHT} bytes, unchanged,")
print("  because the update arithmetic still needs the accuracy. This is why 8-bit")
print("  training does not halve memory — the optimizer, not the model, is most of it.")

RESULTS["section15"] = {
    "block_size": MXFP8_BLOCK,
    "fp8_bytes": FP8_BYTES,
    "scale_overhead_bytes_per_weight": scale_overhead,
    "mxfp8_bytes_per_weight": mxfp8_bpw,
    "bf16_bytes_per_weight": BYTES_PER_WEIGHT,
    "mxfp8_30b_gib": mxfp8_bpw * V5_PARAMS / GiB,
    "bf16_30b_gib": BYTES_PER_WEIGHT * V5_PARAMS / GiB,
    "reduction_fraction": reduction,
    "lesson_reduction": 0.121,
    "lesson_30b_gib": 392.9,
}

# %% [markdown]
"""
## 16. Pros and cons, stage by stage

The measurements are done. This is the judgement they support — what each arrangement costs,
what it buys, and when it is the right choice. Everything below is a consequence of numbers
this notebook produced, not a summary of the lesson.

### Data parallelism

**For:** the simplest thing that works, and the only one with no extra machinery at all —
one all-reduce per step and every rank can checkpoint or evaluate on its own, because every
rank has the whole model. Communication is the 2P floor; nothing beats it.

**Against:** it stores `N` bit-identical copies of the training state. §5 measured that
directly — the largest difference between any two of the 32 weight copies was exactly
0.000000 across all 40 steps, which is another way of saying 31 of the 32 copies were
redundant. At 30B it needs 447.0 GiB per card against a 74.5 GiB card, which is a factor of 6
out, and **adding GPUs does not help at all** — the line in §11 is flat.

**Use it when** the model comfortably fits, which for a 74.5 GiB card and 16 bytes per weight
means roughly 5B parameters or fewer.

### ZeRO-1

**For:** free. It shards the 12 optimizer bytes and costs nothing extra in communication —
§6 measured 1.9375P for both, identical to a tenth of a byte. That is not a coincidence, it
is §4's identity: data parallelism's all-reduce already *is* a reduce-scatter plus an
all-gather, and stage 1 keeps the intermediate slice instead of discarding it.

**Against:** it is the stage that helps least where help is most needed. It leaves weights
and gradients replicated — 4 bytes per weight — and §11 showed that floor is 111.8 GiB at 30B
**at any world size**, so ZeRO-1 never fits a large model no matter how many GPUs are bought.
Its 3.7x saving at N=32 is real but lands on the wrong side of the card.

**Use it when** the model already fits under data parallelism and the optimizer state is what
is squeezing you — a genuinely common case at small scale, and a pointless one at 30B.

### ZeRO-2

**For:** also free, for the same reason, and it breaks the 4-byte floor that traps stage 1.
Shipping each gradient to its owner and forgetting it takes gradients to `2/N`. Measured
2.4375 bytes/weight at N=32 against data parallelism's 16.0000 — **6.6x less state at
identical communication**. For a large model this is the best return on complexity in the
whole session.

**Against:** weights are still replicated at 2 bytes each, so per-GPU memory approaches a
floor of 2 bytes/weight — 55.9 GiB at 30B — and stops improving however many GPUs are added.
It needs 32 GPUs before it fits at all. The steady-state figure also understates the true
peak slightly, since a real implementation holds one bucket of full gradients transiently
during the backward pass.

**Use it when** the model fits at the world size you already have. §11 puts that at 32 GPUs
or more for 30B.

### ZeRO-3

**For:** the only arrangement with no floor. Every byte is sharded, so per-GPU memory is
`16/N` and keeps falling — 0.5000 bytes/weight at N=32, measured, a **32x reduction**. It is
the only option that fits 30B on 8 GPUs, and the only one whose memory problem can be solved
by buying more cards.

**Against:** it costs **1.5x the communication** — measured 2.9062P against stage 2's
1.9375P, exactly the "half as much again" the lesson describes, because the weights have to
be gathered for the forward pass and again for the backward. On
top of that there is transient memory: gathering one transformer block at a time added
0.4847 bytes/weight of peak on this model, which is comparable to the entire steady-state
figure and is a real cost that the ladder table does not show. It is also the arrangement
most dependent on prefetching being configured correctly, since a gather that has not arrived
stalls the layer that needs it.

**Use it when** the model does not fit any other way, or when the GPU count is fixed and
low. §11 puts that at 8 GPUs or more for 30B.

### The decision, for V5's 30B

§11 and §12 narrow it to two options and rule out two entirely. Data parallelism and ZeRO-1
never fit at any world size. ZeRO-2 fits from 32 GPUs; ZeRO-3 fits from 8. So the real
question is the one the lesson's §13 leaves open — **ZeRO-2 on 32 GPUs, or ZeRO-3 on 8** —
and the trade is now concrete: stage 3 costs 50% more traffic per step and brings transient
gather memory the table does not show, in exchange for working on a quarter of the hardware.
Settling it needs a measured step time for both on the real architecture with activation
memory included, which is a different measurement from any in this notebook.
"""

# %%
### 16a. The summary table
print(f"{'arrangement':<15} {'bytes/wt':>9} {'vs DP':>7} {'comm':>9} "
      f"{'30B @ 32':>10} {'fits?':>7} {'fits from':>10}")
for s in STAGES:
    bpw = runs[s]["bytes_per_weight"]
    gib = RESULTS["section12"]["projected"][s][WORLD_SIZE]
    ff = fits_from[s]
    print(f"{STAGE_LABEL[s]:<15} {bpw:>9.4f} "
          f"{runs['dp']['bytes_per_weight'] / bpw:>6.1f}x {runs[s]['comm_multiple_of_P']:>8.4f}P "
          f"{gib:>8.1f}G {'yes' if gib <= CARD_GiB else 'no':>7} "
          f"{(str(ff) + ' GPUs') if ff else 'never':>10}")

print(f"\nall four trained {STEPS} steps on {WORLD_SIZE} virtual GPUs to an identical "
      f"loss of {runs['dp']['final_loss']:.4f}")
print(f"largest weight difference between any two arrangements: "
      f"{max(equiv[s]['max_weight_diff'] for s in STAGES):.1e}")

# %%
### 16b. Save every number this notebook produced
RESULTS["meta"] = {
    "torch_version": torch.__version__,
    "world_size": WORLD_SIZE,
    "n_params": N_PARAMS,
    "steps": STEPS,
    "stage_labels": STAGE_LABEL,
    "total_runtime_seconds": time.perf_counter() - t_start,
}
pathlib.Path("results.json").write_text(json.dumps(RESULTS, indent=2, default=str))
n_keys = sum(len(v) if isinstance(v, dict) else 1 for v in RESULTS.values())
print(f"wrote results.json — {len(RESULTS)} sections, {n_keys} top-level keys")
print(f"total notebook runtime: {RESULTS['meta']['total_runtime_seconds'] / 60:.1f} minutes")
