#!/usr/bin/env python3
"""Train one arm of the input-path ablation and score bits-per-byte per lane.

    .venv/bin/python train_arm.py --arm fourier_2048 --tokens 40000000

Arms differ in the embedding module and in nothing else. Architecture, optimiser, schedule,
seed, sequence length, token budget and the *token stream itself* are pinned across arms, so
a difference in bits-per-byte is attributable to the input path.

Metric is bits-per-byte rather than cross-entropy per token, for the reason S5 gave: bpb is
comparable across languages, per-token loss is not -- a tokenizer that splits Hindi into more
pieces makes per-token loss look better while the model does the same work per unit of text.

    bpb = (sum of NLL in nats over scored positions) / ln(2) / (UTF-8 bytes of those targets)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from kv2.codecs import FourierCodec, KroneckerCodec, NaiveSumCodec
from kv2.data import TRAIN_MIX, VAL_LANES, MixtureSampler, load_val
from kv2.embedding import CodecEmbedding
from kv2.model import Config, Transformer
from kv2.vocab import load_vocab, token_bytes

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "runs")

# bf16 needs sm_80+; the box this runs on is a T4 (sm_75), so pick per-device rather than
# assuming. Getting this wrong does not error, it just runs slowly in emulation.
AMP_DTYPE = (torch.bfloat16 if torch.cuda.is_available()
             and torch.cuda.get_device_capability()[0] >= 8 else torch.float16)

# Every arm's input path, defined in one place so the ablation is auditable at a glance.
ARMS = {
    "dense":         {"kind": "dense"},
    "kronecker_32":  {"kind": "kron",    "pos_dim": 32},
    "kronecker_48":  {"kind": "kron",    "pos_dim": 48},
    "fourier_2048":  {"kind": "fourier", "code_dim": 2048},
    "fourier_8192":  {"kind": "fourier", "code_dim": 8192},
    "naive_2048":    {"kind": "naive",   "code_dim": 2048},
}


def build_embedding(arm: str, cfg: Config, byte_seqs: list[bytes], seed: int) -> nn.Module:
    spec = ARMS[arm]
    kind = spec["kind"]
    if kind == "dense":
        return nn.Embedding(cfg.vocab_size, cfg.d_model)
    if kind == "kron":
        codec = KroneckerCodec(pos_dim=spec["pos_dim"])
    elif kind == "fourier":
        codec = FourierCodec(n_freq=spec["code_dim"] // 2, seed=seed)
    elif kind == "naive":
        codec = NaiveSumCodec(n_freq=spec["code_dim"] // 2, seed=seed)
    else:
        raise ValueError(kind)
    return CodecEmbedding(codec, byte_seqs, cfg.d_model)


def lr_at(step, total, base_lr, warmup_frac=0.02, min_frac=0.1):
    w = max(int(total * warmup_frac), 1)
    if step < w:
        return base_lr * (step + 1) / w
    prog = (step - w) / max(total - w, 1)
    return base_lr * (min_frac + (1 - min_frac) * 0.5 * (1 + math.cos(math.pi * prog)))


@torch.no_grad()
def evaluate(model, device, cfg, bytes_per_id, max_tokens, long_ids, batch_seqs=8) -> dict:
    """Per-lane bits-per-byte, plus bpb restricted to long (>32 byte) target tokens.

    The long-token slice is the one place a 32-byte crop is structurally unable to compete:
    those targets are exactly the ones whose input representation Kronecker truncates.
    """
    model.eval()
    out = {}
    T = cfg.seq_len
    for lane in VAL_LANES:
        try:
            toks = load_val(lane)
        except FileNotFoundError:
            continue
        n_seq = min(max_tokens // T, max(len(toks) // (T + 1), 1))
        if n_seq == 0:
            continue
        nll_sum = 0.0
        byte_sum = 0.0
        long_nll = 0.0
        long_bytes = 0.0
        long_n = 0
        for s in range(0, n_seq, batch_seqs):
            k = min(batch_seqs, n_seq - s)
            rows = []
            for j in range(k):
                i = (s + j) * (T + 1)
                seg = toks[i:i + T + 1]
                if len(seg) < T + 1:
                    seg = np.pad(seg, (0, T + 1 - len(seg)))
                rows.append(seg)
            batch = torch.from_numpy(np.stack(rows)).to(device)
            x, y = batch[:, :-1], batch[:, 1:]
            with torch.amp.autocast("cuda", dtype=AMP_DTYPE,
                                    enabled=device.type == "cuda"):
                logits = model(x)
            # Loss in fp32 regardless: bits-per-byte differences between arms are small
            # enough that half-precision accumulation would be a meaningful share of them.
            nll = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                y.reshape(-1), reduction="none",
            )
            yf = y.reshape(-1)
            b = bytes_per_id[yf]
            nll_sum += float(nll.sum())
            byte_sum += float(b.sum())
            mask = long_ids[yf]
            if bool(mask.any()):
                long_nll += float(nll[mask].sum())
                long_bytes += float(b[mask].sum())
                long_n += int(mask.sum())
        out[lane] = {
            "bpb": nll_sum / math.log(2) / max(byte_sum, 1),
            "tokens": n_seq * T,
        }
        if long_n:
            out[lane]["bpb_long_tokens"] = long_nll / math.log(2) / max(long_bytes, 1)
            out[lane]["long_token_positions"] = long_n
    model.train()
    if out:
        out["macro_avg_bpb"] = float(np.mean([v["bpb"] for k, v in out.items() if isinstance(v, dict)]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--tokens", type=int, default=40_000_000)
    ap.add_argument("--seed", type=int, default=20260810)
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--n-layers", type=int, default=8)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch-seqs", type=int, default=32, help="sequences per optimiser step")
    ap.add_argument("--micro-seqs", type=int, default=8, help="sequences per forward pass")
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--eval-every-frac", type=float, default=0.25)
    ap.add_argument("--light-eval-tokens", type=int, default=200_000)
    ap.add_argument("--final-eval-tokens", type=int, default=1_000_000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed % (2**32))
    device = torch.device(args.device)

    tokens = load_vocab()
    byte_seqs = [token_bytes(t) for t in tokens]
    cfg = Config(vocab_size=len(tokens), d_model=args.d_model, n_layers=args.n_layers,
                 n_heads=args.n_heads, seq_len=args.seq_len)

    # Byte length per token id: the denominator of bits-per-byte, and the definition of
    # "long" (a token the 32-byte window cannot hold).
    bytes_per_id = torch.tensor([max(len(b), 1) for b in byte_seqs],
                                dtype=torch.float32, device=device)
    long_ids = torch.tensor([len(b) > 32 for b in byte_seqs], dtype=torch.bool, device=device)

    embedding = build_embedding(args.arm, cfg, byte_seqs, args.seed)
    model = Transformer(cfg, embedding).to(device)
    pc = model.param_counts()

    print(f"arm            {args.arm}")
    print(f"device         {device}")
    print(f"input path     {pc['input_path']:,} trainable")
    print(f"output head    {pc['output_head']:,}")
    print(f"backbone       {pc['backbone']:,}")
    print(f"total          {pc['total_trainable']:,}")
    rank_info = None
    if isinstance(embedding, CodecEmbedding):
        const = int(embedding.constant_input_rows().sum())
        rank_info = embedding.effective_rank()
        print(f"codec          {embedding.codec.describe()}")
        print(f"constant rows  {const:,} of {embedding.codec.code_dim:,}")
        print(f"effective rank {rank_info['numerical_rank']:,} of {embedding.codec.code_dim:,} "
              f"(99% energy at {rank_info['rank_99pct_energy']:,})")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=0.1, eps=1e-8)
    # Same seed for every arm => every arm sees the identical token stream.
    sampler = MixtureSampler(TRAIN_MIX, cfg.seq_len, seed=args.seed)

    tokens_per_step = args.batch_seqs * cfg.seq_len
    total_steps = max(args.tokens // tokens_per_step, 1)
    accum = max(args.batch_seqs // args.micro_seqs, 1)
    eval_every = max(int(total_steps * args.eval_every_frac), 1)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(f"steps          {total_steps:,} x {tokens_per_step:,} tok = {total_steps*tokens_per_step:,}")
    print(f"mixture        {TRAIN_MIX}")
    print("-" * 78, flush=True)

    history = []
    t0 = time.time()
    for step in range(total_steps):
        lr = lr_at(step, total_steps, args.lr)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        running = 0.0
        for _ in range(accum):
            batch = torch.from_numpy(sampler.batch(args.micro_seqs)).to(device)
            x, y = batch[:, :-1], batch[:, 1:]
            # fp16 not bf16: this runs on a T4 (sm_75), which has no native bf16 support.
            with torch.amp.autocast("cuda", dtype=AMP_DTYPE, enabled=use_amp):
                logits = model(x)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                                       y.reshape(-1))
            scaler.scale(loss / accum).backward()
            running += float(loss) / accum
        scaler.unscale_(opt)
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()

        if step % 25 == 0 or step == total_steps - 1:
            el = time.time() - t0
            done = (step + 1) * tokens_per_step
            print(f"  step {step+1:>5}/{total_steps}  loss {running:.4f}  "
                  f"lr {lr:.2e}  gnorm {float(gnorm):.2f}  "
                  f"{done/1e6:.1f}M tok  {el/60:.1f}m", flush=True)
        if (step + 1) % eval_every == 0 and step + 1 < total_steps:
            ev = evaluate(model, device, cfg, bytes_per_id, args.light_eval_tokens, long_ids)
            history.append({"step": step + 1, "eval": ev})
            print(f"    [eval] macro bpb {ev.get('macro_avg_bpb', float('nan')):.4f}", flush=True)

    final = evaluate(model, device, cfg, bytes_per_id, args.final_eval_tokens, long_ids)
    elapsed = time.time() - t0

    report = {
        "arm": args.arm,
        "arm_spec": ARMS[args.arm],
        "seed": args.seed,
        "device": str(device),
        "config": vars(cfg) | {"d_ffn": cfg.d_ffn, "d_head": cfg.d_head},
        "param_counts": pc,
        "codec": embedding.codec.describe() if isinstance(embedding, CodecEmbedding) else None,
        "constant_input_rows": (int(embedding.constant_input_rows().sum())
                                if isinstance(embedding, CodecEmbedding) else None),
        "effective_rank": rank_info,
        "tokens_trained": total_steps * tokens_per_step,
        "steps": total_steps,
        "declared_mixture": TRAIN_MIX,
        "realised_mixture": sampler.realised_mixture(),
        "history": history,
        "final_eval": final,
        "elapsed_min": round(elapsed / 60, 2),
    }
    tag = f"_{args.tag}" if args.tag else ""
    path = os.path.join(RESULTS, f"{args.arm}{tag}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    print("-" * 78)
    print(f"final macro bpb {final.get('macro_avg_bpb', float('nan')):.4f}   "
          f"({elapsed/60:.1f} min)")
    for lane in VAL_LANES:
        if lane in final:
            row = final[lane]
            extra = (f"   long-token bpb {row['bpb_long_tokens']:.4f} "
                     f"(n={row['long_token_positions']})" if "bpb_long_tokens" in row else "")
            print(f"  {lane:<24} bpb {row['bpb']:.4f}{extra}")
    print(f"wrote {os.path.relpath(path, HERE)}")


if __name__ == "__main__":
    main()
