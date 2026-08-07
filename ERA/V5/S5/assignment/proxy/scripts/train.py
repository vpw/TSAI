#!/usr/bin/env python3
"""Train one arm of the mixture ablation and score per-domain bits-per-byte.

Everything except the data mixture is held fixed across arms: architecture, optimiser,
schedule, seed, sequence length and total trained tokens.

Metric is bits-per-byte, not cross-entropy per token: bpb is comparable across languages and
tokenizers, which cross-entropy is not — a tokenizer that splits Hindi into more pieces makes
per-token loss look better while the model is doing the same work per unit of text.

  bpb = (sum of NLL in nats over scored positions) / ln(2) / (UTF-8 bytes of those positions)

For the agentic set only assistant tokens are scored; system prompts, user turns and tool
observations are context, exactly as they would be in training.
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch

import arms as ARMS_MOD
from data import MixtureSampler, load_val, token_stats
from model import Config, Transformer

HERE = os.path.dirname(os.path.abspath(__file__))
PROXY = os.path.dirname(HERE)
RESULTS = os.path.join(PROXY, "results")

VAL_LANES = ["general_web", "code", "stem", "reasoning", "agentic", "long_context",
             "indic_A_verified", "indic_B_unverified", "indic_C_translated",
             "indic_D_synthetic"]


def lr_at(step, total, base_lr, warmup_frac=0.02, min_frac=0.1):
    w = max(int(total * warmup_frac), 1)
    if step < w:
        return base_lr * (step + 1) / w
    p = (step - w) / max(total - w, 1)
    return base_lr * (min_frac + (1 - min_frac) * 0.5 * (1 + math.cos(math.pi * p)))


@torch.no_grad()
def evaluate(model, device, seq_len, max_tokens, lanes=VAL_LANES, batch_seqs=8):
    model.eval()
    out = {}
    for lane in lanes:
        v = load_val(lane, max_tokens)
        if v is None:
            continue
        t, m, b = v["tokens"], v["mask"], v["tbytes"]
        n_blocks = max((len(t) - 1) // seq_len, 1)
        nll_sum, byte_sum, tok_sum = 0.0, 0, 0
        for s in range(0, n_blocks, batch_seqs):
            idx = [(i * seq_len, i * seq_len + seq_len + 1)
                   for i in range(s, min(s + batch_seqs, n_blocks))]
            xb = np.stack([t[a:b_] for a, b_ in idx if b_ <= len(t)])
            if xb.size == 0:
                continue
            mb = np.stack([m[a:b_] for a, b_ in idx if b_ <= len(t)])
            bb = np.stack([b[a:b_] for a, b_ in idx if b_ <= len(t)])
            x = torch.from_numpy(xb[:, :-1]).to(device)
            y = torch.from_numpy(xb[:, 1:]).to(device)
            keep = torch.from_numpy(mb[:, 1:].astype(np.bool_)).to(device)
            nbytes = bb[:, 1:][mb[:, 1:].astype(bool)].sum()
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(x)
            ls = torch.nn.functional.cross_entropy(
                logits.float().view(-1, logits.size(-1)), y.reshape(-1), reduction="none")
            ls = ls.view(y.shape)[keep]
            nll_sum += float(ls.sum())
            byte_sum += int(nbytes)
            tok_sum += int(keep.sum())
        if byte_sum > 0:
            out[lane] = {"bpb": nll_sum / math.log(2) / byte_sum,
                         "nll_per_token": nll_sum / max(tok_sum, 1),
                         "scored_tokens": tok_sum, "scored_bytes": byte_sum}
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--tokens", type=int, default=120_000_000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--n-layers", type=int, default=8)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--batch-seqs", type=int, default=16, help="sequences per optimiser step")
    ap.add_argument("--micro-seqs", type=int, default=8, help="sequences per forward pass")
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--eval-every-frac", type=float, default=0.25)
    ap.add_argument("--light-eval-tokens", type=int, default=250_000)
    ap.add_argument("--final-eval-tokens", type=int, default=1_500_000)
    ap.add_argument("--tag", default="")
    ap.add_argument("--benchmark-steps", type=int, default=0,
                    help="if set, time this many steps and exit without training")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    ts = token_stats()
    cfg = Config(vocab_size=ts["vocab_size"], d_model=args.d_model, n_layers=args.n_layers,
                 n_heads=args.n_heads, seq_len=args.seq_len)
    model = Transformer(cfg).to(device)
    pc = model.param_counts()

    # ---------------------------------------------------------------- arm definition
    is_transition = args.arm in ARMS_MOD.E_ARMS
    if is_transition:
        e = ARMS_MOD.E_ARMS[args.arm]
        mix = ARMS_MOD.mix_from_plan(ARMS_MOD.MAIN)
        mix2 = ARMS_MOD.mix_from_plan(ARMS_MOD.ANNEAL)
        caps = ARMS_MOD.unique_caps("A_proposed", mix, args.tokens)
        note = e["note"]
        freeze_emb, warm_frac = e["freeze_embeddings"], e["warmup_frac"]
    else:
        spec = ARMS_MOD.build(args.tokens)[args.arm]
        mix, caps, note = spec["mixture_pct"], spec["caps"], spec["note"]
        mix2, freeze_emb, warm_frac = None, False, 0.0

    if freeze_emb:
        model.embed.weight.requires_grad_(False)

    sampler = MixtureSampler(mix, caps, args.seq_len, args.seed)

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim < 2 else decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.95), eps=1e-8)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    tokens_per_step = args.batch_seqs * args.seq_len
    total_steps = max(args.tokens // tokens_per_step, 1)
    accum = max(args.batch_seqs // args.micro_seqs, 1)
    fpt = model.flops_per_token()

    print(f"arm={args.arm} {note}")
    print(f"params total={pc['total'] / 1e6:.1f}M  non-embedding={pc['non_embedding'] / 1e6:.1f}M"
          f"  d_model={cfg.d_model} layers={cfg.n_layers} d_ffn={cfg.d_ffn}")
    print(f"tokens={args.tokens / 1e6:.0f}M  steps={total_steps}  tokens/step={tokens_per_step}"
          f"  accum={accum}  flops/token={fpt / 1e6:.0f}M", flush=True)

    # ------------------------------------------------------------------- benchmark mode
    if args.benchmark_steps:
        model.train()
        t0 = None
        for step in range(args.benchmark_steps + 3):
            if step == 3:
                torch.cuda.synchronize() if device.type == "cuda" else None
                t0 = time.time()
            opt.zero_grad(set_to_none=True)
            for _ in range(accum):
                xb, mb = sampler.batch(args.micro_seqs)
                x = torch.from_numpy(xb[:, :-1]).to(device, non_blocking=True)
                y = torch.from_numpy(xb[:, 1:]).to(device, non_blocking=True)
                keep = torch.from_numpy(mb[:, 1:].astype(np.bool_)).to(device)
                with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                    logits = model(x)
                    ls = torch.nn.functional.cross_entropy(
                        logits.view(-1, logits.size(-1)), y.reshape(-1), reduction="none")
                    ls = (ls.view(y.shape) * keep).sum() / keep.sum().clamp(min=1) / accum
                scaler.scale(ls).backward()
            scaler.step(opt)
            scaler.update()
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
        tps = args.benchmark_steps * tokens_per_step / dt
        print(json.dumps({"benchmark": True, "steps": args.benchmark_steps,
                          "seconds": round(dt, 2), "tokens_per_second": round(tps),
                          "achieved_tflops": round(tps * fpt / 1e12, 2),
                          "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)
                          if device.type == "cuda" else 0}, indent=1))
        return

    # ------------------------------------------------------------------------ training
    switch_step = total_steps // 2 if is_transition else None
    warm_steps = int(total_steps * warm_frac) if is_transition else 0
    log, grad_norms, curves = [], [], []
    t_start = time.time()
    model.train()

    for step in range(total_steps):
        if is_transition and step >= switch_step:
            if warm_steps > 0 and step < switch_step + warm_steps:
                a = (step - switch_step + 1) / warm_steps      # linear blend over the band
                blended = {k: (1 - a) * mix.get(k, 0.0) + a * mix2.get(k, 0.0)
                           for k in set(mix) | set(mix2)}
                sampler.set_mixture(blended)
            elif step == switch_step + warm_steps:
                sampler.set_mixture(mix2)
            elif warm_steps == 0 and step == switch_step:
                sampler.set_mixture(mix2)

        lr = lr_at(step, total_steps, args.lr)
        for g in opt.param_groups:
            g["lr"] = lr

        opt.zero_grad(set_to_none=True)
        loss_acc = 0.0
        for _ in range(accum):
            xb, mb = sampler.batch(args.micro_seqs)
            x = torch.from_numpy(xb[:, :-1]).to(device, non_blocking=True)
            y = torch.from_numpy(xb[:, 1:]).to(device, non_blocking=True)
            keep = torch.from_numpy(mb[:, 1:].astype(np.bool_)).to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(x)
                ls = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)), y.reshape(-1), reduction="none")
                ls = (ls.view(y.shape) * keep).sum() / keep.sum().clamp(min=1) / accum
            scaler.scale(ls).backward()
            loss_acc += float(ls) * accum
        scaler.unscale_(opt)
        gn = float(torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0))
        scaler.step(opt)
        scaler.update()
        grad_norms.append(gn)

        if step % 25 == 0 or step == total_steps - 1:
            el = time.time() - t_start
            done = (step + 1) * tokens_per_step
            log.append({"step": step, "loss": loss_acc / accum, "lr": lr, "grad_norm": gn,
                        "tokens": done, "seconds": round(el, 1)})
            if step % 200 == 0 or step == total_steps - 1:
                tps = done / max(el, 1e-9)
                print(f"  step {step:5d}/{total_steps}  loss {loss_acc / accum:6.3f}  "
                      f"gn {gn:6.2f}  {tps:7.0f} tok/s  "
                      f"eta {(total_steps - step) * tokens_per_step / tps / 60:5.1f} min",
                      flush=True)

        if (not is_transition and args.eval_every_frac > 0
                and step > 0 and step % max(int(total_steps * args.eval_every_frac), 1) == 0):
            ev = evaluate(model, device, args.seq_len, args.light_eval_tokens)
            curves.append({"step": step, "tokens": (step + 1) * tokens_per_step,
                           "bpb": {k: round(v["bpb"], 5) for k, v in ev.items()}})
            print(f"    [eval @ {100 * step / total_steps:.0f}%] " +
                  "  ".join(f"{k[:9]}={v['bpb']:.4f}" for k, v in ev.items()), flush=True)

    elapsed = time.time() - t_start
    final = evaluate(model, device, args.seq_len, args.final_eval_tokens)

    # transition diagnostics: peak grad norm after the switch vs the settled level before it
    trans = None
    if is_transition:
        pre = grad_norms[max(switch_step - 200, 0):switch_step]
        post = grad_norms[switch_step:switch_step + 300]
        # fp16 + GradScaler can emit inf/nan norms on steps the scaler then skips. Those are
        # real instability but they make a ratio meaningless, so they are counted separately
        # and the ratio is taken over the finite values.
        pre_f = [g for g in pre if np.isfinite(g)]
        post_f = [g for g in post if np.isfinite(g)]
        n_nonfinite = sum(1 for g in pre + post if not np.isfinite(g))
        base = float(np.median(pre_f)) if pre_f else float("nan")
        peak = float(np.max(post_f)) if post_f else float("nan")
        ratio = peak / base if (base and np.isfinite(base) and np.isfinite(peak)) else None
        trans = {"switch_step": switch_step, "warmup_steps": warm_steps,
                 "baseline_grad_norm_median_pre": round(base, 4),
                 "peak_grad_norm_post": round(peak, 4),
                 "peak_ratio_x": round(ratio, 3) if ratio is not None else None,
                 "nonfinite_grad_norm_steps": n_nonfinite,
                 "threshold_x": 3.0,
                 "verdict": ("controlled" if ratio is not None and ratio <= 3.0
                             else "unstable")}

    os.makedirs(RESULTS, exist_ok=True)
    name = args.arm + (f"_{args.tag}" if args.tag else "")
    out = {
        "arm": args.arm, "tag": args.tag, "note": note, "seed": args.seed,
        "config": {"d_model": cfg.d_model, "n_layers": cfg.n_layers, "n_heads": cfg.n_heads,
                   "d_ffn": cfg.d_ffn, "seq_len": cfg.seq_len, "vocab_size": cfg.vocab_size,
                   "lr": args.lr, "batch_seqs": args.batch_seqs, "tokens": args.tokens,
                   "steps": total_steps, "tokenizer": ts["tokenizer"]},
        "params": pc, "flops_per_token": fpt,
        "total_flops": fpt * args.tokens,
        "wall_seconds": round(elapsed, 1),
        "tokens_per_second": round(args.tokens / elapsed),
        "achieved_tflops": round(args.tokens * fpt / elapsed / 1e12, 2),
        "mixture_declared_pct": mix,
        "mixture_realised": sampler.report(),
        "final_bpb": {k: round(v["bpb"], 6) for k, v in final.items()},
        "final_detail": final,
        "bpb_curve": curves,
        "loss_log": log,
        "transition": trans,
        "is_transition_arm": is_transition,
    }
    with open(os.path.join(RESULTS, f"{name}.json"), "w") as f:
        json.dump(out, f, indent=1)
    np.save(os.path.join(RESULTS, f"{name}.gradnorms.npy"), np.asarray(grad_norms, np.float32))

    print(f"\n{name}: {elapsed / 60:.1f} min  "
          f"{out['tokens_per_second']:.0f} tok/s  {out['achieved_tflops']:.2f} TFLOP/s")
    for k, v in out["final_bpb"].items():
        print(f"  bpb {k:22s} {v:.4f}")
    if trans:
        print(f"  transition peak {trans['peak_ratio_x']}x baseline -> {trans['verdict']}")


if __name__ == "__main__":
    main()
