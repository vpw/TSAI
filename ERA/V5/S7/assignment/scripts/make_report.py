#!/usr/bin/env python3
"""Turn runs/*.json into a readable RESULTS.md.

Deliberately dumb: it reports what the arms produced, including arms that lost. The point of
pinning everything but the embedding is that the table can be read at face value, so the
report does no smoothing and flags anything that would make a comparison unfair.
"""

from __future__ import annotations

import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(HERE, "runs")

ORDER = ["dense", "kronecker_32", "kronecker_48", "fourier_2048", "fourier_8192", "naive_2048"]

BLURB = {
    "dense": "full V x d_model table (control, upper bound)",
    "kronecker_32": "shipped byte-Kronecker grid, 32-byte window",
    "kronecker_48": "same grid, window widened to 48 (the session's own remedy)",
    "fourier_2048": "phase-bound Fourier code, 4x smaller than the grid",
    "fourier_8192": "phase-bound Fourier code at matched dimension",
    "naive_2048": "unbound wave sum (negative control, order-blind)",
}


def load() -> dict:
    out = {}
    for p in sorted(glob.glob(os.path.join(RUNS, "*.json"))):
        with open(p, encoding="utf-8") as fh:
            r = json.load(fh)
        out[r["arm"]] = r
    return out


def fmt(x, nd=4):
    return "-" if x is None else f"{x:.{nd}f}"


def main() -> None:
    runs = load()
    if not runs:
        print("no runs found in runs/ -- nothing to report", file=sys.stderr)
        sys.exit(1)

    arms = [a for a in ORDER if a in runs] + [a for a in sorted(runs) if a not in ORDER]
    ref = runs[arms[0]]
    lanes = [k for k in ref["final_eval"] if isinstance(ref["final_eval"][k], dict)]

    lines = ["# Ablation results — input path, everything else pinned", ""]

    cfg = ref["config"]
    lines += [
        f"Model: d_model {cfg['d_model']}, {cfg['n_layers']} layers, {cfg['n_heads']} heads, "
        f"seq_len {cfg['seq_len']}, vocab {cfg['vocab_size']:,}.",
        f"Budget: {ref['tokens_trained']:,} tokens per arm, seed {ref['seed']}, "
        f"device {ref['device']}.",
        f"Mixture (declared): {ref['declared_mixture']}",
        f"Mixture (realised, first arm): {ref['realised_mixture']}",
        "",
        "Every arm shares architecture, optimiser, schedule, seed, sequence length, token "
        "budget and token stream. The embedding module is the only difference.",
        "",
        "## Parameters",
        "",
        "| arm | what it is | code dim | input path | dead rows | total trainable |",
        "|---|---|---|---|---|---|",
    ]
    for a in arms:
        r = runs[a]
        pc = r["param_counts"]
        codec = r.get("codec") or {}
        dead = r.get("dead_input_rows")
        cd = codec.get("code_dim")
        dead_s = "-" if dead is None else f"{dead:,} ({100*dead/cd:.1f}%)" if cd else f"{dead:,}"
        lines.append(
            f"| `{a}` | {BLURB.get(a,'')} | {cd or '-'} | {pc['input_path']:,} | {dead_s} | "
            f"{pc['total_trainable']:,} |"
        )

    lines += ["", "## Bits per byte (lower is better)", "",
              "| arm | " + " | ".join(lanes) + " | **macro** |",
              "|---|" + "---|" * (len(lanes) + 1)]
    for a in arms:
        fe = runs[a]["final_eval"]
        row = [fmt(fe[l]["bpb"]) for l in lanes]
        lines.append(f"| `{a}` | " + " | ".join(row) + f" | **{fmt(fe.get('macro_avg_bpb'))}** |")

    # Long-token slice: the positions a 32-byte window structurally cannot represent.
    has_long = any("bpb_long_tokens" in runs[a]["final_eval"].get(l, {})
                   for a in arms for l in lanes)
    if has_long:
        lines += ["", "## Bits per byte on long targets (>32 UTF-8 bytes)", "",
                  "These are exactly the tokens the 32-byte window truncates.", "",
                  "| arm | " + " | ".join(lanes) + " |",
                  "|---|" + "---|" * len(lanes)]
        for a in arms:
            fe = runs[a]["final_eval"]
            row = [fmt(fe.get(l, {}).get("bpb_long_tokens")) for l in lanes]
            lines.append(f"| `{a}` | " + " | ".join(row) + " |")
        supports = {l: ref["final_eval"].get(l, {}).get("long_token_positions") for l in lanes}
        lines += ["", f"Support (scored positions per lane): {supports}", "",
                  "*A small support means a wide error bar. Read this slice as directional "
                  "unless the counts are large.*"]

    # Arms are cheap to compare on the wrong axis. Grouping by input-path parameter count makes
    # the like-for-like comparisons explicit, so a headline picked from the unmatched axis cannot
    # be mistaken for one of these.
    groups: dict[int, list[str]] = {}
    for a in arms:
        groups.setdefault(runs[a]["param_counts"]["input_path"], []).append(a)
    matched = {p: g for p, g in groups.items() if len(g) > 1}
    if matched:
        lines += ["", "## Matched-parameter head-to-heads", "",
                  "Arms sharing an input-path parameter count. These are the like-for-like "
                  "comparisons; anything across groups trades parameters for quality and must "
                  "say so.", ""]
        for p in sorted(matched):
            g = sorted(matched[p], key=lambda a: runs[a]["final_eval"].get("macro_avg_bpb", 9e9))
            lines.append(f"**{p:,} input-path params** — "
                         + ", ".join(f"`{a}` {fmt(runs[a]['final_eval'].get('macro_avg_bpb'))}"
                                     for a in g))
            best, worst = g[0], g[-1]
            bv = runs[best]["final_eval"].get("macro_avg_bpb")
            wv = runs[worst]["final_eval"].get("macro_avg_bpb")
            if bv and wv and best != worst:
                # Normalised by the better arm, so this reads on the same scale as the
                # "change vs baseline" percentages below rather than a slightly smaller one.
                lines.append(f"  → `{worst}` is {100*(wv-bv)/bv:.2f}% worse than `{best}`")
            lines.append("")

    base = "kronecker_32"
    if base in runs:
        lines += ["", f"## Change vs `{base}` (macro bpb, negative = better)", "",
                  "*Not parameter-matched — see the section above. `fourier_2048` uses a quarter "
                  "of the baseline's input-path parameters, `kronecker_48` fifty percent more.*",
                  ""]
        b = runs[base]["final_eval"].get("macro_avg_bpb")
        for a in arms:
            if a == base:
                continue
            v = runs[a]["final_eval"].get("macro_avg_bpb")
            if v is None or b is None:
                continue
            lines.append(f"- `{a}`: {v-b:+.4f} bpb ({100*(v-b)/b:+.2f}%)")

    lines += ["", "## Wall clock", ""]
    for a in arms:
        lines.append(f"- `{a}`: {runs[a]['elapsed_min']} min")

    path = os.path.join(HERE, "RESULTS.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {os.path.relpath(path, HERE)}")


if __name__ == "__main__":
    main()
