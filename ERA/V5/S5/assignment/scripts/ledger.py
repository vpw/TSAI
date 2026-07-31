#!/usr/bin/env python3
"""Supply ledger for the V5 mixture.

Every number in SUPPLY_LEDGER.md and in the README's tables is computed here from
data/inventory.json (which is transcribed from the session's own dataset-inventory
widget) plus the mixture declared in MIXTURE below. Nothing is typed by hand.

Run:  python3 scripts/ledger.py
Emits: data/ledger.json, SUPPLY_LEDGER.md
"""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
INV = json.loads((ROOT / "data" / "inventory.json").read_text())

B = 1_000_000_000
T = 1_000_000_000_000

# ---------------------------------------------------------------- the decisions

# Trained-token budget. The instructor's stated range is 2.4-4T; the curriculum
# widget names a 120B model. 120B x 20 tok/param = 2.4T (Chinchilla-optimal).
# We take the bottom of the range deliberately -- see SUPPLY_LEDGER for why the
# over-training is bought with OPUS's effective multiplier instead of raw tokens.
BUDGET = 2.4 * T

# Stage split, from the training-lifecycle widget (95 / 2 / <1 / <1 / <1).
STAGES = {
    "pretraining": 0.95,
    "anneal": 0.02,
    "sft": 0.01,
    "reasoning_training": 0.01,
    "preference": 0.01,
}

# Main pretraining mixture. Composer default for reference:
#   web 34 / code 24 / agentic 2 / reasoning 6 / long-ctx 6 / indic 16 / stem 12
MAIN = {
    "general_web": 32,
    "code": 24,
    "stem": 12,
    "indic": 16,
    "reasoning": 8,
    "long_context": 6,
    "agentic": 2,
}

# Anneal mixture = the composer's own "V5 anneal" preset, unchanged.
ANNEAL = {
    "general_web": 8,
    "code": 20,
    "stem": 10,
    "indic": 28,
    "reasoning": 18,
    "long_context": 8,
    "agentic": 8,
}

# Protected always-on lane: share of EVERY batch that bypasses OPUS entirely.
ALWAYS_ON = {"indic": 7.0, "agentic": 2.0, "reasoning": 1.0}  # = 10% total

# Mixture floors the selector may not push a lane below (composer's own floors).
FLOORS = {"indic": 12.0, "agentic": 2.0}

# Epoch ceiling before a lane is called over-repeated.
EPOCH_CEILING = 4.0

# Indic tier plan: epochs to run on each real tier, synthetic fills the rest.
INDIC_TIER_EPOCHS = {"A_verified": 2.5, "B_unverified": 2.0, "C_translated": 2.0}

# ------------------------------------------------------------------ supply side

def slot_supply():
    """Unique real tokens per lane, from the inventory widget."""
    s = INV["slots"]
    web_stem = s["general_web_and_stem"]["datasets"]
    return {
        "code": sum(d["tokens"] for d in s["code"]["datasets"]),
        "agentic": sum(d["tokens"] for d in s["agentic"]["datasets"]),
        "reasoning": sum(d["tokens"] for d in s["reasoning"]["datasets"]),
        "long_context": sum(d["tokens"] for d in s["long_context"]["datasets"]),
        "indic": sum(d["tokens"] for d in s["indic"]["datasets"]),
        "general_web": sum(d["tokens"] for d in web_stem if d["kind"] == "web"),
        "stem": sum(d["tokens"] for d in web_stem if d["kind"] == "stem"),
    }


def indic_tiers():
    """Real tokens per verified/unverified/translated/synthetic tier."""
    out = {"A_verified": 0, "B_unverified": 0, "C_translated": 0, "D_synthetic": 0}
    for d in INV["slots"]["indic"]["datasets"]:
        out[d["indic_tier"]] += d["tokens"]
    return out


def agentic_shapes():
    """Split the agentic slot into real multi-step trajectories vs single calls,
    and convert both to SUPERVISED tokens using the trajectory widget's measured
    supervised fractions. Raw tokens overstate what the lane actually teaches."""
    sup = INV["agentic_supervision"]
    traj = sup["long_trajectory"]["supervised_fraction"]
    call = sup["one_shot_call"]["supervised_fraction"]
    out = {"trajectory": 0, "function-call": 0}
    for d in INV["slots"]["agentic"]["datasets"]:
        out[d["shape"]] += d["tokens"]
    out["function_call"] = out.pop("function-call")
    return {
        "trajectory_raw": out["trajectory"],
        "function_call_raw": out["function_call"],
        "trajectory_supervised": out["trajectory"] * traj,
        "function_call_supervised": out["function_call"] * call,
        "supervised_total": out["trajectory"] * traj + out["function_call"] * call,
        "traj_frac": traj,
        "call_frac": call,
    }


# ------------------------------------------------------------------ demand side

def demand():
    pre = BUDGET * STAGES["pretraining"]
    ann = BUDGET * STAGES["anneal"]
    d = {}
    for lane in MAIN:
        d[lane] = pre * MAIN[lane] / 100.0 + ann * ANNEAL[lane] / 100.0
    return d, pre, ann


def verdict(dem, sup):
    if sup <= 0:
        return "must synthesise", float("inf")
    ep = dem / sup
    if ep <= 1.0:
        return "covered by unique tokens", ep
    if ep <= EPOCH_CEILING:
        return f"needs repetition ({ep:.2f} epochs)", ep
    return "must synthesise", ep


def main():
    sup = slot_supply()
    dem, pre, ann = demand()
    tiers = indic_tiers()
    ag = agentic_shapes()

    lanes = {}
    for lane in MAIN:
        v, ep = verdict(dem[lane], sup[lane])
        lanes[lane] = {
            "main_share_pct": MAIN[lane],
            "anneal_share_pct": ANNEAL[lane],
            "demand_tokens": dem[lane],
            "unique_supply_tokens": sup[lane],
            "epochs_required": ep,
            "verdict": v,
            "shortfall_tokens": max(0.0, dem[lane] - sup[lane] * EPOCH_CEILING),
        }

    # Indic tier plan: spend the real tiers at their declared epoch counts,
    # synthetic absorbs whatever remains.
    ind_dem = dem["indic"]
    plan = {}
    spent = 0.0
    for t, ep in INDIC_TIER_EPOCHS.items():
        got = tiers[t] * ep
        plan[t] = {"unique": tiers[t], "epochs": ep, "tokens": got}
        spent += got
    synth_needed = ind_dem - spent
    plan["D_synthetic"] = {
        "unique": tiers["D_synthetic"],
        "epochs": synth_needed / tiers["D_synthetic"] if tiers["D_synthetic"] else 0.0,
        "tokens": synth_needed,
    }
    for t in plan:
        plan[t]["share_of_indic_pct"] = 100.0 * plan[t]["tokens"] / ind_dem

    # Agentic: how much has to be built rather than collected.
    ag_dem = dem["agentic"]
    ag_real_at_ceiling = sup["agentic"] * EPOCH_CEILING
    ag_built = ag_dem - ag_real_at_ceiling

    opus = INV["opus"]
    keep = opus["v4_production_keep_fraction"]
    mult = opus["v4_effective_multiplier"]

    # Candidate-pool feasibility. Only the tokens that go THROUGH the selector need
    # the 1/keep multiple; the always-on lane bypasses OPUS and is consumed 1:1.
    ao = sum(ALWAYS_ON.values()) / 100.0
    candidates = BUDGET * (1 - ao) / keep + BUDGET * ao
    total_unique = sum(sup.values())
    # Largest trained budget the corpus can actually feed through the selector:
    #   budget * ((1-ao)/keep + ao) <= total_unique
    max_budget = total_unique / ((1 - ao) / keep + ao)

    out = {
        "budget_tokens": BUDGET,
        "stage_tokens": {k: BUDGET * v for k, v in STAGES.items()},
        "pretraining_tokens": pre,
        "anneal_tokens": ann,
        "main_mixture_pct": MAIN,
        "anneal_mixture_pct": ANNEAL,
        "always_on_lane_pct": ALWAYS_ON,
        "always_on_total_pct": sum(ALWAYS_ON.values()),
        "floors_pct": FLOORS,
        "epoch_ceiling": EPOCH_CEILING,
        "lanes": lanes,
        "indic_tier_plan": plan,
        "indic_tier_unique": tiers,
        "agentic": {
            **ag,
            "demand_tokens": ag_dem,
            "unique_supply_tokens": sup["agentic"],
            "real_at_epoch_ceiling": ag_real_at_ceiling,
            "must_be_built_tokens": ag_built,
            "built_fraction": ag_built / ag_dem,
            "shortfall_multiple": ag_dem / sup["agentic"],
        },
        "opus": {
            "keep_fraction": keep,
            "effective_multiplier": mult,
            "candidate_tokens_required": candidates,
            "always_on_bypass_frac": ao,
            "total_unique_supply": total_unique,
            "candidate_headroom": total_unique - candidates,
            "max_feasible_budget": max_budget,
            "effective_tokens": BUDGET * mult,
            "effective_tokens_per_param_at_120B": BUDGET * mult / 120e9,
            "raw_tokens_per_param_at_120B": BUDGET / 120e9,
            "compute_overhead": opus["compute_overhead"],
        },
        "checks": {
            "main_sums_to_100": sum(MAIN.values()) == 100,
            "anneal_sums_to_100": sum(ANNEAL.values()) == 100,
            "floors_respected": all(MAIN[k] >= v for k, v in FLOORS.items()),
            "always_on_within_floors": ALWAYS_ON["indic"] <= MAIN["indic"]
            and ALWAYS_ON["agentic"] <= MAIN["agentic"],
            "indic_tiers_sum_to_demand": abs(
                sum(p["tokens"] for p in plan.values()) - ind_dem
            )
            < 1.0,
            "candidate_pool_fits_in_supply": candidates <= total_unique,
        },
    }

    (ROOT / "data" / "ledger.json").write_text(json.dumps(out, indent=2))
    write_markdown(out)
    for k, v in out["checks"].items():
        print(f"{'ok ' if v else 'FAIL'} {k}")
    return out


def fmt(n):
    if n == float("inf"):
        return "inf"
    if n >= T:
        return f"{n/T:.2f}T"
    if n >= B:
        return f"{n/B:.1f}B"
    return f"{n/1e6:.0f}M"


def write_markdown(o):
    L = o["lanes"]
    order = ["general_web", "code", "stem", "indic", "reasoning", "long_context", "agentic"]
    names = {
        "general_web": "General web",
        "code": "Code",
        "stem": "STEM / math",
        "indic": "Indic",
        "reasoning": "Reasoning traces",
        "long_context": "Long-context",
        "agentic": "Agentic / tool-use",
    }
    r = []
    r.append("# Supply ledger\n")
    r.append(
        "Generated by `scripts/ledger.py` from `data/inventory.json`. "
        "Do not edit by hand.\n"
    )
    r.append(
        f"Trained-token budget **{fmt(o['budget_tokens'])}** · "
        f"pretraining **{fmt(o['pretraining_tokens'])}** · "
        f"anneal **{fmt(o['anneal_tokens'])}**\n"
    )
    r.append("## Per-lane demand against real supply\n")
    r.append(
        "| Lane | Main % | Anneal % | Demand | Unique supply | Epochs | Verdict |"
    )
    r.append("|---|---:|---:|---:|---:|---:|---|")
    for k in order:
        d = L[k]
        ep = "-" if d["epochs_required"] == float("inf") else f"{d['epochs_required']:.2f}"
        r.append(
            f"| {names[k]} | {d['main_share_pct']} | {d['anneal_share_pct']} | "
            f"{fmt(d['demand_tokens'])} | {fmt(d['unique_supply_tokens'])} | {ep} | {d['verdict']} |"
        )
    r.append("")
    r.append("## Indic, split across the four tiers\n")
    r.append("| Tier | Unique real | Epochs | Tokens contributed | Share of Indic lane |")
    r.append("|---|---:|---:|---:|---:|")
    tn = {
        "A_verified": "A verified native",
        "B_unverified": "B unverified crawl",
        "C_translated": "C translated",
        "D_synthetic": "D synthetic",
    }
    for k in ["A_verified", "B_unverified", "C_translated", "D_synthetic"]:
        p = o["indic_tier_plan"][k]
        r.append(
            f"| {tn[k]} | {fmt(p['unique'])} | {p['epochs']:.2f} | "
            f"{fmt(p['tokens'])} | {p['share_of_indic_pct']:.1f}% |"
        )
    r.append("")
    a = o["agentic"]
    r.append("## Agentic, in supervised tokens\n")
    r.append(f"- Raw slot supply: **{fmt(a['unique_supply_tokens'])}**")
    r.append(
        f"- Multi-step trajectories: {fmt(a['trajectory_raw'])} raw → "
        f"**{fmt(a['trajectory_supervised'])} supervised** (at {a['traj_frac']:.0%})"
    )
    r.append(
        f"- Single function calls: {fmt(a['function_call_raw'])} raw → "
        f"**{fmt(a['function_call_supervised'])} supervised** (at {a['call_frac']:.0%})"
    )
    r.append(f"- Total supervised: **{fmt(a['supervised_total'])}**")
    r.append(f"- Demand at the floor share: **{fmt(a['demand_tokens'])}**")
    r.append(f"- Shortfall against raw supply: **{a['shortfall_multiple']:.0f}x**")
    r.append(
        f"- Even at the {o['epoch_ceiling']:.0f}-epoch ceiling, "
        f"**{fmt(a['must_be_built_tokens'])} ({a['built_fraction']:.1%})** must be built."
    )
    r.append("")
    op = o["opus"]
    r.append("## OPUS arithmetic\n")
    r.append(
        f"- Keep fraction **{op['keep_fraction']:.0%}**, with "
        f"**{op['always_on_bypass_frac']:.0%}** of every batch bypassing the selector "
        f"→ candidate pool required **{fmt(op['candidate_tokens_required'])}**"
    )
    r.append(
        f"- Total unique supply across all lanes **{fmt(op['total_unique_supply'])}** "
        f"→ headroom **{fmt(op['candidate_headroom'])}**. "
        "This is the tightest constraint in the whole plan: the selector must be fed "
        "far more tokens than it trains on, and the corpus barely covers it."
    )
    r.append(
        f"- **The budget is derived, not chosen.** The largest trained budget this corpus "
        f"can feed through a {op['keep_fraction']:.0%}-keep selector is "
        f"**{fmt(op['max_feasible_budget'])}**. We train **{fmt(o['budget_tokens'])}**. "
        f"A 4T run would need {fmt(4*T*((1-op['always_on_bypass_frac'])/op['keep_fraction']+op['always_on_bypass_frac']))} "
        f"of candidates against {fmt(op['total_unique_supply'])} of supply, and is not fundable."
    )
    r.append(
        f"- Effective multiplier **{op['effective_multiplier']}x** → "
        f"**{fmt(op['effective_tokens'])}** effective tokens"
    )
    r.append(
        f"- At 120B parameters: **{op['raw_tokens_per_param_at_120B']:.0f}** raw tok/param, "
        f"**{op['effective_tokens_per_param_at_120B']:.0f}** effective tok/param"
    )
    r.append(f"- Compute overhead **{op['compute_overhead']:.1%}**")
    r.append("")
    r.append("## Checks\n")
    for k, v in o["checks"].items():
        r.append(f"- {'PASS' if v else 'FAIL'} — `{k}`")
    r.append("")
    (ROOT / "SUPPLY_LEDGER.md").write_text("\n".join(r))


if __name__ == "__main__":
    main()
