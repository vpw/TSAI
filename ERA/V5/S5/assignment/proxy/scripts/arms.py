#!/usr/bin/env python3
"""The five ablation arms, derived from the ledger rather than typed in.

Two things make this a test of *the plan* and not just of seven numbers:

1. Mixture weights come straight out of `data/ledger.json`, so the arms move if the plan
   moves.
2. Each lane carries a `unique_cap` — the number of unique tokens the arm is allowed to
   draw from before it must start repeating — set so the proxy repeats each lane at the
   same epoch count the full-scale plan implies (capped at the plan's own 4.0 ceiling).
   Without this the proxy would silently hand every lane fresh tokens and would not be
   testing the repetition pressure the plan is mostly about.

Run `python3 proxy/scripts/arms.py` to print the arm table.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PROXY = os.path.dirname(HERE)
ROOT = os.path.dirname(PROXY)
LEDGER = json.load(open(os.path.join(ROOT, "data", "ledger.json")))

EPOCH_CEILING = LEDGER["epoch_ceiling"]
MAIN = LEDGER["main_mixture_pct"]
ANNEAL = LEDGER["anneal_mixture_pct"]
TIERS = LEDGER["indic_tier_plan"]
ALWAYS_ON = LEDGER["always_on_lane_pct"]

# Proxy lane names. Indic is split into its four tiers; every other plan lane maps 1:1.
INDIC_TIER_LANES = {
    "A_verified": "indic_A_verified",
    "B_unverified": "indic_B_unverified",
    "C_translated": "indic_C_translated",
    "D_synthetic": "indic_D_synthetic",
}

# Full-scale epochs per lane, from the ledger. These set the proxy's repetition pressure.
PLAN_EPOCHS = {ln: LEDGER["lanes"][ln]["epochs_required"] for ln in LEDGER["lanes"]}
for tier, lane in INDIC_TIER_LANES.items():
    PLAN_EPOCHS[lane] = TIERS[tier]["epochs"]


def _explode_indic(indic_pct):
    """Split an Indic share across the four tiers using the plan's own tier shares."""
    return {INDIC_TIER_LANES[t]: indic_pct * TIERS[t]["share_of_indic_pct"] / 100.0
            for t in TIERS}


def _norm(w):
    tot = sum(w.values())
    return {k: 100.0 * v / tot for k, v in w.items() if v > 0}


def mix_from_plan(plan_mix):
    """Expand a 7-lane plan mixture into proxy lanes."""
    out = {k: float(v) for k, v in plan_mix.items() if k != "indic"}
    out.update(_explode_indic(plan_mix["indic"]))
    return _norm(out)


# --------------------------------------------------------------------------- the arms
def arm_A():
    return mix_from_plan(MAIN), "the proposed V5 main-pretraining mixture"


def arm_B():
    """V4's starting point: web-heavy, no agentic lane, no reasoning lane, no long-context.
    Web/code/STEM are the composer's V4 numbers; the remaining 8 points are V4's documented
    8% always-on Indic floor, which is the only lane protection V4 had."""
    plan = {"general_web": 72, "code": 13, "stem": 7, "indic": 8,
            "reasoning": 0, "long_context": 0, "agentic": 0}
    return mix_from_plan(plan), "naive web-heavy V4 start (web 72 / code 13 / STEM 7 / Indic 8)"


def arm_C():
    """Arm A with the protected always-on floor removed and the selector left free.

    We do not re-run OPUS here; we use OPUS's own measured behaviour. On the selector widget
    at keep 40% with the floor off and an English-shaped proxy, the trained share of Indic is
    0.0% and of agentic 0.0%. So arm C is arm A with those two lanes deleted and their share
    redistributed across the lanes the selector does keep, in arm A's proportions."""
    plan = dict(MAIN)
    plan["indic"] = 0
    plan["agentic"] = 0
    return mix_from_plan(plan), "arm A, protected floor removed (OPUS floor-off: Indic 0%, agentic 0%)"


def arm_D():
    """Arm A with the Indic lane collapsed to verified-only at the same total Indic share.

    At full scale this means the 378.2B Indic lane is drawn from the 64B verified tier alone,
    i.e. 5.9 epochs on verified. The proxy reproduces that: arm D's verified pool is capped so
    it repeats 5.9 times (subject to the plan's 4.0 ceiling), which is the cost the four-tier
    split is supposed to be buying its way out of."""
    plan = dict(MAIN)
    mix = {k: float(v) for k, v in plan.items() if k != "indic"}
    mix["indic_A_verified"] = float(plan["indic"])
    demand = LEDGER["lanes"]["indic"]["demand_tokens"]
    verified_unique = LEDGER["indic_tier_unique"]["A_verified"]
    epochs = demand / verified_unique
    return _norm(mix), f"arm A, Indic collapsed to verified-only ({epochs:.1f} epochs on verified)"


ARMS = {
    "A_proposed": arm_A,
    "B_web_heavy": arm_B,
    "C_no_floor": arm_C,
    "D_verified_only": arm_D,
}

# Arm E is not a bits-per-byte arm: it is a transition-stability probe. Each E run trains a
# short schedule that switches mixture from MAIN to ANNEAL at the midpoint, and differs only
# in how the switch is made. The widget's claim is that frozen embeddings, not the size of
# the shift, dominate the spike (151x frozen vs 8.0x trainable at the same shift).
E_ARMS = {
    "E1_hard_frozen": {"warmup_frac": 0.0, "freeze_embeddings": True,
                       "note": "hard step, embeddings frozen"},
    "E2_hard_trainable": {"warmup_frac": 0.0, "freeze_embeddings": False,
                          "note": "hard step, embeddings trainable"},
    "E3_warm_trainable": {"warmup_frac": 0.10, "freeze_embeddings": False,
                          "note": "10% warmup band, embeddings trainable"},
}


def arm_D_verified_epochs():
    return (LEDGER["lanes"]["indic"]["demand_tokens"]
            / LEDGER["indic_tier_unique"]["A_verified"])


def unique_caps(arm_name, mix, total_tokens):
    """Unique tokens each lane may draw from, so proxy epochs match full-scale epochs."""
    caps = {}
    for lane, pct in mix.items():
        demand = total_tokens * pct / 100.0
        ep = PLAN_EPOCHS.get(lane, 1.0)
        if arm_name == "D_verified_only" and lane == "indic_A_verified":
            ep = arm_D_verified_epochs()
        ep = max(min(ep, EPOCH_CEILING), 1e-6)
        caps[lane] = {"epochs_target": round(ep, 3),
                      "unique_cap_tokens": int(demand / ep)}
    return caps


def build(total_tokens):
    out = {}
    for name, fn in ARMS.items():
        mix, note = fn()
        out[name] = {"mixture_pct": {k: round(v, 4) for k, v in mix.items()},
                     "note": note,
                     "caps": unique_caps(name, mix, total_tokens)}
    return out


if __name__ == "__main__":
    import sys
    total = int(sys.argv[1]) if len(sys.argv) > 1 else 120_000_000
    spec = build(total)
    lanes = sorted({ln for a in spec.values() for ln in a["mixture_pct"]})
    print(f"arms at {total / 1e6:.0f}M tokens/arm\n")
    print(f"{'lane':22s}" + "".join(f"{n[:13]:>15s}" for n in spec))
    for ln in lanes:
        row = "".join(f"{spec[n]['mixture_pct'].get(ln, 0):15.2f}" for n in spec)
        print(f"{ln:22s}{row}")
    print(f"{'TOTAL':22s}" + "".join(
        f"{sum(spec[n]['mixture_pct'].values()):15.2f}" for n in spec))
    print()
    for n, a in spec.items():
        print(f"{n}: {a['note']}")
        for ln, c in sorted(a["caps"].items()):
            print(f"    {ln:22s} {c['epochs_target']:6.2f} epochs on "
                  f"{c['unique_cap_tokens'] / 1e6:8.2f}M unique tokens")
