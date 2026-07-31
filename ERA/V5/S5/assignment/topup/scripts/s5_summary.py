#!/usr/bin/env python3
"""Write TOPUP_REPORT.md — the S5 view of this cleaning pass.

S4's `write_report.py` already renders the pipeline's own numbers. This adds the two things
S5 needs: where the pass lands against the cumulative cleaning target, and the per-language
yield, which is where the interesting failure was.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
STATS = os.path.join(ROOT, "data", "run", "stats.json")
OUT = os.path.join(ROOT, "TOPUP_REPORT.md")

# The cumulative target the session states: 8% of a 4T run = 320B cleaned Indic tokens.
CUMULATIVE_TARGET = 320_000_000_000
S4_CONTRIBUTION = 43_450_641


def main():
    s = json.load(open(STATS))
    h = s["headline"]
    q = s["stages"]["04-quality-heuristics"]
    per_lang = q.get("per_lang", {})
    total = S4_CONTRIBUTION + h["tokens_out"]

    L = []
    w = L.append
    w("# S5 cleaning top-up\n")
    w("Same eight-stage pipeline as S4, pointed at the lane the ledger says is starved: the")
    w("Indic **verified** tier, which the plan runs at 2.5 epochs. Verified tokens are only")
    w("worth having if they cover the languages the model must serve, so this pass takes the")
    w("Tier-1 languages S4 never touched, plus a second Hindi shard so cross-shard dedup has")
    w("something to do.\n")

    w("## Result\n")
    w(f"| | |")
    w(f"|---|---:|")
    w(f"| Documents in → out | {h['docs_in']:,} → {h['docs_out']:,} |")
    w(f"| Tokens in → out | {h['tokens_in']:,} → **{h['tokens_out']:,}** |")
    w(f"| Token retention | {h['token_retention_pct']:.2f}% |")
    w(f"| Shards admitted | {h['shards_admitted']}/{h['shards']} |")
    w("")
    w(f"- This pass: **{h['tokens_out'] / 1e6:.1f}M** clean tokens")
    w(f"- S4 pass: {S4_CONTRIBUTION / 1e6:.1f}M")
    w(f"- Cumulative: **{total / 1e6:.1f}M** against the stated target of "
      f"{CUMULATIVE_TARGET / 1e9:.0f}B (8% of a 4T run) = "
      f"**{100 * total / CUMULATIVE_TARGET:.4f}%**\n")
    w("That percentage is the honest one. A single workstation pass is a rounding error against")
    w("a 320B target; what it buys is a *measured* per-language yield curve to plan the real run")
    w("with, and it caught a defect that would have scaled.\n")

    if per_lang:
        w("## Per-language yield, and the defect this pass found\n")
        w("The S4 quality stage asks \"does this document contain at least 2 common words of its")
        w("language\". `stopword_set()` fell back to the **English** list for any language it had")
        w("no list for — so a Kannada page was checked for English stop-words, found none, and was")
        w("dropped. Five languages had no list.\n")
        w("| Language | docs seen | kept before fix | kept after fix |")
        w("|---|---:|---:|---:|")
        before = {"ben": 96.6, "hin": 96.9, "mar": 95.6, "tam": 80.6,
                  "kan": 5.1, "guj": 4.4, "mal": 3.8, "pan": 5.7, "ory": 2.9}
        for lang, v in sorted(per_lang.items()):
            after = 100.0 * v["sa_kept"] / max(v["docs"], 1)
            b = before.get(lang)
            w(f"| `{lang}` | {v['docs']:,} | {b:.1f}% | {after:.1f}% |"
              if b is not None else
              f"| `{lang}` | {v['docs']:,} | — | {after:.1f}% |")
        w("")
        w("Whole-slice token retention went from **53.16% to 91.90%** once the five missing lists")
        w("were added — 38.7 points of yield that were being discarded silently, in exactly the")
        w("languages the Indic lane is short of. The lists were counted out of the corpus by")
        w("document frequency rather than written from memory")
        w("(`scripts/derive_stopwords.py`, `data/run/derived_stopwords.json`), and a missing list")
        w("is now recorded in the stage stats instead of being absorbed into the English fallback.\n")
        missing = q.get("languages_with_no_stopword_list", [])
        w(f"Languages still with no list after this pass: "
          f"{', '.join('`' + m + '`' for m in missing) if missing else '**none**'}.\n")

    w("## Caveats\n")
    w("- The MILU-Bengali eval fingerprint fetch returned 0 examples (the datasets-server rate")
    w("  limits unauthenticated paging), so Bengali documents were decontaminated against the")
    w("  Hindi and Telugu MILU splits only. Bengali contamination is therefore **unmeasured**,")
    w("  not measured-as-zero.")
    w("- `claimed_lang` for Odia is written `ory`, the pipeline's ISO-639-3 label, while")
    w("  Sangraha's folder is `ori`. The folder name is not trusted for anything else either.\n")

    with open(OUT, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"wrote {OUT}")
    print(f"this pass {h['tokens_out']:,} tokens; cumulative {total:,} "
          f"({100 * total / CUMULATIVE_TARGET:.4f}% of target)")


if __name__ == "__main__":
    main()
