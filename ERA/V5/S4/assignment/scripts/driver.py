"""Run the whole pipeline and assemble data/run/stats.json.

    python scripts/driver.py --smoke     # ~2k docs, minutes
    python scripts/driver.py             # the full ~50M-token slice

Stages 1-8 are one script chain, as the session frames them. Each writes its own
stats block; this collects them, computes the yield descent, and runs the asserts
that would have caught the V4 defects.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import subprocess
import sys
import time
from pathlib import Path

from common import RUN, STAGE, STAGE_STATS, jsonl_read, log, read_stage_stats

HERE = Path(__file__).resolve().parent

# (stage key, script, extra args). Extraction is stage 1 and is inherited from
# Session 3 -- Sangraha ships extracted text -- so it is accounted for, not re-run.
CHAIN = [
    ("00-fetch", "fetch_corpus.py", []),
    ("fixtures", "clean_fixtures.py", []),
    ("02-normalize", "normalize.py", []),
    ("03-langid", "langid_stage.py", []),
    ("04-quality-heuristics", "quality.py", []),
    ("04b-quality-classifier", "quality_clf.py", []),
    ("05-dedup", "dedup.py", []),
    ("06-pii", "pii.py", []),
    ("07-decontaminate", "decontaminate.py", []),
    ("08-manifest", "manifest.py", []),
]

# The session's own reference curve, for the widget to overlay ours against.
SESSION_REFERENCE_YIELD = [100, 92, 88, 61, 44, 43, 42, 42, 42]

ZERO_WIDTH_BAD = re.compile(r"[​⁠᠎­﻿‎‏‪-‮⁦-⁩�]")
ZWNJ, ZWJ = "‌", "‍"


def run(script: str, extra: list[str], smoke: bool) -> float:
    cmd = [sys.executable, str(HERE / script), *extra]
    if smoke and script == "fetch_corpus.py":
        cmd.append("--smoke")
    print(f"\n=== {script} {' '.join(extra)} ===", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, cwd=HERE, env={**__import__("os").environ, "PYTHONPATH": str(HERE)})
    if r.returncode:
        raise SystemExit(f"{script} failed with exit code {r.returncode}")
    return time.time() - t0


def integrity_asserts() -> dict:
    """The checks that would have caught V4's defects, run over the shipped corpus."""
    bad_chars = 0
    docs_checked = 0

    raw_ids_with_joiners = set()
    for rec in jsonl_read(STAGE / "00-raw.jsonl.gz"):
        if ZWNJ in rec["text"] or ZWJ in rec["text"]:
            raw_ids_with_joiners.add(rec["id"])

    shipped_ids = set()
    shipped_with_joiners = set()
    for shard in sorted((RUN / "shards").glob("shard-*.jsonl")):
        with open(shard, encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                docs_checked += 1
                shipped_ids.add(rec["id"])
                t = rec["text"]
                bad_chars += len(ZERO_WIDTH_BAD.findall(t))
                if ZWNJ in t or ZWJ in t:
                    shipped_with_joiners.add(rec["id"])

    # Every document that carried a joiner in the raw slice AND survived to the
    # shipped corpus must still carry it. Documents dropped by a later stage are
    # not counted against the cleaner.
    expected = raw_ids_with_joiners & shipped_ids
    lost = expected - shipped_with_joiners
    # Joiners that exist only after unescaping: the raw text carried `&zwj;`, which
    # html.unescape() turns into a real U+200D. They are only preserved because
    # unescaping runs before noise stripping and the noise pattern exempts joiners.
    from_unescape = shipped_with_joiners - raw_ids_with_joiners

    return {
        "docs_checked": docs_checked,
        "noise_chars_surviving": bad_chars,
        "no_noise_survives": bad_chars == 0,
        "raw_docs_with_indic_joiners": len(raw_ids_with_joiners),
        "raw_joiner_docs_surviving_to_output": len(expected),
        "shipped_docs_with_indic_joiners": len(shipped_with_joiners),
        "joiner_docs_from_html_unescape": len(from_unescape),
        "joiner_docs_that_lost_them": len(lost),
        "joiners_preserved": not lost,
        "note": (
            "A cleaner that stripped every invisible character would ship zero "
            "documents containing ZWNJ or ZWJ. Joiners present as literal characters "
            "were preserved, and so were joiners that only appear after HTML "
            "unescaping turns `&zwj;` into U+200D -- which works only because "
            "unescaping runs before noise stripping and the noise pattern exempts "
            "the two joiners."
        ),
    }


def assemble(durations: dict, smoke: bool) -> dict:
    keys = [k for k, _s, _e in CHAIN]
    stages = {}
    for k in keys:
        p = STAGE_STATS / f"{k}.json"
        if p.exists():
            stages[k] = read_stage_stats(k)

    fetch = stages["00-fetch"]
    start_docs = fetch["docs_out"]
    start_tokens = fetch["tokens_out"]

    # The yield descent. Stage 1 (extract) is inherited from the source corpus and
    # carried at 100% here, and is labelled as such rather than invented.
    descent = [
        {
            "stage": "0. Raw slice",
            "docs": start_docs,
            "pct_docs": 100.0,
            "dropped": 0,
        },
        {
            "stage": "1. Extract",
            "docs": start_docs,
            "pct_docs": 100.0,
            "dropped": 0,
            "inherited": True,
            "note": "Sangraha ships extracted text; extraction was Session 3's stage.",
        },
    ]
    labels = {
        "02-normalize": "2. Normalize",
        "03-langid": "3. Language ID",
        "04-quality-heuristics": "4a. Quality (Gopher/C4)",
        "04b-quality-classifier": "4b. Quality (classifier)",
        "05-dedup": "5. Deduplicate",
        "06-pii": "6. PII scrub",
        "07-decontaminate": "7. Decontaminate",
        "08-manifest": "8. Manifest",
    }
    for k, label in labels.items():
        s = stages.get(k)
        if not s:
            continue
        out = s.get("docs_out", 0)
        descent.append(
            {
                "stage": label,
                "docs": out,
                "pct_docs": round(100 * out / max(1, start_docs), 2),
                "dropped": s.get("docs_dropped", 0),
            }
        )

    man = stages.get("08-manifest", {})
    final_tokens = man.get("tokens_total", 0)

    return {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "smoke": smoke,
        "durations_seconds": {k: round(v, 1) for k, v in durations.items()},
        "total_runtime_seconds": round(sum(durations.values()), 1),
        "headline": {
            "dataset": fetch["dataset"],
            "revision": fetch["revision"],
            "license": fetch["license"],
            "docs_in": start_docs,
            "tokens_in": start_tokens,
            "docs_out": man.get("docs_out", 0),
            "tokens_out": final_tokens,
            "doc_retention_pct": round(100 * man.get("docs_out", 0) / max(1, start_docs), 2),
            "token_retention_pct": round(100 * final_tokens / max(1, start_tokens), 2),
            "shards": man.get("shards", 0),
            "shards_admitted": man.get("shards_admitted", 0),
        },
        "yield_descent": descent,
        "session_reference_yield": SESSION_REFERENCE_YIELD,
        "stages": stages,
        "integrity": integrity_asserts(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--from-stage", default=None, help="resume at this script name")
    ap.add_argument(
        "--assemble-only",
        action="store_true",
        help="re-derive stats.json from existing stage outputs, running nothing",
    )
    args = ap.parse_args()

    if args.assemble_only:
        prev = json.loads((RUN / "stats.json").read_text(encoding="utf-8"))
        stats = assemble(prev.get("durations_seconds", {}), prev.get("smoke", False))
        (RUN / "stats.json").write_text(
            json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"integrity: {stats['integrity']}")
        print(f"rewrote {RUN / 'stats.json'}")
        return

    durations = {}
    started = args.from_stage is None
    for key, script, extra in CHAIN:
        if not started:
            if script == args.from_stage:
                started = True
            else:
                continue
        durations[key] = run(script, extra, args.smoke)

    stats = assemble(durations, args.smoke)
    (RUN / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    h = stats["headline"]
    print("\n" + "=" * 62)
    print(
        f"{h['docs_in']:,} docs / {h['tokens_in']:,} tokens  ->  "
        f"{h['docs_out']:,} docs / {h['tokens_out']:,} tokens"
    )
    print(
        f"retention: {h['doc_retention_pct']}% of docs, "
        f"{h['token_retention_pct']}% of tokens | "
        f"{h['shards_admitted']}/{h['shards']} shards admitted"
    )
    print(f"integrity: {stats['integrity']}")
    print(f"wrote {RUN / 'stats.json'}")


if __name__ == "__main__":
    main()
