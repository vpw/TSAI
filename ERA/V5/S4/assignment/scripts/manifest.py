"""Stage 8 -- shard the clean corpus and emit a lineage manifest for every shard.

No shard enters the corpus without a manifest. Each one records where the data came
from, under what licence, who contributed it, every cleaning script that touched it
with that script's own source hash, the shard's sha256, its token count measured with
a named tokenizer, and its language distribution. A gating rule then admits or blocks
the shard.

Two V4 defects are addressed by construction:
  * identifiers are derived from content, not from a row number over a
    non-deterministic ordering, so re-running the pipeline reproduces them exactly;
  * token counts are measured with a real tokenizer rather than estimated as
    `words x 1.3`, a ratio the session flags as wrong for Indic by 2-10x. Both the
    measured count and what the bad ratio would have claimed are recorded.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
from pathlib import Path

from common import (
    PRIMARY_TOKENIZER,
    REPO_ID,
    RUN,
    STAGE,
    Timer,
    count_tokens_batch,
    jsonl_read,
    log,
    script_hash,
    sha256_text,
    write_stage_stats,
)

CONTRIBUTOR = "era5-vardhan"
LICENSE_ALLOWLIST = {"CC-BY-4.0", "CC-BY-SA-4.0", "CC0-1.0", "MIT", "Apache-2.0"}
LICENSE_CLASS = "CC-BY-4.0"  # Sangraha's licence
DOCS_PER_SHARD = 5000

REQUIRED_FIELDS = [
    "shard_id",
    "source_url",
    "license_class",
    "contributor_id",
    "cleaning_scripts",
    "ingest_timestamp",
    "sha256",
    "token_count",
    "lang_distribution",
]

# Every script that touches the data, in order. The transcript is explicit that a
# shard carries a list of (script, hash) pairs, not one.
PIPELINE_SCRIPTS = [
    "fetch_corpus.py",
    "normalize.py",
    "langid_stage.py",
    "quality.py",
    "quality_clf.py",
    "dedup.py",
    "pii.py",
    "decontaminate.py",
    "manifest.py",
]


def cleaning_scripts() -> list[dict]:
    here = Path(__file__).resolve().parent
    out = []
    for name in PIPELINE_SCRIPTS:
        p = here / name
        if p.exists():
            out.append({"script": name, "sha256": script_hash(p)})
    return out


def gate(manifest: dict) -> tuple[str, list[str]]:
    missing = [f for f in REQUIRED_FIELDS if not manifest.get(f)]
    if missing:
        return "BLOCKED", [f"missing: {', '.join(missing)}"]
    if manifest["license_class"] not in LICENSE_ALLOWLIST:
        return "BLOCKED", [f"licence {manifest['license_class']} not on the allow-list"]
    if any(not s.get("sha256") for s in manifest["cleaning_scripts"]):
        return "BLOCKED", ["a cleaning script has no source hash"]
    if manifest["token_count"] <= 0:
        return "BLOCKED", ["shard has no tokens"]
    return "ADMITTED", ["all required fields present; licence on the allow-list"]


def build_shards(inp: Path, out_dir: Path, timestamp: str) -> tuple[list[dict], dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("shard-*.jsonl"):
        stale.unlink()

    scripts = cleaning_scripts()
    manifests: list[dict] = []
    totals = collections.Counter()
    lang_totals = collections.Counter()
    fertility_num = collections.Counter()
    fertility_den = collections.Counter()

    buf: list[dict] = []
    shard_no = 0

    def flush():
        nonlocal shard_no
        if not buf:
            return
        texts = [r["text"] for r in buf]
        # Measured, with a named tokenizer. Not words x 1.3.
        tok_counts = count_tokens_batch(texts)
        words = [len(t.split()) for t in texts]

        body = "\n".join(json.dumps(r, ensure_ascii=False) for r in buf) + "\n"
        digest = sha256_text(body)
        langs = collections.Counter(
            r.get("detected_lang") or r["claimed_lang"] for r in buf
        )
        total = sum(langs.values())

        for r, tc, w in zip(buf, tok_counts, words):
            lang = r.get("detected_lang") or r["claimed_lang"]
            fertility_num[lang] += tc
            fertility_den[lang] += w
            lang_totals[lang] += tc

        path = out_dir / f"shard-{shard_no:04d}.jsonl"
        path.write_text(body, encoding="utf-8")

        m = {
            # Content-derived, so the same input always yields the same id.
            "shard_id": f"shard_{digest[:12]}",
            "source_url": f"https://huggingface.co/datasets/{REPO_ID}",
            "license_class": LICENSE_CLASS,
            "contributor_id": CONTRIBUTOR,
            "cleaning_scripts": scripts,
            "ingest_timestamp": timestamp,
            "sha256": digest,
            "token_count": sum(tok_counts),
            "tokenizer": PRIMARY_TOKENIZER,
            "word_count": sum(words),
            "naive_words_x_1_3_estimate": int(round(sum(words) * 1.3)),
            "lang_distribution": {
                k: round(100 * v / total, 1) for k, v in langs.most_common()
            },
            "documents": len(buf),
            "file": path.name,
        }
        status, reasons = gate(m)
        m["status"] = status
        m["gate_reasons"] = reasons
        manifests.append(m)
        totals["shards"] += 1
        totals["docs"] += len(buf)
        totals["tokens"] += m["token_count"]
        totals["words"] += m["word_count"]
        totals["naive_estimate"] += m["naive_words_x_1_3_estimate"]
        shard_no += 1
        buf.clear()

    for rec in jsonl_read(inp):
        buf.append(
            {
                k: rec[k]
                for k in ("id", "src", "pool", "claimed_lang", "detected_lang", "text")
                if k in rec
            }
        )
        if len(buf) >= DOCS_PER_SHARD:
            flush()
    flush()

    fertility = {
        lang: round(fertility_num[lang] / fertility_den[lang], 3)
        for lang in fertility_num
        if fertility_den[lang]
    }
    return manifests, {
        "totals": dict(totals),
        "tokens_by_lang": dict(lang_totals),
        "fertility_tokens_per_word": fertility,
    }


def determinism_check(inp: Path, timestamp: str) -> dict:
    """Re-shard the same input twice and assert the ids and hashes are identical."""
    a, _ = build_shards(inp, RUN / "_determinism_a", timestamp)
    b, _ = build_shards(inp, RUN / "_determinism_b", timestamp)
    ids_a = [m["shard_id"] for m in a]
    ids_b = [m["shard_id"] for m in b]
    hashes_a = [m["sha256"] for m in a]
    hashes_b = [m["sha256"] for m in b]
    identical = ids_a == ids_b and hashes_a == hashes_b
    for d in (RUN / "_determinism_a", RUN / "_determinism_b"):
        for f in d.glob("shard-*.jsonl"):
            f.unlink()
        d.rmdir()
    return {
        "runs_compared": 2,
        "shard_ids_run1": ids_a[:3],
        "shard_ids_run2": ids_b[:3],
        "sha256_run1": hashes_a[:1],
        "sha256_run2": hashes_b[:1],
        "identical": identical,
        "contrast": (
            "V4 built ids from row_number() over a non-deterministic ordering, so the "
            "same input produced different ids on every run."
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="06-decontaminated.jsonl.gz")
    args = ap.parse_args()

    inp = STAGE / args.inp
    # Fixed by content, not by wall clock, so re-running is byte-identical.
    timestamp = dt.datetime.fromtimestamp(
        inp.stat().st_mtime, dt.timezone.utc
    ).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    out_dir = RUN / "shards"
    with Timer("shard + manifest"):
        manifests, agg = build_shards(inp, out_dir, timestamp)

    with Timer("determinism check"):
        det = determinism_check(inp, timestamp)
    assert det["identical"], "manifest ids/hashes are not reproducible across runs"

    # A deliberately incomplete manifest, to show the gate actually blocks.
    if manifests:
        blocked = dict(manifests[0])
        blocked["cleaning_scripts"] = []
        b_status, b_reasons = gate(blocked)
    else:
        b_status, b_reasons = "BLOCKED", ["no shards"]

    (RUN / "manifests.json").write_text(
        json.dumps(manifests, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    admitted = sum(1 for m in manifests if m["status"] == "ADMITTED")
    t = agg["totals"]
    stats = {
        "stage": "08-manifest",
        "docs_in": t.get("docs", 0),
        "docs_out": t.get("docs", 0),
        "docs_dropped": 0,
        "shards": t.get("shards", 0),
        "shards_admitted": admitted,
        "shards_blocked": t.get("shards", 0) - admitted,
        "docs_per_shard": DOCS_PER_SHARD,
        "required_fields": REQUIRED_FIELDS,
        "license_allowlist": sorted(LICENSE_ALLOWLIST),
        "license_class": LICENSE_CLASS,
        "cleaning_scripts": cleaning_scripts(),
        "tokens_total": t.get("tokens", 0),
        "words_total": t.get("words", 0),
        "naive_words_x_1_3_total": t.get("naive_estimate", 0),
        "naive_estimate_error_pct": (
            round(
                100
                * (t.get("naive_estimate", 0) - t.get("tokens", 0))
                / max(1, t.get("tokens", 1)),
                2,
            )
        ),
        "tokens_by_lang": agg["tokens_by_lang"],
        "fertility_tokens_per_word": agg["fertility_tokens_per_word"],
        "tokenizer": PRIMARY_TOKENIZER,
        "example_manifest_admitted": manifests[0] if manifests else None,
        "example_manifest_blocked": {
            "status": b_status,
            "gate_reasons": b_reasons,
            "what_was_removed": "cleaning_scripts",
        },
        "determinism": det,
    }
    write_stage_stats("08-manifest", stats)
    log(
        f"{t.get('shards', 0)} shards, {admitted} admitted | "
        f"{t.get('tokens', 0):,} tokens measured vs "
        f"{t.get('naive_estimate', 0):,} claimed by words x 1.3 "
        f"({stats['naive_estimate_error_pct']:+.1f}%) | determinism OK"
    )


if __name__ == "__main__":
    main()
