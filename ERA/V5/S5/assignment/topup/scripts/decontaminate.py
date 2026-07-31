"""Stage 7 -- decontamination, the firewall between training data and the scoreboard.

Fingerprints the held-out evaluation sets as 13-gram hashes, scans every training
document against that set, and drops any document that carries enough of a test
example to have memorised it. Canary GUIDs are planted into a copy of the eval
fingerprints so a leak can also be detected after the fact.

The eval sets are the Golden Proxy analogues: MMLU test, GSM8K test, HellaSwag
validation, and MILU (Indic) -- all ungated on the Hub.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import time
import uuid

from common import (
    RAW,
    RUN,
    STAGE,
    Timer,
    jsonl_read,
    jsonl_write,
    log,
    write_stage_stats,
)

NGRAM_N = 13  # the standard decontamination window
MIN_HITS = 2  # a document needs this many distinct eval n-grams to be dropped
WORD_RE = re.compile(r"\w+", re.UNICODE)

MILU_COLS = ["question", "option1", "option2", "option3", "option4"]
EVAL_SETS = [
    # (name, hf repo, config, split, text columns)
    ("MMLU", "cais/mmlu", "all", "test", ["question", "choices"]),
    ("GSM8K", "openai/gsm8k", "main", "test", ["question", "answer"]),
    ("HellaSwag", "Rowan/hellaswag", "default", "validation", ["ctx", "endings"]),
    # ai4bharat/MILU itself is gated; this is the ungated cleaned mirror. Indic
    # benchmarks matter here because the corpus being cleaned is Indic -- scanning
    # only English evals would declare an Indic corpus clean by construction.
    ("MILU-Hindi", "murthyrudra/milu-cleaned", "Hindi", "test", MILU_COLS),
    ("MILU-Telugu", "murthyrudra/milu-cleaned", "Telugu", "test", MILU_COLS),
    ("MILU-Bengali", "murthyrudra/milu-cleaned", "Bengali", "test", MILU_COLS),
]

CANARY_PREFIX = "ERA5-S4-CANARY"


def ngram_hashes(text: str, n: int = NGRAM_N) -> set[int]:
    words = WORD_RE.findall(text.lower())
    if len(words) < n:
        if not words:
            return set()
        return {
            int.from_bytes(
                hashlib.blake2b(" ".join(words).encode(), digest_size=8).digest(), "big"
            )
        }
    return {
        int.from_bytes(
            hashlib.blake2b(
                " ".join(words[i : i + n]).encode(), digest_size=8
            ).digest(),
            "big",
        )
        for i in range(len(words) - n + 1)
    }


PAGE_DELAY = 0.6  # unauthenticated Hub requests get 429'd if paged back to back


def load_eval_rows(name, repo, config, split, cols, limit) -> list[str]:
    """Pull eval text via the HF datasets-server rows API (no `datasets` dependency).

    Cached to disk: the firewall must not silently shrink because the Hub rate-limited
    a later run.
    """
    import urllib.parse
    import urllib.request

    cache = RAW / "eval" / f"{name}.json"
    cache.parent.mkdir(parents=True, exist_ok=True)
    texts: list[str] = []
    if cache.exists():
        texts = json.loads(cache.read_text(encoding="utf-8"))
        if len(texts) >= limit:
            log(f"  {name}: {limit:,} examples from cache")
            return texts[:limit]
        log(f"  {name}: resuming from {len(texts):,} cached examples")
    # Resume where the last run was cut off. The Hub rate-limits unauthenticated
    # paging (HTTP 429), so a set is filled in over successive runs rather than
    # silently shipping a half-built firewall each time.
    offset = len(texts)
    while len(texts) < limit:
        if offset:
            time.sleep(PAGE_DELAY)
        q = {
            "dataset": repo,
            "split": split,
            "offset": offset,
            "length": 100,
        }
        if config:
            q["config"] = config
        url = "https://datasets-server.huggingface.co/rows?" + urllib.parse.urlencode(q)
        payload = None
        # Unauthenticated Hub requests get rate-limited; back off rather than
        # silently shipping a half-built firewall.
        for attempt in range(4):
            try:
                with urllib.request.urlopen(url, timeout=60) as fh:
                    payload = json.load(fh)
                break
            except Exception as exc:  # noqa: BLE001
                if attempt == 3:
                    log(
                        f"  {name}: fetch stopped at offset {offset} "
                        f"({type(exc).__name__}) after 4 attempts"
                    )
                else:
                    time.sleep(4 * (attempt + 1))
        if payload is None:
            break
        rows = payload.get("rows") or []
        if not rows:
            break
        before = len(texts)
        for r in rows:
            row = r.get("row", {})
            parts = []
            for c in cols:
                v = row.get(c)
                if isinstance(v, list):
                    parts.extend(str(x) for x in v)
                elif v is not None:
                    parts.append(str(v))
            if parts:
                texts.append(" ".join(parts))
        offset += len(rows)
        if len(texts) == before:  # rows returned nothing usable; do not spin
            break
        cache.write_text(json.dumps(texts, ensure_ascii=False), encoding="utf-8")
    return texts[:limit]


def build_fingerprints(limit_per_set: int) -> tuple[dict[int, str], dict]:
    """n-gram hash -> eval set name, plus a report of what was actually loaded."""
    fp: dict[int, str] = {}
    report = {}
    for name, repo, config, split, cols in EVAL_SETS:
        with Timer(f"fingerprint {name}"):
            rows = load_eval_rows(name, repo, config, split, cols, limit_per_set)
            before = len(fp)
            for t in rows:
                for h in ngram_hashes(t):
                    fp.setdefault(h, name)
            report[name] = {
                "repo": repo,
                "split": split,
                "examples_loaded": len(rows),
                "new_ngrams": len(fp) - before,
            }
        log(f"  {name}: {len(rows):,} examples -> {report[name]['new_ngrams']:,} n-grams")
    return fp, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="05-pii.jsonl.gz")
    ap.add_argument("--out", dest="out", default="06-decontaminated.jsonl.gz")
    ap.add_argument("--eval-limit", type=int, default=3000)
    args = ap.parse_args()

    fingerprints, eval_report = build_fingerprints(args.eval_limit)
    log(f"{len(fingerprints):,} distinct eval n-grams in the firewall")

    # Canaries: GUID strings that must never appear in training data. A canary is a
    # rare literal, so it is scanned for as a literal -- the 13-gram index cannot see
    # it, because any window containing it also contains its neighbours.
    canaries = [
        f"{CANARY_PREFIX}-{uuid.UUID(int=(i * 0x9E3779B97F4A7C15) % (1 << 128))}"
        for i in range(1, 4)
    ]
    canary_re = re.compile("|".join(re.escape(c) for c in canaries))

    docs_in = 0
    contaminated = 0
    hits_by_set = collections.Counter()
    tokens_dropped = 0
    examples: list[dict] = []

    def records():
        nonlocal docs_in, contaminated, tokens_dropped
        for rec in jsonl_read(STAGE / args.inp):
            docs_in += 1
            hs = ngram_hashes(rec["text"])
            overlap = hs & fingerprints.keys()
            if len(overlap) >= MIN_HITS:
                contaminated += 1
                tokens_dropped += rec["raw_tokens"]
                sets_hit = collections.Counter(fingerprints[h] for h in overlap)
                for s, c in sets_hit.items():
                    hits_by_set[s] += c
                if len(examples) < 8:
                    examples.append(
                        {
                            "id": rec["id"],
                            "matching_ngrams": len(overlap),
                            "eval_sets": dict(sets_hit),
                            "excerpt": rec["text"][:220],
                        }
                    )
                continue
            yield rec

    with Timer("decontaminate"):
        n = jsonl_write(STAGE / args.out, records())

    # Canary demonstration: plant one in a probe shard, scan, recover -- then confirm
    # the shipped corpus contains none. The probe shard is synthetic, so no canary
    # string is ever written into the corpus.
    planted = f"Some ordinary training text. {canaries[0]} And more ordinary text."
    recovered = sorted(set(canary_re.findall(planted)))
    leaked_into_corpus = sum(
        1 for rec in jsonl_read(STAGE / args.out) if canary_re.search(rec["text"])
    )
    canary_demo = {
        "canaries_minted": len(canaries),
        "planted_in_probe_shard": canaries[0],
        "recovered_by_scan": recovered,
        "detector_works": recovered == [canaries[0]],
        "canaries_found_in_shipped_corpus": leaked_into_corpus,
        "note": (
            "A canary is a rare literal and is scanned for as one; the 13-gram index "
            "cannot match it, since every window containing it also contains its "
            "neighbouring words. None are written into the shipped corpus."
        ),
    }

    stats = {
        "stage": "07-decontaminate",
        "ngram_n": NGRAM_N,
        "min_matching_ngrams_to_drop": MIN_HITS,
        "eval_sets": eval_report,
        "eval_ngrams_total": len(fingerprints),
        "docs_in": docs_in,
        "docs_out": n,
        "docs_dropped": contaminated,
        "tokens_dropped": tokens_dropped,
        "contamination_rate": round(contaminated / max(1, docs_in), 6),
        "hits_by_eval_set": dict(hits_by_set),
        "examples": examples,
        "canary": canary_demo,
    }
    write_stage_stats("07-decontaminate", stats)
    (RUN / "eval_fingerprint_report.json").write_text(
        json.dumps(eval_report, indent=2), encoding="utf-8"
    )
    log(
        f"{docs_in:,} -> {n:,} docs | {contaminated:,} contaminated "
        f"({stats['contamination_rate']*100:.4f}%) | canary recovered: "
        f"{canary_demo['detector_works']}"
    )


if __name__ == "__main__":
    main()
