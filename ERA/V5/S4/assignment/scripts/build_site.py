"""Generate site/index.html from the run stats and the strategy taxonomy.

Every number on the page comes from data/run/stats.json. Nothing is typed into the
HTML by hand -- if a figure appears on the page, a stage produced it.

The output is a single self-contained file: data inlined as JSON, no fetch(), no CDN,
no external font. It therefore opens correctly from file:// as well as from Netlify.

    python scripts/build_site.py
"""

from __future__ import annotations

import json
from pathlib import Path

import taxonomy
from common import ROOT, RUN
from pii import scrub

TEMPLATE = Path(__file__).resolve().parent / "site_template.html"
OUT = ROOT / "site" / "index.html"
PLACEHOLDER = "/*__S4_DATA__*/"

# Stages that run before the PII scrubber capture their examples from unredacted
# text, so every corpus excerpt is scrubbed again on its way into the page. The
# clean_text() fixtures are exempt: they are synthetic, and their whole purpose is
# to show the cleaner's behaviour character by character.
EXCERPT_KEYS = {"excerpt", "scrubbed_excerpt", "text"}


def redact_excerpts(node):
    """Recursively scrub every corpus excerpt in a stats subtree."""
    if isinstance(node, dict):
        return {
            k: (scrub(v, 0.30)[0] if k in EXCERPT_KEYS and isinstance(v, str)
                else redact_excerpts(v))
            for k, v in node.items()
        }
    if isinstance(node, list):
        return [redact_excerpts(x) for x in node]
    return node


def compact(stats: dict) -> dict:
    """Shape the run stats into exactly what the page renders."""
    st = stats["stages"]
    fetch = st["00-fetch"]
    norm = st["02-normalize"]
    lid = st["03-langid"]
    qual = st["04-quality-heuristics"]
    clf = st.get("04b-quality-classifier", {})
    dedup = st["05-dedup"]
    pii = st["06-pii"]
    decon = st["07-decontaminate"]
    man = st["08-manifest"]
    fixtures = st.get("fixtures", {})

    return {
        "meta": {
            "generated": stats["generated_at_utc"],
            "smoke": stats["smoke"],
            "runtime_s": stats["total_runtime_seconds"],
            "durations": stats["durations_seconds"],
        },
        "counts": taxonomy.COUNTS,
        "stages": taxonomy.STAGES,
        "techniques": [
            {"name": n, "stage": s, "what": w, "status": st_}
            for n, s, w, st_ in taxonomy.TECHNIQUES
        ],
        "concerns": taxonomy.CONCERNS,
        "rationale": [
            {"stat": a, "why": b} for a, b in taxonomy.STATISTICS_RATIONALE
        ],
        "citations": taxonomy.CITATIONS,
        "session_yield": [
            {"stage": a, "pct": b} for a, b in taxonomy.SESSION_YIELD
        ],
        "headline": stats["headline"],
        "descent": stats["yield_descent"],
        "integrity": stats["integrity"],
        "dataset": {
            "id": fetch["dataset"],
            "revision": fetch["revision"],
            "license": fetch["license"],
            "sources": [
                {
                    "key": s["key"],
                    "pool": s["pool"],
                    "claimed": s["claimed_lang"],
                    "path": s["repo_path"],
                    "rows_in_file": s["rows_in_file"],
                    "mb": round(s["bytes"] / 1e6, 1),
                    "sha256": s["sha256"],
                    "docs": s["docs_taken"],
                    "tokens": s["raw_tokens"],
                    "sampling": s.get("sampling", {}),
                }
                for s in fetch["sources"].values()
            ],
        },
        "normalize": {
            "docs_in": norm["docs_in"],
            "docs_out": norm["docs_out"],
            "noise_by_class": norm["noise_chars_removed_by_class"],
            "noise_total": norm["noise_chars_removed_total"],
            "joiners_kept": norm["joiners_kept"],
            "entity_docs": norm["docs_with_html_entities"],
            "nfc_docs": norm["docs_changed_by_nfc"],
            "ghost_docs": norm["ghost_tag_docs"],
            "ghost_by_kind": norm["ghost_tag_hits_by_kind"],
            "ghost_total": norm["ghost_tag_hits_total"],
            "unified": norm["docs_format_unified"],
            "garbage_before": norm["garbage_vocab_slots_before"],
            "garbage_after": norm["garbage_vocab_slots_after"],
            "byte_fallback": norm.get("byte_fallback_slots_before", 0),
            "garbage_scanned": norm["garbage_audit_docs_scanned"],
            "chars_removed": norm["chars_removed"],
            "examples": norm["examples"][:4],
        },
        "fixtures": {
            "cases": fixtures.get("cases", 0),
            "passed": fixtures.get("passed", 0),
            "results": [
                {
                    "name": r["name"],
                    "kind": r["kind"],
                    "before": r["before"],
                    "after": r["after"],
                    "chars": [r["chars_before"], r["chars_after"]],
                    "removed": r["noise_removed_by_class"],
                    "joiners": r["joiners_kept"],
                    "ghosts": r["ghost_tags_flagged"],
                    "unified": r["unified"],
                    "checks": r["checks"],
                }
                for r in fixtures.get("results", [])
            ],
        },
        "langid": {
            "detector": lid["detector"],
            "docs_in": lid["docs_in"],
            "docs_out": lid["docs_out"],
            "verdicts": lid["verdicts"],
            "mean_conf": lid["mean_confidence"],
            "mismatches": lid["mislabelled_caught_by_detection"],
            "folder_trust_catches": lid["mislabelled_caught_by_folder_trust"],
            "naive_iso_bug": lid["naive_iso1_vs_iso3_false_mismatches"],
            "code_switched": lid["code_switched"],
            "romanized": lid["romanized_indic"],
            "denominator": lid["token_denominator"],
            "pairs": lid["claimed_to_detected_pairs"],
            "mismatch_examples": lid["mismatch_examples"][:5],
            "codeswitch_examples": lid["codeswitch_examples"][:3],
        },
        "quality": {
            "rules": qual["rules"],
            "thresholds": qual["thresholds_base"],
            "overrides": qual["script_overrides"],
            "docs_in": qual["docs_in"],
            "docs_out": qual["docs_out"],
            "kept": qual["kept_by_config"],
            "keep_rate": qual["keep_rate_by_config"],
            "failures": qual["rule_failures_by_config"],
            "rescued": qual["rescued_docs_total"],
            "rescued_tokens": qual["rescued_tokens_total"],
            "rescued_by_lang": qual["rescued_by_lang"],
            "rescued_by_rule": qual["rescued_by_english_rule_that_killed_them"],
            "per_lang": qual["per_lang"],
            "examples": qual["rescued_examples"][:4],
            "stopword_sizes": qual["stopword_list_sizes"],
        },
        "classifier": {
            "skipped": clf.get("skipped", True),
            "model": clf.get("model"),
            "labeller": clf.get("labeller"),
            "threshold": clf.get("threshold"),
            "dist": clf.get("weak_label_distribution", {}),
            "train": clf.get("train_size"),
            "test": clf.get("test_size"),
            "acc": clf.get("heldout_exact_accuracy"),
            "acc1": clf.get("heldout_within_one_accuracy"),
            "docs_in": clf.get("docs_in"),
            "docs_out": clf.get("docs_out"),
            "hist": clf.get("score_histogram", {}),
            "per_lang": clf.get("per_lang", {}),
            "caveat": clf.get("caveat"),
        },
        "dedup": {
            "config": dedup["config"],
            "demo": dedup["session_demo_config"],
            "docs_in": dedup["docs_in"],
            "docs_out": dedup["docs_out"],
            "exact": dedup["exact_duplicates"],
            "exact_cross": dedup["exact_duplicates_cross_shard"],
            "near": dedup["near_duplicates"],
            "near_cross": dedup["near_duplicates_cross_shard"],
            "candidates": dedup["lsh_candidate_pairs_global"],
            "local_caught": dedup["caught_by_local_passes_only"],
            "global_caught": dedup["caught_by_global_pass"],
            "only_global": dedup["missed_by_local_caught_by_global"],
            "tokens_dropped": dedup["tokens_dropped"],
            "by_source": dedup["dropped_by_source"],
            "pairs": dedup["confirmed_pairs"][:6],
            "memory": dedup["index_memory"],
        },
        "pii": {
            "classes": pii["regex_classes"],
            "layer2": pii["name_layer"],
            "docs_in": pii["docs_in"],
            "docs_with_pii": pii["docs_with_pii"],
            "by_kind": pii["spans_masked_by_kind"],
            "total": pii["spans_masked_total"],
            "waste": pii["token_waste"],
            "curve": [
                {k: v for k, v in c.items() if k != "per_case"}
                for c in pii["precision_curve"]
            ],
            "per_case": next(
                (c["per_case"] for c in pii["precision_curve"] if c["aggressiveness"] == 0.30),
                [],
            ),
            "fixture_spans": pii["fixture_spans_annotated"],
            "fixture_traps": pii["fixture_traps_annotated"],
            "scoring_note": pii["scoring_note"],
            "public_figure_caveat": pii["public_figure_caveat"],
            "examples": pii["examples"][:4],
        },
        "decontaminate": {
            "n": decon["ngram_n"],
            "min_hits": decon["min_matching_ngrams_to_drop"],
            "eval_sets": decon["eval_sets"],
            "ngrams": decon["eval_ngrams_total"],
            "docs_in": decon["docs_in"],
            "dropped": decon["docs_dropped"],
            "rate": decon["contamination_rate"],
            "by_set": decon["hits_by_eval_set"],
            "examples": decon["examples"][:4],
            "canary": decon["canary"],
        },
        "manifest": {
            "shards": man["shards"],
            "admitted": man["shards_admitted"],
            "blocked": man["shards_blocked"],
            "docs_per_shard": man["docs_per_shard"],
            "required": man["required_fields"],
            "allowlist": man["license_allowlist"],
            "scripts": man["cleaning_scripts"],
            "tokens": man["tokens_total"],
            "words": man["words_total"],
            "naive": man["naive_words_x_1_3_total"],
            "naive_err": man["naive_estimate_error_pct"],
            "by_lang": man["tokens_by_lang"],
            "fertility": man["fertility_tokens_per_word"],
            "tokenizer": man["tokenizer"],
            "example": man["example_manifest_admitted"],
            "blocked_example": man["example_manifest_blocked"],
            "determinism": man["determinism"],
        },
    }


def main():
    stats = json.loads((RUN / "stats.json").read_text(encoding="utf-8"))
    data = compact(stats)
    fixtures = data.pop("fixtures")  # synthetic; must not be scrubbed
    data = redact_excerpts(data)
    data["fixtures"] = fixtures
    # The headline counts are claims about the lists below them; keep them honest.
    assert data["counts"]["techniques"] == len(data["techniques"])
    assert data["counts"]["named_strategies"] == len(data["stages"])
    assert data["counts"]["cross_cutting_concerns"] == len(data["concerns"])
    blob = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    html = TEMPLATE.read_text(encoding="utf-8")
    if PLACEHOLDER not in html:
        raise SystemExit(f"placeholder {PLACEHOLDER} missing from {TEMPLATE}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html.replace(PLACEHOLDER, blob), encoding="utf-8")
    kb = OUT.stat().st_size / 1024
    print(f"wrote {OUT} ({kb:.0f} KB, data {len(blob)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
