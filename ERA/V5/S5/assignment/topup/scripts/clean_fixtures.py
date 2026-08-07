"""Correctness fixtures for clean_text() -- the session's own worked examples.

Sangraha turns out to be fairly clean at the character level: noise appears in
roughly one document in a thousand. That is a real finding, but it means the corpus
alone cannot demonstrate that the cleaner handles each noise class correctly, and in
particular cannot demonstrate the rule that matters most -- that ZWNJ and ZWJ survive
while every other invisible character does not.

So the cleaner is also run against fixtures rebuilt from the session's Widget 2
examples, with the expected outcome asserted per case. These are labelled as fixtures
everywhere they appear; they are not corpus measurements.

    python scripts/clean_fixtures.py
"""

from __future__ import annotations

import json

from common import RUN, log, write_stage_stats
from normalize import (
    CLASS_RES,
    JOINERS,
    clean_text,
    scan_ghost_tags,
    unify_format,
)

ZWSP, BOM, RLO, ZWNJ, ZWJ, FFFD = (
    "​",
    "﻿",
    "‮",
    "‌",
    "‍",
    "�",
)

FIXTURES = [
    {
        "name": "English web scrape",
        "kind": "web",
        "text": (
            "Cookie notice: We &amp; our 47 partners use cookies. By clicking "
            "&quot;Accept&quot; you consent. It&#8217;s the crawler&#8217;s job to "
            f"fetch every page from the open web. Encoding glitch here {FFFD} "
            "corrupted one byte.\tTabs and    multiple     spaces should collapse."
        ),
        "expect": {
            "no_entities": True,
            "no_replacement": True,
            "single_spaced": True,
        },
    },
    {
        "name": "Chat log (ghost tags)",
        "kind": "conversation",
        "text": (
            "[SYSTEM] You are a helpful data-cleaning assistant.\n"
            "[USER] How do I dedupe a 15-trillion-token corpus?\n"
            "[ASSISTANT] Use MinHash plus LSH, then hash AFTER cleaning.\n"
            "<USER>raw serialized conversation turn</USER>"
        ),
        "expect": {"ghost_flagged": True, "unifies": True},
    },
    {
        "name": "Hindi (Devanagari)",
        "kind": "indic",
        "text": (
            f"{BOM}नमस्ते &amp; डेटा सफाई में स्वागत है। संयुक्त अक्षर: क{ZWNJ}् — ZWNJ ko "
            f"rakhna zaroori.{ZWSP} जोड़ने वाला: र{ZWJ} — ZWJ real Eyelash-Ra. "
            "HTML entity: &#2309; = अ, aur BOM va ZWSP hataane chaahiye."
        ),
        "expect": {"keeps_zwnj": True, "keeps_zwj": True, "drops_bom_zwsp": True},
    },
    {
        "name": "Telugu with bidi override",
        "kind": "indic",
        "text": (
            f"కిరణజన్య సంయోగక్రియ. ZWNJ (క{ZWNJ}్) stops the conjunct, keep it. "
            f"ZWJ (ర{ZWJ}) is valid.\x07 Bidi override {RLO}reversed and stripped."
        ),
        "expect": {"keeps_zwnj": True, "keeps_zwj": True, "drops_bidi_control": True},
    },
    {
        "name": "Code snippet",
        "kind": "code",
        "text": (
            "def clean(s):\n"
            "    if len(s) &gt; 0 &amp;&amp; s[0] &lt; limit:\n"
            "        return s.strip()   # nbsp-padded comment\n"
            "    else:\n"
            "        return None  # CRLF + tab-prefixed\r\n"
        ),
        "expect": {"no_entities": True, "unescapes_operators": True},
    },
    {
        "name": "Mixed / worst case (stress)",
        "kind": "stress",
        "text": (
            f"{BOM}Mixed{ZWSP} case: &amp;amp; double-escaped, {FFFD} bad byte, "
            f"\x01control, {RLO}bidi, private-use, "
            f"Indic joiners क{ZWNJ}् and र{ZWJ} must survive.   [USER] ghost."
        ),
        "expect": {
            "keeps_zwnj": True,
            "keeps_zwj": True,
            "no_replacement": True,
            "ghost_flagged": True,
        },
    },
]


def check(fx: dict) -> dict:
    raw = fx["text"]
    cleaned = clean_text(raw)
    removed = {
        name: len(rx.findall(raw)) for name, rx in CLASS_RES.items() if rx.search(raw)
    }
    ghosts = scan_ghost_tags(cleaned)
    exp = fx["expect"]
    checks = {}

    if exp.get("no_entities"):
        checks["no_entities"] = "&amp;" not in cleaned and "&gt;" not in cleaned
    if exp.get("unescapes_operators"):
        checks["unescapes_operators"] = ">" in cleaned and "&&" in cleaned
    if exp.get("no_replacement"):
        checks["no_replacement"] = FFFD not in cleaned
    if exp.get("single_spaced"):
        checks["single_spaced"] = "  " not in cleaned and "\t" not in cleaned
    if exp.get("keeps_zwnj"):
        checks["keeps_zwnj"] = cleaned.count(ZWNJ) == raw.count(ZWNJ) > 0
    if exp.get("keeps_zwj"):
        checks["keeps_zwj"] = cleaned.count(ZWJ) == raw.count(ZWJ) > 0
    if exp.get("drops_bom_zwsp"):
        checks["drops_bom_zwsp"] = BOM not in cleaned and ZWSP not in cleaned
    if exp.get("drops_bidi_control"):
        checks["drops_bidi_control"] = RLO not in cleaned and "\x07" not in cleaned
    if exp.get("ghost_flagged"):
        checks["ghost_flagged"] = bool(ghosts)
    if exp.get("unifies"):
        checks["unifies"] = "<|user|>" in unify_format(cleaned)

    return {
        "name": fx["name"],
        "kind": fx["kind"],
        "chars_before": len(raw),
        "chars_after": len(cleaned),
        "noise_removed_by_class": removed,
        "joiners_kept": {
            k: cleaned.count(v) for k, v in JOINERS.items() if cleaned.count(v)
        },
        "ghost_tags_flagged": ghosts,
        "before": raw,
        "after": cleaned,
        "unified": unify_format(cleaned) if fx["expect"].get("unifies") else None,
        "checks": checks,
        "passed": all(checks.values()),
    }


def main():
    results = [check(fx) for fx in FIXTURES]
    passed = sum(r["passed"] for r in results)
    stats = {
        "stage": "fixtures",
        "purpose": (
            "Correctness fixtures for clean_text(), rebuilt from the session's own "
            "Widget 2 examples. These are FIXTURES, not corpus measurements."
        ),
        "cases": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "results": results,
    }
    write_stage_stats("fixtures", stats)
    (RUN / "clean_fixtures.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for r in results:
        mark = "ok " if r["passed"] else "FAIL"
        failed = [k for k, v in r["checks"].items() if not v]
        log(
            f"{mark} {r['name']:32s} {r['chars_before']}->{r['chars_after']} chars"
            + (f"  failed: {failed}" if failed else "")
        )
    assert passed == len(results), f"{len(results) - passed} clean_text fixtures failed"


if __name__ == "__main__":
    main()
