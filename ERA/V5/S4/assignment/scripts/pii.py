"""Stage 6 -- PII scrubbing, in two layers with two different failure modes.

Layer 1, regex: emails, phone numbers, IPv4 addresses, Aadhaar-shaped 12-digit
numbers, PAN and GSTIN. Structured identifiers have an exact shape, so precision is
near-perfect and this is what Dolma did.

Layer 2, names: names have no fixed shape. A real pipeline uses NER (Presidio and
friends); with no GPU here this is a gazetteer of Indic and English given names plus
surnames, gated on capitalisation/context, with an explicit exclusion list of Indian
place names. That exclusion list is the whole point -- the session's false-positive
trap is मैसूर (Mysuru) sitting next to two real person names, and an aggressive name
layer masks the city.

Both layers are scored against a hand-labelled sample so precision and recall are
measured numbers with a stated sample size, not claims.
"""

from __future__ import annotations

import argparse
import collections
import json
import re

from common import (
    RUN,
    STAGE,
    Timer,
    count_tokens,
    jsonl_read,
    jsonl_write,
    log,
    write_stage_stats,
)

# ------------------------------------------------------- layer 1: regex

REGEX_LAYER = {
    "EMAIL": re.compile(
        r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}", re.UNICODE
    ),
    # Indian mobile numbers with or without +91 / 0 prefix, and landline forms.
    "PHONE": re.compile(
        r"(?<![\d])(?:\+?91[\-\s]?|0)?[6-9]\d{4}[\-\s]?\d{5}(?![\d])"
        r"|(?<![\d])\+?91[\-\s]?\d{2,4}[\-\s]?\d{6,8}(?![\d])"
    ),
    # The trailing guard is (?!\d|\.\d) and not (?![\d.]) -- an address at the end of
    # a sentence is followed by a full stop, and the stricter form silently misses it.
    "IPV4": re.compile(
        r"(?<![\d.])(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}"
        r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(?!\d|\.\d)"
    ),
    # 12 digits in 4-4-4 or solid form. Deliberately loose, then filtered below.
    "AADHAAR": re.compile(r"(?<![\d])[2-9]\d{3}[\s\-]?\d{4}[\s\-]?\d{4}(?![\d])"),
    "PAN": re.compile(r"(?<![A-Z0-9])[A-Z]{5}\d{4}[A-Z](?![A-Z0-9])"),
    "GSTIN": re.compile(r"(?<![A-Z0-9])\d{2}[A-Z]{5}\d{4}[A-Z]\d[A-Z]{2}(?![A-Z0-9])"),
    "CREDENTIAL": re.compile(
        r"(?i)\b(?:password|passwd|api[_\s\-]?key|secret[_\s\-]?key|token)\b\s*[:=]\s*\S{6,}"
    ),
}

MASK = {k: f"[{k}]" for k in REGEX_LAYER}

# ------------------------------------------------------- layer 2: names

# Given names and surnames common across Indian languages, in Latin and in native
# script. Small on purpose: a gazetteer that tries to be exhaustive becomes a
# false-positive machine, which is exactly the failure mode being measured.
GIVEN_NAMES = {
    "ananya", "rahul", "priya", "amit", "sneha", "vikram", "arjun", "kavya",
    "rohit", "deepak", "anjali", "suresh", "ramesh", "lakshmi", "meera", "aditya",
    "sanjay", "pooja", "manish", "neha", "rajesh", "sunita", "vijay", "shreya",
    "karthik", "divya", "harsha", "nikhil", "swati", "abhishek", "ritu", "gaurav",
    "राहुल", "अनन्या", "प्रिया", "अमित", "स्नेहा", "विक्रम", "अर्जुन", "काव्या",
    "রাহুল", "প্রিয়া", "অমিত", "স্নেহা", "অনন্যা",
    "రాహుల్", "ప్రియ", "అమిత్", "స్నేహ", "అనన్య",
}
SURNAMES = {
    "sharma", "verma", "iyer", "patel", "reddy", "nair", "gupta", "singh", "das",
    "bose", "chatterjee", "banerjee", "mukherjee", "rao", "menon", "pillai",
    "kulkarni", "joshi", "desai", "shah", "agarwal", "mehta", "bhat", "naidu",
    "शर्मा", "वर्मा", "गुप्ता", "सिंह", "पटेल", "मेहता",
    "শর্মা", "দাস", "বসু", "ব্যানার্জী",
    "శర్మ", "రెడ్డి", "నాయుడు", "రావు",
}

# The false-positive trap, made explicit. Place names, months, deities and honorifics
# collide with given names constantly in Indian text.
NOT_A_PERSON = {
    "mysuru", "mysore", "mumbai", "delhi", "chennai", "kolkata", "bengaluru",
    "bangalore", "hyderabad", "pune", "jaipur", "lucknow", "kerala", "punjab",
    "gujarat", "assam", "bihar", "odisha", "goa", "india", "bharat", "ganga",
    "himalaya", "ashoka", "gandhi", "nehru", "tagore", "ambedkar", "kalam",
    "मैसूर", "मुंबई", "दिल्ली", "कोलकाता", "बेंगलुरु", "भारत", "गंगा", "केरल",
    "মুম্বাই", "দিল্লি", "কলকাতা", "ভারত",
    "మైసూరు", "ముంబై", "ఢిల్లీ", "భారత", "హైదరాబాద్",
}

TOKEN_RE = re.compile(r"[\wऀ-෿؀-ۿ]+", re.UNICODE)
LATIN_CAP_RE = re.compile(r"^[A-Z][a-z]{2,}$")


def name_spans(text: str, aggressiveness: float) -> list[tuple[int, int, str]]:
    """Find personal-name spans. `aggressiveness` in [0,1] widens the net.

    At 0.30 (the session's default dial position) a token must be a known given name
    or a known surname preceded by a known given name. Above 0.60 any capitalised
    unknown word following a known given name is also taken, which is where the
    place-name false positives start.
    """
    spans: list[tuple[int, int, str]] = []
    toks = list(TOKEN_RE.finditer(text))
    prev_was_given = False
    for m in toks:
        w = m.group(0)
        low = w.lower()
        if low in NOT_A_PERSON:
            # Below the trip point the exclusion list protects place names, deities
            # and public figures. Above it the model "fires on weaker signals" and
            # the exclusion list stops being consulted -- which is precisely when it
            # starts masking a city. This is the trade the session's dial shows.
            if aggressiveness < 0.60:
                prev_was_given = False
                continue
            spans.append((m.start(), m.end(), "AMBIGUOUS"))
            prev_was_given = False
            continue
        if low in GIVEN_NAMES:
            spans.append((m.start(), m.end(), "GIVEN"))
            prev_was_given = True
            continue
        if low in SURNAMES and (prev_was_given or aggressiveness >= 0.45):
            spans.append((m.start(), m.end(), "SURNAME"))
            prev_was_given = False
            continue
        if (
            aggressiveness >= 0.60
            and prev_was_given
            and LATIN_CAP_RE.match(w)
        ):
            spans.append((m.start(), m.end(), "FOLLOWER"))
        prev_was_given = False
    return spans


def scrub(text: str, aggressiveness: float, use_names: bool = True):
    """Returns (scrubbed_text, {kind: count})."""
    counts: collections.Counter = collections.Counter()

    def sub(kind, rx, s):
        def repl(m):
            counts[kind] += 1
            return MASK[kind]

        return rx.sub(repl, s)

    out = text
    # Credentials first, then emails (an email inside a credential line is one hit).
    for kind in ("CREDENTIAL", "EMAIL", "IPV4", "GSTIN", "PAN", "AADHAAR", "PHONE"):
        out = sub(kind, REGEX_LAYER[kind], out)

    if use_names:
        spans = name_spans(out, aggressiveness)
        if spans:
            pieces, last = [], 0
            for s, e, _kind in spans:
                if s < last:
                    continue
                pieces.append(out[last:s])
                pieces.append("[NAME]")
                counts["NAME"] += 1
                last = e
            pieces.append(out[last:])
            out = "".join(pieces)
    return out, dict(counts)


# ------------------------------------------------- precision / recall fixtures
#
# Precision and recall need ground truth, and there is none for a web crawl. So the
# scrubber is scored two ways, separately, and neither is dressed up as the other:
#
#   1. LABELLED FIXTURES -- hand-written documents (including the session's own PII
#      example) where every true span is annotated. Gives real precision and recall,
#      including the false-positive trap: an Indic place name sitting next to two
#      person names.
#   2. CORPUS TALLY -- what actually fired across the real corpus, reported as counts
#      by class, with no precision claim attached.

FIXTURES = [
    {
        "name": "session example (forum post, en+hi)",
        "text": (
            "From Ananya Sharma (ananya.sharma@gmail.com), posting on the migration "
            "thread. Reply-to set to r.iyer@wipro.co.in. Callback number "
            "+91 98450 12345, request logged from 203.0.113.47. Ticket raised by "
            "राहुल वर्मा while travelling through मैसूर — the Mysuru data centre. "
            "No further contact details on file."
        ),
        "truth": [
            "Ananya", "Sharma", "ananya.sharma@gmail.com", "r.iyer@wipro.co.in",
            "+91 98450 12345", "203.0.113.47", "राहुल", "वर्मा",
        ],
        "traps": ["मैसूर", "Mysuru"],
    },
    {
        "name": "support ticket with credentials",
        "text": (
            "Reported by Priya Nair. Contact 9845012345 or priya.nair@example.org. "
            "Server at 10.0.0.14 rejected the login; api_key = sk_live_9f2b7c11a4 "
            "was rotated. PAN ABCDE1234F on file. Escalated from the Bengaluru office."
        ),
        "truth": [
            "Priya", "Nair", "9845012345", "priya.nair@example.org", "10.0.0.14",
            "api_key = sk_live_9f2b7c11a4", "ABCDE1234F",
        ],
        "traps": ["Bengaluru"],
    },
    {
        "name": "public-figure and place-name trap (no private PII)",
        "text": (
            "महात्मा गांधी और जवाहरलाल नेहरू के बारे में यह लेख दिल्ली और मुंबई के "
            "पाठकों के लिए है। अर्जुन महाभारत का पात्र है। Kerala and Punjab are "
            "states, not people."
        ),
        "truth": [],
        "traps": ["गांधी", "नेहरू", "दिल्ली", "मुंबई", "Kerala", "Punjab", "अर्जुन"],
    },
    {
        "name": "clean prose, no identifiers",
        "text": (
            "किरणजन्य संयोगक्रिया पौधों में होने वाली एक महत्वपूर्ण प्रक्रिया है जिसमें "
            "प्रकाश ऊर्जा रासायनिक ऊर्जा में बदल जाती है।"
        ),
        "truth": [],
        "traps": [],
    },
]


def score_fixtures(aggressiveness: float) -> dict:
    """Real precision/recall against annotated spans."""
    tp = fp = fn = 0
    trap_hits = 0
    per_case = []
    for fx in FIXTURES:
        text = fx["text"]
        found: list[str] = []
        # Same layer ordering as scrub(): structured identifiers are masked first, so
        # the name layer never re-fires on a name sitting inside an email address.
        masked = text
        for kind in ("CREDENTIAL", "EMAIL", "IPV4", "GSTIN", "PAN", "AADHAAR", "PHONE"):
            rx = REGEX_LAYER[kind]
            found.extend(m.group(0) for m in rx.finditer(masked))
            masked = rx.sub(MASK[kind], masked)
        for s, e, _k in name_spans(masked, aggressiveness):
            found.append(masked[s:e])

        truth = list(fx["truth"])
        c_tp = c_fp = 0
        for f in found:
            matched = next(
                (t for t in truth if t.strip() in f.strip() or f.strip() in t.strip()),
                None,
            )
            if matched:
                truth.remove(matched)
                c_tp += 1
            else:
                c_fp += 1
                if any(tr in f for tr in fx["traps"]):
                    trap_hits += 1
        tp += c_tp
        fp += c_fp
        fn += len(truth)
        per_case.append(
            {
                "case": fx["name"],
                "true_spans": len(fx["truth"]),
                "found": len(found),
                "tp": c_tp,
                "fp": c_fp,
                "missed": len(truth),
                "missed_spans": truth,
            }
        )
    prec = tp / (tp + fp) if (tp + fp) else None
    rec = tp / (tp + fn) if (tp + fn) else None
    return {
        "aggressiveness": aggressiveness,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "trap_spans_wrongly_masked": trap_hits,
        "precision": round(prec, 4) if prec is not None else None,
        "recall": round(rec, 4) if rec is not None else None,
        "f1": (
            round(2 * prec * rec / (prec + rec), 4)
            if prec and rec and (prec + rec)
            else None
        ),
        "per_case": per_case,
    }


# --------------------------------------------------------------- driver


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="04-dedup.jsonl.gz")
    ap.add_argument("--out", dest="out", default="05-pii.jsonl.gz")
    ap.add_argument("--aggressiveness", type=float, default=0.30)
    ap.add_argument("--sample", type=int, default=200)
    args = ap.parse_args()

    docs_in = 0
    masked = collections.Counter()
    docs_with_pii = 0
    docs_by_kind = collections.Counter()
    chars_removed = 0
    token_waste_before = token_waste_after = 0
    waste_docs = 0
    examples: list[dict] = []

    def records():
        nonlocal docs_in, docs_with_pii, chars_removed
        nonlocal token_waste_before, token_waste_after, waste_docs
        for rec in jsonl_read(STAGE / args.inp):
            docs_in += 1
            before = rec["text"]
            after, counts = scrub(before, args.aggressiveness)
            if counts:
                docs_with_pii += 1
                for k, v in counts.items():
                    masked[k] += v
                    docs_by_kind[k] += 1
                chars_removed += max(0, len(before) - len(after))
                # The compute-waste angle: identifiers are token-expensive.
                if waste_docs < 500:
                    token_waste_before += count_tokens(before[:4000])
                    token_waste_after += count_tokens(after[:4000])
                    waste_docs += 1
                if len(examples) < 8 and len(counts) >= 2:
                    # Only the scrubbed form is ever recorded.
                    examples.append(
                        {
                            "id": rec["id"],
                            "masked": counts,
                            "scrubbed_excerpt": after[:260],
                        }
                    )
            rec["text"] = after
            if counts:
                rec["pii_masked"] = counts
            yield rec

    with Timer("pii scrub"):
        n = jsonl_write(STAGE / args.out, records())

    log("scoring both layers against the labelled fixtures…")
    curve = [score_fixtures(a) for a in (0.10, 0.30, 0.45, 0.60, 0.80)]
    for c in curve:
        log(
            f"  dial {c['aggressiveness']:.2f}: P={c['precision']} R={c['recall']} "
            f"traps wrongly masked={c['trap_spans_wrongly_masked']}"
        )
    # The most frequent "names" in an Indic web corpus are public figures, not
    # private individuals. Measure that rather than assume it away.
    public_figure_surfaces = sorted(
        s for s in NOT_A_PERSON if s in {"gandhi", "nehru", "tagore", "ambedkar", "kalam"}
    )

    stats = {
        "stage": "06-pii",
        "regex_classes": list(REGEX_LAYER),
        "name_layer": {
            "kind": "gazetteer + context gating (no GPU available for NER)",
            "given_names": len(GIVEN_NAMES),
            "surnames": len(SURNAMES),
            "place_name_exclusions": len(NOT_A_PERSON),
            "aggressiveness_used": args.aggressiveness,
        },
        "docs_in": docs_in,
        "docs_out": n,
        "docs_dropped": 0,
        "docs_with_pii": docs_with_pii,
        "spans_masked_by_kind": dict(masked),
        "spans_masked_total": sum(masked.values()),
        "chars_removed": chars_removed,
        "token_waste": {
            "docs_measured": waste_docs,
            "tokens_before": token_waste_before,
            "tokens_after": token_waste_after,
            "pct_saved": (
                round(
                    100 * (token_waste_before - token_waste_after) / token_waste_before,
                    2,
                )
                if token_waste_before
                else 0
            ),
        },
        "precision_curve": curve,
        "fixture_cases": [f["name"] for f in FIXTURES],
        "fixture_spans_annotated": sum(len(f["truth"]) for f in FIXTURES),
        "fixture_traps_annotated": sum(len(f["traps"]) for f in FIXTURES),
        "scoring_note": (
            "Precision and recall come from hand-annotated fixtures, including the "
            "session's own PII example -- a web crawl has no ground truth. The corpus "
            "numbers above are a tally of what fired, with no precision claim "
            "attached to them. The two are reported separately on purpose."
        ),
        "public_figure_caveat": (
            "The name layer cannot tell a private individual from a public figure, "
            "and in an Indic web corpus the most frequent person-names are public "
            "figures, deities and mythological characters. Masking those costs real "
            "knowledge; keeping them risks real people. The exclusion list holds "
            f"{len(NOT_A_PERSON)} entries including {', '.join(public_figure_surfaces)}, "
            "which is a blunt instrument and is reported as one."
        ),
        "examples": examples,
    }
    write_stage_stats("06-pii", stats)
    (RUN / "pii_precision_curve.json").write_text(
        json.dumps(curve, indent=2), encoding="utf-8"
    )
    log(
        f"{docs_in:,} docs | {sum(masked.values()):,} spans masked in "
        f"{docs_with_pii:,} docs | {dict(masked)}"
    )


if __name__ == "__main__":
    main()
