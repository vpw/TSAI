"""Stage 3 -- language identification and validation.

The folder tells you where a file came from, not what it contains. Sangraha encodes
language in the path (`verified/asm/...`), which is precisely what V4 trusted. Here
every document is detected at runtime with fastText lid.176 and validated against the
code its path claims; mismatches are quarantined before they reach the tokenizer.

Two things get measured that the "trust the path" pipeline cannot see at all:

  * the corrupted denominator -- per-language token totals as the folder claims them
    versus as detection finds them (a Bengali doc counted as Assamese inflates the
    Assamese budget and starves the Bengali one);
  * the `te` / `tel` code bug -- fastText emits ISO 639-1 while the folders use
    ISO 639-3, so a naive string comparison marks *every* Telugu document a mismatch.
    We count how many documents that bug would have destroyed.
"""

from __future__ import annotations

import argparse
import collections

from common import (
    ISO1_TO_ISO3,
    LANG_SCRIPT,
    MODELS,
    STAGE,
    Timer,
    jsonl_read,
    jsonl_write,
    log,
    script_profile,
    write_stage_stats,
)

LID_MODEL = "lid.176.ftz"
CONFIDENCE_FLOOR = 0.50  # below this the detector is not trusted either way
DETECT_CHARS = 3000  # prefix handed to the detector
BATCH = 2000

# A document whose claimed script holds the majority but which still carries this much
# Latin is code-switched (the "यह dataset बहुत large scale पर train होता है" case).
CODESWITCH_LO, CODESWITCH_HI = 0.15, 0.85


def load_detector():
    import fasttext

    return fasttext.load_model(str(MODELS / LID_MODEL))


def flatten(s: str) -> str:
    """fastText refuses newlines."""
    return s[:DETECT_CHARS].replace("\n", " ").replace("\r", " ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="01-normalized.jsonl.gz")
    ap.add_argument("--out", dest="out", default="02-langid.jsonl.gz")
    ap.add_argument("--quarantine", default="02-langid-quarantine.jsonl.gz")
    args = ap.parse_args()

    model = load_detector()

    docs_in = 0
    verdicts = collections.Counter()
    detected_pairs = collections.Counter()  # (claimed, detected) -> docs
    tokens_claimed = collections.Counter()  # folder-trust view of the corpus
    tokens_detected = collections.Counter()  # runtime-detection view
    naive_mismatch = 0  # what an ISO1-vs-ISO3 string compare would have thrown away
    codeswitched = 0
    romanized = 0
    low_conf = 0
    conf_sum = 0.0
    mismatch_examples: list[dict] = []
    codeswitch_examples: list[dict] = []
    quarantined: list[dict] = []

    def process():
        nonlocal docs_in, naive_mismatch, codeswitched, romanized, low_conf, conf_sum

        buf: list[dict] = []

        def flush():
            nonlocal naive_mismatch, codeswitched, romanized, low_conf, conf_sum
            if not buf:
                return
            labels, probs = model.predict([flatten(r["text"]) for r in buf], k=2)
            for rec, labs, prs in zip(buf, labels, probs):
                iso1 = labs[0].replace("__label__", "")
                conf = float(prs[0])
                iso3 = ISO1_TO_ISO3.get(iso1, iso1)
                claimed = rec["claimed_lang"]
                conf_sum += conf

                # The V4 bug, made visible: comparing the raw detector output to the
                # folder code without normalising 639-1 -> 639-3. Counted only where
                # the mapped code DOES agree, so this is purely the encoding bug --
                # documents a naive comparison would have thrown away for no reason.
                if iso1 != claimed and iso3 == claimed:
                    naive_mismatch += 1

                prof = script_profile(rec["text"])
                claimed_script = LANG_SCRIPT.get(claimed)
                latn = prof.get("Latn", 0.0)
                own = prof.get(claimed_script, 0.0) if claimed_script else 0.0

                rec["detected_lang"] = iso3
                rec["detect_conf"] = round(conf, 4)
                rec["script_profile"] = {k: round(v, 3) for k, v in prof.items()}

                if conf < CONFIDENCE_FLOOR:
                    verdict = "LOW_CONFIDENCE"
                    low_conf += 1
                elif iso3 == claimed:
                    if (
                        claimed_script
                        and claimed_script != "Latn"
                        and CODESWITCH_LO < latn < CODESWITCH_HI
                    ):
                        verdict = "CODE_SWITCHED"
                        codeswitched += 1
                        if len(codeswitch_examples) < 6:
                            codeswitch_examples.append(
                                {
                                    "id": rec["id"],
                                    "claimed": claimed,
                                    "detected": iso3,
                                    "latin_fraction": round(latn, 3),
                                    "excerpt": rec["text"][:220],
                                }
                            )
                    else:
                        verdict = "MATCH"
                elif claimed_script and claimed_script != "Latn" and own < 0.15 and latn > 0.6:
                    # Claimed an Indic language, written in Latin script.
                    verdict = "ROMANIZED"
                    romanized += 1
                else:
                    verdict = "MISMATCH"
                    if len(mismatch_examples) < 8:
                        mismatch_examples.append(
                            {
                                "id": rec["id"],
                                "src": rec["src"],
                                "claimed": claimed,
                                "detected": iso3,
                                "confidence": round(conf, 3),
                                "script_profile": rec["script_profile"],
                                "excerpt": rec["text"][:220],
                            }
                        )

                rec["lang_verdict"] = verdict
                verdicts[verdict] += 1
                detected_pairs[f"{claimed}->{iso3}"] += 1
                tokens_claimed[claimed] += rec["raw_tokens"]
                tokens_detected[iso3 if verdict != "LOW_CONFIDENCE" else claimed] += rec[
                    "raw_tokens"
                ]

                if verdict in ("MISMATCH", "ROMANIZED", "LOW_CONFIDENCE"):
                    quarantined.append(
                        {k: rec[k] for k in ("id", "src", "claimed_lang", "detected_lang", "lang_verdict")}
                    )
                    continue
                yield_buf.append(rec)
            buf.clear()

        yield_buf: list[dict] = []
        for rec in jsonl_read(STAGE / args.inp):
            docs_in += 1
            buf.append(rec)
            if len(buf) >= BATCH:
                flush()
                while yield_buf:
                    yield yield_buf.pop(0)
        flush()
        while yield_buf:
            yield yield_buf.pop(0)

    with Timer("language id"):
        n = jsonl_write(STAGE / args.out, process())
    jsonl_write(STAGE / args.quarantine, iter(quarantined))

    # The corrupted denominator, side by side.
    langs = sorted(set(tokens_claimed) | set(tokens_detected))
    denominator = {
        lang: {
            "tokens_if_folder_trusted": tokens_claimed.get(lang, 0),
            "tokens_after_detection": tokens_detected.get(lang, 0),
            "delta": tokens_detected.get(lang, 0) - tokens_claimed.get(lang, 0),
        }
        for lang in langs
    }

    stats = {
        "stage": "03-langid",
        "detector": f"fastText {LID_MODEL} (176 languages)",
        "confidence_floor": CONFIDENCE_FLOOR,
        "docs_in": docs_in,
        "docs_out": n,
        "docs_dropped": docs_in - n,
        "verdicts": dict(verdicts),
        "mean_confidence": round(conf_sum / max(1, docs_in), 4),
        "quarantined": len(quarantined),
        "code_switched": codeswitched,
        "romanized_indic": romanized,
        "low_confidence": low_conf,
        "mislabelled_caught_by_detection": verdicts.get("MISMATCH", 0),
        "mislabelled_caught_by_folder_trust": 0,
        "naive_iso1_vs_iso3_false_mismatches": naive_mismatch,
        "iso_code_map_used": ISO1_TO_ISO3,
        "claimed_to_detected_pairs": dict(detected_pairs.most_common(40)),
        "token_denominator": denominator,
        "mismatch_examples": mismatch_examples,
        "codeswitch_examples": codeswitch_examples,
    }
    write_stage_stats("03-langid", stats)
    log(
        f"{docs_in:,} -> {n:,} docs | mismatches {verdicts.get('MISMATCH', 0):,} | "
        f"code-switched {codeswitched:,} | romanized {romanized:,} | "
        f"naive te/tel bug would have flagged {naive_mismatch:,}"
    )


if __name__ == "__main__":
    main()
