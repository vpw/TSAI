"""Stage 4a -- the Gopher/C4 heuristic cascade, run twice.

The nine rules are the session's, at the session's exact thresholds. They are applied
under two configurations over the same documents:

  english_tuned  -- one English stop-word list and one set of thresholds for every
                    language, which is what V4's selector effectively did;
  script_aware   -- per-language stop-word lists, per-script word-count floors and
                    mean-word-length bands, and sentence terminators that include the
                    danda. Same nine rules, calibrated per script.

The difference between the two is the headline number of this assignment: how many
genuinely good Indic documents an English-tuned filter destroys. The session asserts
this happens; here it is counted.
"""

from __future__ import annotations

import argparse
import collections
import re

from common import (
    LANG_SCRIPT,
    STAGE,
    Timer,
    jsonl_read,
    jsonl_write,
    log,
    write_stage_stats,
)

# --------------------------------------------------------------- stop-words

STOPWORDS = {
    "eng": {
        "the", "be", "to", "of", "and", "that", "have", "with", "is", "in", "it",
        "for", "not", "on", "as", "are", "was", "this", "by", "from", "or", "an",
        "but", "they", "we", "you", "he", "she", "at", "which", "their", "has",
    },
    "hin": {
        "का", "के", "की", "है", "में", "से", "हैं", "को", "पर", "इस", "कि", "जो",
        "और", "ने", "नहीं", "तो", "ही", "या", "हो", "था", "एक", "यह", "भी", "लिए",
        "तक", "साथ", "बाद", "कुछ", "थे", "वह", "करने", "गया", "होता", "रहा",
    },
    "mar": {
        "आणि", "आहे", "या", "तो", "ती", "ते", "एक", "मध्ये", "पण", "होते", "करून",
        "आहेत", "साठी", "नाही", "हे", "त्या", "म्हणून", "व", "का", "असे", "केले",
    },
    "nep": {
        "र", "छ", "मा", "को", "का", "की", "हो", "यो", "त्यो", "भन्ने", "पनि",
        "गर्न", "भएको", "छन्", "लागि", "तर", "एक", "हुन", "गरेको", "थियो",
    },
    "san": {
        "च", "एव", "तु", "हि", "अपि", "यत्", "तत्", "स्य", "इति", "न", "वा",
        "सः", "किम्", "अथ", "यथा", "तथा",
    },
    "ben": {
        "এবং", "এই", "সেই", "করা", "হয়", "থেকে", "যে", "তার", "তিনি", "আমি",
        "আমরা", "না", "ও", "কিন্তু", "জন্য", "সঙ্গে", "মধ্যে", "একটি", "করে", "হয়েছে",
        "তা", "কে", "এর", "বলে", "আছে",
    },
    "asm": {
        "আৰু", "এই", "সেই", "কৰা", "হয়", "পৰা", "যে", "তেওঁ", "মই", "আমি",
        "নহয়", "কিন্তু", "বাবে", "লগত", "মাজত", "এটা", "কৰি", "হৈছে", "তাৰ", "ৰ",
        "ত", "কৰে", "আছে", "বুলি", "নাই",
    },
    "tel": {
        "మరియు", "ఒక", "ఈ", "ఆ", "అని", "కూడా", "ఉంది", "లో", "కు", "తో",
        "నుండి", "వారు", "అతను", "ఆమె", "గా", "ను", "కి", "పై", "వల్ల", "చేసి",
        "అయితే", "కాని", "ఉన్న", "చేయ", "అందుకు",
    },
    "tam": {
        "மற்றும்", "ஒரு", "இந்த", "அந்த", "என்று", "உள்ளது", "இல்", "கு", "உடன்",
        "இருந்து", "அவர்", "அவள்", "ஆக", "மேல்", "ஆனால்",
    },
    "urd": {
        "اور", "کے", "کی", "کا", "ہے", "میں", "سے", "کو", "پر", "یہ", "وہ",
        "نہیں", "بھی", "ہیں", "تھا", "ایک", "لیے", "کیا",
    },
    # --- S5 top-up ------------------------------------------------------------
    # These five languages had no list in S4, and `stopword_set` fell back to the
    # English one -- so a Kannada page was checked for English stop-words, found
    # none, and was dropped. That silently rejected ~95% of kan/guj/mal/pan/ory on
    # the top-up slice. Counted out of the corpus by document frequency; see
    # scripts/derive_stopwords.py and data/run/derived_stopwords.json.
    "kan": {
        "ಈ", "ಎಂದು", "ಮತ್ತು", "ಹಾಗೂ", "ಅವರು", "ಆದರೆ", "ಬಗ್ಗೆ", "ಮೇಲೆ", "ಒಂದು", "ಅವರ",
        "ಮೂಲಕ", "ಇದು", "ತಮ್ಮ", "ಮಾಡಿ", "ಎಂಬ", "ನಂತರ", "ಆ", "ಹೆಚ್ಚು", "ಯಾವುದೇ", "ಎರಡು",
        "ತನ್ನ", "ಕೂಡ", "ನಮ್ಮ", "ಈಗ", "ಇದೆ"
    },
    "guj": {
        "છે", "અને", "આ", "પણ", "કે", "કરી", "માટે", "એક", "સાથે", "તે", "પર", "જ",
        "હતી", "આવી", "જે", "નથી", "હતા", "એ", "કરવામાં", "હતું", "તો", "હતો", "થઈ",
        "કરવા", "દ્વારા"
    },
    "mal": {
        "ഒരു", "ഈ", "എന്ന", "തന്നെ", "പറഞ്ഞു", "നിന്ന്", "കഴിഞ്ഞ", "ഏറ്റവും", "ഇത്",
        "വലിയ", "ചെയ്തു", "ശേഷം", "പുതിയ", "ദിവസം", "ആ", "എന്ന്", "അത്", "നിന്നും",
        "എന്നാൽ", "വരെ", "രണ്ട്", "അദ്ദേഹം", "ആണ്", "പറയുന്നു", "എന്നാല്"
    },
    "pan": {
        "ਦੇ", "ਨੂੰ", "ਦੀ", "ਹੈ", "ਦਾ", "ਕਿ", "ਤੇ", "ਇਸ", "ਨੇ", "ਅਤੇ", "ਨਾਲ", "ਤੋਂ",
        "ਲਈ", "ਵੀ", "ਹਨ", "ਕੀਤਾ", "ਕਰ", "ਇਹ", "ਗਿਆ", "ਕਰਨ", "ਕੇ", "ਵਿੱਚ", "ਹੀ", "ਕੀਤੀ",
        "ਹੋ"
    },
    "ory": {
        "ଏହି", "ପାଇଁ", "ଓ", "ଏକ", "ମଧ୍ୟ", "କରି", "ପରେ", "ବୋଲି", "ସେ", "କରିବା", "ଏବଂ",
        "ଏହା", "ସହ", "କରିଛନ୍ତି", "ତେବେ", "ନେଇ", "କରିଥିଲେ", "ମଧ୍ୟରେ", "ଯେ", "ଉପରେ",
        "ଏବେ", "ବେଳେ", "ହେବ", "ନାହିଁ", "ହୋଇଛି"
    },
}

# --------------------------------------------------------------- thresholds

BASE = {
    "mean_word_len": (3.0, 10.0),
    "symbol_ratio_max": 0.10,
    "terminal_punct_min": 0.30,
    "dup_line_max": 0.30,
    "top_2gram_max": 0.20,
    "stopwords_min": 2,
    "bullet_ratio_max": 0.90,
    "ellipsis_ratio_max": 0.30,
    "word_count": (50, 100_000),
}

# Per-script recalibration. Dravidian and Indo-Aryan scripts pack more meaning into
# each orthographic word, so an English word-count floor of 50 and a mean-word-length
# ceiling of 10 both mis-fire on perfectly good text.
SCRIPT_OVERRIDES = {
    "Telu": {"word_count": (25, 100_000), "mean_word_len": (3.0, 16.0)},
    "Taml": {"word_count": (25, 100_000), "mean_word_len": (3.0, 16.0)},
    "Knda": {"word_count": (25, 100_000), "mean_word_len": (3.0, 16.0)},
    "Mlym": {"word_count": (20, 100_000), "mean_word_len": (3.0, 20.0)},
    "Deva": {"word_count": (35, 100_000), "mean_word_len": (3.0, 12.0)},
    "Beng": {"word_count": (35, 100_000), "mean_word_len": (3.0, 12.0)},
    "Guru": {"word_count": (35, 100_000), "mean_word_len": (3.0, 12.0)},
    "Gujr": {"word_count": (35, 100_000), "mean_word_len": (3.0, 12.0)},
    "Orya": {"word_count": (35, 100_000), "mean_word_len": (3.0, 12.0)},
    "Arab": {"word_count": (35, 100_000), "mean_word_len": (3.0, 12.0)},
}

RULE_NAMES = [
    "mean_word_length",
    "symbol_to_word_ratio",
    "lines_end_in_terminal_punct",
    "duplicate_line_fraction",
    "top_2gram_fraction",
    "common_stopwords_present",
    "bullet_line_ratio",
    "ellipsis_line_ratio",
    "document_word_count",
]

SYMBOL_RE = re.compile(r"[#…]|\.\.\.")
WORD_RE = re.compile(r"\S+")


def thresholds_for(lang: str, script_aware: bool) -> dict:
    t = dict(BASE)
    if script_aware:
        t.update(SCRIPT_OVERRIDES.get(LANG_SCRIPT.get(lang, "Latn"), {}))
    return t


# Languages the script-aware config was asked for but has no list for. Falling back to
# English here is what silently deleted five languages in S4, so the gap is recorded instead
# of being absorbed, and the stage reports it.
MISSING_STOPWORD_LANGS: set[str] = set()


def stopword_set(lang: str, script_aware: bool) -> set[str]:
    if not script_aware:
        return STOPWORDS["eng"]  # the bias, stated plainly
    if lang not in STOPWORDS:
        MISSING_STOPWORD_LANGS.add(lang)
        return STOPWORDS["eng"]
    return STOPWORDS[lang]


def doc_features(rec: dict) -> dict:
    """Everything the nine rules need, computed once per document."""
    text = rec["text"]
    words = WORD_RE.findall(text)
    nw = len(words)
    ls = rec.get("line_stats") or {"n_lines": 0}
    nl = ls.get("n_lines", 0) or 0

    if nw:
        mean_len = sum(len(w) for w in words) / nw
        symbols = len(SYMBOL_RE.findall(text))
        symbol_ratio = symbols / nw
    else:
        mean_len = 0.0
        symbol_ratio = 1.0

    if nw >= 2:
        grams = collections.Counter(zip(words, words[1:]))
        top = grams.most_common(1)[0][1]
        top_2gram = (top * 2) / nw
    else:
        top_2gram = 0.0

    return {
        "n_words": nw,
        "mean_word_len": mean_len,
        "symbol_ratio": symbol_ratio,
        "top_2gram": top_2gram,
        "n_lines": nl,
        "dup_line_frac": (ls.get("dup_lines", 0) / nl) if nl else 0.0,
        "term_ascii_frac": (ls.get("term_ascii", 0) / nl) if nl else 0.0,
        "term_any_frac": (
            (ls.get("term_ascii", 0) + ls.get("term_indic", 0)) / nl if nl else 0.0
        ),
        "bullet_frac": (ls.get("bullet_lines", 0) / nl) if nl else 0.0,
        "ellipsis_frac": (ls.get("ellipsis_lines", 0) / nl) if nl else 0.0,
        "words": words,
    }


def evaluate(feat: dict, lang: str, script_aware: bool) -> tuple[bool, list[str]]:
    """Run the nine rules. Returns (passed, list of failed rule names)."""
    t = thresholds_for(lang, script_aware)
    sw = stopword_set(lang, script_aware)
    fails = []

    lo, hi = t["mean_word_len"]
    if not (lo <= feat["mean_word_len"] <= hi):
        fails.append("mean_word_length")
    if feat["symbol_ratio"] >= t["symbol_ratio_max"]:
        fails.append("symbol_to_word_ratio")

    # The English-tuned config only recognises . ! ? as sentence enders; the
    # script-aware one also accepts the danda and the Urdu full stop.
    term = feat["term_any_frac"] if script_aware else feat["term_ascii_frac"]
    if feat["n_lines"] and term < t["terminal_punct_min"]:
        fails.append("lines_end_in_terminal_punct")
    if feat["dup_line_frac"] >= t["dup_line_max"]:
        fails.append("duplicate_line_fraction")
    if feat["top_2gram"] >= t["top_2gram_max"]:
        fails.append("top_2gram_fraction")

    hits = 0
    for w in feat["words"][:600]:
        if w.strip(".,;:!?()[]{}\"'“”‘’।॥") in sw:
            hits += 1
            if hits >= t["stopwords_min"]:
                break
    if hits < t["stopwords_min"]:
        fails.append("common_stopwords_present")

    if feat["bullet_frac"] >= t["bullet_ratio_max"]:
        fails.append("bullet_line_ratio")
    if feat["ellipsis_frac"] >= t["ellipsis_ratio_max"]:
        fails.append("ellipsis_line_ratio")

    wlo, whi = t["word_count"]
    if not (wlo <= feat["n_words"] <= whi):
        fails.append("document_word_count")

    return (not fails), fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="02-langid.jsonl.gz")
    ap.add_argument("--out", dest="out", default="03-quality.jsonl.gz")
    args = ap.parse_args()

    docs_in = 0
    kept = {"english_tuned": 0, "script_aware": 0}
    tokens_kept = {"english_tuned": 0, "script_aware": 0}
    fails = {c: collections.Counter() for c in kept}
    # Documents the script-aware filter keeps but the English-tuned filter destroys.
    rescued = collections.Counter()
    rescued_tokens = collections.Counter()
    rescued_by_rule = collections.Counter()
    per_lang = collections.defaultdict(
        lambda: {"docs": 0, "en_kept": 0, "sa_kept": 0, "tokens": 0, "sa_tokens": 0}
    )
    rescued_examples: list[dict] = []

    def records():
        nonlocal docs_in
        for rec in jsonl_read(STAGE / args.inp):
            docs_in += 1
            lang = rec.get("detected_lang") or rec["claimed_lang"]
            feat = doc_features(rec)

            en_ok, en_fails = evaluate(feat, lang, script_aware=False)
            sa_ok, sa_fails = evaluate(feat, lang, script_aware=True)

            for cfg, ok, f in (
                ("english_tuned", en_ok, en_fails),
                ("script_aware", sa_ok, sa_fails),
            ):
                for name in f:
                    fails[cfg][name] += 1
                if ok:
                    kept[cfg] += 1
                    tokens_kept[cfg] += rec["raw_tokens"]

            pl = per_lang[lang]
            pl["docs"] += 1
            pl["tokens"] += rec["raw_tokens"]
            pl["en_kept"] += int(en_ok)
            pl["sa_kept"] += int(sa_ok)
            if sa_ok:
                pl["sa_tokens"] += rec["raw_tokens"]

            if sa_ok and not en_ok:
                rescued[lang] += 1
                rescued_tokens[lang] += rec["raw_tokens"]
                for name in en_fails:
                    rescued_by_rule[name] += 1
                if len(rescued_examples) < 10:
                    rescued_examples.append(
                        {
                            "id": rec["id"],
                            "lang": lang,
                            "failed_under_english_rules": en_fails,
                            "n_words": feat["n_words"],
                            "mean_word_len": round(feat["mean_word_len"], 2),
                            "term_ascii_frac": round(feat["term_ascii_frac"], 3),
                            "term_any_frac": round(feat["term_any_frac"], 3),
                            "excerpt": rec["text"][:240],
                        }
                    )

            rec["quality"] = {
                "english_tuned_pass": en_ok,
                "script_aware_pass": sa_ok,
                "failed_rules": sa_fails,
            }
            # The corpus we actually ship uses the script-aware filter.
            if sa_ok:
                yield rec

    with Timer("quality filter"):
        n = jsonl_write(STAGE / args.out, records())

    stats = {
        "stage": "04-quality-heuristics",
        "rules": RULE_NAMES,
        "thresholds_base": {k: list(v) if isinstance(v, tuple) else v for k, v in BASE.items()},
        "script_overrides": {
            s: {k: list(v) if isinstance(v, tuple) else v for k, v in o.items()}
            for s, o in SCRIPT_OVERRIDES.items()
        },
        "stopword_list_sizes": {k: len(v) for k, v in STOPWORDS.items()},
        "languages_with_no_stopword_list": sorted(MISSING_STOPWORD_LANGS),
        "docs_in": docs_in,
        "docs_out": n,
        "docs_dropped": docs_in - n,
        "kept_by_config": kept,
        "tokens_kept_by_config": tokens_kept,
        "keep_rate_by_config": {
            c: round(kept[c] / max(1, docs_in), 4) for c in kept
        },
        "rule_failures_by_config": {c: dict(v) for c, v in fails.items()},
        "rescued_docs_total": sum(rescued.values()),
        "rescued_tokens_total": sum(rescued_tokens.values()),
        "rescued_by_lang": dict(rescued),
        "rescued_tokens_by_lang": dict(rescued_tokens),
        "rescued_by_english_rule_that_killed_them": dict(rescued_by_rule),
        "per_lang": {k: dict(v) for k, v in per_lang.items()},
        "rescued_examples": rescued_examples,
        "shipped_config": "script_aware",
    }
    write_stage_stats("04-quality-heuristics", stats)
    log(
        f"{docs_in:,} -> {n:,} docs | english-tuned would keep {kept['english_tuned']:,} "
        f"| script-aware keeps {kept['script_aware']:,} | "
        f"rescued {sum(rescued.values()):,} docs / {sum(rescued_tokens.values()):,} tokens"
    )


if __name__ == "__main__":
    main()
