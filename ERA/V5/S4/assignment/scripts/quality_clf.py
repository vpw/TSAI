"""Stage 4b -- the trained classifier gate.

The session's recipe is: have a large model label a small sample for educational
value 0-5, train a cheap model to imitate those labels, then run the cheap model over
the whole corpus. FineWeb-Edu used a BERT regression head; Llama 3 and DCLM used
fastText.

There is no GPU on this box and no educational-value labeller for Indic, so the
expensive labeller is replaced by a weak one: a set of structural signals that are
deliberately *disjoint* from the nine Gopher rules already applied upstream, so the
classifier contributes signal rather than restating the cascade. Everything else is
the real recipe -- a fastText supervised model, trained on a held-out split, with its
accuracy reported so the reader can discount it appropriately.

Label semantics (0-5, same axis as FineWeb-Edu, keep at >= 3.0):
    0-1  boilerplate, link farms, navigation, near-empty pages
    2    thin content, heavy templating
    3    ordinary prose with real information
    4-5  substantive expository prose
"""

from __future__ import annotations

import argparse
import collections
import math
import random
import re

from common import (
    MODELS,
    STAGE,
    Timer,
    jsonl_read,
    jsonl_write,
    log,
    write_stage_stats,
)

KEEP_THRESHOLD = 3.0  # the session's classifier gate
SAMPLE_SIZE = 40_000
TRAIN_FRACTION = 0.8
FT_TEXT_CHARS = 1200

WORD_RE = re.compile(r"\S+", re.UNICODE)
URL_RE = re.compile(r"https?://|www\.")
DIGIT_RE = re.compile(r"\d")

# Boilerplate the extraction stage is supposed to have removed and often has not.
BOILERPLATE = re.compile(
    r"(?i)\b(?:cookie|privacy policy|terms of (?:use|service)|all rights reserved|"
    r"subscribe|newsletter|click here|read more|sign in|log in|advertisement|"
    r"share on|follow us|copyright|©)\b"
    r"|कुकी|गोपनीयता नीति|सर्वाधिकार सुरक्षित|यहाँ क्लिक|और पढ़ें|सदस्यता"
    r"|కుకీ|గోప్యతా విధానం|ఇక్కడ క్లిక్|మరింత చదవండి"
    r"|কুকি|গোপনীয়তা নীতি|ইয়াত ক্লিক|অধিক পঢ়ক"
)


def weak_label(text: str) -> int:
    """A 0-5 educational-value proxy from structural signals.

    Signals chosen to be independent of the nine Gopher rules: boilerplate density,
    link density, digit density, lexical variety, and sentence-length regularity.
    """
    words = WORD_RE.findall(text)
    nw = len(words)
    if nw < 5:
        return 0

    n_chars = max(1, len(text))
    boiler = len(BOILERPLATE.findall(text)) / (nw / 100)  # hits per 100 words
    urls = len(URL_RE.findall(text)) / (nw / 100)
    digits = len(DIGIT_RE.findall(text)) / n_chars
    variety = len({w.lower() for w in words}) / nw  # type-token ratio
    long_words = sum(1 for w in words if len(w) >= 7) / nw

    sentences = [s for s in re.split(r"[.!?।॥۔]\s", text) if s.strip()]
    if len(sentences) >= 2:
        lens = [len(WORD_RE.findall(s)) for s in sentences]
        mean = sum(lens) / len(lens)
        var = sum((x - mean) ** 2 for x in lens) / len(lens)
        # Real prose varies its sentence length; templated pages do not.
        regularity = math.sqrt(var) / mean if mean else 0.0
    else:
        mean, regularity = float(nw), 0.0

    score = 3.0
    score -= min(2.5, boiler * 0.9)
    score -= min(1.5, urls * 0.5)
    score -= 1.5 if digits > 0.14 else 0.0
    score += 0.9 if variety > 0.55 else (-1.0 if variety < 0.28 else 0.0)
    score += 0.6 if long_words > 0.22 else 0.0
    score += 0.7 if 8 <= mean <= 40 else -0.7
    score += 0.5 if regularity > 0.35 else -0.4
    score += 0.5 if nw > 250 else 0.0

    return max(0, min(5, int(round(score))))


def ft_line(text: str, label: int | None = None) -> str:
    body = " ".join(text[:FT_TEXT_CHARS].split())
    return (f"__label__{label} {body}" if label is not None else body)


def expected_score(labels, probs) -> float:
    """Expectation over the label distribution -- a 0-5 score, not just argmax."""
    tot = 0.0
    for lab, p in zip(labels, probs):
        tot += int(lab.replace("__label__", "")) * float(p)
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="03-quality.jsonl.gz")
    ap.add_argument("--out", dest="out", default="03b-quality-clf.jsonl.gz")
    ap.add_argument("--sample", type=int, default=SAMPLE_SIZE)
    ap.add_argument("--threshold", type=float, default=KEEP_THRESHOLD)
    args = ap.parse_args()

    import fasttext

    # ------------------------------------------------ weak-label a sample
    rng = random.Random(1729)
    sample: list[tuple[str, int]] = []
    total_seen = 0
    for rec in jsonl_read(STAGE / args.inp):
        total_seen += 1
        if len(sample) < args.sample:
            sample.append((rec["text"], weak_label(rec["text"])))
        else:
            j = rng.randrange(total_seen)  # reservoir, so the sample spans the corpus
            if j < args.sample:
                sample[j] = (rec["text"], weak_label(rec["text"]))
    rng.shuffle(sample)
    label_dist = collections.Counter(lbl for _t, lbl in sample)
    log(f"weak-labelled {len(sample):,} of {total_seen:,} docs: {dict(sorted(label_dist.items()))}")

    if len(sample) < 200 or len(label_dist) < 2:
        log("sample too small or single-class -- classifier gate skipped")
        n = jsonl_write(STAGE / args.out, jsonl_read(STAGE / args.inp))
        write_stage_stats(
            "04b-quality-classifier",
            {
                "stage": "04b-quality-classifier",
                "skipped": True,
                "reason": "insufficient or single-class weak-label sample",
                "docs_in": total_seen,
                "docs_out": n,
                "docs_dropped": 0,
            },
        )
        return

    cut = int(len(sample) * TRAIN_FRACTION)
    train, test = sample[:cut], sample[cut:]
    train_path = MODELS / "quality_train.txt"
    test_path = MODELS / "quality_test.txt"
    train_path.write_text(
        "\n".join(ft_line(t, l) for t, l in train) + "\n", encoding="utf-8"
    )
    test_path.write_text(
        "\n".join(ft_line(t, l) for t, l in test) + "\n", encoding="utf-8"
    )

    with Timer("train fastText quality classifier"):
        model = fasttext.train_supervised(
            input=str(train_path),
            epoch=8,
            lr=0.35,
            wordNgrams=2,
            dim=64,
            minCount=3,
            loss="softmax",
            verbose=0,
        )
    model.save_model(str(MODELS / "quality_clf.bin"))

    n_test, prec, rec = model.test(str(test_path))
    # Off-by-one accuracy matters more than exact-bucket accuracy on an ordinal scale.
    within_one = 0
    for t, l in test:
        pred = int(model.predict(ft_line(t))[0][0].replace("__label__", ""))
        if abs(pred - l) <= 1:
            within_one += 1

    log(
        f"held-out: n={n_test:,} exact acc={prec:.3f} within-1 acc={within_one/max(1,len(test)):.3f}"
    )

    # ------------------------------------------------ score the whole corpus
    docs_in = 0
    kept = 0
    tokens_kept = 0
    score_hist = collections.Counter()
    per_lang = collections.defaultdict(lambda: {"docs": 0, "kept": 0, "score_sum": 0.0})
    dropped_examples: list[dict] = []

    def records():
        nonlocal docs_in, kept, tokens_kept
        batch, recs = [], []

        def flush():
            nonlocal kept, tokens_kept
            if not recs:
                return
            labels, probs = model.predict(batch, k=6)
            for r, labs, prs in zip(recs, labels, probs):
                s = expected_score(labs, prs)
                score_hist[round(s)] += 1
                lang = r.get("detected_lang") or r["claimed_lang"]
                pl = per_lang[lang]
                pl["docs"] += 1
                pl["score_sum"] += s
                r["edu_score"] = round(s, 3)
                if s >= args.threshold:
                    pl["kept"] += 1
                    kept += 1
                    tokens_kept += r["raw_tokens"]
                    out.append(r)
                elif len(dropped_examples) < 8:
                    dropped_examples.append(
                        {
                            "id": r["id"],
                            "lang": lang,
                            "edu_score": round(s, 3),
                            "excerpt": r["text"][:200],
                        }
                    )
            batch.clear()
            recs.clear()

        out: list[dict] = []
        for rec in jsonl_read(STAGE / args.inp):
            docs_in += 1
            batch.append(ft_line(rec["text"]))
            recs.append(rec)
            if len(batch) >= 2000:
                flush()
                while out:
                    yield out.pop(0)
        flush()
        while out:
            yield out.pop(0)

    with Timer("classifier gate"):
        n = jsonl_write(STAGE / args.out, records())

    stats = {
        "stage": "04b-quality-classifier",
        "skipped": False,
        "recipe": (
            "weak structural labels -> fastText supervised -> expected-value score, "
            "the session's recipe with a heuristic labeller standing in for an LLM"
        ),
        "labeller": "structural weak labels (boilerplate, link, digit, variety, sentence regularity)",
        "model": "fastText supervised, dim=64, wordNgrams=2, epoch=8",
        "threshold": args.threshold,
        "weak_label_distribution": dict(sorted(label_dist.items())),
        "train_size": len(train),
        "test_size": len(test),
        "heldout_exact_accuracy": round(prec, 4),
        "heldout_within_one_accuracy": round(within_one / max(1, len(test)), 4),
        "docs_in": docs_in,
        "docs_out": n,
        "docs_dropped": docs_in - n,
        "tokens_kept": tokens_kept,
        "score_histogram": dict(sorted(score_hist.items())),
        "per_lang": {
            k: {
                "docs": v["docs"],
                "kept": v["kept"],
                "mean_score": round(v["score_sum"] / max(1, v["docs"]), 3),
            }
            for k, v in per_lang.items()
        },
        "dropped_examples": dropped_examples,
        "caveat": (
            "The labels are weak/heuristic, not LLM-generated. The classifier "
            "reproduces the mechanism and its cost profile, not FineWeb-Edu's quality."
        ),
    }
    write_stage_stats("04b-quality-classifier", stats)
    log(f"{docs_in:,} -> {n:,} docs kept at score >= {args.threshold}")


if __name__ == "__main__":
    main()
