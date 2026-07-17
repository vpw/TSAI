"""Export everything the static frontend needs into site/data/*.json:

  vocab.json           - id->token array + ordered merge list (rank = index)
                          for the WINNING tokenizer (whichever of mr/bn was
                          picked), plus per-token script classification and
                          which language(s)' unique word lists use each id.
  stats.json            - { mr: {...}, bn: {...}, winner: "mr"|"bn" } so the
                          Overview tab can show the head-to-head comparison
                          MODIFY.md asked for, not just the winner alone.
  samples.json          - a quick-pick sample paragraph per language (winner's
                          4 languages only) plus corpus metadata, for the
                          Coverage Inspector tab.
  run_config.json       - the winning run's exact knobs (repeat_counts,
                          min_frequency, vocab_size, fourth language) so the
                          Methodology tab shows numbers pulled from the real
                          artifact, not hardcoded prose.
  fidelity_probes.json  - the round-trip probe strings from
                          scripts/verify_roundtrip.py, run live client-side
                          by the Fidelity Check tab (not precomputed -- the
                          point is the browser's own bpe.js proving it,
                          live, in front of the grader).

Everything here is static data computed once at build time; the browser only
ever loads JSON and runs BPE merges client-side (see site/bpe.js).
"""

import argparse
import json
import pathlib

import regex
from tokenizers import Tokenizer

from common import TOKENIZER_DIR, langset, load_corpora
from compute_metrics import LANG_NAMES, compute_stats
from verify_roundtrip import KNOWN_UNSAFE_PROBES, PROBES

SITE_DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "site" / "data"
STATS_DIR = TOKENIZER_DIR.parent
WORD_PATTERN = regex.compile(r"[\p{L}\p{M}\p{N}]+")

SOURCE_TITLES = {"en": "India", "hi": "भारत", "te": "భారతదేశం", "mr": "भारत", "bn": "ভারত"}


def classify_script(token: str) -> str:
    for ch in token:
        cp = ord(ch)
        if 0x0900 <= cp <= 0x097F:
            return "devanagari"
        if 0x0C00 <= cp <= 0x0C7F:
            return "telugu"
        if 0x0980 <= cp <= 0x09FF:
            return "bengali"
        if ("A" <= ch <= "Z") or ("a" <= ch <= "z"):
            return "latin"
    if token.strip("▁") == "":
        return "whitespace"
    if any(ch.isdigit() for ch in token):
        return "digit"
    return "punct/other"


def build_vocab_export(fourth: str, texts: dict[str, str]) -> dict:
    langs = langset(fourth)
    path = TOKENIZER_DIR / fourth / "tokenizer.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    vocab: dict[str, int] = raw["model"]["vocab"]
    merges = raw["model"]["merges"]

    id_to_token = [None] * len(vocab)
    for tok, idx in vocab.items():
        id_to_token[idx] = tok

    tokenizer = Tokenizer.from_file(str(path))
    token_langs: list[set] = [set() for _ in id_to_token]
    for lang in langs:
        words = sorted(set(WORD_PATTERN.findall(texts[lang])))
        for word in words:
            for tid in tokenizer.encode(word).ids:
                if tid < len(token_langs):
                    token_langs[tid].add(lang)

    tokens_export = []
    for idx, tok in enumerate(id_to_token):
        tokens_export.append(
            {
                "id": idx,
                "token": tok,
                "script": classify_script(tok),
                "langs": sorted(token_langs[idx]),
                "isMerged": idx >= (len(id_to_token) - len(merges)),
            }
        )

    return {
        "vocab_size": len(id_to_token),
        "tokens": tokens_export,
        "merges": [list(pair) for pair in merges],
    }


def build_samples_export(fourth: str, texts: dict[str, str]) -> dict:
    langs = langset(fourth)
    out = {}
    for lang in langs:
        text = texts[lang]
        words = WORD_PATTERN.findall(text)
        cutoff = 0
        word_count = 0
        for m in regex.finditer(r"[\p{L}\p{M}\p{N}]+", text):
            word_count += 1
            if word_count >= 60:
                cutoff = m.end()
                break
        sample = text[:cutoff] if cutoff else text[:400]
        out[lang] = {
            "name": LANG_NAMES[lang],
            "source_title": SOURCE_TITLES[lang],
            "char_count": len(text),
            "word_count": len(words),
            "sample": sample,
        }
    return out


def build_run_config(fourth: str, repeat_counts: dict, min_frequency: int, vocab_size: int) -> dict:
    return {
        "fourth": fourth,
        "langs": langset(fourth),
        "repeat_counts": repeat_counts,
        "min_frequency": min_frequency,
        "vocab_size": vocab_size,
    }


def build_fidelity_probes() -> dict:
    return {"probes": PROBES, "known_unsafe": KNOWN_UNSAFE_PROBES}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--winner", default="mr", choices=["mr", "bn"])
    parser.add_argument("--min-frequency", type=int, default=1)
    args = parser.parse_args()

    SITE_DATA_DIR.mkdir(parents=True, exist_ok=True)

    both_stats = {}
    for fourth in ["mr", "bn"]:
        tok_path = TOKENIZER_DIR / fourth / "tokenizer.json"
        if tok_path.exists():
            both_stats[fourth] = compute_stats(fourth)
    both_stats["winner"] = args.winner
    (SITE_DATA_DIR / "stats.json").write_text(
        json.dumps(both_stats, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[stats.json] mr+bn comparison written, winner={args.winner}")

    winner_langs = langset(args.winner)
    texts = load_corpora(winner_langs)

    vocab_export = build_vocab_export(args.winner, texts)
    (SITE_DATA_DIR / "vocab.json").write_text(
        json.dumps(vocab_export, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[vocab.json] {len(vocab_export['tokens'])} tokens, {len(vocab_export['merges'])} merges")

    samples_export = build_samples_export(args.winner, texts)
    (SITE_DATA_DIR / "samples.json").write_text(
        json.dumps(samples_export, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("[samples.json] written")

    sweep_path = STATS_DIR / f"tune_sweep_{args.winner}.json"
    repeat_counts = json.loads(sweep_path.read_text())["best"]["repeat_counts"] if sweep_path.exists() else {}
    run_config = build_run_config(args.winner, repeat_counts, args.min_frequency, vocab_export["vocab_size"])
    (SITE_DATA_DIR / "run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("[run_config.json] written")

    fidelity = build_fidelity_probes()
    (SITE_DATA_DIR / "fidelity_probes.json").write_text(
        json.dumps(fidelity, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("[fidelity_probes.json] written")

    files = ["vocab.json", "stats.json", "samples.json", "run_config.json", "fidelity_probes.json"]
    total_bytes = sum((SITE_DATA_DIR / f).stat().st_size for f in files)
    print(f"Total site/data size: {total_bytes / 1024:.1f} KB")


if __name__ == "__main__":
    main()
