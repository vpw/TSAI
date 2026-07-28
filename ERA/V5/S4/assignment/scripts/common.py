"""Shared plumbing for the S4 cleaning pipeline.

Every stage reads a gzipped JSONL shard, writes a gzipped JSONL shard, and drops a
stats block into data/run/stages/<name>.json. The driver stitches those into the
single stats.json the widget is generated from, so no number in the widget is typed
by hand.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
STAGE = DATA / "stage"
RUN = DATA / "run"
STAGE_STATS = RUN / "stages"
MODELS = ROOT / "models"

for _d in (RAW, STAGE, RUN, STAGE_STATS, MODELS):
    _d.mkdir(parents=True, exist_ok=True)

# The slice of ai4bharat/sangraha we clean. `claimed` is the language the *folder path*
# asserts -- which is exactly the thing stage 3 refuses to trust.
SOURCES = [
    # key            pool          claimed  file                            token budget
    ("unv-hin", "unverified", "hin", "unverified/hin/data-0.parquet", 15_000_000),
    ("ver-hin", "verified", "hin", "verified/hin/data-0.parquet", 12_000_000),
    ("ver-tel", "verified", "tel", "verified/tel/data-0.parquet", 10_000_000),
    ("ver-eng", "verified", "eng", "verified/eng/data-0.parquet", 8_000_000),
    ("ver-asm", "verified", "asm", "verified/asm/data-0.parquet", 5_000_000),
]

REPO_ID = "ai4bharat/sangraha"
REPO_REVISION_FILE = RUN / "source_revision.txt"

# ISO 639-1 (what fastText emits) -> ISO 639-3 (what Sangraha's folders use).
# The V4 bug was exactly this mapping being absent: Telugu arrived as `te` where the
# pipeline expected `tel`, and only worked because a fallback happened to return the
# right value.
ISO1_TO_ISO3 = {
    "as": "asm", "bn": "ben", "gu": "guj", "hi": "hin", "kn": "kan",
    "ks": "kas", "ml": "mal", "mr": "mar", "ne": "nep", "or": "ory",
    "pa": "pan", "sa": "san", "sd": "snd", "ta": "tam", "te": "tel",
    "ur": "urd", "en": "eng",
}

# Script ranges, used for romanised-Indic detection and per-script thresholds.
SCRIPT_RANGES = {
    "Deva": (0x0900, 0x097F),
    "Beng": (0x0980, 0x09FF),
    "Guru": (0x0A00, 0x0A7F),
    "Gujr": (0x0A80, 0x0AFF),
    "Orya": (0x0B00, 0x0B7F),
    "Taml": (0x0B80, 0x0BFF),
    "Telu": (0x0C00, 0x0C7F),
    "Knda": (0x0C80, 0x0CFF),
    "Mlym": (0x0D00, 0x0D7F),
    "Arab": (0x0600, 0x06FF),
    "Latn": (0x0041, 0x024F),
}

LANG_SCRIPT = {
    "hin": "Deva", "mar": "Deva", "nep": "Deva", "san": "Deva",
    "ben": "Beng", "asm": "Beng", "tel": "Telu", "tam": "Taml",
    "kan": "Knda", "mal": "Mlym", "guj": "Gujr", "pan": "Guru",
    "ory": "Orya", "urd": "Arab", "eng": "Latn",
}


def script_of(ch: str) -> str | None:
    cp = ord(ch)
    for name, (lo, hi) in SCRIPT_RANGES.items():
        if lo <= cp <= hi:
            return name
    return None


def script_profile(text: str, sample: int = 4000) -> dict[str, float]:
    """Fraction of letter characters belonging to each script, on a prefix sample."""
    counts: dict[str, int] = {}
    total = 0
    for ch in text[:sample]:
        if not ch.isalpha():
            continue
        s = script_of(ch)
        if s is None:
            continue
        counts[s] = counts.get(s, 0) + 1
        total += 1
    if not total:
        return {}
    return {k: v / total for k, v in counts.items()}


# ---------------------------------------------------------------- io


def jsonl_write(path: Path, records):
    """Stream records to a gzipped JSONL file. Returns the count written."""
    n = 0
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=4) as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False))
            fh.write("\n")
            n += 1
    return n


def jsonl_read(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def script_hash(path: Path) -> str:
    """Hash of a cleaning script's own source -- a required manifest field."""
    return sha256_file(path)


def write_stage_stats(name: str, stats: dict) -> Path:
    out = STAGE_STATS / f"{name}.json"
    out.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def read_stage_stats(name: str) -> dict:
    return json.loads((STAGE_STATS / f"{name}.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------- tokenizers

_TOKENIZERS: dict[str, object] = {}

# Two tokenizers on purpose. Sarvam-1 is Indic-tuned and is what we budget with;
# Qwen2.5 is a strong general (English-centric) BPE, kept as the comparator so the
# fertility gap the session warns about is measured rather than asserted.
TOKENIZER_FILES = {
    "sarvam1": "tokenizer-sarvam1.json",
    "qwen25": "tokenizer-qwen25.json",
}
PRIMARY_TOKENIZER = "sarvam1"


def get_tokenizer(name: str = PRIMARY_TOKENIZER):
    if name not in _TOKENIZERS:
        from tokenizers import Tokenizer

        _TOKENIZERS[name] = Tokenizer.from_file(str(MODELS / TOKENIZER_FILES[name]))
    return _TOKENIZERS[name]


def count_tokens(text: str, name: str = PRIMARY_TOKENIZER) -> int:
    return len(get_tokenizer(name).encode(text, add_special_tokens=False).ids)


def count_tokens_batch(texts: list[str], name: str = PRIMARY_TOKENIZER) -> list[int]:
    enc = get_tokenizer(name).encode_batch(texts, add_special_tokens=False)
    return [len(e.ids) for e in enc]


class Timer:
    def __init__(self, label: str):
        self.label = label

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        self.seconds = time.time() - self.t0
        print(f"[{self.label}] {self.seconds:.1f}s", flush=True)


def log(msg: str):
    print(f"  {msg}", flush=True)
