"""Stage 2 -- normalize, and the ghost-tag trap.

Implements the session's ~15-line `clean_text()` exactly, with the one rule that makes
it a cleaner *for these languages*: ZWNJ (U+200C) and ZWJ (U+200D) are legitimate
Brahmic script controls and are always kept, while ZWSP, BOM, bidi overrides and
C0/C1 controls are pure noise and always go. "Strip all invisible characters" would
silently corrupt Indic text.

Also reproduces the audit's 46-garbage-vocab-tokens finding by measuring, on our own
corpus, how many distinct tokenizer vocab slots are occupied by pure-noise tokens
before cleaning versus after.

The content hash is computed AFTER cleaning, so two documents differing only in
invisible junk collapse to the same hash downstream.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import html
import re
import unicodedata

from common import (
    STAGE,
    Timer,
    get_tokenizer,
    jsonl_read,
    jsonl_write,
    log,
    write_stage_stats,
)

# --------------------------------------------------------------- clean_text()

# Noise classes, kept separate so each can be counted rather than lumped together.
NOISE_CLASSES = {
    # C0 controls except \t \n \r (whitespace collapse handles those), plus DEL and C1.
    "control": r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]",
    # Zero-width and invisible spacing -- but NOT U+200C ZWNJ / U+200D ZWJ.
    "zero_width": r"[​⁠᠎­﻿]",
    # Bidirectional overrides and marks.
    "bidi": r"[‎‏‪-‮⁦-⁩]",
    # Private use area -- 4 of V4's 46 garbage tokens were these.
    "private_use": r"[-]",
    # The replacement character: a byte that failed to decode.
    "replacement": r"�",
}
NOISE_RE = re.compile("|".join(f"(?:{p})" for p in NOISE_CLASSES.values()))
CLASS_RES = {k: re.compile(v) for k, v in NOISE_CLASSES.items()}

# Always kept. Counting them is the point: it proves the cleaner did not eat them.
JOINERS = {"ZWNJ": "‌", "ZWJ": "‍"}

WS_RE = re.compile(r"\s+")
ENTITY_RE = re.compile(r"&(?:[a-zA-Z][a-zA-Z0-9]{1,31}|#\d{1,7}|#[xX][0-9a-fA-F]{1,6});")

# The session's own clean_text() collapses every run of whitespace to one space,
# including newlines -- correct for a single crawled paragraph, but it turns any
# document with real line structure (lists, poems, code, quoted dialogue) into one
# unreadable line. So whitespace collapses *within* each line, not across the
# document: a line's leading indentation is kept (capped, so a formatting artifact
# can't ship 200 leading spaces), its internal runs of horizontal whitespace collapse
# to one space, and runs of blank lines collapse to a single blank separator.
LINE_WS_RE = re.compile(r"[^\S\n]+")  # horizontal whitespace, not the newline itself
BLANK_RUN_RE = re.compile(r"\n{3,}")
MAX_INDENT = 16


def denoise(s: str) -> str:
    """Everything clean_text() does except the final whitespace collapse."""
    s = unicodedata.normalize("NFC", s)
    s = html.unescape(s)
    return NOISE_RE.sub("", s)  # KEEPS U+200C ZWNJ and U+200D ZWJ


def collapse_whitespace(s: str) -> str:
    """Collapse per line, keeping line breaks and indentation intact."""
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    lines = []
    for ln in s.split("\n"):
        rest = ln.lstrip(" \t")
        indent = ln[: len(ln) - len(rest)].expandtabs(4)
        if len(indent) > MAX_INDENT:
            indent = " " * MAX_INDENT
        lines.append(indent + LINE_WS_RE.sub(" ", rest).rstrip())
    return BLANK_RUN_RE.sub("\n\n", "\n".join(lines)).strip("\n \t")


def clean_text(s: str) -> str:
    """The session's cleaner, with one change: the whitespace collapse is per-line."""
    return collapse_whitespace(denoise(s))


# ------------------------------------------------- line structure, captured early
#
# Line-level counts are taken on the denoised text, before collapse_whitespace() caps
# indentation and folds blank-line runs, so stage 4's Gopher/C4 thresholds see the
# document's structure exactly as it arrived.

ASCII_TERMINAL = (".", "!", "?", '."', ".'", '!"', '?"')
# Danda and double danda end a Devanagari sentence; Urdu uses its own full stop and
# question mark. An English-tuned rule sees none of these and concludes the document
# has no sentences at all.
INDIC_TERMINAL = ("।", "॥", "۔", "؟", "।\"", "॥\"")
BULLET_PREFIX = ("*", "-", "•", "·", "‣", "▪", "–", "—", "‧", "○", "●")


def line_stats(denoised: str) -> dict:
    lines = [ln.strip() for ln in denoised.split("\n")]
    lines = [ln for ln in lines if ln]
    n = len(lines)
    if not n:
        return {"n_lines": 0}
    seen: set[str] = set()
    dup = 0
    ascii_term = indic_term = bullet = ellipsis = 0
    for ln in lines:
        if ln in seen:
            dup += 1
        else:
            seen.add(ln)
        if ln.endswith(ASCII_TERMINAL):
            ascii_term += 1
        if ln.endswith(INDIC_TERMINAL):
            indic_term += 1
        if ln.startswith(BULLET_PREFIX):
            bullet += 1
        if ln.endswith(("...", "…")):
            ellipsis += 1
    return {
        "n_lines": n,
        "dup_lines": dup,
        "term_ascii": ascii_term,
        "term_indic": indic_term,
        "bullet_lines": bullet,
        "ellipsis_lines": ellipsis,
    }


# --------------------------------------------------------------- ghost tags

# Literal conversation markers that show up as ordinary subwords in pretraining text.
# The audit found [USER] x6, [SYSTEM] x2, <|endoftext|> x3 sitting inside V4's shards.
GHOST_PATTERNS = {
    "bracket_role": re.compile(r"\[(?:USER|SYSTEM|ASSISTANT|INST|/INST)\]"),
    "xml_role": re.compile(r"</?(?:USER|SYSTEM|ASSISTANT|user|system|assistant)>"),
    "special_token": re.compile(
        r"<\|(?:endoftext|im_start|im_end|user|assistant|system|end|begin_of_text)\|>"
    ),
    "alpaca_header": re.compile(
        r"###\s*(?:Instruction|Response|Input|Output|Topic|Question|Answer)\s*:"
    ),
    "chat_prefix": re.compile(r"^\s*(?:Human|AI|Assistant|Bot)\s*:\s", re.M),
}

# The one canonical format everything is rewritten into, per the session's fix.
CANONICAL = {
    "user": "<|user|>",
    "assistant": "<|assistant|>",
    "system": "<|system|>",
    "end": "<|end|>",
}
ROLE_REWRITE = [
    # Closing tags first: </USER> ends a turn, so it becomes <|end|>, not another
    # <|user|>. Rewriting it as an opener would invent a turn that was never there.
    (re.compile(r"</(?:user|system|assistant)>|\[/INST\]", re.I), CANONICAL["end"]),
    (re.compile(r"\[SYSTEM\]|<system>|###\s*Topic\s*:", re.I), CANONICAL["system"]),
    (
        re.compile(
            r"\[USER\]|\[INST\]|<user>|###\s*(?:Instruction|Question|Input)\s*:"
            r"|^\s*Human\s*:\s",
            re.I | re.M,
        ),
        CANONICAL["user"],
    ),
    (
        re.compile(
            r"\[ASSISTANT\]|<assistant>|###\s*(?:Response|Answer|Output)\s*:"
            r"|^\s*(?:AI|Assistant|Bot)\s*:\s",
            re.I | re.M,
        ),
        CANONICAL["assistant"],
    ),
    (re.compile(r"<\|(?:endoftext|im_end|begin_of_text|im_start)\|>"), ""),
]


def scan_ghost_tags(s: str) -> dict[str, int]:
    hits: dict[str, int] = {}
    for name, rx in GHOST_PATTERNS.items():
        n = len(rx.findall(s))
        if n:
            hits[name] = n
    return hits


def unify_format(s: str) -> str:
    """Rewrite every source's own role markers into the one canonical form."""
    for rx, repl in ROLE_REWRITE:
        s = rx.sub(repl, s)
    return WS_RE.sub(" ", s).strip()


# --------------------------------------------------- garbage vocab-token audit

PURE_NOISE_RE = re.compile(
    r"^(?:"
    + "|".join(f"(?:{p})" for p in NOISE_CLASSES.values())
    + r"|\s)+$"
)


BYTE_FALLBACK_RE = re.compile(r"^(?:<0x[0-9A-Fa-f]{2}>)+$")


def is_garbage_token(piece: str) -> bool:
    """A vocab slot spent on characters that carry no linguistic information.

    Byte-fallback pieces (`<0xE0>`) are deliberately NOT counted here: a byte-level
    fallback firing on well-formed Indic text is the tokenizer working as designed,
    not dirt. They are tracked separately.
    """
    if not piece:
        return False
    body = piece.replace("▁", "")  # sentencepiece word-boundary marker
    return bool(body) and bool(PURE_NOISE_RE.match(body))


def scan_vocab_slots(texts: list[str], tok) -> tuple[set[int], set[int]]:
    """Distinct (garbage, byte-fallback) vocab ids observed in `texts`."""
    garbage: set[int] = set()
    byte_fb: set[int] = set()
    for enc in tok.encode_batch(texts, add_special_tokens=False):
        for tid, piece in zip(enc.ids, enc.tokens):
            if tid in garbage or tid in byte_fb:
                continue
            if is_garbage_token(piece):
                garbage.add(tid)
            elif BYTE_FALLBACK_RE.match(piece):
                byte_fb.add(tid)
    return garbage, byte_fb


# --------------------------------------------------------------- driver


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="00-raw.jsonl.gz")
    ap.add_argument("--out", dest="out", default="01-normalized.jsonl.gz")
    ap.add_argument(
        "--garbage-sample",
        type=int,
        default=40_000,
        help="documents scanned for the garbage-vocab-slot audit",
    )
    args = ap.parse_args()

    tok = get_tokenizer()

    removed = collections.Counter()  # noise chars removed, by class
    joiners_kept = collections.Counter()
    ghost_hits = collections.Counter()
    ghost_docs = 0
    unified_docs = 0
    entity_docs = 0
    nfc_changed = 0
    docs_in = docs_out = 0
    chars_in = chars_out = 0
    dropped_empty = 0
    examples: list[dict] = []
    garbage_raw: set[int] = set()
    garbage_clean: set[int] = set()
    byte_fallback_raw: set[int] = set()
    garbage_scanned = 0
    ghost_examples: list[dict] = []

    def records():
        nonlocal docs_in, docs_out, chars_in, chars_out, dropped_empty
        nonlocal ghost_docs, unified_docs, entity_docs, nfc_changed, garbage_scanned

        batch_raw: list[str] = []
        batch_clean: list[str] = []

        def flush_garbage():
            nonlocal garbage_scanned
            if not batch_raw:
                return
            g_raw, b_raw = scan_vocab_slots(batch_raw, tok)
            g_clean, _b_clean = scan_vocab_slots(batch_clean, tok)
            garbage_raw.update(g_raw)
            byte_fallback_raw.update(b_raw)
            garbage_clean.update(g_clean)
            garbage_scanned += len(batch_raw)
            batch_raw.clear()
            batch_clean.clear()

        for rec in jsonl_read(STAGE / args.inp):
            docs_in += 1
            raw = rec.pop("text")
            chars_in += len(raw)

            # Per-class accounting, measured before removal.
            per_class = {}
            for name, rx in CLASS_RES.items():
                n = len(rx.findall(raw))
                if n:
                    per_class[name] = n
                    removed[name] += n
            for jname, jch in JOINERS.items():
                n = raw.count(jch)
                if n:
                    joiners_kept[jname] += n
            if ENTITY_RE.search(raw):
                entity_docs += 1
            if unicodedata.normalize("NFC", raw) != raw:
                nfc_changed += 1

            denoised = denoise(raw)
            lstats = line_stats(denoised)
            cleaned = collapse_whitespace(denoised)

            hits = scan_ghost_tags(cleaned)
            if hits:
                ghost_docs += 1
                for k, v in hits.items():
                    ghost_hits[k] += v
                # A document carrying two or more distinct role markers is a
                # conversation in some source's private format: rewrite it into the
                # one canonical form rather than shipping a competing format.
                if len(hits) >= 2 or hits.get("bracket_role", 0) >= 2:
                    before = cleaned
                    cleaned = unify_format(cleaned)
                    unified_docs += 1
                    if len(ghost_examples) < 6:
                        ghost_examples.append(
                            {
                                "id": rec["id"],
                                "markers": hits,
                                "before": before[:280],
                                "after": cleaned[:280],
                            }
                        )
                elif len(ghost_examples) < 6:
                    ghost_examples.append(
                        {"id": rec["id"], "markers": hits, "before": cleaned[:280]}
                    )

            if not cleaned:
                dropped_empty += 1
                continue

            if garbage_scanned < args.garbage_sample:
                batch_raw.append(raw)
                batch_clean.append(cleaned)
                if len(batch_raw) >= 1000:
                    flush_garbage()

            if per_class and len(examples) < 8 and sum(per_class.values()) >= 3:
                examples.append(
                    {
                        "id": rec["id"],
                        "src": rec["src"],
                        "removed": per_class,
                        "joiners_in_doc": {
                            k: raw.count(v) for k, v in JOINERS.items() if raw.count(v)
                        },
                        "chars_before": len(raw),
                        "chars_after": len(cleaned),
                    }
                )

            docs_out += 1
            chars_out += len(cleaned)
            rec["text"] = cleaned
            rec["line_stats"] = lstats
            # Hash AFTER cleaning -- dedup and the manifest both trust this.
            rec["hash"] = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()
            if hits:
                rec["ghost_tags"] = hits
            yield rec

        flush_garbage()

    with Timer("normalize"):
        n = jsonl_write(STAGE / args.out, records())

    stats = {
        "stage": "02-normalize",
        "docs_in": docs_in,
        "docs_out": n,
        "docs_dropped": docs_in - n,
        "dropped_empty_after_clean": dropped_empty,
        "chars_in": chars_in,
        "chars_out": chars_out,
        "chars_removed": chars_in - chars_out,
        "noise_chars_removed_by_class": dict(removed),
        "noise_chars_removed_total": sum(removed.values()),
        "joiners_kept": dict(joiners_kept),
        "docs_with_html_entities": entity_docs,
        "docs_changed_by_nfc": nfc_changed,
        "ghost_tag_docs": ghost_docs,
        "ghost_tag_hits_by_kind": dict(ghost_hits),
        "ghost_tag_hits_total": sum(ghost_hits.values()),
        "docs_format_unified": unified_docs,
        "canonical_format": CANONICAL,
        "garbage_vocab_slots_before": len(garbage_raw),
        "garbage_vocab_slots_after": len(garbage_clean),
        "byte_fallback_slots_before": len(byte_fallback_raw),
        "garbage_audit_docs_scanned": garbage_scanned,
        "garbage_audit_note": (
            "A garbage slot is a vocab entry whose piece is nothing but noise "
            "characters. Byte-fallback pieces (<0xE0>) are counted separately: on "
            "well-formed Indic text they are the tokenizer working, not dirt."
        ),
        "examples": examples,
        "ghost_examples": ghost_examples,
        "hash_order": "sha256 computed AFTER clean_text(), not before",
    }
    write_stage_stats("02-normalize", stats)
    log(
        f"{docs_in:,} -> {n:,} docs | removed {sum(removed.values()):,} noise chars | "
        f"kept {sum(joiners_kept.values()):,} joiners | "
        f"garbage vocab slots {len(garbage_raw)} -> {len(garbage_clean)}"
    )


if __name__ == "__main__":
    main()
