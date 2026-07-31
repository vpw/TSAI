#!/usr/bin/env python3
"""Turn the raw downloads into one text pool per lane.

Output: `proxy/data/pools/<lane>.jsonl`, one JSON object per line.
  normal lanes : {"text": "..."}
  agentic lane : {"segments": [{"role": ..., "text": ..., "sup": 0|1}, ...]}

The agentic lane keeps role structure because the plan's central agentic claim is about
*supervised* tokens, not raw ones: assistant turns (including the tool calls they emit) are
supervised; system prompts, user turns and tool observations are context only. That mask is
carried all the way through to the bits-per-byte metric.

The long-context lane is built by *repacking* the code and web lanes, not from new supply —
which is exactly what the inventory says the real long-context slot is.
"""
import gzip
import json
import os
import re
import sys
from collections import defaultdict

import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
PROXY = os.path.dirname(HERE)
RAW = os.path.join(PROXY, "data", "raw")
POOLS = os.path.join(PROXY, "data", "pools")
S4_SHARDS = os.path.normpath(os.path.join(PROXY, "..", "..", "..", "S4", "assignment",
                                          "data", "run", "shards"))

# Character budgets per lane. Tokens are counted for real in tokenize_pools.py; these are
# only download-to-disk caps, set well above what any single arm consumes.
BUDGET = {
    "general_web": 900_000_000,
    "code": 520_000_000,
    "stem": 260_000_000,
    "reasoning": 260_000_000,
    "agentic": 10**12,          # take everything: this lane is supply-limited, as in the plan
    "long_context": 170_000_000,
    "indic_A_verified": 10**12,  # all of S4's verified output
    "indic_B_unverified": 10**12,
    "indic_C_translated": 70_000_000,
    "indic_D_synthetic": 170_000_000,
}
MIN_DOC_CHARS = 200


class Pool:
    def __init__(self, lane):
        self.lane = lane
        self.path = os.path.join(POOLS, lane + ".jsonl")
        self.f = open(self.path, "w")
        self.chars = 0
        self.docs = 0
        self.budget = BUDGET[lane]

    def full(self):
        return self.chars >= self.budget

    def add(self, text):
        if len(text) < MIN_DOC_CHARS or self.full():
            return False
        self.f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
        self.chars += len(text)
        self.docs += 1
        return True

    def add_segments(self, segs):
        if self.full():
            return False
        n = sum(len(s["text"]) for s in segs)
        if n < MIN_DOC_CHARS:
            return False
        self.f.write(json.dumps({"segments": segs}, ensure_ascii=False) + "\n")
        self.chars += n
        self.docs += 1
        return True

    def close(self):
        self.f.close()
        print(f"  {self.lane:22s} {self.docs:8d} docs  {self.chars / 1e6:9.1f} M chars")
        return {"lane": self.lane, "docs": self.docs, "chars": self.chars}


def iter_parquet(path, col, batch=512):
    pf = pq.ParquetFile(path)
    for b in pf.iter_batches(batch_size=batch, columns=[col]):
        for v in b.column(col).to_pylist():
            if v:
                yield v


# --------------------------------------------------------------------------- web / stem
def build_simple(lane, path, col, stats):
    p = Pool(lane)
    for text in iter_parquet(path, col):
        if not p.add(text):
            if p.full():
                break
    stats.append(p.close())


# --------------------------------------------------------------------------------- code
def build_code(stats, keep_repo_index):
    p = Pool("code")
    index_chars = 0
    with gzip.open(os.path.join(RAW, "codeparrot-001.json.gz"), "rt") as f:
        for line in f:
            d = json.loads(line)
            text = d.get("content") or ""
            added = p.add(text)
            # Keep a bounded slice for repo-packing the long-context lane.
            if added and index_chars < BUDGET["long_context"] and len(text) < 20_000:
                keep_repo_index.append((d.get("repo_name", ""), text))
                index_chars += len(text)
            if p.full():
                break
    # CommitPack is a distinct shape (diff-shaped edits), so it gets a share of the lane
    cp = os.path.join(RAW, "commitpackft-python.jsonl")
    if os.path.exists(cp):
        p.budget += 40_000_000
        with open(cp) as f:
            for line in f:
                d = json.loads(line)
                text = (f"# {d.get('subject', '')}\n"
                        f"{d.get('old_contents', '')}\n# ->\n{d.get('new_contents', '')}")
                p.add(text)
                if p.full():
                    break
    stats.append(p.close())


# ---------------------------------------------------------------------------- reasoning
def build_reasoning(stats):
    p = Pool("reasoning")
    path = os.path.join(RAW, "openr1-math-000.parquet")
    pf = pq.ParquetFile(path)
    cols = set(pf.schema_arrow.names)
    prob = "problem" if "problem" in cols else None
    gen = "generations" if "generations" in cols else None
    sol = "solution" if "solution" in cols else None
    use = [c for c in (prob, gen, sol) if c]
    for b in pf.iter_batches(batch_size=256, columns=use):
        rows = b.to_pylist()
        for r in rows:
            trace = r.get(gen) if gen else None
            if isinstance(trace, list):
                trace = trace[0] if trace else None
            trace = trace or (r.get(sol) if sol else None)
            if not trace:
                continue
            text = f"Problem:\n{r.get(prob, '')}\n\nReasoning:\n{trace}"
            p.add(text)
        if p.full():
            break
    stats.append(p.close())


# ------------------------------------------------------------------------------ agentic
ASSIST = {"assistant", "gpt", "model"}
OBS = {"tool", "function response", "observation", "function"}


def _segs_from_conversations(conv, sysmsg=None):
    segs = []
    if sysmsg:
        segs.append({"role": "system", "text": str(sysmsg), "sup": 0})
    for turn in conv:
        if not isinstance(turn, dict):
            continue
        role = str(turn.get("from") or turn.get("role") or "").strip().lower()
        text = turn.get("value") or turn.get("content") or ""
        if not isinstance(text, str) or not text.strip():
            continue
        if role in ASSIST:
            segs.append({"role": "assistant", "text": text, "sup": 1})
        elif role in OBS:
            segs.append({"role": "tool", "text": text, "sup": 0})
        elif role == "system":
            segs.append({"role": "system", "text": text, "sup": 0})
        else:
            segs.append({"role": "user", "text": text, "sup": 0})
    return segs


GLAIVE_SPLIT = re.compile(r"(USER:|ASSISTANT:|FUNCTION RESPONSE:)")


def _segs_from_glaive(system, chat):
    segs = []
    if system:
        segs.append({"role": "system", "text": system.strip(), "sup": 0})
    parts = GLAIVE_SPLIT.split(chat or "")
    i = 1
    while i < len(parts):
        marker, body = parts[i], parts[i + 1] if i + 1 < len(parts) else ""
        body = body.replace("<|endoftext|>", "").strip()
        i += 2
        if not body:
            continue
        if marker == "ASSISTANT:":
            segs.append({"role": "assistant", "text": body, "sup": 1})
        elif marker == "FUNCTION RESPONSE:":
            segs.append({"role": "tool", "text": body, "sup": 0})
        else:
            segs.append({"role": "user", "text": body, "sup": 0})
    return segs


def _load_json_any(path):
    """These agentic files are variously a JSON array, or JSON-lines."""
    with open(path) as f:
        head = f.read(2048)
        f.seek(0)
        if head.lstrip().startswith("["):
            try:
                return json.load(f)
            except json.JSONDecodeError:
                f.seek(0)
        out = []
        for line in f:
            line = line.strip().rstrip(",")
            if not line or line in "[]":
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out


def build_agentic(stats):
    p = Pool("agentic")
    per_source = defaultdict(int)

    for fname in ("glaive-fc-v2.json", "toolace.json", "hermes-fc.json", "hermes-fc-single.json"):
        path = os.path.join(RAW, fname)
        if not os.path.exists(path):
            print(f"  (missing {fname}, skipping)")
            continue
        rows = _load_json_any(path)
        n0 = p.docs
        for r in rows:
            if not isinstance(r, dict):
                continue
            if "chat" in r:
                segs = _segs_from_glaive(r.get("system"), r.get("chat"))
            else:
                conv = r.get("conversations") or r.get("messages") or []
                segs = _segs_from_conversations(conv, r.get("system"))
            if any(s["sup"] for s in segs):
                p.add_segments(segs)
        per_source[fname] = p.docs - n0
        print(f"    {fname:26s} -> {per_source[fname]:6d} dialogues")
    s = p.close()
    s["per_source"] = dict(per_source)
    stats.append(s)


# ------------------------------------------------------------------------- long context
def build_long_context(stats, repo_index):
    """Repack existing lanes into long documents. No new supply, by design."""
    p = Pool("long_context")
    by_repo = defaultdict(list)
    for repo, text in repo_index:
        if repo:
            by_repo[repo].append(text)
    packed_repo = packed_web = 0
    for repo, files in by_repo.items():
        if len(files) < 3:
            continue
        doc = f"# repository: {repo}\n\n" + "\n\n# ---- file ----\n\n".join(files)
        if len(doc) < 40_000:      # only keep genuinely long packs
            continue
        if p.add(doc):
            packed_repo += 1
        if p.chars >= p.budget * 0.6:
            break

    web_path = os.path.join(POOLS, "general_web.jsonl")
    buf, buflen = [], 0
    with open(web_path) as f:
        for line in f:
            t = json.loads(line)["text"]
            buf.append(t)
            buflen += len(t)
            if buflen >= 60_000:
                if p.add("\n\n".join(buf)):
                    packed_web += 1
                buf, buflen = [], 0
                if p.full():
                    break
    s = p.close()
    s["packed_repo_docs"] = packed_repo
    s["packed_web_docs"] = packed_web
    stats.append(s)


# -------------------------------------------------------------------------------- indic
def build_indic_from_s4(stats):
    # On the GPU box the S4 shards are not present: the two Indic pools are uploaded
    # ready-built instead. Never clobber them with empties.
    if not os.path.isdir(S4_SHARDS):
        have = [p for p in ("indic_A_verified", "indic_B_unverified")
                if os.path.exists(os.path.join(POOLS, p + ".jsonl"))]
        if have:
            print(f"  S4 shards absent; keeping uploaded pools: {', '.join(have)}")
            for lane in have:
                n = sum(1 for _ in open(os.path.join(POOLS, lane + ".jsonl")))
                sz = os.path.getsize(os.path.join(POOLS, lane + ".jsonl"))
                print(f"  {lane:22s} {n:8d} docs  (uploaded, {sz / 1e6:.1f} MB on disk)")
                stats.append({"lane": lane, "docs": n, "chars": None, "source": "uploaded"})
            return
        print(f"  !! S4 shards not found at {S4_SHARDS} and no uploaded pools")

    pools = {"indic_A_verified": Pool("indic_A_verified"),
             "indic_B_unverified": Pool("indic_B_unverified")}
    langs = defaultdict(int)
    if not os.path.isdir(S4_SHARDS):
        pass
    else:
        for fn in sorted(os.listdir(S4_SHARDS)):
            if not fn.endswith(".jsonl"):
                continue
            with open(os.path.join(S4_SHARDS, fn)) as f:
                for line in f:
                    d = json.loads(line)
                    lane = ("indic_A_verified" if d.get("pool") == "verified"
                            else "indic_B_unverified")
                    if pools[lane].add(d.get("text", "")):
                        langs[(lane, d.get("claimed_lang"))] += 1
    for lane in pools:
        s = pools[lane].close()
        s["by_lang"] = {k[1]: v for k, v in langs.items() if k[0] == lane}
        stats.append(s)


def build_indic_translated(stats):
    p = Pool("indic_C_translated")
    path = os.path.join(RAW, "samanantar-hi-000.parquet")
    pf = pq.ParquetFile(path)
    cols = set(pf.schema_arrow.names)
    src = "src" if "src" in cols else ("idx" if "idx" in cols else list(cols)[0])
    tgt = "tgt" if "tgt" in cols else ("hi" if "hi" in cols else list(cols)[-1])
    print(f"    samanantar columns={sorted(cols)} using src={src} tgt={tgt}")
    # Sentence pairs are tiny; glue them into paragraph-sized documents so the lane has the
    # same document shape as the others.
    buf, buflen = [], 0
    for b in pf.iter_batches(batch_size=1024, columns=[src, tgt]):
        rows = b.to_pylist()
        for r in rows:
            a, c = r.get(src) or "", r.get(tgt) or ""
            if not c:
                continue
            buf.append(f"{a}\n{c}" if isinstance(a, str) and a else c)
            buflen += len(buf[-1])
            if buflen >= 3000:
                p.add("\n".join(buf))
                buf, buflen = [], 0
        if p.full():
            break
    stats.append(p.close())


def main():
    os.makedirs(POOLS, exist_ok=True)
    stats = []
    print("building pools:")

    build_simple("general_web", os.path.join(RAW, "fineweb-edu-000.parquet"), "text", stats)
    repo_index = []
    build_code(stats, repo_index)
    build_simple("stem", os.path.join(RAW, "openwebmath-000.parquet"), "text", stats)
    build_reasoning(stats)
    build_agentic(stats)
    build_long_context(stats, repo_index)
    build_indic_from_s4(stats)
    build_indic_translated(stats)
    build_simple("indic_D_synthetic", os.path.join(RAW, "sangraha-synth-hin-000.parquet"),
                 "text", stats)

    with open(os.path.join(POOLS, "pool_stats.json"), "w") as f:
        json.dump({"pools": stats}, f, indent=1)
    total = sum(s["chars"] for s in stats)
    print(f"\ntotal {total / 1e6:.0f} M chars across {len(stats)} pools")
    return 0


if __name__ == "__main__":
    sys.exit(main())
