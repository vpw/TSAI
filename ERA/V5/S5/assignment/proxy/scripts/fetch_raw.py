#!/usr/bin/env python3
"""Download the raw source files the proxy ablation trains on.

Every source here is a dataset the S5 inventory (`data/inventory.json`) actually names,
so the proxy is fed from the same supply the plan is written against. Files land in
`proxy/data/raw/` and are recorded with size + sha256 in `proxy/data/raw/sources.json`.

Downloads resume: a file already present at the expected size is skipped.
"""
import hashlib
import json
import os
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
PROXY = os.path.dirname(HERE)
RAW = os.path.join(PROXY, "data", "raw")

HF = "https://huggingface.co/datasets"

# (local name, url, lane it feeds, inventory row it stands in for)
SOURCES = [
    ("fineweb-edu-000.parquet",
     f"{HF}/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT/000_00000.parquet",
     "general_web", "FineWeb-Edu (inventory: 1.3T tokens)"),

    ("codeparrot-001.json.gz",
     f"{HF}/codeparrot/codeparrot-clean/resolve/main/file-000000000001.json.gz",
     "code", "stands in for The Stack v2 (gated); same shape, permissive Python"),

    ("commitpackft-python.jsonl",
     f"{HF}/bigcode/commitpackft/resolve/main/data/python/data.jsonl",
     "code", "CommitPack / CommitPackFT (inventory: 4B tokens)"),

    ("openwebmath-000.parquet",
     f"{HF}/open-web-math/open-web-math/resolve/main/data/train-00000-of-00114-5a023365406cb9c4.parquet",
     "stem", "proof-pile-2 component (inventory: 55B tokens)"),

    ("openr1-math-000.parquet",
     f"{HF}/open-r1/OpenR1-Math-220k/resolve/main/data/train-00000-of-00010.parquet",
     "reasoning", "OpenR1-Math, R1-distilled (inventory: 1.6B tokens)"),

    ("glaive-fc-v2.json",
     f"{HF}/glaiveai/glaive-function-calling-v2/resolve/main/glaive-function-calling-v2.json",
     "agentic", "Glaive function-calling v2 (inventory: 50M tokens)"),

    ("toolace.json",
     f"{HF}/Team-ACE/ToolACE/resolve/main/data.json",
     "agentic", "ToolACE (inventory: 60M tokens)"),

    ("hermes-fc.json",
     f"{HF}/NousResearch/hermes-function-calling-v1/resolve/main/func-calling.json",
     "agentic", "Hermes function-calling, multi-turn (inventory: 22M tokens)"),

    ("hermes-fc-single.json",
     f"{HF}/NousResearch/hermes-function-calling-v1/resolve/main/func-calling-singleturn.json",
     "agentic", "Hermes function-calling, single-turn"),

    ("samanantar-hi-000.parquet",
     f"{HF}/ai4bharat/samanantar/resolve/main/hi/train-00000-of-00008.parquet",
     "indic_C_translated", "Samanantar (inventory: 2B tokens, tier C)"),

    ("sangraha-synth-hin-000.parquet",
     f"{HF}/ai4bharat/sangraha/resolve/main/synthetic/hin_Deva/wiki_hin_Deva_0000_of_0063.parquet",
     "indic_D_synthetic", "Sangraha synthetic (inventory: 162B tokens, tier D)"),
]


def head_size(url):
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers.get("Content-Length", 0))


def download(url, dest, expect, attempts=4):
    """Resumable download. A stream that ends early is a silent truncation unless the
    final size is checked against Content-Length, so it is checked and resumed."""
    tmp = dest + ".part"
    for attempt in range(1, attempts + 1):
        start = os.path.getsize(tmp) if os.path.exists(tmp) else 0
        if expect and start >= expect:
            break
        req = urllib.request.Request(url)
        if start:
            req.add_header("Range", f"bytes={start}-")
            print(f"      resuming at {start / 1e6:.0f} MB (attempt {attempt})")
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "ab") as f:
                got = start
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    got += len(chunk)
                    if got % (64 << 20) < (1 << 20):
                        mb = got / 1e6
                        print(f"      {mb:8.0f} MB  "
                              f"{mb / max(time.time() - t0, 1e-9):5.1f} MB/s", flush=True)
        except Exception as e:
            print(f"      stream error: {e}")
        if not expect or os.path.getsize(tmp) >= expect:
            break
    got = os.path.getsize(tmp)
    if expect and got != expect:
        raise IOError(f"truncated: got {got} of {expect} bytes after {attempts} attempts")
    os.replace(tmp, dest)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    os.makedirs(RAW, exist_ok=True)
    manifest = []
    for name, url, lane, note in SOURCES:
        dest = os.path.join(RAW, name)
        print(f"[{lane}] {name}")
        try:
            remote = head_size(url)
        except Exception as e:
            print(f"   !! HEAD failed: {e}")
            remote = 0
        if os.path.exists(dest) and remote and os.path.getsize(dest) == remote:
            print(f"   already have {os.path.getsize(dest) / 1e6:.0f} MB")
        else:
            if os.path.exists(dest):
                # wrong size on disk: restart from whatever the .part file has
                os.remove(dest)
            print(f"   downloading {remote / 1e6:.0f} MB ...")
            try:
                download(url, dest, remote)
            except Exception as e:
                print(f"   !! FAILED: {e}")
                manifest.append({"name": name, "url": url, "lane": lane,
                                 "note": note, "status": f"failed: {e}"})
                continue
        manifest.append({
            "name": name, "url": url, "lane": lane, "note": note,
            "bytes": os.path.getsize(dest), "sha256": sha256(dest), "status": "ok",
        })
        print(f"   ok  {manifest[-1]['bytes'] / 1e6:.0f} MB  sha256 {manifest[-1]['sha256'][:16]}")

    with open(os.path.join(RAW, "sources.json"), "w") as f:
        json.dump({"fetched_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   "sources": manifest}, f, indent=1)
    bad = [m for m in manifest if m["status"] != "ok"]
    print(f"\n{len(manifest) - len(bad)}/{len(SOURCES)} sources ok")
    for m in bad:
        print("  FAILED:", m["name"], m["status"])
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
