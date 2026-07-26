"""Stage 0 -- acquire the corpus slice.

Downloads five parquet files from ai4bharat/sangraha at a pinned revision, measures
their size / row count / sha256 (never copy-pastes them: V4's manifest defect was
copy-pasted dataset sizes), and builds a ~50M-token slice by taking a deterministic
strided sample across the whole of each file rather than its first N rows.

Usage:
    python scripts/fetch_corpus.py             # download + build
    python scripts/fetch_corpus.py --smoke     # tiny slice for a pipeline dry run
"""

from __future__ import annotations

import argparse

import pyarrow.parquet as pq

from common import (
    RAW,
    REPO_ID,
    RUN,
    SOURCES,
    STAGE,
    TOKENIZER_FILES,
    MODELS,
    count_tokens_batch,
    jsonl_write,
    log,
    sha256_file,
    write_stage_stats,
    Timer,
)

SMOKE_DIVISOR = 30  # smoke run targets ~1/30th of the token budget (~2k docs)
# Nominal documents per contiguous sampling block. Sized from a measurement: in
# verified/hin the median near-duplicate pair sits ~2,500 rows apart, so blocks
# have to be at least that long for both members of a pair to be co-sampled.
BLOCK_DOCS = 2500
MAX_BLOCKS = 12  # spread across the shard, so the sample still spans the file


def ensure_tokenizers():
    from huggingface_hub import hf_hub_download
    import shutil

    repos = {"sarvam1": "sarvamai/sarvam-1", "qwen25": "Qwen/Qwen2.5-0.5B"}
    for name, fname in TOKENIZER_FILES.items():
        dest = MODELS / fname
        if not dest.exists():
            src = hf_hub_download(repos[name], "tokenizer.json")
            shutil.copy(src, dest)
            log(f"tokenizer {name} -> {dest.name}")


def resolve_revision() -> str:
    """Pin the exact dataset commit, so the run is reproducible against a moving repo."""
    from huggingface_hub import HfApi

    info = HfApi().dataset_info(REPO_ID)
    return info.sha


def download(revision: str) -> dict[str, dict]:
    from huggingface_hub import hf_hub_download

    files = {}
    for key, pool, claimed, path, _budget in SOURCES:
        local = RAW / path  # local_dir mirrors the repo layout, so keep the same path
        if not local.exists():
            with Timer(f"download {key}"):
                # local_dir downloads straight into data/raw -- the HF cache lives on a
                # different filesystem here, so a hardlink out of it would fail.
                hf_hub_download(
                    REPO_ID,
                    path,
                    repo_type="dataset",
                    revision=revision,
                    local_dir=str(RAW),
                )
        pf = pq.ParquetFile(local)
        files[key] = {
            "key": key,
            "pool": pool,
            "claimed_lang": claimed,
            "repo_path": path,
            "url": f"https://huggingface.co/datasets/{REPO_ID}/blob/{revision}/{path}",
            "bytes": local.stat().st_size,
            "sha256": sha256_file(local),
            "rows_in_file": pf.metadata.num_rows,
            "row_groups": pf.metadata.num_row_groups,
        }
        log(
            f"{key}: {files[key]['rows_in_file']:,} rows, "
            f"{files[key]['bytes']/1e6:.0f} MB, sha256 {files[key]['sha256'][:12]}…"
        )
    return files


# Every Sangraha file is a single row group of ~1.5 GB decompressed, so it is read in
# batches rather than whole -- there is ~2 GB of RAM free on this box.
READ_BATCH = 2000


def columns_of(path) -> list[str]:
    """`unverified/` shards have no `type` column; `verified/` ones do."""
    have = {f.name for f in pq.ParquetFile(path).schema_arrow}
    return [c for c in ("doc_id", "type", "text") if c in have]


def probe_tokens_per_doc(path, n: int = 300) -> float:
    """Mean tokens/doc from the head of the file, used to size the sampling stride."""
    pf = pq.ParquetFile(path)
    batch = next(pf.iter_batches(batch_size=n, columns=["text"]))
    texts = [t for t in batch.column("text").to_pylist() if t]
    if not texts:
        return 500.0
    return max(1.0, sum(count_tokens_batch(texts)) / len(texts))


def build_slice(files: dict[str, dict], smoke: bool) -> tuple[int, dict]:
    """Write data/stage/00-raw.jsonl.gz and return (docs, per-source stats)."""
    out_path = STAGE / "00-raw.jsonl.gz"
    per_source: dict[str, dict] = {}

    def records():
        for key, pool, claimed, path, budget in SOURCES:
            if smoke:
                budget = max(20_000, budget // SMOKE_DIVISOR)
            local = RAW / path
            meta = files[key]
            mean_tok = probe_tokens_per_doc(local)
            target_docs = max(1, int(budget / mean_tok))
            # Sample contiguous BLOCKS spread across the file, not every Nth row.
            # A 1-in-3000 stride would take at most one copy of any duplicated
            # document, so it would erase exactly the duplicate structure stage 5
            # exists to find. Blocks keep local structure while still spanning
            # the whole shard.
            n_blocks = max(1, min(MAX_BLOCKS, target_docs // BLOCK_DOCS))
            block_len = max(1, target_docs // n_blocks)
            block_gap = max(block_len, meta["rows_in_file"] // n_blocks)
            log(
                f"{key}: budget {budget:,} tok, ~{mean_tok:.0f} tok/doc -> target "
                f"{target_docs:,} docs in {n_blocks} blocks of {block_len} "
                f"(every {block_gap} rows)"
            )

            def in_sample(idx: int) -> bool:
                return (idx % block_gap) < block_len

            pf = pq.ParquetFile(local)
            cols = columns_of(local)
            taken = tokens = chars = 0
            global_idx = -1
            for rb in pf.iter_batches(batch_size=READ_BATCH, columns=cols):
                if tokens >= budget:
                    break
                ids = rb.column("doc_id").to_pylist()
                texts = rb.column("text").to_pylist()
                types = (
                    rb.column("type").to_pylist()
                    if "type" in cols
                    else [None] * len(texts)
                )
                batch, batch_meta = [], []
                for i in range(len(texts)):
                    global_idx += 1
                    if not in_sample(global_idx):
                        continue
                    t = texts[i]
                    if not t:
                        continue
                    batch.append(t)
                    batch_meta.append((ids[i], types[i]))
                if not batch:
                    continue
                counts = count_tokens_batch(batch)
                for (doc_id, dtype), text, ntok in zip(batch_meta, batch, counts):
                    if tokens >= budget:
                        break
                    taken += 1
                    tokens += ntok
                    chars += len(text)
                    yield {
                        "id": f"{key}:{doc_id}",
                        "src": key,
                        "pool": pool,
                        # The language the FOLDER PATH claims. Stage 3 does not trust it.
                        "claimed_lang": claimed,
                        "type": dtype,
                        "raw_tokens": ntok,
                        "raw_chars": len(text),
                        "text": text,
                    }
            per_source[key] = {
                **meta,
                "docs_taken": taken,
                "raw_tokens": tokens,
                "raw_chars": chars,
                "sampling": {
                    "mode": "contiguous blocks spread across the shard",
                    "blocks": n_blocks,
                    "block_len_docs": block_len,
                    "block_gap_rows": block_gap,
                },
                "token_budget": budget,
            }
            log(f"{key}: took {taken:,} docs / {tokens:,} tokens")

    with Timer("build slice"):
        n = jsonl_write(out_path, records())
    return n, per_source


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    ensure_tokenizers()
    revision = resolve_revision()
    (RUN / "source_revision.txt").write_text(revision + "\n", encoding="utf-8")
    log(f"pinned {REPO_ID} @ {revision}")

    files = download(revision)
    docs, per_source = build_slice(files, args.smoke)

    stats = {
        "stage": "00-fetch",
        "dataset": REPO_ID,
        "revision": revision,
        "license": "CC-BY-4.0",
        "smoke": args.smoke,
        "docs_in": 0,
        "docs_out": docs,
        "tokens_out": sum(s["raw_tokens"] for s in per_source.values()),
        "chars_out": sum(s["raw_chars"] for s in per_source.values()),
        "sources": per_source,
        "output": str((STAGE / "00-raw.jsonl.gz").relative_to(RUN.parent.parent)),
    }
    write_stage_stats("00-fetch", stats)
    log(f"TOTAL {docs:,} docs / {stats['tokens_out']:,} tokens")


if __name__ == "__main__":
    main()
