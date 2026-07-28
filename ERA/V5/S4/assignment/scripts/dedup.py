"""Stage 5 -- deduplication. The stage V4's Indic crawl skipped entirely.

Three moves, as the session frames them:

  1. exact  -- content hashes (computed after cleaning, so documents differing only in
     invisible junk collapse together);
  2. shingle + MinHash -- each document becomes a set of overlapping k-word shingles,
     then a fixed-length signature of minimum hash values whose expected agreement
     rate equals the true Jaccard of the sets;
  3. LSH -- the signature is split into b bands of r rows and only documents colliding
     in at least one band are compared, so we never do the N^2 comparison.

Candidate pairs surfaced by LSH are then confirmed against the true Jaccard of the
full shingle sets and dropped at the transcript's stated threshold of 0.67.

The run is done twice: once per source shard in isolation (what each contributor
could do alone) and once globally (what only a central pass can do). The gap between
them is the point of the whole exercise.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import re

import numpy as np

from common import (
    STAGE,
    Timer,
    jsonl_read,
    jsonl_write,
    log,
    write_stage_stats,
)

SHINGLE_K = 5  # words per shingle, as in the session's simulator
JACCARD_DROP = 0.67  # the transcript's rule: "more than that, throw it away"
# b and r are chosen so the LSH S-curve inflects at (1/b)^(1/r) = 0.662, just BELOW
# the 0.67 drop threshold. The banding stage has to be slightly more permissive than
# the confirmation stage, or it silently discards true duplicates before they are
# ever scored. 18 x 7 = 126 permutations.
BANDS = 18
ROWS = 7
NUM_PERM = BANDS * ROWS  # 126
MERSENNE = (1 << 61) - 1
# Shingles are taken over the first SHINGLE_WORDS words of a document. A prefix
# rather than a subsample, because two documents must shingle the *same* way for
# their Jaccard to mean anything -- and a repost shares its opening. This also keeps
# the retained shingle sets inside a few hundred MB on a 16 GB box.
SHINGLE_WORDS = 1200

WORD_RE = re.compile(r"\S+")

# The session's teaching configuration, reported alongside ours for comparison.
DEMO_CONFIG = {
    "k": 5,
    "bands": 6,
    "rows": 4,
    "permutations": 24,
    "lsh_threshold": round((1 / 6) ** (1 / 4), 3),
}


def lsh_threshold(bands: int, rows: int) -> float:
    """Similarity at which a pair becomes more likely than not to be a candidate."""
    return (1 / bands) ** (1 / rows)


def shingles(text: str, k: int = SHINGLE_K) -> np.ndarray:
    """Sorted unique 32-bit hashes of the document's overlapping k-word shingles."""
    words = WORD_RE.findall(text)[:SHINGLE_WORDS]
    if not words:
        return np.empty(0, dtype=np.uint32)
    if len(words) < k:
        grams = [" ".join(words)]
    else:
        grams = [" ".join(words[i : i + k]) for i in range(len(words) - k + 1)]
    out = np.fromiter(
        (
            int.from_bytes(
                hashlib.blake2b(g.encode("utf-8"), digest_size=4).digest(), "big"
            )
            for g in grams
        ),
        dtype=np.uint32,
        count=len(grams),
    )
    return np.unique(out)


class MinHasher:
    """h_i(x) = ((a_i * x + b_i) mod (2^61 - 1)) mod 2^32, vectorised over shingles."""

    def __init__(self, num_perm: int = NUM_PERM, seed: int = 20260726):
        rng = np.random.default_rng(seed)
        # a and b stay under 2^32 so that a*x + b cannot overflow uint64 for a 32-bit
        # shingle hash x -- the standard MinHash parameterisation.
        self.a = rng.integers(1, 1 << 32, size=num_perm, dtype=np.uint64)
        self.b = rng.integers(0, 1 << 32, size=num_perm, dtype=np.uint64)
        self.num_perm = num_perm

    def signature(self, sh: np.ndarray) -> np.ndarray:
        if sh.size == 0:
            return np.full(self.num_perm, np.iinfo(np.uint32).max, dtype=np.uint32)
        x = sh.astype(np.uint64)
        h = (self.a[:, None] * x[None, :] + self.b[:, None]) % MERSENNE
        return (h.min(axis=1) % (1 << 32)).astype(np.uint32)


def band_keys(sig: np.ndarray, bands: int = BANDS, rows: int = ROWS) -> list[bytes]:
    return [sig[i * rows : (i + 1) * rows].tobytes() for i in range(bands)]


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 and b.size == 0:
        return 1.0
    inter = np.intersect1d(a, b, assume_unique=True).size
    union = a.size + b.size - inter
    return inter / union if union else 0.0


def index_memory_model(n_docs: int) -> dict:
    """The session's sizing arithmetic, applied to an arbitrary corpus size."""
    sig_bytes = 128 * 8  # the session prices a signature at 128 x 8 B = 1 KB
    resident = n_docs * sig_bytes * 1.6  # x1.6 for LSH band tables + doc ids
    return {
        "docs": n_docs,
        "signature_bytes_per_doc": sig_bytes,
        "overhead_factor": 1.6,
        "resident_bytes": int(resident),
        "resident_gib": round(resident / (1 << 30), 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="03b-quality-clf.jsonl.gz")
    ap.add_argument("--out", dest="out", default="04-dedup.jsonl.gz")
    args = ap.parse_args()

    hasher = MinHasher()

    ids: list[str] = []
    srcs: list[str] = []
    hashes: list[str] = []
    tokens: list[int] = []
    sigs: list[np.ndarray] = []
    shingle_sets: list[np.ndarray] = []

    with Timer("minhash signatures"):
        for rec in jsonl_read(STAGE / args.inp):
            sh = shingles(rec["text"])
            ids.append(rec["id"])
            srcs.append(rec["src"])
            hashes.append(rec["hash"])
            tokens.append(rec["raw_tokens"])
            shingle_sets.append(sh)
            sigs.append(hasher.signature(sh))
    n = len(ids)
    log(f"{n:,} documents signed with {NUM_PERM} permutations")

    # ---------------------------------------------------------------- exact
    first_seen: dict[str, int] = {}
    exact_dup_of: dict[int, int] = {}
    exact_cross_shard = 0
    for i, h in enumerate(hashes):
        if h in first_seen:
            j = first_seen[h]
            exact_dup_of[i] = j
            if srcs[i] != srcs[j]:
                exact_cross_shard += 1
        else:
            first_seen[h] = i

    # ---------------------------------------------------- near, global pass
    def run_lsh(indices: list[int]) -> tuple[dict[int, int], list[dict], int]:
        """Returns (duplicate -> kept representative, confirmed pairs, candidates).

        Buckets are processed one at a time rather than materialised into a global
        pair set: with this corpus's duplication density a full candidate set would
        be tens of millions of pairs. Once a document is assigned to a duplicate
        group it is skipped, which prunes the rest of its group's pairs.
        """
        buckets: dict[bytes, list[int]] = collections.defaultdict(list)
        for i in indices:
            if i in exact_dup_of:
                continue
            for band, key in enumerate(band_keys(sigs[i])):
                buckets[bytes([band]) + key].append(i)

        dup_of: dict[int, int] = {}
        confirmed: list[dict] = []
        seen_pairs: set[tuple[int, int]] = set()
        candidates = 0

        for members in buckets.values():
            if len(members) < 2:
                continue
            members.sort()
            for x, i in enumerate(members):
                if i in dup_of:
                    continue
                for j in members[x + 1 :]:
                    if j in dup_of:
                        continue
                    pair = (i, j)
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)
                    candidates += 1
                    jac = jaccard(shingle_sets[i], shingle_sets[j])
                    if jac < JACCARD_DROP:
                        continue
                    est = float((sigs[i] == sigs[j]).mean())
                    dup_of[j] = i
                    if len(confirmed) < 12:
                        confirmed.append(
                            {
                                "kept": ids[i],
                                "dropped": ids[j],
                                "same_shard": srcs[i] == srcs[j],
                                "true_jaccard": round(jac, 4),
                                "minhash_estimate": round(est, 4),
                                "estimate_error": round(abs(jac - est), 4),
                                "shingles_kept": int(shingle_sets[i].size),
                                "shingles_dropped": int(shingle_sets[j].size),
                            }
                        )
        return dup_of, confirmed, candidates

    all_idx = list(range(n))
    with Timer("global LSH pass"):
        global_dup, confirmed_pairs, global_candidates = run_lsh(all_idx)

    # -------------------------------------------- near, per-shard local pass
    local_dup: dict[int, int] = {}
    local_candidates = 0
    by_src: dict[str, list[int]] = collections.defaultdict(list)
    for i, s in enumerate(srcs):
        by_src[s].append(i)
    with Timer("per-shard local passes"):
        for s, idx in by_src.items():
            d, _, c = run_lsh(idx)
            local_dup.update(d)
            local_candidates += c

    # Exact duplicates a local-only pass would also have missed.
    local_exact = {
        i: j for i, j in exact_dup_of.items() if srcs[i] == srcs[j]
    }
    caught_locally = set(local_dup) | set(local_exact)
    caught_globally = set(global_dup) | set(exact_dup_of)
    only_global = caught_globally - caught_locally

    dropped = caught_globally
    cross_shard_near = sum(
        1 for i, j in global_dup.items() if srcs[i] != srcs[j]
    )

    tokens_dropped = sum(tokens[i] for i in dropped)
    per_src_dropped = collections.Counter(srcs[i] for i in dropped)

    keep = [i for i in range(n) if i not in dropped]
    keep_set = set(keep)

    def records():
        k = 0
        for rec in jsonl_read(STAGE / args.inp):
            if k in keep_set:
                yield rec
            k += 1

    with Timer("write deduped"):
        written = jsonl_write(STAGE / args.out, records())

    stats = {
        "stage": "05-dedup",
        "config": {
            "shingle_k": SHINGLE_K,
            "permutations": NUM_PERM,
            "bands": BANDS,
            "rows_per_band": ROWS,
            "lsh_threshold": round(lsh_threshold(BANDS, ROWS), 4),
            "jaccard_drop_threshold": JACCARD_DROP,
            "shingle_window_words": SHINGLE_WORDS,
        },
        "session_demo_config": DEMO_CONFIG,
        "docs_in": n,
        "docs_out": written,
        "docs_dropped": len(dropped),
        "tokens_dropped": tokens_dropped,
        "exact_duplicates": len(exact_dup_of),
        "exact_duplicates_cross_shard": exact_cross_shard,
        "near_duplicates": len(global_dup),
        "near_duplicates_cross_shard": cross_shard_near,
        "lsh_candidate_pairs_global": global_candidates,
        "lsh_candidate_pairs_local_total": local_candidates,
        "caught_by_local_passes_only": len(caught_locally),
        "caught_by_global_pass": len(caught_globally),
        "missed_by_local_caught_by_global": len(only_global),
        "dropped_by_source": dict(per_src_dropped),
        "confirmed_pairs": confirmed_pairs,
        "index_memory": {
            "this_run": index_memory_model(n),
            "at_500M_docs": index_memory_model(500_000_000),
        },
        "note": (
            "Local per-shard dedup does not produce a globally deduplicated corpus. "
            "The gap is 'missed_by_local_caught_by_global'."
        ),
    }
    write_stage_stats("05-dedup", stats)
    log(
        f"{n:,} -> {written:,} docs | exact {len(exact_dup_of):,} "
        f"(cross-shard {exact_cross_shard:,}) | near {len(global_dup):,} "
        f"(cross-shard {cross_shard_near:,}) | only the global pass caught {len(only_global):,}"
    )


if __name__ == "__main__":
    main()
