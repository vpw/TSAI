#!/usr/bin/env python3
"""Tokenize each lane pool into flat arrays, holding out a validation slice per lane.

Uses S4's `tokenizer-sarvam1.json` unchanged, so fertility (tokens per word) stays
comparable with the S4 cleaning run and the Indic lane is not unfairly penalised by an
English-shaped vocabulary.

Per lane it writes to `proxy/data/tokens/`:
  <lane>.train.bin      uint32 flat token ids, documents separated by EOS
  <lane>.trainmask.bin  uint8  1 = token contributes to the loss (agentic only; others all 1)
  <lane>.val.npz        tokens / mask / tbytes  — tbytes is the exact UTF-8 byte length of
                        each token's surface form, taken from the tokenizer's offsets, so
                        bits-per-byte is summed over exactly the positions that were scored.

Shapes and dtypes are recorded in `token_stats.json`; training memory-maps the .bin files.
Output streams to disk as it goes — the pools are larger than this machine's free RAM.

Validation documents are chosen by a hash of the text and removed from train by the same
hash, so no validation document is trained on.
"""
import hashlib
import json
import os
import sys
import time

import numpy as np
from tokenizers import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
PROXY = os.path.dirname(HERE)
POOLS = os.path.join(PROXY, "data", "pools")
TOK = os.path.join(PROXY, "data", "tokens")
# S4's tokenizer, unchanged. Local checkout reads it from the S4 tree; the GPU box gets an
# uploaded copy under proxy/models/.
_TOKENIZER_CANDIDATES = [
    os.path.join(PROXY, "models", "tokenizer-sarvam1.json"),
    os.path.normpath(os.path.join(PROXY, "..", "..", "..", "S4", "assignment",
                                  "models", "tokenizer-sarvam1.json")),
]
TOKENIZER = next((p for p in _TOKENIZER_CANDIDATES if os.path.exists(p)),
                 _TOKENIZER_CANDIDATES[0])

VAL_TOKEN_TARGET = 1_500_000     # per lane; enough for a stable bpb, cheap to score
VAL_HASH_BUCKETS = 97            # ~1% of documents are candidates for validation
BATCH = 500

LANES = ["general_web", "code", "stem", "reasoning", "agentic", "long_context",
         "indic_A_verified", "indic_B_unverified", "indic_C_translated", "indic_D_synthetic"]


def doc_hash(text):
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8", "ignore"),
                                          digest_size=8).digest(), "big")


def tok_bytes_from_offsets(text, offsets):
    """Exact UTF-8 byte length of each token's surface form."""
    out = np.zeros(len(offsets), dtype=np.uint16)
    for i, (a, b) in enumerate(offsets):
        out[i] = min(len(text[a:b].encode("utf-8")), 65535)
    return out


def main():
    os.makedirs(TOK, exist_ok=True)
    tk = Tokenizer.from_file(TOKENIZER)
    eos = tk.token_to_id("</s>")
    if eos is None:
        eos = tk.token_to_id("<|endoftext|>") or 2
    print(f"tokenizer vocab={tk.get_vocab_size()} eos_id={eos}", flush=True)

    stats = {}
    for lane in LANES:
        path = os.path.join(POOLS, lane + ".jsonl")
        if not os.path.exists(path):
            print(f"!! missing pool {lane}, skipping")
            continue
        t0 = time.time()
        is_agentic = lane == "agentic"

        f_ids = open(os.path.join(TOK, f"{lane}.train.bin"), "wb")
        f_msk = open(os.path.join(TOK, f"{lane}.trainmask.bin"), "wb")
        val_ids, val_mask, val_tb = [], [], []
        counters = {"train_tokens": 0, "val_tokens": 0, "val_docs": 0,
                    "sup_bytes": 0, "tot_bytes": 0, "sup_tokens": 0}
        n_docs = 0
        raw_chars = 0

        def flush(batch_docs):
            flat = [t for _, segs in batch_docs for t, _ in segs]
            if not flat:
                return
            encs = tk.encode_batch(flat, add_special_tokens=False)
            k = 0
            for is_val, segs in batch_docs:
                ids_doc, mask_doc, tb_doc = [], [], []
                for text, sup in segs:
                    e = encs[k]
                    k += 1
                    ids_doc.extend(e.ids)
                    mask_doc.extend([sup] * len(e.ids))
                    if is_val:
                        tb_doc.append(tok_bytes_from_offsets(text, e.offsets))
                    nb = len(text.encode("utf-8"))
                    counters["tot_bytes"] += nb
                    if sup:
                        counters["sup_bytes"] += nb
                        counters["sup_tokens"] += len(e.ids)
                ids_doc.append(eos)
                mask_doc.append(0)          # never score the separator
                if is_val:
                    tb_doc.append(np.zeros(1, dtype=np.uint16))
                    val_ids.append(np.asarray(ids_doc, dtype=np.uint32))
                    val_mask.append(np.asarray(mask_doc, dtype=np.uint8))
                    val_tb.append(np.concatenate(tb_doc))
                    counters["val_tokens"] += len(ids_doc)
                    counters["val_docs"] += 1
                else:
                    np.asarray(ids_doc, dtype=np.uint32).tofile(f_ids)
                    np.asarray(mask_doc, dtype=np.uint8).tofile(f_msk)
                    counters["train_tokens"] += len(ids_doc)

        batch = []
        val_hashes = set()
        with open(path) as fh:
            for line in fh:
                d = json.loads(line)
                if is_agentic:
                    segs = [(s["text"], int(s["sup"])) for s in d["segments"]]
                    whole = "".join(s[0] for s in segs)
                else:
                    segs = [(d["text"], 1)]
                    whole = d["text"]
                raw_chars += len(whole)
                h = doc_hash(whole)
                if h in val_hashes:
                    continue                        # exact duplicate of a held-out doc
                is_val = (h % VAL_HASH_BUCKETS == 0
                          and counters["val_tokens"] < VAL_TOKEN_TARGET)
                if is_val:
                    val_hashes.add(h)
                batch.append((is_val, segs))
                n_docs += 1
                if len(batch) >= BATCH:
                    flush(batch)
                    batch = []
        flush(batch)
        f_ids.close()
        f_msk.close()

        if val_ids:
            np.savez(os.path.join(TOK, f"{lane}.val.npz"),
                     tokens=np.concatenate(val_ids),
                     mask=np.concatenate(val_mask),
                     tbytes=np.concatenate(val_tb))
        del val_ids, val_mask, val_tb

        n_tok = counters["train_tokens"] + counters["val_tokens"]
        s = {
            "docs": n_docs, "val_docs": counters["val_docs"],
            "train_tokens": counters["train_tokens"], "val_tokens": counters["val_tokens"],
            "chars": raw_chars, "bytes_total": counters["tot_bytes"],
            "chars_per_token": round(raw_chars / max(n_tok, 1), 3),
            "bytes_per_token": round(counters["tot_bytes"] / max(n_tok, 1), 3),
            "seconds": round(time.time() - t0, 1),
        }
        if is_agentic:
            s["supervised_token_fraction"] = round(counters["sup_tokens"] / max(n_tok, 1), 4)
            s["supervised_byte_fraction"] = round(
                counters["sup_bytes"] / max(counters["tot_bytes"], 1), 4)
        stats[lane] = s
        print(f"  {lane:22s} train {s['train_tokens'] / 1e6:8.2f}M  "
              f"val {s['val_tokens'] / 1e6:5.2f}M  chars/tok {s['chars_per_token']:5.2f}  "
              f"{s['seconds']:6.1f}s"
              + (f"  supervised {s['supervised_token_fraction']:.1%}" if is_agentic else ""),
              flush=True)

    with open(os.path.join(TOK, "token_stats.json"), "w") as f:
        json.dump({"tokenizer": os.path.basename(TOKENIZER),
                   "vocab_size": tk.get_vocab_size(), "eos_id": eos,
                   "train_dtype": "uint32", "mask_dtype": "uint8",
                   "lanes": stats}, f, indent=1)
    total = sum(v["train_tokens"] for v in stats.values())
    print(f"\ntotal train tokens across lanes: {total / 1e6:.1f}M")
    return 0


if __name__ == "__main__":
    sys.exit(main())
