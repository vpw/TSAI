"""Generate RUN_REPORT.md from data/run/stats.json, so the prose cannot drift from the run."""

from __future__ import annotations

import json

from common import ROOT, RUN

OUT = ROOT / "RUN_REPORT.md"


def n(v):
    return f"{v:,}" if isinstance(v, int) else v


def main():
    s = json.loads((RUN / "stats.json").read_text(encoding="utf-8"))
    st, H = s["stages"], s["headline"]
    fetch, norm = st["00-fetch"], st["02-normalize"]
    lid, qual = st["03-langid"], st["04-quality-heuristics"]
    clf, ded = st.get("04b-quality-classifier", {}), st["05-dedup"]
    pii, dec, man = st["06-pii"], st["07-decontaminate"], st["08-manifest"]
    fx = st.get("fixtures", {})

    L = []
    w = L.append

    w("# Run report — cleaning a slice of ai4bharat/sangraha\n")
    w(f"Generated {s['generated_at_utc']} from `data/run/stats.json`. "
      f"{'**Smoke run.**' if s['smoke'] else 'Full run.'} "
      f"Total runtime {s['total_runtime_seconds']/60:.1f} min on 12 CPU cores, no GPU.\n")

    w("## Headline\n")
    w("| | |")
    w("|---|---:|")
    w(f"| Dataset | `{fetch['dataset']}` @ `{fetch['revision'][:12]}` |")
    w(f"| Licence | {fetch['license']} |")
    w(f"| Documents in → out | {n(H['docs_in'])} → {n(H['docs_out'])} |")
    w(f"| Tokens in → out ({man['tokenizer']}) | {n(H['tokens_in'])} → {n(man['tokens_total'])} |")
    w(f"| Document retention | {H['doc_retention_pct']}% |")
    w(f"| Token retention | {H['token_retention_pct']}% |")
    w(f"| Shards emitted / admitted | {n(man['shards'])} / {n(man['shards_admitted'])} |")
    w("")

    w("## Yield descent\n")
    w("| Stage | Docs out | Dropped | % of raw |")
    w("|---|---:|---:|---:|")
    for d in s["yield_descent"]:
        w(f"| {d['stage']}{' *(inherited)*' if d.get('inherited') else ''} "
          f"| {n(d['docs'])} | {n(d['dropped']) if d['dropped'] else '—'} | {d['pct_docs']}% |")
    w("")
    w(f"Session reference curve, for shape comparison: "
      f"{' → '.join(str(x) for x in s['session_reference_yield'])}.\n")

    w("## Stage 2 — normalize\n")
    w(f"- Noise characters removed: **{n(norm['noise_chars_removed_total'])}** "
      f"({norm['noise_chars_removed_by_class'] or 'none'})")
    w(f"- Indic joiners kept: {norm['joiners_kept'] or 'none present in this slice'}")
    w(f"- Documents changed by NFC: {n(norm['docs_changed_by_nfc'])}; "
      f"with HTML entities: {n(norm['docs_with_html_entities'])}")
    w(f"- Ghost conversation markers found: **{n(norm['ghost_tag_hits_total'])}** across "
      f"{n(norm['ghost_tag_docs'])} documents; {n(norm['docs_format_unified'])} rewritten to the "
      f"canonical format")
    w(f"- Garbage vocab slots: {norm['garbage_vocab_slots_before']} → "
      f"{norm['garbage_vocab_slots_after']} (scanned {n(norm['garbage_audit_docs_scanned'])} docs; "
      f"byte-fallback slots counted separately: {norm.get('byte_fallback_slots_before', 0)})")
    w(f"- `clean_text()` correctness fixtures: **{fx.get('passed')}/{fx.get('cases')} pass**\n")

    w("## Stage 3 — language ID\n")
    w(f"Detector: {lid['detector']}. Verdicts: `{lid['verdicts']}`\n")
    w(f"- Mislabelled documents caught by runtime detection: **{n(lid['mislabelled_caught_by_detection'])}**")
    w(f"- Caught by trusting the folder path: **{lid['mislabelled_caught_by_folder_trust']}**")
    w(f"- Code-switched, flagged: {n(lid['code_switched'])}; romanised Indic: {n(lid['romanized_indic'])}")
    w(f"- Documents a naive ISO 639-1 vs 639-3 comparison would have wrongly rejected "
      f"(the `te`/`tel` bug): **{n(lid['naive_iso1_vs_iso3_false_mismatches'])}**\n")
    w("### The corrupted denominator\n")
    w("| Language | If folder trusted | After detection | Delta |")
    w("|---|---:|---:|---:|")
    for lang, d in lid["token_denominator"].items():
        w(f"| {lang} | {n(d['tokens_if_folder_trusted'])} | {n(d['tokens_after_detection'])} "
          f"| {d['delta']:+,} |")
    w("")

    w("## Stage 4a — quality filter, English-tuned vs script-aware\n")
    w(f"Same nine rules, same documents, two calibrations. English-tuned keeps "
      f"**{n(qual['kept_by_config']['english_tuned'])}** documents "
      f"({100*qual['keep_rate_by_config']['english_tuned']:.1f}%); script-aware keeps "
      f"**{n(qual['kept_by_config']['script_aware'])}** ({100*qual['keep_rate_by_config']['script_aware']:.1f}%). "
      f"The gap is **{n(qual['rescued_docs_total'])} documents / "
      f"{n(qual['rescued_tokens_total'])} tokens** of good text an English-tuned filter destroys.\n")
    w("| Rule | Fails · English-tuned | Fails · script-aware |")
    w("|---|---:|---:|")
    for r in qual["rules"]:
        w(f"| `{r}` | {n(qual['rule_failures_by_config']['english_tuned'].get(r, 0))} "
          f"| {n(qual['rule_failures_by_config']['script_aware'].get(r, 0))} |")
    w("")
    w("| Language | Docs | Kept · English-tuned | Kept · script-aware |")
    w("|---|---:|---:|---:|")
    for lang, p in qual["per_lang"].items():
        w(f"| {lang} | {n(p['docs'])} | {n(p['en_kept'])} ({100*p['en_kept']/max(1,p['docs']):.1f}%) "
          f"| {n(p['sa_kept'])} ({100*p['sa_kept']/max(1,p['docs']):.1f}%) |")
    w("")

    if not clf.get("skipped"):
        w("## Stage 4b — trained classifier gate\n")
        w(f"{clf['model']}, weak labels from {clf['labeller']}. "
          f"Held-out exact accuracy **{clf['heldout_exact_accuracy']}**, within-one "
          f"**{clf['heldout_within_one_accuracy']}** on {n(clf['test_size'])} documents. "
          f"Dropped {n(clf['docs_in'] - clf['docs_out'])} below score {clf['threshold']}.\n")
        w(f"> {clf['caveat']}\n")

    w("## Stage 5 — deduplication\n")
    c = ded["config"]
    w(f"k={c['shingle_k']} words, n={c['permutations']} permutations, "
      f"b={c['bands']} × r={c['rows_per_band']}, LSH threshold {c['lsh_threshold']}, "
      f"drop at true Jaccard ≥ {c['jaccard_drop_threshold']}.\n")
    w(f"- Exact duplicates: **{n(ded['exact_duplicates'])}** "
      f"({n(ded['exact_duplicates_cross_shard'])} across shards)")
    w(f"- Near-duplicates: **{n(ded['near_duplicates'])}** "
      f"({n(ded['near_duplicates_cross_shard'])} across shards)")
    w(f"- LSH candidate pairs evaluated: {n(ded['lsh_candidate_pairs_global'])}")
    w(f"- Caught by per-shard local passes: {n(ded['caught_by_local_passes_only'])}; "
      f"by one global pass: {n(ded['caught_by_global_pass'])}; "
      f"**only the global pass: {n(ded['missed_by_local_caught_by_global'])}**")
    w(f"- Tokens removed: {n(ded['tokens_dropped'])}")
    w(f"- Index memory model: {ded['index_memory']['this_run']['resident_gib']} GiB for this run; "
      f"{ded['index_memory']['at_500M_docs']['resident_gib']} GiB at 500M documents\n")

    w("## Stage 6 — PII\n")
    w(f"{n(pii['spans_masked_total'])} spans masked across {n(pii['docs_with_pii'])} documents "
      f"({pii['spans_masked_by_kind']}). Masking saved {pii['token_waste']['pct_saved']}% of tokens "
      f"on the {n(pii['token_waste']['docs_measured'])} documents measured.\n")
    w("Precision and recall, against hand-annotated fixtures:\n")
    w("| Dial | Precision | Recall | F1 | Traps wrongly masked |")
    w("|---:|---:|---:|---:|---:|")
    for cv in pii["precision_curve"]:
        w(f"| {cv['aggressiveness']:.2f} | {cv['precision']} | {cv['recall']} | {cv['f1']} "
          f"| {cv['trap_spans_wrongly_masked']} |")
    w("")
    w(f"> {pii['scoring_note']}\n")
    w(f"> {pii['public_figure_caveat']}\n")

    w("## Stage 7 — decontamination\n")
    w(f"{n(dec['eval_ngrams_total'])} distinct {dec['ngram_n']}-gram fingerprints; a document is "
      f"dropped at {dec['min_matching_ngrams_to_drop']} distinct matches. "
      f"**{n(dec['docs_dropped'])} documents dropped** "
      f"({dec['contamination_rate']*100:.4f}%).\n")
    w("| Eval set | Split | Examples | n-grams | Hits |")
    w("|---|---|---:|---:|---:|")
    for k, v in dec["eval_sets"].items():
        w(f"| {k} (`{v['repo']}`) | {v['split']} | {n(v['examples_loaded'])} "
          f"| {n(v['new_ngrams'])} | {n(dec['hits_by_eval_set'].get(k, 0))} |")
    w("")
    w(f"Canary: minted {dec['canary']['canaries_minted']}, recovered by scan "
      f"{dec['canary']['detector_works']}, present in shipped corpus "
      f"{dec['canary'].get('canaries_found_in_shipped_corpus', 0)}.\n")

    w("## Stage 8 — manifest\n")
    w(f"{n(man['shards'])} shards, {n(man['shards_admitted'])} admitted. "
      f"{n(man['tokens_total'])} tokens measured with `{man['tokenizer']}`; "
      f"`words × 1.3` would have claimed {n(man['naive_words_x_1_3_total'])} "
      f"(**{man['naive_estimate_error_pct']:+}% error**).\n")
    w("| Language | Tokens shipped | Fertility (tokens/word) | vs the assumed 1.3 |")
    w("|---|---:|---:|---:|")
    for lang, f in sorted(man["fertility_tokens_per_word"].items(), key=lambda x: -x[1]):
        w(f"| {lang} | {n(man['tokens_by_lang'].get(lang, 0))} | {f} | {f/1.3:.1f}× |")
    w("")
    d = man["determinism"]
    w(f"Determinism: two independent runs over the same input produced identical shard ids and "
      f"hashes — **{d['identical']}** (`{d['shard_ids_run1'][0] if d['shard_ids_run1'] else ''}`).\n")

    w("## Integrity assertions\n")
    ig = s["integrity"]
    w(f"- No noise character survives: **{ig['no_noise_survives']}** "
      f"({ig['noise_chars_surviving']} found in {n(ig['docs_checked'])} documents)")
    w(f"- Indic joiners preserved: **{ig['joiners_preserved']}** "
      f"({ig['raw_docs_with_indic_joiners']} raw documents carried one; "
      f"{ig['shipped_docs_with_indic_joiners']} shipped documents still do)")
    w(f"- Shard ids reproduce across runs: **{d['identical']}**")
    w(f"- clean_text() fixtures: **{fx.get('passed')}/{fx.get('cases')}**\n")

    OUT.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
