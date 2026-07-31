"""The strategy taxonomy: the assignment's "how many strategies, and what are they".

Everything here is sourced from the session writeup, the live-class transcript, and
the extracted widget data in resources/. Run-time numbers are never stored here --
they come from data/run/stats.json and are merged in by build_site.py.

Counting note. The material gives two different lists of eight, and they are not the
same eight:

  A. the pipeline map (Widget 1, section 2): extract, normalize, language ID, quality
     filter, deduplicate, PII scrub, decontaminate, manifest;
  B. the session's own closing commitment (section 14): "normalization, format
     discipline, quality filtering, deduplication, language validation, PII removal,
     decontamination, and the manifest" -- which drops Extract (studied in Session 3)
     and promotes format discipline to a stage of its own.

Union: 9. Asked whether eight was all of it, the instructor answered "This is a
minimal set... This is the minimum that you have to do."
"""

from __future__ import annotations

COUNTS = {
    "pipeline_stages": 8,
    "named_strategies": 9,
    "techniques": 30,
    "cross_cutting_concerns": 8,
    "why": (
        "Eight is the pipeline map and the instructor's stated minimum. Nine is the "
        "union of that map with the session's own closing list, which drops Extract "
        "and promotes format discipline. Thirty is the count of distinct named "
        "techniques inside those stages, most of which live only in the widgets."
    ),
    "instructor_quote": (
        "Asked whether the eight sections were all that was required: "
        "“This is a minimal set… This is the minimum that you have to do.”"
    ),
    "list_a": [
        "Extract", "Normalize", "Language ID", "Quality filter",
        "Deduplicate", "PII scrub", "Decontaminate", "Manifest",
    ],
    "list_b": [
        "Normalization", "Format discipline", "Quality filtering", "Deduplication",
        "Language validation", "PII removal", "Decontamination", "The manifest",
    ],
    "union_note": (
        "List A has Extract and folds format discipline inside Normalize. List B "
        "drops Extract and makes format discipline first-class. Nine distinct "
        "strategies between them."
    ),
}

# The reference yield descent from the session's pipeline widget, for overlay.
SESSION_YIELD = [
    ("Raw", 100), ("Extract", 92), ("Normalize", 88), ("Language ID", 61),
    ("Quality filter", 44), ("Deduplicate", 43), ("PII scrub", 42),
    ("Decontaminate", 42), ("Manifest", 42),
]

STAGES = [
    {
        "n": 1,
        "key": "extract",
        "name": "Extract",
        "session_kept": 92,
        "what": "Turn raw HTML into clean prose: pull the article body out of the page and drop everything around it.",
        "why": "Every page of a site repeats its header, menu, footer and legal boilerplate. Training on that is pure waste, and naive HTML stripping keeps it as if it were content.",
        "v4_defect": "Extraction quality was compared in Session 3: naive stripping keeps nav and legal text as content.",
        "applied": "inherited",
        "applied_note": "Sangraha ships extracted text, and extraction was Session 3's stage. Accounted for at 100% here rather than re-run, and labelled as inherited everywhere it appears.",
    },
    {
        "n": 2,
        "key": "normalize",
        "name": "Normalize",
        "session_kept": 88,
        "what": "NFC unicode normalization, HTML unescaping, removal of the replacement character, stripping of control / zero-width / bidi / BOM characters, whitespace collapse — and preservation of ZWNJ and ZWJ.",
        "why": "A byte-level tokenizer sees every stray character, so an uncleaned corpus spends permanent vocabulary slots on zero-width spaces and broken byte fragments. But ZWNJ (U+200C) and ZWJ (U+200D) are legitimate Brahmic script controls: a cleaner that strips all invisible characters corrupts Indic text while believing it is helping.",
        "v4_defect": "V4 had no clean_text() in any of its six ingestion scripts, which produced 46 garbage vocab tokens (18 zero-width, 20 broken utf-8, 4 HTML artifact, 4 private-use) plus ghost tags baked into the data.",
        "applied": "yes",
    },
    {
        "n": 3,
        "key": "format",
        "name": "Format discipline",
        "session_kept": None,
        "what": "Detect conversation markers written into the text as ordinary characters, and rewrite every source into one canonical format using the tokenizer's real special tokens.",
        "why": "During pretraining the model learns literal markers as ordinary subwords; during SFT it meets the real special tokens, and holds two competing ideas of what a conversation looks like. The audit measured the waste at 15 pretrain tokens where 10 were wanted, and 23 where 8 were wanted.",
        "v4_defect": "Four sources arrived in four incompatible formats and none used the tokenizer's real special tokens — the P0 root cause. The audit found [USER] x6, [SYSTEM] x2 and <|endoftext|> x3 inside the pretraining shards.",
        "applied": "yes",
        "applied_note": "Folded inside Normalize by the pipeline map; counted separately here because the session's own closing list makes it first-class.",
    },
    {
        "n": 4,
        "key": "langid",
        "name": "Language ID and validation",
        "session_kept": 61,
        "what": "Detect the language of every document at runtime and validate it against the code its source claims. Quarantine mismatches; flag code-switched and romanised documents rather than shipping them into one bucket.",
        "why": "Web-crawled data is mislabelled often enough that trusting the path pollutes the per-language pools. A Bengali document counted as Assamese inflates the Assamese total and starves the Bengali one, so every downstream sampling decision is computed on a corrupted denominator.",
        "v4_defect": "V4 trusted the folder name (verified/asm/) with no runtime detection. Telugu was coded te where the pipeline expected tel, and only worked because a fallback happened to return the right value — the kind of bug that is most dangerous, the kind that works.",
        "applied": "yes",
    },
    {
        "n": 5,
        "key": "quality",
        "name": "Quality filtering",
        "session_kept": 44,
        "what": "A cascade of nine Gopher/C4 heuristic rules, then a trained classifier scoring educational value 0–5, keeping documents at or above 3.0.",
        "why": "Most of the web is not worth training on. But filtering is not a neutral cleaning step — it is a decision about which languages the model will be able to speak, and a chain of rules tuned on English scores much of the low-resource web as garbage.",
        "v4_defect": "V4's OPUS selector used an English-heavy proxy that systematically under-valued Indic text, which forced an Always-ON bypass just to stop it discarding good Indic data.",
        "applied": "yes",
    },
    {
        "n": 6,
        "key": "dedup",
        "name": "Deduplication",
        "session_kept": 43,
        "what": "Exact content-hash dedup, then near-duplicate detection: shingle each document, reduce to a MinHash signature, band the signatures with LSH to find candidates, confirm against true Jaccard. Then do it again globally rather than per shard.",
        "why": "The naive picture — dropping identical documents — misses almost all real duplication, which is near-identical reposts. And duplication is global: two contributors can each dedup their own shard perfectly and still leave a document that appears in both.",
        "v4_defect": "Sangraha, the Indic web crawl, had zero deduplication at any level. The audit flagged it as both wasted compute and a memorization risk at scale.",
        "applied": "yes",
    },
    {
        "n": 7,
        "key": "pii",
        "name": "PII removal",
        "session_kept": 42,
        "what": "A regex layer for structured identifiers (emails, phones, IPs, national ids, credentials) and a name layer for the things that have no fixed shape.",
        "why": "It matters for the people whose data would otherwise sit in the corpus and for the legal usability of the corpus itself. The two layers have two different failure modes: regex is near-exact, while the name layer trades precision against recall — and that trade is sharper for Indic, where place names and given names overlap.",
        "v4_defect": "Dolma ran regex PII scrubbing; the Indic pipeline had none, so identifiers passed straight through into the training set.",
        "applied": "yes",
    },
    {
        "n": 8,
        "key": "decontaminate",
        "name": "Decontamination",
        "session_kept": 42,
        "what": "Fingerprint the evaluation sets as n-grams, scan every shard against those fingerprints, remove any training document that overlaps a test example, and plant canary strings so a later leak is still detectable.",
        "why": "It protects every number the project will ever report. Leaking the held-out set into training moved a reported benchmark score from 41% to 88% in the session's own demonstration.",
        "v4_defect": "V4 kept a Golden Proxy of the test splits (6.8M tokens, 11 shards) that was never trained on, so decontamination could be verified rather than assumed. Band B2 was dropped once it showed 18.7% leakage.",
        "applied": "yes",
    },
    {
        "n": 9,
        "key": "manifest",
        "name": "Manifest and provenance",
        "session_kept": 42,
        "what": "Emit per-shard provenance — source, licence, contributor, every cleaning script and its source hash, ingest timestamp, sha256, measured token count, language distribution — and gate on it. Identifiers derive from content, so the same input reproduces them.",
        "why": "A clean corpus is only trustworthy if you can say where every part came from and reproduce it on demand. The manifest is the datasheet published with the corpus and the audit trail behind the paper, and it is what the gating rule enforces: a contribution that cannot produce a manifest has not shipped clean data.",
        "v4_defect": "V4 copy-pasted dataset sizes and built ids from row_number() over a non-deterministic ordering, so the same input produced different ids every run. Its token estimate of words x 1.3 was wrong for Indic by 2–10x.",
        "applied": "yes",
    },
]

# The 24 techniques. `stage` links each to the strategy above; `metric` names the key
# in the run stats that reports our measured result for it.
TECHNIQUES = [
    ("Article-body extraction", "extract", "Keep the article, drop nav/footer/cookie banners.", "inherited"),
    ("NFC unicode normalization", "normalize", "One canonical encoding per character. NFC, not NFKC — the session is explicit.", "applied"),
    ("Control / C0-C1 stripping", "normalize", "Remove control bytes that carry no meaning.", "applied"),
    ("Zero-width and BOM stripping", "normalize", "ZWSP, word joiner, soft hyphen, BOM — pure noise.", "applied"),
    ("Bidi override stripping", "normalize", "LRM/RLM/LRO/RLO reverse reading order and corrupt text.", "applied"),
    ("ZWNJ / ZWJ preservation", "normalize", "U+200C and U+200D are legitimate Brahmic controls and are always kept. This is the rule that separates a cleaner built for these languages from one that was not.", "applied"),
    ("HTML entity unescaping", "normalize", "&amp; becomes &, so the tokenizer never learns the escape.", "applied"),
    ("U+FFFD removal", "normalize", "The replacement character marks a byte that failed to decode.", "applied"),
    ("Whitespace collapse", "normalize", "Runs of whitespace become one space.", "applied"),
    ("Hash after cleaning", "normalize", "sha256 is computed on the cleaned text, so two documents differing only in invisible junk dedupe to the same hash. Dedup and the manifest both trust this ordering.", "applied"),
    ("Ghost special-token flagging", "format", "Literal [USER] / <|endoftext|> / ### Instruction: markers are flagged, never silently deleted.", "applied"),
    ("Canonical format unification", "format", "Every source's role markers rewritten to one canonical <|user|> / <|assistant|> form.", "applied"),
    ("Runtime language detection", "langid", "fastText lid.176 on every document, never the folder path.", "applied"),
    ("Language-code normalization", "langid", "ISO 639-1 to 639-3 mapping — the te / tel bug, fixed rather than survived by accident.", "applied"),
    ("Code-switch and romanisation flagging", "langid", "Script-mix profiling separates code-switched and Latin-script Indic documents from clean monolingual ones.", "applied"),
    ("Gopher/C4 heuristic cascade", "quality", "Nine rules at the session's exact thresholds.", "applied"),
    ("Script-aware thresholds", "quality", "Per-language stop-word lists, per-script word-count floors and mean-word-length bands, and the danda as a sentence terminator.", "applied"),
    ("Trained classifier gate", "quality", "Weak structural labels train a fastText model that scores educational value 0–5; keep at 3.0. The session's recipe with a heuristic labeller in place of an LLM.", "applied-substitute"),
    ("Exact content-hash dedup", "dedup", "Identical cleaned text collapses to one document.", "applied"),
    ("Shingling", "dedup", "Overlapping k-word shingles turn a document into a set.", "applied"),
    ("MinHash signatures", "dedup", "Fixed-length signatures whose agreement rate estimates the true Jaccard.", "applied"),
    ("LSH banding", "dedup", "Bands and rows tuned so the S-curve inflects just below the drop threshold.", "applied"),
    ("Global vs. local dedup", "dedup", "Per-shard passes run alongside one global pass, to measure what only the global pass catches.", "applied"),
    ("PII regex layer", "pii", "Emails, Indian phone formats, IPv4, Aadhaar-shaped ids, PAN, GSTIN, credentials.", "applied"),
    ("PII name layer", "pii", "Gazetteer plus context gating, with a place-name exclusion list, scored across an aggressiveness dial.", "applied-substitute"),
    ("n-gram decontamination", "decontaminate", "13-gram fingerprints of the held-out eval sets, including Indic ones.", "applied"),
    ("Canary strings", "decontaminate", "Minted, planted in a probe shard, recovered by scan, and confirmed absent from the shipped corpus.", "applied"),
    ("Manifest provenance and licence gating", "manifest", "Eleven required fields, a licence allow-list, and an admit/block verdict per shard.", "applied"),
    ("Deterministic content-derived ids", "manifest", "Re-running the pipeline reproduces every shard id and hash byte for byte.", "applied"),
    ("Measured token counts", "manifest", "A named tokenizer, with per-language fertility reported — never words x 1.3.", "applied"),
]

# Assignment bullet 4: "any other strategy or concern was cleaned up?"
CONCERNS = [
    {
        "name": "Filter bias against low-resource scripts",
        "status": "handled",
        "detail": "The central finding of this run. The nine rules were executed twice, English-tuned and script-aware, and the gap between them measured per language.",
    },
    {
        "name": "Licence and attribution",
        "status": "handled",
        "detail": "Sangraha is CC-BY-4.0: shippable with an attribution obligation, which is recorded on every shard and checked against an allow-list before admission. The session's own caution stands — a project wanting an MIT-licensed model should not train on it.",
    },
    {
        "name": "Determinism and reproducibility",
        "status": "handled",
        "detail": "Source revision pinned, ids derived from content, timestamps derived from input rather than wall clock, and a two-run assert that shard ids and hashes come back byte-identical.",
    },
    {
        "name": "Memorisation risk from duplication",
        "status": "handled",
        "detail": "Near-duplicate removal at Jaccard 0.67, the transcript's stated threshold, with the local-versus-global gap measured rather than assumed.",
    },
    {
        "name": "Undeclared machine-translated content",
        "status": "flagged",
        "detail": "Roughly 65% of Sangraha is machine-translated or synthetic, and the split it sits in does not say which documents. Detecting translationese is out of scope here; the risk is recorded on the datasheet instead of being quietly inherited.",
    },
    {
        "name": "Toxicity and values filtering",
        "status": "deferred",
        "detail": "Deferred deliberately, matching the session: asked whether the quality cascade handles political or ideological bias, the answer was “Not yet — the LLM-labelled classifier needs to kick in here.” Doing it with a keyword list would be worse than not doing it.",
    },
    {
        "name": "Malicious and poisoned data",
        "status": "deferred",
        "detail": "An acknowledged gap in the session too — no stage owns it, and the answer given was that the cohort has to write that script itself. Named here rather than left implied.",
    },
    {
        "name": "Code-aware cleaning",
        "status": "not-applicable",
        "detail": "Whitespace must never be collapsed in code, and licence headers should go while code comments stay. This slice is prose, so the rule is stated but not exercised; a code pool would need a separate cleaner.",
    },
]

# Why these statistics and not others -- the instructor made this choice part of the
# grade: "What do you think is the right statistics for the cleanup — that itself,
# this is the most important thing."
STATISTICS_RATIONALE = [
    ("Per-stage survival, in documents and tokens", "A cleanup is a sequence of decisions to discard. The only honest summary is how much each decision discarded and at what point, which is why the descent is reported per stage rather than as one retention figure."),
    ("Tokens as well as documents", "Documents and tokens do not fall together. A stage that drops many short documents costs little; one that drops a few long ones costs a lot. Reporting only documents hides which is which."),
    ("Counterfactual, not just outcome", "For the two stages where the design choice is contested — the quality filter and dedup — the alternative was also run: English-tuned versus script-aware, and local versus global. A retention number without its counterfactual cannot tell you whether the filter or the text was broken."),
    ("Per-language, never only in aggregate", "The whole risk in an Indic corpus is a stage that behaves differently by language. An aggregate keep-rate would have concealed the single largest finding in this run."),
    ("Measured, with the measuring instrument named", "Token counts come from a named tokenizer with its per-language fertility published, because words x 1.3 is wrong for Indic by 2–10x and that error propagates into every budget."),
    ("Negative and near-zero results kept", "Stages that found little are reported at what they found. A cleanup report that only lists large numbers is selecting its own evidence."),
]

CITATIONS = [
    "Session 4 — Data Cleaning and Deduplication (writeup, live transcript, and all 10 interactive widgets)",
    "ai4bharat/sangraha — CC-BY-4.0, revision pinned in the manifest",
    "Gopher (Rae et al. 2021) and C4 (Raffel et al. 2020) heuristic filters",
    "FineWeb-Edu classifier recipe; Llama 3 and DCLM use fastText for the same gate",
    "MinHash (Broder 1997) and LSH banding (Indyk & Motwani 1998)",
]
