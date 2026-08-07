# Design Notes — 40B India-First Model Data Strategy

Working document behind the submitted report (`site/index.html`). All reasoning and
arithmetic lives here; the report is the condensed version. Sources: `resources/s3-session.md`,
`resources/s3-transcript.md`, `research/` notes, S2 tokenizer results.

## 1. Token budget

- Chinchilla floor: 40B × 20 = **800B tokens** (the transcript's "200B" is a transcription
  error; 40 × 20 = 800).
- Gemma-class peers massively overtrain: Gemma-2/3 27B ≈ 13–14T tokens (≈500 tok/param).
  Instructor's observation: big good models cluster at 100–150 tok/param; V5 program hunch
  10–30T total.
- We are data-constrained on the objective that matters (Indic), so: **8T consumed tokens
  (200 tok/param)**, stretched by OPUS-style selection on general pools (V4 measured ~6x
  effective multiplier at the stage it ran; we conservatively claim 2–3x → effective
  ~16–20T on general capability). Indic pools are exempt from the selector (V4 lesson) so
  they get no multiplier — their leverage comes from fertility instead (§5).

## 2. Pre-training mixture (8T consumed)

| Pool | Share | Tokens | What / why |
|---|---|---|---|
| D1 Web foundation | 34% | 2.7T | FineWeb-Edu/DCLM-grade filtered global web. Language, world knowledge. |
| D2 Web diverse + India-English | 14% | 1.1T | Wikipedia, news, forums; **~0.45T Indian-perspective English**: Indian newspaper archives, court judgments, Parliament/PIB records, NCERT-lineage textbooks, Indian historiography. The "India-first" worldview largely lives in *English written in India* — collect it deliberately. |
| D3 Code | 22% | 1.75T | Stack-v2-class repos + issues + docs + tests + execution traces. Cross-dataset deduped. Code also buys reasoning on non-code tasks (session §5). |
| D4 STEM | 14% | 1.1T | arXiv/proof-pile-class math & science, worked solutions, DeepSeekMath-style mined math. |
| D5 Indic | 12% | 1.0T | ~300B **verified human** tokens across 12 languages (Sangraha-verified slivers, IndicCorp, news, books, govt., ASR-from-speech) × ~3 epochs (≤4-epoch ceiling) + synthetic top-up capped ≤40% (translation/rewrite/Q&A/textbook, all flagged). |
| D6 Agentic | 4% | 0.3T | Tool-call logs, terminal sessions, browser trajectories, API docs, structured JSON/XML, multi-step task transcripts. |

- **Always-on channel: 10% of every batch** guaranteed to D5 + India-English regardless of
  the selector (V4 used 8%; we widen because the model has two protected objectives:
  language *and* worldview).
- **Curriculum** (seeded growth as V4): early stages web-heavy (D1≈45%), final stage
  code+STEM+Indic-heavy; D-share table per stage decided by proxy ablation.
- **Anneal reserve, planned day one**: final ~15% of schedule uses held-back top-decile
  data (math with solutions, tested code, documentary-register Hindi/English, native Indic
  textbooks). OLMo-2 precedent: GSM 24→67 with a tiny reserve.

## 3. Post-training

- **SFT ~1M pairs**: 30% Indic *native-authored* (not translated — translation teaches
  translationese, we need native instruction-following), 25% coding w/ execution feedback,
  15% agentic multi-turn tool trajectories, 15% math/science with reasoning tags
  (reasoning enters at SFT — transcript), 15% general/writing/safety.
- **Preference ~400K comparisons**, incl. a dedicated **~100K India preference corpus**
  written by Indian annotators: honorifics/register per language, festivals/food/culture,
  neutrality across Indian political parties, Indian legal-ethical frames, colloquial
  Indian English accepted as valid register. This is where "views the world from an Indian
  perspective" is actually installed — pretraining supplies the knowledge, preference
  optimisation supplies the stance.
- **RLVR ~80K verifiable prompts**: 30K math (checkable answers), 25K code (unit tests),
  15K agentic (sandboxed browser/terminal tasks with programmatic success predicates),
  10K Indic (FLORES-anchored translation fidelity + script-correctness verifiers).
  A faulty verifier teaches wrong behaviour — verifiers get their own test suite.

## 4. Cleaning pipeline (ordered; every stage proxy-ablated at 140M–1B before scale)

1. **Extraction**: rendering-based extractor (not naive HTML strip), NFC normalization,
   per-paragraph script ID, code-mixed (Hinglish) kept as first-class.
2. **Language ID** per paragraph, 12 Indic + English + code-mixed classes.
3. **Quality filter — script-aware**: English pools use FineWeb-Edu-style classifier;
   each Indic language gets its own fastText classifier trained on ~5K native-speaker-labeled
   seed docs, expanded by DeepSeekMath-style mining. English-tuned filters are *never*
   applied to Indic (V4's selector erased Indic; that's why always-on exists).
   Colloquial Indian English ("prepone", "do the needful") whitelisted.
4. **Dedup**: exact hash + MinHash fuzzy, **per snapshot** not global (global removal cost
   ~58% of tokens and scored worse in FineWeb); cross-dataset dedup for code corpora.
5. **Decontamination**: source-level exclusion + exact + n-gram + embedding-similarity
   against all target benchmarks and our golden proxies; canary GUIDs planted to detect
   leaks post-hoc.
6. **PII/toxicity**: script-aware, per-language lists.
7. **Provenance & license at collection time**: permissive / attribution (CC-BY, e.g.
   Sangraha) / restricted buckets; the blend's shippability computed continuously, since
   one unknown-provenance source flags the whole mixture.

## 5. Languages, fertility targets, vocab size

**Languages (by speakers + script coverage), two tiers:**
- Tier 1: Hindi, Bengali, Marathi, Telugu, Tamil, Urdu — full data + SFT + preference + RLVR.
- Tier 2: Gujarati, Kannada, Malayalam, Odia, Punjabi, Assamese — pretraining + SFT.
- 12 languages ≈ 95%+ of Indian L1 speakers, and cover all 9 Brahmic script blocks + Perso-Arabic.

**Fertility targets** (tokens/word, FLORES-200 word standard — the class norm; "target
fertility" = lower is better, ideal 1.0):

| Domain | Target | Rationale |
|---|---|---|
| English | 1.20 | o200k_base parity (1.232) — don't regress the Gemma-class base. |
| Hindi / Urdu / Punjabi / Marathi | 1.45–1.55 | Devanagari/Gurmukhi/Perso-Arabic, moderately inflected. |
| Bengali / Assamese / Gujarati / Odia | 1.60–1.70 | |
| Telugu / Tamil / Kannada / Malayalam | 1.80–1.90 | Agglutinative: words are longer, so equal *per-character* cost implies higher per-word fertility; LP allocates by measured character-level deficit, not raw fertility. |
| Code | ≤1.00 | Identifiers/keywords mostly single tokens; whitespace runs merged. Metric: tokens per code-word on HumanEval/MBPP corpora, beat o200k by ≥5%. |
| Math | ≤1.30 | LaTeX commands (`\frac`, `\sum`, `^{`) as single tokens — math is *presented* to the model as markdown/LaTeX. |
| Science | ≤1.35 | Units, chemical formulas, Greek letters. |
| Agentic | ≤1.10 | JSON keys, schema punctuation runs, markdown/HTML structures as single tokens. Agentic traces are the longest contexts we train; low fertility here buys the most effective context. |

**Weighted mixture fertility** ≈ 0.48×1.2 + 0.22×1.0 + 0.14×1.3 + 0.12×1.65 + 0.04×1.1 ≈
**1.22** (V4 baseline ≈ 1.33) → 8T tokens ≈ 6.5T words, i.e. ~9% more content for the
same compute, before any data is collected.

**Vocab size — derived, not chosen:**
- Start from BrahmicTokenizer-131K's retrofit method (prune out-of-scope scripts, LP-allocate
  slots to measured deficits). Its 2,372 Indic slots close Odia-class gaps but leave
  Dravidian fertility ≥2.0.
- Needed slots: ~45K Indic subwords (LP across 12 languages ≈ 3–4K high-frequency
  morpheme-level units each), ~6K math/science, ~4K agentic/markup, ~2K code-mixing;
  ~10K reclaimed by pruning further non-target scripts.
- Total ≈ 178K → round up to **196,608 = 3 × 2^16** (GPU-friendly, headroom for the LP).
- Cost check: softmax head 196,608 × 5,120 ≈ **1.0B params (2.5% of 40B)**; input side via
  Kronecker embeddings ≈ 50M trainable (vs ~1B for a naive tied table) — the Kronecker
  paper is what makes 196K cheap.
- Training-signal check: 8T / 196,608 ≈ 41M mean occurrences/token; LP constraint that no
  allocated slot falls below ~50K expected occurrences (rare-token undertraining guard).
- Why not 131K: Dravidian stuck ≥2.0 → ~7% effective-corpus loss on the flagship objective.
  Why not 262K: <2% marginal fertility gain, +0.34B head params, fatter undertrained tail.

## 6. Evaluation (≤12 benchmarks — "pick 10, be selective")

General: MMLU-Pro, GPQA-Diamond. Code: LiveCodeBench, SWE-bench Verified.
Math: MATH-500 + AIME set. Agentic: BFCL (function calling), terminal/τ-bench-class tasks.
Indic: MILU, IndicGenBench, FLORES-200 (translation + the fertility meter itself).
India-first (must build): **"BharatDrishti" golden proxy** — freshly-authored India-centric
QA (history from Indian historiography, civics, law, culture) + an India-perspective
preference eval scored by Indian annotators; measured as win-rate vs Gemma 4. Freshly
authored ⇒ unleakable; lives in the golden-proxy tier, never trains, canaried.

Firewall: three tiers as V4 — training pools / benchmark-train splits (format familiarity,
allowed in always-on) / golden proxy (never trains). Per-checkpoint validation on ~5%
benchmark subsets across *all* domains simultaneously (catches capability trade-offs, e.g.
code share hurting language).

Success criterion per objective: beat published Gemma 4 numbers on the code/agentic/general
benchmarks; beat Gemma 4 by ≥10 points on MILU/IndicGenBench; ≥60% win-rate on BharatDrishti.
