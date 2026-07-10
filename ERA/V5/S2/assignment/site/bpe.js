// Client-side re-implementation of the HF `tokenizers` word-level BPE
// encode path (Whitespace pre-tokenizer + plain character-pair BPE merges,
// no continuing-subword-prefix / end-of-word-suffix markers, no unk token).
//
// Must stay in lockstep with scripts/train_tokenizer.py's "word" variant.
// Verified against the Python tokenizer's own output (see
// scripts/verify_bpe_js.py) before being trusted here.

// Mirrors HF's Whitespace pre-tokenizer regex `\w+|[^\w\s]+`, where Rust's
// Unicode-aware `\w` is letters+marks+digits+connector-punctuation (this is
// what keeps Devanagari/Telugu combining marks attached to their base
// consonant instead of being split off).
const PRETOKEN_RE = /[\p{L}\p{M}\p{N}\p{Pc}]+|[^\s\p{L}\p{M}\p{N}\p{Pc}]+/gu;

function preTokenize(text) {
  const chunks = [];
  let m;
  PRETOKEN_RE.lastIndex = 0;
  while ((m = PRETOKEN_RE.exec(text)) !== null) {
    chunks.push({ text: m[0], start: m.index, end: m.index + m[0].length });
  }
  return chunks;
}

// Builds a Map<string, Map<string, number>> so mergeRank.get(a).get(b) gives
// the merge's rank (lower = applied earlier), matching the `merges` array's
// own order from tokenizer.json.
function buildMergeRank(merges) {
  const rank = new Map();
  merges.forEach(([a, b], idx) => {
    if (!rank.has(a)) rank.set(a, new Map());
    rank.get(a).set(b, idx);
  });
  return rank;
}

function buildTokenToId(tokens) {
  const map = new Map();
  for (const t of tokens) map.set(t.token, t.id);
  return map;
}

// Standard BPE merge loop: repeatedly find the lowest-rank adjacent pair
// present anywhere in the symbol sequence, merge *all* its occurrences in
// one pass, and repeat until no mergeable pair remains.
function bpeMergeWord(word, mergeRank) {
  let symbols = Array.from(word); // codepoint-safe split
  if (symbols.length <= 1) return symbols;

  for (;;) {
    let bestRank = Infinity;
    let bestIdx = -1;
    for (let i = 0; i < symbols.length - 1; i++) {
      const r = mergeRank.get(symbols[i])?.get(symbols[i + 1]);
      if (r !== undefined && r < bestRank) {
        bestRank = r;
        bestIdx = i;
      }
    }
    if (bestIdx === -1) break;

    const merged = symbols[bestIdx] + symbols[bestIdx + 1];
    const next = [];
    let i = 0;
    while (i < symbols.length) {
      if (i === bestIdx) {
        next.push(merged);
        i += 2;
      } else {
        next.push(symbols[i]);
        i += 1;
      }
    }
    symbols = next;
  }
  return symbols;
}

class BpeEncoder {
  constructor(vocabData) {
    this.tokens = vocabData.tokens;
    this.vocabSize = vocabData.vocab_size;
    this.tokenToId = buildTokenToId(vocabData.tokens);
    this.mergeRank = buildMergeRank(vocabData.merges);
  }

  // Encodes one pre-tokenized word (no whitespace inside) into subword
  // strings + ids. Any resulting symbol not in vocab (only possible for
  // characters never seen during training) is reported as uncovered rather
  // than silently dropped.
  encodeWord(word) {
    const symbols = bpeMergeWord(word.normalize("NFC"), this.mergeRank);
    const ids = [];
    const uncovered = [];
    const perSymbolId = [];
    for (const sym of symbols) {
      const id = this.tokenToId.get(sym);
      if (id === undefined) {
        uncovered.push(sym);
        perSymbolId.push(null);
      } else {
        ids.push(id);
        perSymbolId.push(id);
      }
    }
    return { symbols, ids, uncovered, perSymbolId };
  }

  // Encodes a full text span, returning per-pretoken results plus rollup
  // stats. This is what the Coverage Inspector renders.
  encode(text) {
    const normalized = text.normalize("NFC");
    const chunks = preTokenize(normalized);
    const pretokens = [];
    let totalTokens = 0;
    const uncoveredChars = new Set();

    for (const chunk of chunks) {
      const { symbols, ids, uncovered, perSymbolId } = this.encodeWord(chunk.text);
      pretokens.push({ ...chunk, symbols, ids, perSymbolId });
      totalTokens += ids.length;
      for (const u of uncovered) uncoveredChars.add(u);
    }

    return {
      text: normalized,
      pretokens,
      wordCount: chunks.length,
      tokenCount: totalTokens,
      fertility: chunks.length ? totalTokens / chunks.length : 0,
      uncoveredChars: Array.from(uncoveredChars),
    };
  }
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = { BpeEncoder, preTokenize, bpeMergeWord };
}
