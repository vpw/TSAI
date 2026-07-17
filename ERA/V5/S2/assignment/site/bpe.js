// Client-side re-implementation of the HF `tokenizers` Metaspace BPE
// encode/decode path (NFKC normalizer + Metaspace pre-tokenizer + plain
// character-pair BPE merges + matching Metaspace decoder).
//
// Must stay in lockstep with scripts/train_tokenizer.py. Verified against
// the Python tokenizer's own encode AND decode output (see
// scripts/verify_bpe_js.py) before being trusted here -- this is the piece
// that failed silently in phase 1 (Whitespace pre-tokenizer + no decoder),
// so decode() correctness here is the whole point.

const METASPACE_REPLACEMENT = "▁";
const UNK_TOKEN = "[UNK]";

// Reproduces HF's Metaspace(prepend_scheme="always") pre-tokenization
// exactly: replace every literal space with the replacement char, prepend
// one more replacement char at the very start unless one is already there,
// then split into pretokens at each occurrence of the replacement char
// (each pretoken keeps its leading replacement char).
function metaspaceTransform(text) {
  const chars = Array.from(text); // codepoint-safe
  const origIndex = []; // origIndex[i] = index into `chars` this transformed char came from, or -1 if synthetic
  const out = [];
  for (let i = 0; i < chars.length; i++) {
    out.push(chars[i] === " " ? METASPACE_REPLACEMENT : chars[i]);
    origIndex.push(i);
  }
  if (out.length === 0 || out[0] !== METASPACE_REPLACEMENT) {
    out.unshift(METASPACE_REPLACEMENT);
    origIndex.unshift(-1);
  }
  return { out, origIndex, chars };
}

function preTokenize(text) {
  if (text.length === 0) return []; // HF: empty input has no pretokens at all
  const { out, origIndex, chars } = metaspaceTransform(text);
  const pretokens = [];
  let spanStart = -1;
  for (let i = 0; i <= out.length; i++) {
    const isBoundary = i === out.length || out[i] === METASPACE_REPLACEMENT;
    if (isBoundary) {
      if (spanStart !== -1) {
        const symbolsText = out.slice(spanStart, i).join("");
        // Original-text span this pretoken covers (for display only):
        // first real (non-synthetic) original index in the span through
        // the last real index + 1.
        const realIdxs = origIndex.slice(spanStart, i).filter((v) => v !== -1);
        const start = realIdxs.length ? realIdxs[0] : (chars.length ? Math.min(chars.length, origIndex[i] ?? chars.length) : 0);
        const end = realIdxs.length ? realIdxs[realIdxs.length - 1] + 1 : start;
        pretokens.push({ text: symbolsText, start, end });
      }
      spanStart = i;
    }
  }
  return pretokens;
}

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
    this.unkId = this.tokenToId.get(UNK_TOKEN);
  }

  // Encodes one Metaspace pretoken (already ▁-prefixed) into subword
  // strings + ids. A symbol not in vocab becomes [UNK] (matching the
  // Python model's unk_token="[UNK]"), not silently dropped -- but is
  // still reported separately as "uncovered" for the Coverage Inspector.
  encodePretoken(ptext) {
    const rawSymbols = bpeMergeWord(ptext.normalize("NFC"), this.mergeRank);
    const symbols = [];
    const ids = [];
    const uncovered = [];
    const perSymbolId = [];
    for (const sym of rawSymbols) {
      const id = this.tokenToId.get(sym);
      if (id === undefined) {
        uncovered.push(sym);
        symbols.push(UNK_TOKEN); // matches HF: displayed token text is literally "[UNK]"
        perSymbolId.push(this.unkId ?? null);
        ids.push(this.unkId ?? null);
      } else {
        symbols.push(sym);
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
      const { symbols, ids, uncovered, perSymbolId } = this.encodePretoken(chunk.text);
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

  // Mirrors HF's Tokenizer.decode(ids) default behavior: skip special
  // tokens ([UNK]), concatenate the rest, replace the Metaspace
  // replacement char with a literal space, then strip exactly one leading
  // space (the artificial prepend_scheme="always" marker).
  decode(ids) {
    const idToToken = this.tokens; // index === id
    let s = "";
    for (const id of ids) {
      if (id === this.unkId) continue;
      const tok = idToToken[id]?.token;
      if (tok !== undefined) s += tok;
    }
    s = s.split(METASPACE_REPLACEMENT).join(" ");
    if (s.startsWith(" ")) s = s.slice(1);
    return s;
  }

  // Convenience: encode then decode, for round-trip fidelity checks.
  roundTrip(text) {
    const { pretokens } = this.encode(text);
    const ids = [];
    for (const p of pretokens) ids.push(...p.ids);
    return this.decode(ids);
  }
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = { BpeEncoder, preTokenize, bpeMergeWord, metaspaceTransform };
}
