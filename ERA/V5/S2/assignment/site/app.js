(function () {
  "use strict";

  const state = {
    vocab: null,
    stats: null,
    samples: null,
    runConfig: null,
    encoder: null,
    vocabFiltered: [],
    vocabPage: 0,
    vocabPageSize: 50,
  };

  const LANGS = ["en", "hi", "te", "mr"];
  const LANG_LABEL = { en: "English", hi: "Hindi", te: "Telugu", mr: "Marathi" };
  const VARIANT_LABEL = { byte: "Byte-level BPE", char: "Char-level BPE", word: "Word-level BPE (winner)", sentencepiece: "SentencePiece BPE" };

  function escapeHtml(s) {
    return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }

  function fmt(n, digits = 4) {
    return typeof n === "number" ? n.toFixed(digits) : n;
  }

  // ---------------------------------------------------------------- tabs --
  function activateTab(name) {
    const btn = document.querySelector(`.tab-btn[data-tab="${name}"]`);
    if (!btn) return;
    document.querySelectorAll(".tab-btn").forEach((b) => { b.classList.remove("active"); b.setAttribute("aria-selected", "false"); });
    document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    btn.setAttribute("aria-selected", "true");
    document.getElementById("tab-" + name).classList.add("active");
  }

  function initTabs() {
    document.querySelectorAll(".tab-btn").forEach((btn) => {
      btn.addEventListener("click", () => activateTab(btn.dataset.tab));
    });
    // Deep-linkable tabs: /site/#vocab jumps straight to a tab.
    const hashTab = location.hash.replace("#", "");
    if (hashTab) activateTab(hashTab);
  }

  // ------------------------------------------------------------ overview --
  function renderOverview() {
    const word = state.stats.word;
    const sorted = word.sorted_by_X;
    const xVals = sorted.map((l) => word.per_language[l].X);

    document.getElementById("score-hero").innerHTML = `
      Sorted X values (${sorted.map((l) => LANG_LABEL[l]).join(" &le; ")}):
      <strong>${xVals.map((v) => fmt(v)).join(" &le; ")}</strong><br/>
      spread = ${fmt(word.X_max)} &minus; ${fmt(word.X_min)} = ${fmt(word.spread)}<br/>
      score = 1000 / ${fmt(word.spread)} = <span class="score-num">${fmt(word.score, 2)}</span>
    `;

    const winnerBody = document.querySelector("#winner-table tbody");
    winnerBody.innerHTML = "";
    for (const lang of LANGS) {
      const pl = word.per_language[lang];
      const tr = document.createElement("tr");
      if (lang === "en") tr.classList.add("highlight-row");
      tr.innerHTML = `<td>${LANG_LABEL[lang]}${lang === "en" ? " (anchor, &le;1.2 target)" : ""}</td>
        <td>${pl.unique_words}</td><td>${pl.distinct_tokens_used}</td><td>${fmt(pl.X)}</td>`;
      winnerBody.appendChild(tr);
    }

    const compareBody = document.querySelector("#compare-table tbody");
    compareBody.innerHTML = "";
    for (const variant of ["byte", "char", "word", "sentencepiece"]) {
      const s = state.stats[variant];
      const tr = document.createElement("tr");
      if (variant === "word") tr.classList.add("highlight-row");
      tr.innerHTML = `<td>${VARIANT_LABEL[variant]}</td><td>${s.vocab_size}</td>
        <td>${fmt(s.per_language.en.X)}</td><td>${fmt(s.per_language.hi.X)}</td>
        <td>${fmt(s.per_language.te.X)}</td><td>${fmt(s.per_language.mr.X)}</td>
        <td>${fmt(s.spread)}</td><td>${fmt(s.score, 2)}</td>`;
      compareBody.appendChild(tr);
    }
  }

  // -------------------------------------------------------- vocab explorer --
  function applyVocabFilter() {
    const query = document.getElementById("vocab-search").value.trim().toLowerCase();
    const scriptFilter = document.getElementById("vocab-script-filter").value;
    const mergedOnly = document.getElementById("vocab-merged-only").checked;
    const checkedLangs = new Set(
      Array.from(document.querySelectorAll(".lang-filter-group input:checked")).map((cb) => cb.value)
    );

    state.vocabFiltered = state.vocab.tokens.filter((t) => {
      if (query && !t.token.toLowerCase().includes(query)) return false;
      if (scriptFilter !== "all" && t.script !== scriptFilter) return false;
      if (mergedOnly && !t.isMerged) return false;
      if (t.langs.length > 0 && !t.langs.some((l) => checkedLangs.has(l))) return false;
      return true;
    });
    state.vocabPage = 0;
    renderVocabPage();
  }

  function renderVocabPage() {
    const { vocabFiltered, vocabPage, vocabPageSize } = state;
    const start = vocabPage * vocabPageSize;
    const pageItems = vocabFiltered.slice(start, start + vocabPageSize);

    document.getElementById("vocab-count").textContent =
      `${vocabFiltered.length.toLocaleString()} token(s) match ("used by" reflects the X-metric's unique-word encoding, so punctuation tokens naturally show no language — they're never part of a "word").`;

    const body = document.querySelector("#vocab-table tbody");
    body.innerHTML = "";
    for (const t of pageItems) {
      const tr = document.createElement("tr");
      tr.innerHTML = `<td>${t.id}</td><td><code>${escapeHtml(t.token)}</code></td>
        <td><span class="script-badge">${t.script}</span></td>
        <td>${t.isMerged ? "merged" : "base"}</td>
        <td>${t.langs.length ? t.langs.join(", ") : "—"}</td>`;
      body.appendChild(tr);
    }

    const totalPages = Math.max(1, Math.ceil(vocabFiltered.length / vocabPageSize));
    document.getElementById("vocab-page-info").textContent = `Page ${vocabPage + 1} / ${totalPages}`;
    document.getElementById("vocab-prev").disabled = vocabPage === 0;
    document.getElementById("vocab-next").disabled = vocabPage >= totalPages - 1;
  }

  function initVocabExplorer() {
    document.getElementById("vocab-search").addEventListener("input", applyVocabFilter);
    document.getElementById("vocab-script-filter").addEventListener("change", applyVocabFilter);
    document.getElementById("vocab-merged-only").addEventListener("change", applyVocabFilter);
    document.querySelectorAll(".lang-filter-group input").forEach((cb) => cb.addEventListener("change", applyVocabFilter));
    document.getElementById("vocab-prev").addEventListener("click", () => {
      if (state.vocabPage > 0) { state.vocabPage--; renderVocabPage(); }
    });
    document.getElementById("vocab-next").addEventListener("click", () => {
      const totalPages = Math.ceil(state.vocabFiltered.length / state.vocabPageSize);
      if (state.vocabPage < totalPages - 1) { state.vocabPage++; renderVocabPage(); }
    });
    applyVocabFilter();
  }

  // ------------------------------------------------------ coverage inspector --
  function runTokenize() {
    const text = document.getElementById("coverage-input").value;
    const warningEl = document.getElementById("coverage-warning");
    const statsEl = document.getElementById("coverage-stats");
    const highlightEl = document.getElementById("coverage-highlight");
    const breakdownBody = document.querySelector("#coverage-breakdown tbody");

    if (!text.trim()) {
      warningEl.textContent = "";
      statsEl.innerHTML = "";
      highlightEl.innerHTML = "<span class=\"muted\">Nothing to tokenize yet.</span>";
      breakdownBody.innerHTML = "";
      return;
    }

    const result = state.encoder.encode(text);

    statsEl.innerHTML = `
      <div class="stat-card"><div class="label">Words</div><div class="value">${result.wordCount}</div></div>
      <div class="stat-card"><div class="label">Tokens</div><div class="value">${result.tokenCount}</div></div>
      <div class="stat-card"><div class="label">Fertility (tok/word)</div><div class="value">${fmt(result.fertility, 3)}</div></div>
      <div class="stat-card"><div class="label">Uncovered chars</div><div class="value">${result.uncoveredChars.length}</div></div>
    `;

    warningEl.textContent = result.uncoveredChars.length
      ? `Not covered by this vocab (dropped, not encoded): ${result.uncoveredChars.map((c) => JSON.stringify(c)).join(", ")}`
      : "";

    const PALETTE_SIZE = 8;
    let html = "";
    let cursor = 0;
    let tokenCounter = 0;
    result.pretokens.forEach((chunk) => {
      if (chunk.start > cursor) {
        html += `<span class="gap">${escapeHtml(result.text.slice(cursor, chunk.start))}</span>`;
      }
      const pieces = chunk.symbols
        .map((sym, si) => {
          const id = chunk.perSymbolId[si];
          const covered = id !== null;
          const cls = covered ? `tok-piece tok-${tokenCounter % PALETTE_SIZE}` : "tok-piece tok-uncovered";
          const title = covered ? `token id ${id}` : "not covered by vocab";
          tokenCounter++;
          return `<span class="${cls}" title="${title}">${escapeHtml(sym)}</span>`;
        })
        .join("");
      html += pieces;
      cursor = chunk.end;
    });
    if (cursor < result.text.length) {
      html += `<span class="gap">${escapeHtml(result.text.slice(cursor))}</span>`;
    }
    highlightEl.innerHTML = html || "<span class=\"muted\">(empty)</span>";

    breakdownBody.innerHTML = "";
    for (const chunk of result.pretokens) {
      const tr = document.createElement("tr");
      tr.innerHTML = `<td>${escapeHtml(chunk.text)}</td><td>${chunk.symbols.map(escapeHtml).join(" + ")}</td><td>${chunk.ids.length}</td>`;
      breakdownBody.appendChild(tr);
    }
  }

  function initCoverageInspector() {
    const quickPicksEl = document.getElementById("quick-picks");
    for (const lang of LANGS) {
      const btn = document.createElement("button");
      btn.textContent = `${LANG_LABEL[lang]} sample`;
      btn.addEventListener("click", () => {
        document.getElementById("coverage-input").value = state.samples[lang].sample;
        runTokenize();
      });
      quickPicksEl.appendChild(btn);
    }
    document.getElementById("tokenize-btn").addEventListener("click", runTokenize);
    document.getElementById("coverage-input").value = state.samples.en.sample;
    runTokenize();
  }

  // ------------------------------------------------------------ methodology --
  function renderMethodology() {
    document.getElementById("meta-titles").textContent = LANGS
      .map((l) => `${LANG_LABEL[l]}: "${state.samples[l].source_title}"`)
      .join(", ");

    const body = document.querySelector("#run-config-table tbody");
    const rc = state.runConfig;
    const rows = [
      ["Winning variant", VARIANT_LABEL[rc.variant]],
      ["Vocab size", rc.vocab_size],
      ["min_frequency", rc.min_frequency],
      ["Repeat counts (en/hi/te/mr)", LANGS.map((l) => `${l}=${rc.repeat_counts[l]}`).join(", ")],
    ];
    body.innerHTML = rows.map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join("");
  }

  // --------------------------------------------------------------- analysis --
  function renderAnalysis() {
    const byte = state.stats.byte;
    const char = state.stats.char;
    const word = state.stats.word;
    const sp = state.stats.sentencepiece;
    const rc = state.runConfig;

    document.getElementById("an-wordcounts").textContent = LANGS
      .map((l) => `${LANG_LABEL[l]}=${state.samples[l].word_count.toLocaleString()}`)
      .join(", ");

    document.getElementById("an-byte-vocab").textContent = byte.vocab_size.toLocaleString();
    document.getElementById("an-byte-spread").textContent = fmt(byte.spread);
    document.getElementById("an-byte-score").textContent = fmt(byte.score, 2);

    document.getElementById("an-char-spread").textContent = fmt(char.spread);
    document.getElementById("an-char-score").textContent = fmt(char.score, 2);

    const x1 = word.per_language.en.X;
    const worstLang = ["hi", "te", "mr"].reduce((worst, l) =>
      Math.abs(word.per_language[l].X - x1) > Math.abs(word.per_language[worst].X - x1) ? l : worst
    , "hi");
    const worstGap = Math.abs(word.per_language[worstLang].X - x1);

    document.getElementById("an-final-config").innerHTML =
      `word-level BPE, min_frequency=${rc.min_frequency}, repeat_counts {${LANGS.map((l) => `${l}=${rc.repeat_counts[l]}`).join(", ")}}`;
    document.getElementById("an-final-x1").textContent = fmt(x1);
    document.getElementById("an-final-gap").textContent = `${fmt(worstGap)} (${LANG_LABEL[worstLang]})`;
    document.getElementById("an-final-spread").textContent = fmt(word.spread);
    document.getElementById("an-final-score").textContent = fmt(word.score, 2);

    document.getElementById("an-sp-score").textContent = fmt(sp.score, 2);
  }

  // ------------------------------------------------------------------ init --
  async function loadData() {
    const [vocab, stats, samples, runConfig] = await Promise.all([
      fetch("data/vocab_word.json").then((r) => r.json()),
      fetch("data/stats.json").then((r) => r.json()),
      fetch("data/samples.json").then((r) => r.json()),
      fetch("data/run_config.json").then((r) => r.json()),
    ]);
    state.vocab = vocab;
    state.stats = stats;
    state.samples = samples;
    state.runConfig = runConfig;
    state.encoder = new BpeEncoder(vocab);
  }

  async function main() {
    initTabs();
    await loadData();
    renderOverview();
    renderAnalysis();
    initVocabExplorer();
    initCoverageInspector();
    renderMethodology();
  }

  main().catch((err) => {
    console.error(err);
    document.querySelector("main").innerHTML = `<p class="warning">Failed to load: ${err.message}</p>`;
  });
})();
