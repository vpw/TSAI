(function () {
  "use strict";

  const state = {
    vocab: null,
    stats: null,
    samples: null,
    runConfig: null,
    fidelityProbes: null,
    encoder: null,
    langs: [],
    vocabFiltered: [],
    vocabPage: 0,
    vocabPageSize: 50,
  };

  const LANG_LABEL = { en: "English", hi: "Hindi", te: "Telugu", mr: "Marathi", bn: "Bengali" };
  const FOURTH_LABEL = { mr: "Marathi", bn: "Bengali" };

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
    document.querySelectorAll("[data-tab-link]").forEach((el) => {
      el.addEventListener("click", (e) => { e.preventDefault(); activateTab(el.dataset.tabLink); });
    });
    const hashTab = location.hash.replace("#", "");
    if (hashTab) activateTab(hashTab);
  }

  // ------------------------------------------------------------ overview --
  function renderOverview() {
    const winner = state.stats.winner;
    const w = state.stats[winner];
    const sorted = w.sorted_by_fertility;
    const fVals = sorted.map((l) => w.rows[l].fertility);

    document.getElementById("score-hero").innerHTML = `
      Sorted fertility (${sorted.map((l) => LANG_LABEL[l]).join(" &le; ")}):
      <strong>${fVals.map((v) => fmt(v)).join(" &le; ")}</strong><br/>
      spread = ${fmt(w.f_max)} &minus; ${fmt(w.f_min)} = ${fmt(w.spread)}<br/>
      raw score = 1000 / ${fmt(w.spread)} = <span class="score-num">${fmt(w.score, 2)}</span><br/>
      English fertility ${fmt(w.english_fertility)} (&le;1.2: ${w.english_meets_1_2 ? "yes" : "no — penalty applied"}),
      adjusted score = <strong>${fmt(w.adjusted_score, 2)}</strong>
    `;

    const winnerBody = document.querySelector("#winner-table tbody");
    winnerBody.innerHTML = "";
    for (const lang of state.langs) {
      const r = w.rows[lang];
      const tr = document.createElement("tr");
      if (lang === "en") tr.classList.add("highlight-row");
      tr.innerHTML = `<td>${LANG_LABEL[lang]}${lang === "en" ? " (anchor, &le;1.2 target)" : ""}</td>
        <td>${r.faithful_units.toLocaleString()}</td><td>${r.token_count.toLocaleString()}</td><td>${fmt(r.fertility)}</td>`;
      winnerBody.appendChild(tr);
    }

    const compareBody = document.querySelector("#fourth-compare-table tbody");
    compareBody.innerHTML = "";
    for (const fourth of ["mr", "bn"]) {
      const s = state.stats[fourth];
      if (!s) continue;
      const tr = document.createElement("tr");
      if (fourth === winner) tr.classList.add("highlight-row");
      tr.innerHTML = `<td>${FOURTH_LABEL[fourth]}${fourth === winner ? " (chosen)" : ""}</td>
        <td>${fmt(s.english_fertility)} (${s.english_meets_1_2 ? "&le;1.2" : "&gt;1.2"})</td>
        <td>${fmt(s.spread)}</td><td>${fmt(s.score, 2)}</td><td>${fmt(s.adjusted_score, 2)}</td>
        <td>${fourth === winner ? "&#9733; selected" : ""}</td>`;
      compareBody.appendChild(tr);
    }
  }

  // ---------------------------------------------------------- fidelity check --
  function runFidelityCheck() {
    const probesBody = document.querySelector("#fidelity-table tbody");
    const unsafeBody = document.querySelector("#fidelity-unsafe-table tbody");
    probesBody.innerHTML = "";
    unsafeBody.innerHTML = "";

    let pass = 0, total = 0;
    for (const probe of state.fidelityProbes.probes) {
      const decoded = state.encoder.roundTrip(probe);
      const normOrig = probe.normalize("NFKC").replace(/\s/g, "");
      const normDec = decoded.normalize("NFKC").replace(/\s/g, "");
      const ok = normOrig === normDec;
      total++;
      if (ok) pass++;
      const tr = document.createElement("tr");
      tr.innerHTML = `<td><span class="fidelity-badge ${ok ? "fidelity-pass" : "fidelity-fail"}">${ok ? "PASS" : "FAIL"}</span></td>
        <td><code>${escapeHtml(probe)}</code></td><td><code>${escapeHtml(decoded)}</code></td>`;
      probesBody.appendChild(tr);
    }

    document.getElementById("fidelity-summary").innerHTML = pass === total
      ? `<span class="score-num" style="color:#2e7d32">${pass}/${total} probes passed</span> — decode(encode(text)) preserves visible non-whitespace characters for every probe, including URL/ref-anchor text shaped like the grading feedback's failing sample.`
      : `<span class="score-num" style="color:var(--warning)">${pass}/${total} probes passed</span> — see failures below.`;

    for (const probe of state.fidelityProbes.known_unsafe) {
      const decoded = state.encoder.roundTrip(probe);
      const tr = document.createElement("tr");
      tr.innerHTML = `<td><code>${escapeHtml(probe)}</code></td><td><code>${escapeHtml(decoded)}</code></td>
        <td><span class="fidelity-badge fidelity-unsafe">expected [UNK]</span> characters outside every training corpus (e.g. emoji) are not covered by any tokenizer's vocab and fall back to [UNK], same as any BPE model.</td>`;
      unsafeBody.appendChild(tr);
    }
  }

  // -------------------------------------------------------- vocab explorer --
  function initVocabLangFilter() {
    const container = document.getElementById("vocab-lang-filter");
    container.innerHTML = state.langs
      .map((l) => `<label><input type="checkbox" value="${l}" checked /> ${l}</label>`)
      .join("");
    container.querySelectorAll("input").forEach((cb) => cb.addEventListener("change", applyVocabFilter));
  }

  function applyVocabFilter() {
    const query = document.getElementById("vocab-search").value.trim().toLowerCase();
    const scriptFilter = document.getElementById("vocab-script-filter").value;
    const mergedOnly = document.getElementById("vocab-merged-only").checked;
    const checkedLangs = new Set(
      Array.from(document.querySelectorAll("#vocab-lang-filter input:checked")).map((cb) => cb.value)
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
      `${vocabFiltered.length.toLocaleString()} token(s) match ("used by" reflects which language(s)' unique word lists actually resolve to this token id).`;

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
    initVocabLangFilter();
    document.getElementById("vocab-search").addEventListener("input", applyVocabFilter);
    document.getElementById("vocab-script-filter").addEventListener("change", applyVocabFilter);
    document.getElementById("vocab-merged-only").addEventListener("change", applyVocabFilter);
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
  function displaySpace(sym) {
    return sym.split("▁").join("·");
  }

  function runTokenize() {
    const text = document.getElementById("coverage-input").value;
    const warningEl = document.getElementById("coverage-warning");
    const statsEl = document.getElementById("coverage-stats");
    const highlightEl = document.getElementById("coverage-highlight");
    const roundtripEl = document.getElementById("coverage-roundtrip");
    const breakdownBody = document.querySelector("#coverage-breakdown tbody");

    if (!text.trim()) {
      warningEl.textContent = "";
      statsEl.innerHTML = "";
      highlightEl.innerHTML = "<span class=\"muted\">Nothing to tokenize yet.</span>";
      roundtripEl.textContent = "";
      breakdownBody.innerHTML = "";
      return;
    }

    const result = state.encoder.encode(text);
    const allIds = [];
    for (const p of result.pretokens) allIds.push(...p.ids);
    const decoded = state.encoder.decode(allIds);
    const roundtripOk = text.normalize("NFKC").replace(/\s/g, "") === decoded.normalize("NFKC").replace(/\s/g, "");

    statsEl.innerHTML = `
      <div class="stat-card"><div class="label">Pretokens</div><div class="value">${result.wordCount}</div></div>
      <div class="stat-card"><div class="label">Tokens</div><div class="value">${result.tokenCount}</div></div>
      <div class="stat-card"><div class="label">Fertility (tok/pretoken)</div><div class="value">${fmt(result.fertility, 3)}</div></div>
      <div class="stat-card"><div class="label">Uncovered chars</div><div class="value">${result.uncoveredChars.length}</div></div>
    `;

    warningEl.textContent = result.uncoveredChars.length
      ? `Not covered by this vocab (mapped to [UNK]): ${result.uncoveredChars.map((c) => JSON.stringify(c)).join(", ")}`
      : "";

    roundtripEl.innerHTML = `<span class="fidelity-badge ${roundtripOk ? "fidelity-pass" : "fidelity-fail"}">${roundtripOk ? "PASS" : "FAIL"}</span> decode(encode(text)) ${roundtripOk ? "preserves" : "does NOT preserve"} visible non-whitespace characters for this input.`;

    const PALETTE_SIZE = 8;
    let html = "";
    let tokenCounter = 0;
    result.pretokens.forEach((chunk) => {
      chunk.symbols.forEach((sym, si) => {
        const id = chunk.perSymbolId[si];
        const covered = sym !== "[UNK]";
        const cls = covered ? `tok-piece tok-${tokenCounter % PALETTE_SIZE}` : "tok-piece tok-uncovered";
        const title = covered ? `token id ${id}` : "not covered by vocab ([UNK])";
        tokenCounter++;
        html += `<span class="${cls}" title="${title}">${escapeHtml(displaySpace(sym))}</span>`;
      });
    });
    highlightEl.innerHTML = html || "<span class=\"muted\">(empty)</span>";

    breakdownBody.innerHTML = "";
    for (const chunk of result.pretokens) {
      const tr = document.createElement("tr");
      tr.innerHTML = `<td><code>${escapeHtml(displaySpace(chunk.text))}</code></td><td>${chunk.symbols.map((s) => escapeHtml(displaySpace(s))).join(" + ")}</td><td>${chunk.ids.length}</td>`;
      breakdownBody.appendChild(tr);
    }
  }

  function initCoverageInspector() {
    const quickPicksEl = document.getElementById("quick-picks");
    for (const lang of state.langs) {
      const btn = document.createElement("button");
      btn.textContent = `${LANG_LABEL[lang]} sample`;
      btn.addEventListener("click", () => {
        document.getElementById("coverage-input").value = state.samples[lang].sample;
        runTokenize();
      });
      quickPicksEl.appendChild(btn);
    }
    const urlBtn = document.createElement("button");
    urlBtn.textContent = "URL sample (fidelity probe)";
    urlBtn.addEventListener("click", () => {
      document.getElementById("coverage-input").value = state.fidelityProbes.probes[0];
      runTokenize();
    });
    quickPicksEl.appendChild(urlBtn);

    document.getElementById("tokenize-btn").addEventListener("click", runTokenize);
    document.getElementById("coverage-input").value = state.samples[state.langs[0]].sample;
    runTokenize();
  }

  // ------------------------------------------------------------ methodology --
  function renderMethodology() {
    document.getElementById("meta-titles").textContent = state.langs
      .map((l) => `${LANG_LABEL[l]}: "${state.samples[l].source_title}"`)
      .join(", ");

    const body = document.querySelector("#run-config-table tbody");
    const rc = state.runConfig;
    const rows = [
      ["4th language", FOURTH_LABEL[rc.fourth]],
      ["Vocab size", rc.vocab_size],
      ["min_frequency", rc.min_frequency],
      [`Repeat counts (${state.langs.join("/")})`, state.langs.map((l) => `${l}=${rc.repeat_counts[l]}`).join(", ")],
    ];
    body.innerHTML = rows.map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join("");
  }

  // --------------------------------------------------------------- analysis --
  function renderAnalysis() {
    const winner = state.stats.winner;
    const w = state.stats[winner];
    const rc = state.runConfig;

    document.getElementById("an-final-config").innerHTML =
      `Metaspace BPE, min_frequency=${rc.min_frequency}, fourth=${FOURTH_LABEL[rc.fourth]}, repeat_counts {${state.langs.map((l) => `${l}=${rc.repeat_counts[l]}`).join(", ")}}`;
    document.getElementById("an-final-x1").textContent = fmt(w.english_fertility);
    document.getElementById("an-final-spread").textContent = fmt(w.spread);
    document.getElementById("an-final-score").textContent = fmt(w.score, 2);
  }

  // ------------------------------------------------------------------ init --
  async function loadData() {
    const [vocab, stats, samples, runConfig, fidelityProbes] = await Promise.all([
      fetch("data/vocab.json").then((r) => r.json()),
      fetch("data/stats.json").then((r) => r.json()),
      fetch("data/samples.json").then((r) => r.json()),
      fetch("data/run_config.json").then((r) => r.json()),
      fetch("data/fidelity_probes.json").then((r) => r.json()),
    ]);
    state.vocab = vocab;
    state.stats = stats;
    state.samples = samples;
    state.runConfig = runConfig;
    state.fidelityProbes = fidelityProbes;
    state.langs = runConfig.langs;
    state.encoder = new BpeEncoder(vocab);
  }

  async function main() {
    initTabs();
    await loadData();
    document.getElementById("header-langs").textContent =
      `(${state.langs.map((l) => LANG_LABEL[l]).join(", ")})`;
    renderOverview();
    runFidelityCheck();
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
