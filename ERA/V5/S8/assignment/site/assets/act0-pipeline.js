/* ===========================================================
   Act 0 — Attention, built once.
   Six tokens through Score → Scale → Mask → Softmax → Weighted sum.

   Every number rendered by this file is computed here from the Q/K/V
   below — nothing is hard-coded for display. The vectors are authored
   (not random) so the resulting attention pattern is readable: keys
   advertise what a token IS, queries advertise what it is LOOKING FOR.
   =========================================================== */
(function () {
  'use strict';

  var TOK  = ['the', 'cat', 'sat', 'on', 'the', 'mat'];
  var N    = TOK.length;
  var DK   = 4;
  var SQDK = Math.sqrt(DK);           // = 2, chosen so the scaling is mental arithmetic

  var QKDIM = ['det', 'noun', 'verb', 'prep'];
  var VDIM  = ['animate', 'action', 'place', 'det'];

  // K — "what I am". A one-hot-ish advertisement of the token's grammatical role.
  var K = [
    [2, 0, 0, 0],   // the   → determiner
    [0, 2, 0, 0],   // cat   → noun
    [0, 0, 2, 0],   // sat   → verb
    [0, 0, 0, 2],   // on    → preposition
    [2, 0, 0, 0],   // the   → determiner
    [0, 2, 0, 0]    // mat   → noun
  ];
  // Q — "what I'm looking for".
  var Q = [
    [0, 1,   0, 0], // the → wants a noun to attach to
    [1, 0,   0, 0], // cat → wants its determiner
    [0, 1.5, 0, 0], // sat → wants its subject noun (asked louder)
    [0, 0,   1, 0], // on  → wants the verb it modifies
    [0, 1,   0, 0], // the → wants a noun
    [1, 0,   0, 1]  // mat → wants its determiner and its preposition
  ];
  // V — "what I pass on if you attend to me".
  var V = [
    [0, 0, 0, 1],   // the  → determiner-ness
    [1, 0, 0, 0],   // cat  → animate thing
    [0, 1, 0, 0],   // sat  → action
    [0, 0, 1, 0],   // on   → place relation
    [0, 0, 0, 1],   // the  → determiner-ness
    [0, 0, 1, 0]    // mat  → place
  ];

  var VCOL = ['var(--cmp)', 'var(--pos)', 'var(--mem)', 'var(--sys)'];

  /* ---------- math ---------- */
  function dot(a, b) { var s = 0; for (var i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }

  function scores() {
    var m = [];
    for (var i = 0; i < N; i++) { m[i] = []; for (var j = 0; j < N; j++) m[i][j] = dot(Q[i], K[j]); }
    return m;
  }
  function scale(m) { return m.map(function (r) { return r.map(function (x) { return x / SQDK; }); }); }
  function mask(m, causal) {
    return m.map(function (r, i) {
      return r.map(function (x, j) { return (causal && j > i) ? -Infinity : x; });
    });
  }
  function softmax(m) {
    return m.map(function (r) {
      var mx = Math.max.apply(null, r.filter(isFinite));
      var ex = r.map(function (x) { return isFinite(x) ? Math.exp(x - mx) : 0; });
      var s = ex.reduce(function (a, b) { return a + b; }, 0);
      return ex.map(function (e) { return e / s; });
    });
  }
  function weighted(W) {
    var out = [];
    for (var i = 0; i < N; i++) {
      out[i] = new Array(DK).fill(0);
      for (var j = 0; j < N; j++) for (var d = 0; d < DK; d++) out[i][d] += W[i][j] * V[j][d];
    }
    return out;
  }

  /* ---------- state ---------- */
  var state = { step: 0, causal: true, sel: null };

  var STEPS = [
    { id: 'inputs', label: 'Inputs', term: null,
      title: 'Three vectors per token, and no notion of order at all',
      body: 'Each token projects into a <strong>query</strong> (what am I looking for), a <strong>key</strong> ' +
            '(what do I offer), and a <strong>value</strong> (what I pass on if you read me). These are the ' +
            '<code>[B, T, D]</code> vectors Session 7 handed over — this is the first time they get to meet each other.' +
            '<p>Note what is <em>not</em> here: any information about position. Shuffle the six tokens and every ' +
            'number below is unchanged. That absence is what the whole position half of the timeline exists to fix.</p>' },

    { id: 'score', label: 'Score', term: 'qk',
      title: 'Every query meets every key — and there is the first bill',
      body: 'The score for query <code>i</code> against key <code>j</code> is just their dot product: high when ' +
            'what token <code>i</code> wants matches what token <code>j</code> offers.' +
            '<p>Count the cells. Six tokens produce <strong>36</strong> comparisons; a thousand produce a million. ' +
            'This grid <em>is</em> the quadratic-compute bill, and half the mechanisms on this page exist to avoid ' +
            'filling all of it in.</p>' },

    { id: 'scale', label: 'Scale', term: 'sq',
      title: 'Divide by √d_k, or softmax saturates',
      body: 'Dot products grow with dimension. Feed large values into softmax and it collapses toward one-hot — ' +
            'near-zero gradients, and the model stops learning. Dividing by <code>√d_k</code> holds the variance ' +
            'steady.' +
            '<p>Here <code>d_k = 4</code>, so this step is a division by <strong>2</strong>. Every number halves; ' +
            'the <em>pattern</em> is untouched. It is a numerical-stability fix, not a modelling decision.</p>' },

    { id: 'mask', term: 'm', label: 'Mask',
      title: 'Add −∞ above the diagonal, and get parallel training for free',
      body: function () {
        var s = 'A token predicting the next word must not read the answer. So every position <code>j &gt; i</code> ' +
                'gets <code>−∞</code> added to its score.' +
                '<p>The trick is what happens next: <code>softmax(−∞) = 0</code> exactly. The forbidden cells become ' +
                'hard zeros, which means all six positions can be trained <strong>in one parallel pass</strong> and ' +
                'still be causally honest. Without this you would train one token at a time.</p>';
        s += state.causal
          ? '<p><strong>Turn the toggle off</strong> and watch the top-right fill in — that is the leak.</p>'
          : '<p><strong>The toggle is off.</strong> Nothing was added, so every score above the diagonal survived ' +
            'intact — outlined in red. Those are the cells a token is not allowed to see.</p>';
        return s;
      } },

    { id: 'softmax', label: 'Softmax', term: 'sm',
      title: 'Scores become a distribution — every row now sums to 1',
      body: 'Exponentiate, then normalise per row. Two consequences worth naming.' +
            '<p>First, attention is a <strong>budget</strong>: each token has exactly 1.0 of it to spend, so ' +
            'attending more to one token necessarily means attending less to another. Second, softmax is sharp — ' +
            'a score gap of 1.5 becomes a weight ratio near 4.5×, which is what makes retrieval selective rather ' +
            'than a blur.</p>' +
            '<p>That sharpness is exactly what linear attention gives up in 2020 to escape the <code>T²</code> bill.</p>' },

    { id: 'out', label: 'Weighted sum', term: 'v',
      title: 'Mix the values by those weights — and the KV cache is born',
      body: function () {
        var W = softmax(mask(scale(scores()), state.causal));
        var toCat = (W[2][1] * 100).toFixed(0), toMat = (W[2][5] * 100).toFixed(0);
        var s = 'Each token\'s output is the weighted sum of every value vector it was allowed to read.';
        s += state.causal
          ? '<p>Look at <strong>sat</strong>: ' + toCat + '% of its attention went to <em>cat</em>, so its output ' +
            'is dominated by <em>animate</em>. The verb has absorbed its subject. That is attention doing its job.</p>'
          : '<p>With the mask off, <strong>sat</strong> splits its attention — ' + toCat + '% to <em>cat</em> and ' +
            'another ' + toMat + '% to <em>mat</em>, a word that has not been generated yet. Its output is now a ' +
            'blend of a token it can legitimately see and one it cannot.</p>';
        s += '<p>And here is the second bill. To generate token 7 you need every earlier <strong>K</strong> and ' +
             '<strong>V</strong> still in memory. They cannot be recomputed cheaply and cannot be shared between ' +
             'users — that stored pile is the KV cache, and it grows with every token.</p>';
        return s;
      } }
  ];

  /* ---------- element helpers ---------- */
  function $(sel) { return document.querySelector(sel); }
  function el(tag, attrs, html) {
    var n = document.createElement(tag);
    if (attrs) Object.keys(attrs).forEach(function (k) { n.setAttribute(k, attrs[k]); });
    if (html != null) n.innerHTML = html;
    return n;
  }
  function fmt(x, p) {
    if (x === -Infinity) return '−∞';
    var s = x.toFixed(p == null ? 2 : p);
    return s === '-0.00' ? '0.00' : s;
  }

  /* ---------- current matrix for a step ---------- */
  function matrixFor(step, causal) {
    var raw = scores();
    if (step <= 1) return { m: raw, kind: 'score' };
    var sc = scale(raw);
    if (step === 2) return { m: sc, kind: 'score' };
    var mk = mask(sc, causal);
    if (step === 3) return { m: mk, kind: 'score' };
    return { m: softmax(mk), kind: 'weight' };
  }

  /* ---------- render: formula ---------- */
  function renderFormula() {
    var s = state.step, causal = state.causal;
    function cls(term, order) {
      var st = STEPS[s].term;
      if (st === term) return 't on';
      return (order < s) ? 't done' : 't';
    }
    $('#f-qk').className  = cls('qk', 1);
    $('#f-sq').className  = cls('sq', 2);
    $('#f-m').className   = cls('m', 3) + (causal ? '' : ' off');
    $('#f-sm').className  = cls('sm', 4);
    $('#f-v').className   = cls('v', 5);
    $('#f-m').setAttribute('title', causal ? 'causal mask on' : 'causal mask OFF — no term added');
  }

  /* ---------- render: matrix ---------- */
  function renderMatrix() {
    var host = $('#m-host');
    host.innerHTML = '';
    if (state.step === 0) { host.appendChild(renderQKV()); return; }

    var res = matrixFor(state.step, state.causal);
    var m = res.m, isW = res.kind === 'weight';

    // heat normalisation
    var finite = [];
    m.forEach(function (r) { r.forEach(function (x) { if (isFinite(x)) finite.push(x); }); });
    var hi = isW ? Math.max.apply(null, finite) : Math.max.apply(null, finite.map(Math.abs));
    if (!hi) hi = 1;

    var wrap = el('div', { class: 'mwrap' });
    wrap.appendChild(el('div', { class: 'axis' }, 'key  j  →  (what each token offers)'));

    var t = el('table', { class: 'matrix' });
    var thead = el('thead'), hr = el('tr');
    hr.appendChild(el('th', { class: 'rowh' }, '<span style="font-size:.66rem">query i ↓</span>'));
    TOK.forEach(function (tk, j) {
      hr.appendChild(el('th', { class: 'colh', scope: 'col' }, tk + '<span class="qi">' + j + '</span>'));
    });
    thead.appendChild(hr); t.appendChild(thead);

    var tb = el('tbody');
    for (var i = 0; i < N; i++) {
      var tr = el('tr');
      tr.appendChild(el('th', { class: 'rowh', scope: 'row' }, TOK[i] + ' <span class="qi" style="display:inline">' + i + '</span>'));
      for (var j = 0; j < N; j++) {
        var x = m[i][j];
        // after softmax the −∞ cells hold exact zeros, but they are still the masked
        // region — keep the hatch so the eye can follow the same triangle across steps
        var masked = (x === -Infinity) || (isW && state.causal && j > i);
        var a = masked ? 0 : (isW ? x / hi : Math.abs(x) / hi);
        var c = el('div', {
          class: 'cell' + (masked ? ' masked' : '') + (a > .62 ? ' hot' : '') +
                 (!state.causal && j > i ? ' leak' : '') +
                 (state.sel && state.sel[0] === i && state.sel[1] === j ? ' sel' : ''),
          role: 'button', tabindex: '0',
          'data-i': i, 'data-j': j,
          'aria-label': 'query ' + TOK[i] + ' ' + i + ', key ' + TOK[j] + ' ' + j +
                        ', value ' + (masked ? 'masked' : fmt(x, isW ? 3 : 2))
        }, (x === -Infinity) ? '−∞' : fmt(x, isW ? 3 : 2));
        if (!masked && a > .02) c.style.background = 'rgba(var(--heat),' + (0.10 + a * 0.82).toFixed(3) + ')';
        var td = el('td'); td.appendChild(c); tr.appendChild(td);
      }
      tb.appendChild(tr);
    }
    t.appendChild(tb); wrap.appendChild(t);

    var leg = el('div', { class: 'mlegend' });
    leg.innerHTML =
      '<span><i style="background:rgba(var(--heat),.92)"></i>' + (isW ? 'high weight' : 'high score') + '</span>' +
      '<span><i style="background:rgba(var(--heat),.14)"></i>' + (isW ? 'low weight' : 'low score') + '</span>' +
      (state.step >= 3 && state.causal
        ? '<span><i style="background:var(--surface-2);border:1px dashed var(--muted)"></i>' +
          (isW ? 'exactly 0 — was −∞' : 'masked (−∞)') + '</span>'
        : '') +
      (!state.causal ? '<span style="color:var(--origin)"><i style="box-shadow:inset 0 0 0 2px var(--origin)"></i>reads the future</span>' : '') +
      '<span style="opacity:.75">click any cell for the arithmetic</span>';
    wrap.appendChild(leg);
    host.appendChild(wrap);
  }

  /* ---------- render: Q/K/V tables (step 0) ---------- */
  function vtable(name, M, dims, note) {
    var d = el('div', { class: 'vec' });
    d.appendChild(el('h4', null, name));
    var t = el('table', { class: 'vt' });
    var hr = el('tr');
    hr.appendChild(el('th', null, 'token'));
    dims.forEach(function (dm) { hr.appendChild(el('th', null, dm)); });
    t.appendChild(hr);
    for (var i = 0; i < N; i++) {
      var tr = el('tr');
      tr.appendChild(el('td', null, TOK[i] + ' <span style="opacity:.5">' + i + '</span>'));
      for (var k = 0; k < dims.length; k++) {
        tr.appendChild(el('td', { class: M[i][k] === 0 ? 'z' : '' }, String(M[i][k])));
      }
      t.appendChild(tr);
    }
    d.appendChild(t);
    if (note) d.appendChild(el('p', { style: 'font-size:.78rem;color:var(--muted);margin:8px 0 0' }, note));
    return d;
  }
  function renderQKV() {
    var box = el('div', { class: 'vecs' });
    box.appendChild(vtable('Q — what I\'m looking for', Q, QKDIM, 'sat asks for a noun, and asks louder (1.5) than the determiners do.'));
    box.appendChild(vtable('K — what I offer', K, QKDIM, 'Each key advertises the token\'s grammatical role.'));
    box.appendChild(vtable('V — what I pass on', V, VDIM, 'The content that actually flows once attention decides who reads whom.'));
    return box;
  }

  /* ---------- render: inspector ---------- */
  function renderInspector() {
    var host = $('#inspect');
    if (state.step === 0) {
      host.innerHTML = '<h4>Why these numbers</h4>' +
        '<p>The vectors are authored rather than random, so the attention pattern is readable instead of noise. ' +
        'A real model learns these projections; the arithmetic downstream is identical either way.</p>' +
        '<p class="hint" style="margin-top:9px">Step forward to start the pipeline.</p>';
      return;
    }
    if (!state.sel) {
      host.innerHTML = '<h4>Cell inspector</h4>' +
        '<p class="hint">Click any cell in the grid to see exactly how its number was produced — ' +
        'no step here is more than one line of arithmetic.</p>';
      return;
    }
    var i = state.sel[0], j = state.sel[1], s = state.step;
    var raw = dot(Q[i], K[j]);
    var sc = raw / SQDK;
    var isMasked = state.causal && j > i;

    var h = '<h4>query <b>' + TOK[i] + '</b>&nbsp;' + i + '  ·  key <b>' + TOK[j] + '</b>&nbsp;' + j + '</h4>';

    if (s === 1 || s === 2) {
      var terms = Q[i].map(function (q, d) { return q + '×' + K[j][d]; }).join(' + ');
      h += '<div class="big">' + fmt(s === 1 ? raw : sc) + '</div>';
      h += '<div class="work">Q<sub>' + i + '</sub> · K<sub>' + j + '</sub><br>= ' + terms +
           '<br>= <b>' + fmt(raw) + '</b>';
      if (s === 2) h += '<br>÷ √d<sub>k</sub> = ' + fmt(raw) + ' ÷ 2 = <b>' + fmt(sc) + '</b>';
      h += '</div>';
    } else if (s === 3) {
      h += '<div class="big">' + (isMasked ? '−∞' : fmt(sc)) + '</div>';
      h += '<div class="work">' + fmt(sc) + (isMasked ? ' + (−∞) = <b>−∞</b>' : ' + 0 = <b>' + fmt(sc) + '</b>') + '</div>';
      h += '<p>' + (isMasked
        ? 'Key ' + j + ' comes after query ' + i + '. Masked — and after softmax this becomes exactly&nbsp;0.'
        : (j > i ? '<strong style="color:var(--origin)">Key ' + j + ' is in the future of query ' + i + ', and the mask is off.</strong> This score survives.'
                 : 'Key ' + j + ' is at or before query ' + i + '. Visible, so nothing is added.')) + '</p>';
    } else {
      var Wm = softmax(mask(scale(scores()), state.causal));
      var w = Wm[i][j];
      var row = mask(scale(scores()), state.causal)[i];
      var mx = Math.max.apply(null, row.filter(isFinite));
      var num = isFinite(row[j]) ? Math.exp(row[j] - mx) : 0;
      var den = row.reduce(function (a, x) { return a + (isFinite(x) ? Math.exp(x - mx) : 0); }, 0);
      h += '<div class="big">' + fmt(w, 3) + '</div>';
      h += '<div class="work">exp(' + fmt(row[j]) + ') = ' + fmt(num, 3) +
           '<br>÷ row total ' + fmt(den, 3) + '<br>= <b>' + fmt(w, 3) + '</b> &nbsp;(' + (w * 100).toFixed(1) + '%)</div>';
      if (s === 5) {
        var contrib = V[j].map(function (v) { return v * w; });
        h += '<p style="margin-bottom:6px">Contribution to <b>' + TOK[i] + '</b>\'s output:</p>' +
             '<div class="work">' + fmt(w, 3) + ' × [' + V[j].join(', ') + '] = [' +
             contrib.map(function (c) { return fmt(c, 2); }).join(', ') + ']</div>';
      } else {
        h += '<p>' + (w < 0.0005
          ? 'Zero — exactly, not approximately. That is <code>softmax(−∞)</code>, and it is what makes parallel causal training possible.'
          : TOK[i] + ' spends ' + (w * 100).toFixed(1) + '% of its attention budget on ' + TOK[j] + '.') + '</p>';
      }
    }
    host.innerHTML = h;
  }

  /* ---------- render: outputs (step 5) ---------- */
  function renderOutputs() {
    var host = $('#outputs');
    if (state.step !== 5) { host.hidden = true; host.innerHTML = ''; return; }
    host.hidden = false;
    var W = softmax(mask(scale(scores()), state.causal));
    var O = weighted(W);
    var h = '<h4 style="font-family:var(--mono);font-size:.7rem;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);font-weight:400;margin-bottom:12px">' +
            'Output vectors — each token\'s new representation</h4>';
    for (var i = 0; i < N; i++) {
      h += '<div class="obar"><span class="lbl">' + TOK[i] + ' <span style="opacity:.5">' + i + '</span></span><span class="track2">';
      for (var d = 0; d < DK; d++) {
        var pct = O[i][d] * 100;
        h += '<span class="seg" style="flex:' + Math.max(pct, 0.001) + ';background:' + VCOL[d] + '" title="' +
             VDIM[d] + ' ' + pct.toFixed(0) + '%">' + (pct > 13 ? VDIM[d] : '') + '</span>';
      }
      h += '</span></div>';
    }
    h += '<div class="mlegend" style="justify-content:flex-start;margin-top:10px">';
    VDIM.forEach(function (d, k) { h += '<span><i style="background:' + VCOL[k] + '"></i>' + d + '</span>'; });
    h += '</div>';
    host.innerHTML = h;
  }

  /* ---------- render: leak warning ---------- */
  function renderLeak() {
    var host = $('#leak');
    if (state.causal || state.step < 3) { host.hidden = true; return; }
    host.hidden = false;
    var W = softmax(mask(scale(scores()), false));
    var future = 0;
    for (var j = 1; j < N; j++) future += W[0][j];
    host.innerHTML =
      '<h4>The mask is off, and token 0 is cheating</h4>' +
      '<p><strong>' + (future * 100).toFixed(0) + '% of <em>the</em>\'s attention now lands on tokens that come after it</strong> — ' +
      (W[0][1] * 100).toFixed(0) + '% of it on <em>cat</em>, the very word it is supposed to help predict. ' +
      'At training time the model would score perfectly by copying the answer it can already see, and learn nothing. ' +
      'At generation time those tokens do not exist yet, so the model would be reading memory that was never written.</p>' +
      '<p style="margin-top:8px">That gap between training and generation is why the mask is not an optimisation — ' +
      'it is what makes the training signal honest.</p>';
  }

  /* ---------- render: step chrome ---------- */
  function renderStep() {
    document.querySelectorAll('.step').forEach(function (b, k) {
      if (k === state.step) b.setAttribute('aria-current', 'step');
      else b.removeAttribute('aria-current');
    });
    var b = STEPS[state.step].body;
    $('#s-title').textContent = STEPS[state.step].title;
    $('#s-body').innerHTML = (typeof b === 'function') ? b() : b;
    $('#prev').disabled = state.step === 0;
    $('#next').disabled = state.step === STEPS.length - 1;
  }

  function render() {
    renderStep(); renderFormula(); renderMatrix();
    renderInspector(); renderOutputs(); renderLeak();
  }

  function go(n) {
    state.step = Math.max(0, Math.min(STEPS.length - 1, n));
    if (state.step === 0) state.sel = null;
    render();
  }

  /* ---------- wire up ---------- */
  function init() {
    var bar = $('#steps');
    STEPS.forEach(function (s, k) {
      var b = el('button', { class: 'step', type: 'button' },
                 '<span class="k">' + k + '</span>' + s.label);
      b.addEventListener('click', function () { go(k); });
      bar.appendChild(b);
    });

    $('#prev').addEventListener('click', function () { go(state.step - 1); });
    $('#next').addEventListener('click', function () { go(state.step + 1); });
    $('#causal').addEventListener('change', function (e) {
      state.causal = e.target.checked;
      if (state.step < 3) go(3); else render();
    });

    $('#m-host').addEventListener('click', function (e) {
      var c = e.target.closest('.cell'); if (!c) return;
      state.sel = [+c.dataset.i, +c.dataset.j];
      renderMatrix(); renderInspector();
    });
    $('#m-host').addEventListener('keydown', function (e) {
      if (e.key !== 'Enter' && e.key !== ' ') return;
      var c = e.target.closest('.cell'); if (!c) return;
      e.preventDefault();
      state.sel = [+c.dataset.i, +c.dataset.j];
      renderMatrix(); renderInspector();
    });

    render();
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
