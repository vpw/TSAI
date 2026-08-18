/* ===========================================================
   Act 1 — The two bills.

   One context slider driving both costs at once:
     compute  ~ T²  (per head, per layer)
     KV cache = 2 × layers × kv_heads × head_dim × T × batch × bytes

   The lesson's yardstick — 48 layers, 8 KV heads, head_dim 128, bf16,
   T = 32,768, one user — must read 6.44 GB. That is a regression test,
   not a decoration: every mechanism later on this page is scored on this
   same meter, so an off-by-one here would corrupt every comparison.

   Note on units: 6,442,450,944 bytes is 6.44 GB decimal (÷1e9) and
   exactly 6.00 GiB binary (÷2^30). The lesson quotes 6.44, so this meter
   is decimal throughout. Mixing the two would be a silent 7% error.
   =========================================================== */
(function () {
  'use strict';

  var T_STEPS = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144];
  var U_STEPS = [1, 2, 4, 8, 16, 32, 64];
  var Q_HEADS = 64;                       // query heads, held fixed across presets
  var GPU = 80e9, NODE = 640e9;           // one 80 GB accelerator; an 8-way node

  var KV_PRESETS = [
    { kv: 64, name: 'MHA',    note: 'one per query head' },
    { kv: 16, name: 'GQA-16', note: '4 queries each' },
    { kv: 8,  name: 'GQA-8',  note: 'the yardstick' },
    { kv: 2,  name: 'GQA-2',  note: '32 queries each' },
    { kv: 1,  name: 'MQA',    note: 'one for all' }
  ];
  var LINE_KV = [64, 8, 2, 1];
  var LINE_COL = ['var(--origin)', 'var(--mem)', 'var(--cmp)', 'var(--pos)'];

  var YARDSTICK = { layers: 48, kv: 8, hd: 128, bytes: 2 };

  var st = { ti: 5, ui: 0, layers: 48, kv: 8, hd: 128, bytes: 2 };  // ti=5 → T=32768

  /* ---------- the formula ---------- */
  function perToken(cfg) { return 2 * cfg.layers * cfg.kv * cfg.hd * cfg.bytes; }
  function cacheBytes(cfg, T, users) { return perToken(cfg) * T * users; }

  var T = function () { return T_STEPS[st.ti]; };
  var U = function () { return U_STEPS[st.ui]; };

  /* ---------- formatting ---------- */
  function gb(b) { return b / 1e9; }
  function fgb(b) {
    var g = gb(b);
    if (g >= 1000) return (g / 1000).toFixed(2) + ' TB';
    if (g >= 100) return g.toFixed(1) + ' GB';
    // two decimals through the 1–100 GB band so the lesson's own figures land
    // exactly: 6.44 GB at one user, 51.54 GB at eight. One decimal here would
    // round 51.54 to 51.5 and quietly disagree with the source material.
    if (g >= 1) return g.toFixed(2) + ' GB';
    return (g * 1000).toFixed(0) + ' MB';
  }
  function fcount(n) {
    if (n >= 1e12) return (n / 1e12).toFixed(2) + ' trillion';
    if (n >= 1e9)  return (n / 1e9).toFixed(2) + ' billion';
    if (n >= 1e6)  return (n / 1e6).toFixed(2) + ' million';
    return n.toLocaleString('en-US');
  }
  function ftok(t) {
    if (t >= 1048576) return (t / 1048576).toFixed(t >= 10485760 ? 0 : 1) + 'M';
    return t >= 1024 ? (t / 1024) + 'K' : String(t);
  }
  function commas(n) { return Math.round(n).toLocaleString('en-US'); }

  function $(s) { return document.querySelector(s); }

  /* ---------- readouts ---------- */
  function renderCards() {
    var t = T(), u = U();
    var cmp = t * t;
    var cmpAll = cmp * st.layers * Q_HEADS;
    var one = cacheBytes(st, t, 1);
    var all = cacheBytes(st, t, u);
    var fits = Math.floor(GPU / one);

    $('#b1-big').innerHTML = fcount(cmp) + '<small>comparisons</small>';
    $('#b1-sub').innerHTML =
      'That is <b>T²</b> for a single head in a single layer. Across ' + st.layers +
      ' layers × ' + Q_HEADS + ' query heads: <b>' + fcount(cmpAll) + '</b>.';
    $('#b1-tag').textContent =
      'Double the context and this quadruples. But it is arithmetic — it parallelises, and you can buy more of it.';

    $('#b2-big').innerHTML = fgb(all) + '<small>' + (u === 1 ? 'one user' : u + ' users') + '</small>';
    $('#b2-sub').innerHTML =
      'Per user: <b>' + fgb(one) + '</b>. Every concurrent request carries its own copy — ' +
      'nothing here is shared between them.';
    $('#b2-tag').textContent = fits >= 1
      ? 'An 80 GB accelerator holds ' + fits + ' such cache' + (fits === 1 ? '' : 's') + ' — before the model weights get any room.'
      : 'One user\'s cache alone already exceeds an 80 GB accelerator.';

    $('#fsub-body').innerHTML =
      '2 × <b>' + st.layers + '</b> layers × <b>' + st.kv + '</b> kv_heads × <b>' + st.hd +
      '</b> head_dim × <b>' + commas(t) + '</b> tokens × <b>' + u + '</b> ' + (u === 1 ? 'user' : 'users') +
      ' × <b>' + st.bytes + '</b> bytes<br>= <b>' + commas(all) + '</b> bytes = <b>' + fgb(all) + '</b>' +
      '<span style="color:var(--muted)">&nbsp;&nbsp;(' + (all / 1073741824).toFixed(2) + ' GiB)</span>';

    // "if you double the context" line
    var t2 = t * 2;
    $('#dbl').innerHTML = t2 <= 4194304
      ? 'Going from <b>' + ftok(t) + '</b> to <b>' + ftok(t2) + '</b>: compute <b>×4</b> (' +
        fcount(cmp) + ' → ' + fcount(t2 * t2) + '), cache <b>×2</b> (' + fgb(all) + ' → ' +
        fgb(cacheBytes(st, t2, u)) + ').'
      : '';
  }

  /* ---------- chart ---------- */
  function renderChart() {
    var host = $('#chart'), t = T(), u = U();
    var W = 720, H = 330, L = 58, R = 96, TP = 18, B = 40;
    var xmin = T_STEPS[0], xmax = T_STEPS[T_STEPS.length - 1];

    // Log on BOTH axes, deliberately. Cache is linear in T, and on log-log a linear
    // function is a straight line of slope 1 — so the four configurations come out as
    // parallel straight climbs. On a linear-x/log-y plot the same data curves and
    // appears to flatten, which reads as "growth is slowing" — the exact opposite of
    // what the lesson says about head sharing.
    var maxB = Math.max.apply(null, LINE_KV.map(function (k) {
      return cacheBytes({ layers: st.layers, kv: k, hd: st.hd, bytes: st.bytes }, xmax, u);
    }));
    var lo = -2, hi = Math.max(Math.ceil(Math.log10(gb(maxB))), lo + 1);
    var lx0 = Math.log10(xmin), lx1 = Math.log10(xmax);

    var X = function (v) { return L + (Math.log10(v) - lx0) / (lx1 - lx0) * (W - L - R); };
    var Y = function (g) {
      var e = Math.log10(Math.max(g, Math.pow(10, lo)));
      return TP + (1 - (e - lo) / (hi - lo)) * (H - TP - B);
    };

    var s = '<svg class="chart" viewBox="0 0 ' + W + ' ' + H + '" role="img" ' +
            'aria-label="KV cache size against context length for four KV-head configurations, ' +
            'log-log axes, showing four parallel straight lines that each cross the 80 GB limit ' +
            'at a different context length">';

    // decade gridlines
    for (var e = lo; e <= hi; e++) {
      var y = Y(Math.pow(10, e)), g = Math.pow(10, e);
      s += '<line class="gl" x1="' + L + '" y1="' + y + '" x2="' + (W - R) + '" y2="' + y + '"/>';
      var lab = g >= 1000 ? (g / 1000) + ' TB' : (g >= 1 ? g + ' GB' : (g * 1000) + ' MB');
      s += '<text class="lbl" x="' + (L - 8) + '" y="' + (y + 3.5) + '" text-anchor="end">' + lab + '</text>';
    }
    T_STEPS.forEach(function (tv) {
      var x = X(tv);
      s += '<line class="gl" x1="' + x + '" y1="' + TP + '" x2="' + x + '" y2="' + (H - B) + '" opacity=".55"/>';
      s += '<text class="lbl" x="' + x + '" y="' + (H - B + 15) + '" text-anchor="middle">' + ftok(tv) + '</text>';
    });
    s += '<text class="lbl" x="' + ((L + W - R) / 2) + '" y="' + (H - 6) +
         '" text-anchor="middle" style="letter-spacing:.1em">CONTEXT LENGTH T  ·  LOG SCALE</text>';
    s += '<line class="ax" x1="' + L + '" y1="' + (H - B) + '" x2="' + (W - R) + '" y2="' + (H - B) + '"/>';

    // right-edge labels get collected first, then pushed apart before drawing
    var labels = [];

    [[GPU, 'one 80 GB GPU'], [NODE, '8-GPU node']].forEach(function (r) {
      var g = gb(r[0]);
      if (Math.log10(g) > hi || Math.log10(g) < lo) return;
      var y = Y(g);
      s += '<line class="ref" x1="' + L + '" y1="' + y + '" x2="' + (W - R) + '" y2="' + y + '"/>';
      labels.push({ y: y, text: r[1], color: 'var(--origin)', weight: 400 });
    });

    LINE_KV.forEach(function (k, idx) {
      var cfg = { layers: st.layers, kv: k, hd: st.hd, bytes: st.bytes };
      var pts = [];
      for (var i = 0; i <= 48; i++) {
        var tv = Math.pow(10, lx0 + (lx1 - lx0) * (i / 48));
        pts.push(X(tv).toFixed(1) + ',' + Y(gb(cacheBytes(cfg, tv, u))).toFixed(1));
      }
      var active = (k === st.kv);
      s += '<polyline class="ln' + (active ? '' : ' dim') + '" points="' + pts.join(' ') +
           '" stroke="' + LINE_COL[idx] + '"/>';
      labels.push({
        y: Y(gb(cacheBytes(cfg, xmax, u))),
        text: (KV_PRESETS.filter(function (p) { return p.kv === k; })[0] || { name: k + ' kv' }).name,
        color: LINE_COL[idx], weight: active ? 700 : 400
      });
    });

    // de-overlap: sort top-to-bottom and enforce a minimum vertical gap
    labels.sort(function (a, b) { return a.y - b.y; });
    for (var i = 1; i < labels.length; i++) {
      if (labels[i].y - labels[i - 1].y < 12) labels[i].y = labels[i - 1].y + 12;
    }
    labels.forEach(function (lb) {
      s += '<text class="lbl" x="' + (W - R + 6) + '" y="' + (lb.y + 3.5) + '" fill="' + lb.color +
           '" style="font-weight:' + lb.weight + '">' + lb.text + '</text>';
    });

    var cx = X(t);
    s += '<line class="cur" x1="' + cx + '" y1="' + TP + '" x2="' + cx + '" y2="' + (H - B) + '"/>';
    var cy = Y(gb(cacheBytes(st, t, u)));
    s += '<circle class="dot" cx="' + cx + '" cy="' + cy + '" r="5" fill="' +
         (LINE_COL[LINE_KV.indexOf(st.kv)] || 'var(--ink)') + '"/>';

    s += '</svg>';

    var leg = '<div class="chartlegend">';
    LINE_KV.forEach(function (k, i) {
      var p = KV_PRESETS.filter(function (x) { return x.kv === k; })[0];
      leg += '<span><i style="background:' + LINE_COL[i] + '"></i>' + p.name + ' — ' + k + ' kv head' +
             (k === 1 ? '' : 's') + '</span>';
    });
    leg += '<span style="opacity:.8">' + (u === 1 ? '1 user' : u + ' concurrent users') + ' · log scale</span></div>';

    host.innerHTML = s + leg;

    // where each configuration runs out of one 80 GB accelerator
    var outs = LINE_KV.map(function (k) {
      var per = perToken({ layers: st.layers, kv: k, hd: st.hd, bytes: st.bytes }) * u;
      var name = KV_PRESETS.filter(function (p) { return p.kv === k; })[0].name;
      return name + ' ' + ftok(Math.round(GPU / per / 1024) * 1024);
    });
    $('#crossings').innerHTML =
      'Tokens that fit in 80 GB at ' + (u === 1 ? 'one user' : u + ' users') + ': <b>' +
      outs.join('</b> · <b>') + '</b>. Every line crosses the limit — sharing only changes <em>when</em>.';
  }

  /* ---------- controls ---------- */
  function renderControls() {
    $('#t-val').textContent = commas(T()) + ' tokens';
    $('#u-val').textContent = U() + (U() === 1 ? ' user' : ' users');
    document.querySelectorAll('.kv').forEach(function (b) {
      b.setAttribute('aria-pressed', String(+b.dataset.kv === st.kv));
    });
    $('#c-layers').value = st.layers;
    $('#c-hd').value = st.hd;
    $('#c-bytes').value = st.bytes;
    var isY = st.layers === YARDSTICK.layers && st.kv === YARDSTICK.kv &&
              st.hd === YARDSTICK.hd && st.bytes === YARDSTICK.bytes;
    $('#yard').hidden = isY && st.ti === 5 && st.ui === 0;
  }

  function render() { renderControls(); renderCards(); renderChart(); }

  function init() {
    var kvrow = $('#kvrow');
    KV_PRESETS.forEach(function (p) {
      var b = document.createElement('button');
      b.className = 'kv'; b.type = 'button'; b.dataset.kv = p.kv;
      b.setAttribute('aria-pressed', 'false');
      b.innerHTML = p.name + '<small>' + p.kv + ' kv · ' + p.note + '</small>';
      b.addEventListener('click', function () { st.kv = p.kv; render(); });
      kvrow.appendChild(b);
    });

    $('#t-sl').addEventListener('input', function (e) { st.ti = +e.target.value; render(); });
    $('#u-sl').addEventListener('input', function (e) { st.ui = +e.target.value; render(); });
    $('#c-layers').addEventListener('change', function (e) { st.layers = +e.target.value; render(); });
    $('#c-hd').addEventListener('change', function (e) { st.hd = +e.target.value; render(); });
    $('#c-bytes').addEventListener('change', function (e) { st.bytes = +e.target.value; render(); });
    $('#yard').addEventListener('click', function () {
      st.layers = YARDSTICK.layers; st.kv = YARDSTICK.kv; st.hd = YARDSTICK.hd;
      st.bytes = YARDSTICK.bytes; st.ti = 5; st.ui = 0;
      $('#t-sl').value = 5; $('#u-sl').value = 0;
      render();
    });

    render();
  }

  // exposed so the arithmetic can be asserted from outside the widget
  window.__billMeter = { perToken: perToken, cacheBytes: cacheBytes, state: st, steps: T_STEPS };

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
