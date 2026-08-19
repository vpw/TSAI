/* ===========================================================
   Act 2 — the timeline, and Act 3's dossier template.

   Rendered entirely from window.MECHANISMS. Nothing here restates
   a date; if a date is wrong it is wrong in one place.

   The organising idea: markers sit at their TRUE proportional
   position in time, and labels are allowed to move. A de-overlap
   pass pushes colliding labels apart and draws a leader line back
   to the dot, so readability never costs positional honesty —
   which matters because the gaps are a third of the answer to the
   bonus question.

   The "even spacing" toggle is the bonus question made interactive:
   it collapses the timeline into what a plain ordered list would
   show, so you can watch the information disappear.
   =========================================================== */
(function () {
  'use strict';

  var M = window.MECHANISMS, CTX = window.CONTEXT_MARKER, LANES = window.LANES;
  var LANE_ORDER = ['origin', 'pos', 'mem', 'cmp', 'sys'];
  var LANE_SUB = {
    origin: 'where it all starts',
    pos:    'how far can it reach',
    mem:    'what must stay resident',
    cmp:    'what must be computed',
    sys:    'how fast can it run'
  };

  var W = 1000, PADL = 146, PADR = 34, TOP = 46, LANE_H = 64, AXIS_H = 46;
  var H = TOP + LANE_ORDER.length * LANE_H + AXIS_H;

  var st = { sel: null, even: false };

  var ALL = M.concat([CTX]);
  var t0 = Date.parse(M[0].date);
  var t1 = Date.parse(CTX.date);

  function el(tag, at, txt) {
    var n = document.createElementNS('http://www.w3.org/2000/svg', tag);
    if (at) Object.keys(at).forEach(function (k) { n.setAttribute(k, at[k]); });
    if (txt != null) n.textContent = txt;
    return n;
  }
  function laneY(l) { return TOP + LANE_ORDER.indexOf(l) * LANE_H + LANE_H / 2; }

  /* x position — the whole point of the widget */
  function xOf(m, i) {
    var span = W - PADL - PADR;
    if (st.even) return PADL + (ALL.length === 1 ? 0 : (i / (ALL.length - 1)) * span);
    return PADL + (Date.parse(m.date) - t0) / (t1 - t0) * span;
  }

  function fmtDate(d) {
    var p = d.split('-'), mo = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
    return mo[+p[1] - 1] + ' ' + (+p[2]) + ', ' + p[0];
  }

  /* ---------- the timeline ---------- */
  function renderTimeline() {
    var host = document.getElementById('tl-host');
    host.innerHTML = '';
    var svg = el('svg', {
      id: 'tl', viewBox: '0 0 ' + W + ' ' + H, role: 'img',
      'aria-label': 'Timeline of 21 attention mechanisms from November 2016 to December 2025, ' +
                    'grouped into four lanes by which cost they address'
    });

    // lane bands + labels
    LANE_ORDER.forEach(function (l, i) {
      var y = TOP + i * LANE_H;
      if (i % 2 === 0) svg.appendChild(el('rect', { class: 'band', x: 0, y: y, width: W, height: LANE_H, rx: 6 }));
      var lab = el('text', { class: 'lanelab', x: PADL - 14, y: laneY(l) - 2, 'text-anchor': 'end' });
      lab.innerHTML = LANES[l].name;
      lab.setAttribute('fill', LANES[l].color);
      svg.appendChild(lab);
      svg.appendChild(el('text', { class: 'lanesub', x: PADL - 14, y: laneY(l) + 10, 'text-anchor': 'end' },
                         LANE_SUB[l]));
    });

    // year gridlines (only meaningful when time is proportional)
    var axisY = TOP + LANE_ORDER.length * LANE_H + 12;
    if (!st.even) {
      for (var y2 = 2017; y2 <= 2026; y2++) {
        var xx = PADL + (Date.parse(y2 + '-01-01') - t0) / (t1 - t0) * (W - PADL - PADR);
        if (xx < PADL - 2) continue;
        svg.appendChild(el('line', { class: 'yr', x1: xx, y1: TOP - 8, x2: xx, y2: axisY }));
        svg.appendChild(el('text', { class: 'yrlab', x: xx, y: axisY + 15 }, String(y2)));
      }
    } else {
      svg.appendChild(el('text', {
        class: 'yrlab', x: (PADL + W - PADR) / 2, y: axisY + 15,
        style: 'letter-spacing:.12em'
      }, 'ORDER ONLY — TIME REMOVED'));
    }
    svg.appendChild(el('line', { class: 'axis', x1: PADL - 14, y1: axisY, x2: W - PADR, y2: axisY }));

    // positions for every entry
    var pos = {};
    ALL.forEach(function (m, i) { pos[m.id] = { x: xOf(m, i), y: laneY(m.lane || 'sys') }; });
    pos[CTX.id].y = laneY('sys') + 0;

    /* lineage arcs — drawn before markers so dots sit on top */
    [
      { a: 'delta', b: 'deltanet', col: 'var(--cmp)', lab: '3 years dormant', up: false },
      { a: 'ntk-parts', b: 'yarn', col: 'var(--pos)', lab: 'YaRN builds on this one', up: true, lift: 14 }
    ].forEach(function (arc) {
      var A = pos[arc.a], B = pos[arc.b];
      if (!A || !B) return;
      var dy = (arc.up ? -26 : 26) - (arc.lift || 0);
      var mx = (A.x + B.x) / 2, my = (A.y + B.y) / 2 + dy;
      var d = 'M' + A.x + ',' + A.y + ' Q' + mx + ',' + my + ' ' + B.x + ',' + B.y;
      svg.appendChild(el('path', { class: 'arc', d: d, stroke: arc.col }));
      var t = el('text', { class: 'arclab', x: mx, y: my + (arc.up ? -3 : 11), fill: arc.col }, arc.lab);
      svg.appendChild(t);
    });

    /* labels: alternate above/below within a lane, then de-overlap horizontally */
    var byLane = {};
    ALL.forEach(function (m, i) {
      var l = m.lane || 'sys';
      (byLane[l] = byLane[l] || []).push({ m: m, i: i, x: pos[m.id].x });
    });
    var placed = {};
    Object.keys(byLane).forEach(function (l) {
      var arr = byLane[l].sort(function (a, b) { return a.x - b.x; });
      var rows = [[], []];                       // 0 = above, 1 = below
      arr.forEach(function (o, k) { rows[k % 2].push(o); });
      rows.forEach(function (row, ri) {
        var lastRight = -1e9;
        row.forEach(function (o) {
          var name = o.m.name;
          var wpx = name.length * 5.15 + 8;
          var lx = o.x;
          if (lx - wpx / 2 < lastRight + 5) lx = lastRight + 5 + wpx / 2;   // push right
          if (lx + wpx / 2 > W - PADR) lx = W - PADR - wpx / 2;             // clamp
          lastRight = lx + wpx / 2;
          placed[o.m.id] = { lx: lx, ly: laneY(l) + (ri === 0 ? -17 : 25), row: ri };
        });
      });
    });

    /* markers */
    ALL.forEach(function (m, i) {
      var p = pos[m.id], pl = placed[m.id];
      var isCtx = (m.id === CTX.id);
      var col = isCtx ? 'var(--muted)' : LANES[m.lane].color;
      var g = el('g', {
        class: 'mk', tabindex: '0', role: 'button',
        'aria-label': m.name + ', ' + fmtDate(m.date) + (isCtx ? ', context marker' : '')
      });

      // leader line when the label had to move
      if (Math.abs(pl.lx - p.x) > 1.5) {
        var midY = p.y + (pl.row === 0 ? -8 : 12);
        g.appendChild(el('path', {
          class: 'leader',
          d: 'M' + p.x + ',' + (p.y + (pl.row === 0 ? -7 : 7)) + ' L' + p.x + ',' + midY +
             ' L' + pl.lx + ',' + midY + ' L' + pl.lx + ',' + (pl.ly + (pl.row === 0 ? 3 : -8))
        }));
      }

      var nonPaper = (m.src === 'reddit' || m.src === 'github');
      var dot;
      if (isCtx) {
        dot = el('circle', { class: 'dot ctxmk', cx: p.x, cy: p.y, r: 6, stroke: col });
      } else if (nonPaper) {
        // diamond = not a paper
        var r = 6.4;
        dot = el('rect', {
          class: 'dot', x: p.x - r, y: p.y - r, width: r * 2, height: r * 2, rx: 1.5,
          fill: col, transform: 'rotate(45 ' + p.x + ' ' + p.y + ')'
        });
      } else {
        dot = el('circle', { class: 'dot', cx: p.x, cy: p.y, r: m.bonus ? 7 : 6, fill: col });
      }
      if (m.bonus) dot.setAttribute('stroke-dasharray', '2.5 2');
      if (st.sel === m.id) dot.setAttribute('class', dot.getAttribute('class') + ' sel');
      g.appendChild(dot);

      var t = el('text', {
        class: 'mlab' + (st.sel === m.id ? ' on' : ''),
        x: pl.lx, y: pl.ly, 'text-anchor': 'middle'
      }, m.name);
      if (isCtx) t.setAttribute('font-style', 'italic');
      g.appendChild(t);

      function open() { st.sel = m.id; renderTimeline(); renderDossier(); }
      g.addEventListener('click', open);
      g.addEventListener('keydown', function (e) {
        if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); open(); }
      });
      svg.appendChild(g);
    });

    var inner = document.createElement('div');
    inner.className = 'tl-inner';
    inner.appendChild(svg);
    var scroll = document.createElement('div');
    scroll.className = 'tl-scroll';
    scroll.appendChild(inner);
    host.appendChild(scroll);

    var leg = document.createElement('div');
    leg.className = 'tl-legend';
    leg.innerHTML =
      LANE_ORDER.map(function (l) {
        return '<span><i style="background:' + LANES[l].color + '"></i>' + LANES[l].short + '</span>';
      }).join('') +
      '<span><i class="sq" style="background:var(--muted)"></i>not a paper</span>' +
      '<span><i class="ring"></i>context, not a mechanism</span>' +
      '<span style="opacity:.8">click any marker</span>';
    host.appendChild(leg);

    renderNote();
  }

  /* ---------- the note under the timeline, which is the point of the toggle ---------- */
  function renderNote() {
    var n = document.getElementById('tl-note');
    if (st.even) {
      n.innerHTML =
        '<p><b>This is what a list gives you.</b> Same twenty-one mechanisms, same order — and every ' +
        'gap has been thrown away. MQA now looks like it sits comfortably next to GQA instead of ' +
        '<b>three and a half years</b> before it. The delta rule looks adjacent to DeltaNet rather than ' +
        'stranded for three years waiting on a training algorithm. And 2023 looks like four ordinary ' +
        'steps rather than the pile-up it was.</p>' +
        '<p>Switch back and watch the argument reappear.</p>';
    } else {
      n.innerHTML =
        '<p><b>The gaps are the content.</b> Two long silences dominate this picture. Between ' +
        '<b>MQA (Nov 2019)</b> and <b>GQA (May 2023)</b> the KV-cache problem sits almost untouched for ' +
        'three and a half years — the answer existed the whole time; what was missing was a reason to ' +
        'care, and that arrived with serving cost. Between the <b>delta rule (Feb 2021)</b> and ' +
        '<b>DeltaNet (Jun 2024)</b> a good idea waits three years for a training algorithm that suits ' +
        'the hardware.</p>' +
        '<p>Then look at <b>2023</b>: five entries in nine months, four of them about length. And look ' +
        'at the <span style="color:var(--pos)">position lane</span> across the whole span — eight ' +
        'attempts in nine years, ending with a method whose proposal is to <em>delete</em> positional ' +
        'embeddings entirely.</p>';
    }
  }

  /* ---------- Act 3's dossier template ---------- */
  function renderDossier() {
    var host = document.getElementById('dossier');
    if (!st.sel) {
      host.hidden = false;
      host.innerHTML = '<p class="dshint">Select a mechanism above to open its dossier — ' +
        'what it does, what it buys, what it costs, and how its date was checked.</p>';
      return;
    }
    var isCtx = st.sel === CTX.id;
    var m = isCtx ? CTX : M.filter(function (x) { return x.id === st.sel; })[0];
    if (!m) return;
    host.hidden = false;

    var lane = isCtx ? null : LANES[m.lane];
    var srcLabel = m.src === 'reddit' ? 'reddit · r/LocalLLaMA'
                 : m.src === 'github' ? 'github · jquesnelle/yarn#1'
                 : 'arXiv ' + m.src;

    var idx = M.map(function (x) { return x.id; }).indexOf(st.sel);
    var prev = idx > 0 ? M[idx - 1] : null;
    var next = (idx >= 0 && idx < M.length - 1) ? M[idx + 1] : null;

    var h = '<div class="dsh"><div class="top">' +
      '<h3><span class="n">' + (isCtx ? 'context' : m.n) + '</span>' + m.name + '</h3>' +
      (lane ? '<span class="badge ' + m.lane + '">' + lane.short + '</span>' : '<span class="badge">not a mechanism</span>') +
      '</div>' +
      '<p class="meta">' + fmtDate(m.date) + ' · <a href="' + m.url + '" target="_blank" rel="noopener">' +
      srcLabel + '</a> · ' + m.paper + (m.who ? ' — ' + m.who : '') + '</p>' +
      '<p class="one">' + m.one + '</p></div><div class="dsb">';

    if (isCtx) {
      h += '<div class="blk"><h4>Why it is on this page at all</h4><p>The lesson leans on "V4" for its ' +
           'depth-schedule motif, its KV-cache figures, its Memory Stream and its DroPE application. ' +
           'It is a real published model, so those numbers are citable rather than course folklore. ' +
           'It is drawn differently because it is a <em>model</em>, not a mechanism.</p></div>';
    } else {
      h += '<div class="blk"><h4>What it does</h4><p>' + m.idea + '</p></div>' +
        '<div class="pcw">' +
          '<div class="buy"><h4>What it buys</h4><p>' + m.buys + '</p></div>' +
          '<div class="cost"><h4>What it costs</h4><p>' + m.costs + '</p></div>' +
          '<div class="pick"><h4>When you would pick it</h4><p>' + m.pick + '</p></div>' +
        '</div>';
      if (m.note) h += '<div class="blk lineage"><h4>Lineage &amp; priority</h4><p>' + m.note + '</p></div>';
    }
    // block six, on every dossier without exception
    h += '<div class="blk verif"><h4>How this date was checked</h4><p>' + m.verified + '</p></div>';
    h += '</div>';

    h += '<div class="dsnav">' +
      (prev ? '<button class="btn" type="button" data-go="' + prev.id + '">← ' + prev.name + '</button>' : '<span></span>') +
      (next ? '<button class="btn" type="button" data-go="' + next.id + '">' + next.name + ' →</button>' : '<span></span>') +
      '</div>';

    host.innerHTML = h;
    host.querySelectorAll('[data-go]').forEach(function (b) {
      b.addEventListener('click', function () {
        st.sel = b.dataset.go; renderTimeline(); renderDossier();
        document.getElementById('dossier').scrollIntoView({ block: 'nearest' });
      });
    });
  }

  function init() {
    document.getElementById('tl-even').addEventListener('change', function (e) {
      st.even = e.target.checked; renderTimeline();
    });
    document.getElementById('tl-clear').addEventListener('click', function () {
      st.sel = null; renderTimeline(); renderDossier();
    });
    renderTimeline();
    renderDossier();

    // deep links: #rope opens that dossier
    var hash = location.hash.replace('#', '');
    if (hash && M.some(function (x) { return x.id === hash; })) {
      st.sel = hash; renderTimeline(); renderDossier();
    }
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
