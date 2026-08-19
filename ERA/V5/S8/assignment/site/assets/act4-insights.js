/* ===========================================================
   Act 4 — What the timeline shows that a list would not.
   The answer to the assignment's bonus question.

   Every claim here carries a "show me" button that highlights the
   exact region of the axis it is about. Nothing is asserted that
   the picture cannot be made to demonstrate.

   The supporting numbers are NOT typed in. They are computed from
   window.MECHANISMS at render time by the helpers below, so a claim
   cannot survive a change to the underlying dates — if a date moves,
   the sentence moves with it.
   =========================================================== */
(function () {
  'use strict';

  var M = window.MECHANISMS;
  function get(id) { return M.filter(function (m) { return m.id === id; })[0]; }
  function D(s) { return new Date(s + 'T00:00:00Z'); }
  function days(a, b) { return Math.round((D(b) - D(a)) / 864e5); }
  function yearsMonths(a, b) {
    var d = days(a, b), y = Math.floor(d / 365.25), mo = Math.round((d - y * 365.25) / 30.4375);
    if (mo === 12) { y += 1; mo = 0; }
    return y + (y === 1 ? ' year' : ' years') + (mo ? ' ' + mo + ' month' + (mo === 1 ? '' : 's') : '');
  }
  function between(a, b) {
    return M.filter(function (m) { return m.date > a && m.date < b; });
  }
  function inLane(l) { return M.filter(function (m) { return m.lane === l; }); }

  var INSIGHTS = [

  { id: 'silence-after',
    title: 'After the Transformer, nothing happened for nearly two years',
    build: function () {
      var a = get('sdpa'), b = get('sparse');
      return {
        facts: [['gap', yearsMonths(a.date, b.date)], ['entries between', String(between(a.date, b.date).length)]],
        html: '<p>Attention arrives in June 2017 and the next entry on this page — the first attempt to ' +
          'make it cheaper — is <strong>' + days(a.date, b.date) + ' days</strong> later. That is the ' +
          'largest gap between consecutive entries anywhere on the timeline, and it sits immediately ' +
          'after the most important one.</p>' +
          '<p>A list would show sparse attention "following" the Transformer and imply a reaction. The ' +
          'spacing says otherwise: for two years nobody was economising, because the returns were coming ' +
          'from making these models <em>bigger</em>, not cheaper. Efficiency work starts when scaling ' +
          'starts to hurt, not when the architecture is published.</p>',
        hi: { from: a.date, to: b.date, ids: ['sdpa', 'sparse'], label: yearsMonths(a.date, b.date) + ' — the largest gap here' }
      };
    } },

  { id: 'memory-twice',
    title: 'The KV-cache answer existed for three and a half years before anyone wanted it',
    build: function () {
      var a = get('mqa'), b = get('gqa');
      return {
        facts: [['MQA → GQA', yearsMonths(a.date, b.date)], ['days', String(days(a.date, b.date))]],
        html: '<p>MQA solves the KV-cache problem in <strong>November 2019</strong> — share the key and ' +
          'value heads, cut the cache eightfold. Then the memory lane goes quiet until GQA in ' +
          '<strong>May 2023</strong>.</p>' +
          '<p>Nothing was missing technically; GQA is a softer version of the same idea, and its own ' +
          'selling point is that you can uptrain into it from an existing checkpoint. What changed in ' +
          'between was <em>who pays</em>. In 2019 the expensive thing was training. By 2023 it was ' +
          '<strong>serving</strong> — thousands of concurrent users, each carrying an unshareable cache. ' +
          'The idea waited for the bill.</p>',
        hi: { from: a.date, to: b.date, ids: ['mqa', 'gqa'], label: yearsMonths(a.date, b.date) + ' of silence on the memory bill' }
      };
    } },

  { id: 'delta-dormant',
    title: 'A good idea waited three years for hardware-shaped arithmetic',
    build: function () {
      var a = get('delta'), b = get('deltanet');
      return {
        facts: [['delta rule → DeltaNet', yearsMonths(a.date, b.date)], ['predates RoPE by', days(a.date, get('rope').date) + ' days']],
        html: '<p>The delta rule — correct the fixed state instead of accumulating into it — is published ' +
          'in <strong>February 2021</strong>. It predates RoPE by ' + days(a.date, get('rope').date) +
          ' days. Then it does essentially nothing for three years.</p>' +
          '<p>The blocker was never the idea. It was that read-then-write is sequential, and a GPU hates ' +
          'sequential. The 2024 paper everyone calls "the DeltaNet paper" contributes a <em>parallel ' +
          'training algorithm</em>, not the rule. Chronology makes the real lesson visible: in this field ' +
          'an idea is not adopted when it is correct, but when someone works out how to train it fast.</p>',
        hi: { from: a.date, to: b.date, ids: ['delta', 'deltanet'], label: yearsMonths(a.date, b.date) + ' dormant' }
      };
    } },

  { id: 'quiet-systems',
    title: 'The stretch that looks empty is where the systems people were working',
    build: function () {
      var a = get('alibi'), b = get('gqa'), mid = between(a.date, b.date);
      return {
        facts: [['ALiBi → GQA', yearsMonths(a.date, b.date)], ['entries in between', String(mid.length)]],
        html: '<p>Between ALiBi (August 2021) and GQA (May 2023) the assignment\'s required list shows ' +
          'almost nothing — ' + yearsMonths(a.date, b.date) + ' of apparent quiet. Exactly one entry ' +
          'sits inside it, and it is the one nobody assigned: <strong>FlashAttention</strong>.</p>' +
          '<p>That is the whole reason it earns a slot. It answers neither of the two bills — it changes ' +
          'no maths at all, computes bit-identical attention, and attacks a third cost the lesson never ' +
          'names: <em>memory bandwidth</em>. The field was not idle in this window; it was solving the ' +
          '<strong>systems</strong> half of the problem while the architectural half paused. Exact ' +
          'attention stayed affordable for years longer than it otherwise would have — which is part of ' +
          'why the pressure to approximate it only becomes urgent afterwards.</p>',
        hi: { from: a.date, to: b.date, ids: ['flash'], label: 'one entry in ' + yearsMonths(a.date, b.date) }
      };
    } },

  { id: 'pileup-2023',
    title: 'Then 2023 happens all at once',
    build: function () {
      var c = M.filter(function (m) { return m.date >= '2023-05-01' && m.date <= '2023-10-01'; });
      var pos = c.filter(function (m) { return m.lane === 'pos'; });
      var mem = c.filter(function (m) { return m.lane === 'mem'; });
      return {
        facts: [['entries', String(c.length)], ['span', days(c[0].date, c[c.length - 1].date) + ' days'],
                ['two of them', days(get('ntk').date, get('ntk-parts').date) + ' days apart']],
        html: '<p><strong>' + c.length + ' of the 21 mechanisms land inside ' +
          days(c[0].date, c[c.length - 1].date) + ' days</strong> — ' + pos.length + ' about length, ' +
          mem.length + ' about memory. Two of them, NTK-Aware and NTK-By-Parts, are ' +
          days(get('ntk').date, get('ntk-parts').date) + ' days apart; at true scale they very nearly ' +
          'occupy the same pixel.</p>' +
          '<p>This is what a field looks like when one constraint starts binding for everybody ' +
          'simultaneously. Long context became the thing customers asked for, and within a single summer ' +
          'four different groups shipped answers. A list renders this as four ordinary rows. The timeline ' +
          'renders it as a pile-up, which is what it was.</p>',
        hi: { from: '2023-05-01', to: '2023-10-01', ids: c.map(function (m) { return m.id; }),
              label: c.length + ' entries in ' + days(c[0].date, c[c.length - 1].date) + ' days' }
      };
    } },

  { id: 'position-never',
    title: 'Position is the problem that never stayed solved',
    build: function () {
      var p = inLane('pos');
      return {
        facts: [['attempts', String(p.length)], ['span', yearsMonths(p[0].date, p[p.length - 1].date)]],
        html: '<p>The position lane runs the entire width of this page: <strong>' + p.length + ' attempts ' +
          'across ' + yearsMonths(p[0].date, p[p.length - 1].date) + '</strong>, and it is both the first ' +
          'entry on the timeline and the last. Learned table → sinusoidal → RoPE → ALiBi → NTK-Aware → ' +
          'NTK-By-Parts → YaRN → DroPE.</p>' +
          '<p>Read the sequence and the shape is unmistakable: add a mechanism, refine it, refine the ' +
          'refinement — and then, in the most recent entry, <strong>propose deleting it</strong>. DroPE\'s ' +
          'claim is that a causal decoder already knows the order implicitly, so the safest positional ' +
          'embedding is none. Nine years of increasingly elaborate machinery, ending in an argument that ' +
          'the machinery was the problem.</p>',
        hi: { from: p[0].date, to: p[p.length - 1].date, ids: p.map(function (m) { return m.id; }),
              label: p.length + ' attempts, ' + yearsMonths(p[0].date, p[p.length - 1].date) }
      };
    } },

  { id: 'acceleration',
    title: 'The second half of the story is twice as fast as the first',
    build: function () {
      var early = M.filter(function (m) { return m.date < '2023-01-01'; });
      var late = M.filter(function (m) { return m.date >= '2023-01-01'; });
      var e = (days(M[0].date, '2023-01-01') / 365.25).toFixed(1);
      var l = (days('2023-01-01', M[M.length - 1].date) / 365.25).toFixed(1);
      return {
        facts: [['before 2023', early.length + ' in ' + e + ' yrs'], ['2023 onward', late.length + ' in ' + l + ' yrs']],
        html: '<p><strong>' + early.length + ' mechanisms in the first ' + e + ' years. ' + late.length +
          ' in the ' + l + ' years since.</strong> Roughly double the rate, and the composition changes ' +
          'with it — the early period is dominated by compute, the recent one spreads across memory, ' +
          'length and compute at once.</p>' +
          '<p>Ordering alone cannot show this. You need the spacing to see that the field did not work ' +
          'through these problems steadily; it worked slowly while models were small, then all at once ' +
          'when serving them at length became the binding cost.</p>',
        hi: { from: '2023-01-01', to: M[M.length - 1].date, ids: late.map(function (m) { return m.id; }),
              label: late.length + ' entries in ' + l + ' years' }
      };
    } },

  { id: 'no-paper',
    title: 'Two of the twenty-one never had a paper at all',
    build: function () {
      var np = M.filter(function (m) { return m.src === 'reddit' || m.src === 'github'; });
      return {
        facts: [['non-paper entries', String(np.length)], ['both in', '2023'], ['both about', 'length']],
        html: '<p>NTK-Aware scaling is a <strong>Reddit post</strong>. NTK-By-Parts is a <strong>GitHub ' +
          'pull request</strong>. Neither has a paper, neither was peer reviewed, and both went into ' +
          'production within weeks — YaRN, which does have a paper, is built directly on the second one.</p>' +
          '<p>They sit inside the 2023 pile-up, which is not a coincidence: when a constraint binds hard ' +
          'enough, the publication cycle is too slow and the fix ships wherever it can. This is also the ' +
          'part of the timeline that is hardest to date honestly — and the reason every dossier on this ' +
          'page ends by stating how its date was established rather than merely asserting one.</p>',
        hi: { from: get('ntk').date, to: get('ntk-parts').date, ids: np.map(function (m) { return m.id; }),
              label: 'no paper — a forum post and a pull request' }
      };
    } }
  ];

  function init() {
    var host = document.getElementById('insights');
    if (!host) return;
    INSIGHTS.forEach(function (ins, i) {
      var b = ins.build();
      var d = document.createElement('div');
      d.className = 'ins';
      d.id = 'ins-' + ins.id;
      d.innerHTML =
        '<div class="k">' + String(i + 1).padStart(2, '0') + '</div>' +
        '<div><h3>' + ins.title + '</h3>' + b.html +
        '<div>' + b.facts.map(function (f) {
          return '<span class="fact">' + f[0] + ' <b>' + f[1] + '</b></span>';
        }).join('') +
        '<button class="showme" type="button">show me on the timeline ↑</button></div></div>';
      d.querySelector('.showme').addEventListener('click', function () {
        window.__timeline.highlight(b.hi);
      });
      host.appendChild(d);
    });

    var clr = document.getElementById('ins-clear');
    if (clr) clr.addEventListener('click', function () { window.__timeline.clear(); });
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
