/* ===========================================================
   Act 5 — the sources table.

   Every row is generated from window.MECHANISMS. Nothing here is
   typed by hand, including the counts in the stat strip: this table
   is the page auditing itself, so a hand-maintained copy would defeat
   the point of having it.

   tools/gen-readme-table.js emits the README's table from the same
   file, which is why the two cannot disagree.
   =========================================================== */
(function () {
  var M = window.MECHANISMS, CTX = window.CONTEXT_MARKER, LANES = window.LANES;

  var MON = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
  function fmtDate(iso) {
    var p = iso.split('-');
    return MON[+p[1] - 1] + ' ' + (+p[2]) + ', ' + p[0];
  }

  /* The source link reads as its own citation: the arXiv id, or the platform
     the mechanism actually shipped on for the two entries with no paper. */
  function srcLabel(m) {
    return m.src === 'reddit' ? 'r/LocalLLaMA post'
         : m.src === 'github' ? 'github PR #1'
         : 'arXiv:' + m.src;
  }

  function row(m, isCtx) {
    var lane = isCtx ? null : LANES[m.lane];
    var corrected = /corrected during the/i.test(m.verified);
    var cls = 'id' + (isCtx ? ' ctxrow' : '') + (corrected ? ' mark' : '');

    var h = '<tr class="' + cls + '">' +
      '<td class="num">' + (isCtx ? '—' : m.n) + '</td>' +
      '<td class="date">' + fmtDate(m.date) + '</td>' +
      '<td>' +
        '<a class="mech" href="#' + m.id + '">' + m.name + '</a>' +
        (lane ? '<span class="badge ' + m.lane + '">' + lane.short + '</span>'
              : '<span class="badge">context, not a mechanism</span>') +
      '</td>' +
      '<td>' +
        '<span class="paper">' + m.paper + '</span>' +
        (m.who ? '<span class="who">' + m.who + '</span><br>' : '') +
        '<a class="srclink" href="' + m.url + '" target="_blank" rel="noopener">' +
          srcLabel(m) + ' ↗</a>' +
        (m.v1 ? '<span class="v1">v1 · ' + m.v1 + '</span>' : '') +
      '</td>' +
      '<td><span class="badge t' + m.tier + '">tier ' + m.tier + '</span></td>' +
      '</tr>';

    h += '<tr class="ev"><td></td><td colspan="4"><div class="how">' +
      m.verified + '</div></td></tr>';
    return h;
  }

  function stats(rows) {
    var arx = rows.filter(function (m) { return !!m.v1; });
    var ids = {};
    arx.forEach(function (m) { ids[m.src] = 1; });
    var t = {A: 0, B: 0, C: 0};
    rows.forEach(function (m) { t[m.tier]++; });
    var corrected = rows.filter(function (m) {
      return /corrected during the/i.test(m.verified);
    }).length;

    return [
      [rows.length, 'rows — 21 mechanisms plus one model quoted for its numbers'],
      [arx.length, 'dated from an arXiv <b>v1</b> submission record, to the second'],
      [Object.keys(ids).length, 'distinct papers — two rows share <em>Attention Is All You Need</em>'],
      [rows.length - arx.length, 'with no paper at all: a Reddit post and a GitHub pull request'],
      [t.A + ' / ' + t.B + ' / ' + t.C, 'evidence tiers A / B / C'],
      [corrected, 'dates corrected against a first draft that looked right']
    ].map(function (s) {
      var wide = String(s[0]).length > 4 ? ' wide' : '';
      return '<div class="s"><div class="n' + wide + '">' + s[0] + '</div>' +
             '<div class="l">' + s[1] + '</div></div>';
    }).join('');
  }

  function init() {
    var host = document.getElementById('sources');
    if (!host || !M) return;
    var rows = M.concat([CTX]);

    document.getElementById('srcstats').innerHTML = stats(rows);

    host.innerHTML =
      '<table class="src"><thead><tr>' +
        '<th>#</th><th>Date</th><th>Mechanism</th>' +
        '<th>Primary source</th><th>Evidence</th>' +
      '</tr></thead><tbody>' +
      rows.map(function (m) { return row(m, m === CTX); }).join('') +
      '</tbody></table>';

    /* An in-page link to a mechanism means "open its dossier", not "jump to an
       anchor that does not exist" — Act 3 renders in place inside Act 2. */
    host.querySelectorAll('a.mech').forEach(function (a) {
      a.addEventListener('click', function (e) {
        var id = a.getAttribute('href').slice(1);
        if (window.__timeline && window.__timeline.open) {
          e.preventDefault();
          window.__timeline.open(id);
        }
      });
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else { init(); }
})();
