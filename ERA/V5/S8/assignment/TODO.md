# S8 TODO — Modern Attention Variants

Build a chronologically-ordered, visually interactive web app explaining every attention
mechanism from the session, with honest pros/cons and dated primary sources.

`S8-assignment.md` = the task, `resources/s8-session.md` = lesson, `resources/s8-transcript.md` =
live-class transcript.

- [x] Session scaffolding: `CLAUDE.md`, `AGENTS.md`, session + transcript resources.
- [x] Widget-data extraction — deferred (2026-08-18, user-confirmed): optional per CLAUDE.md
      Conventions, nothing load-bearing yet since the app hasn't been built. Revisit only if a
      specific widget's exact default value becomes load-bearing for the app's own numbers.
- [x] Chronology research (2026-08-18): 18 of 19 minimum-coverage mechanisms found + verified on
      arXiv (v1 metadata) and downloaded into the local library — see
      `resources/chronology-research.md` for the full sourced table, sorted chronologically. Two
      findings worth carrying into the app/README: (1) the delta rule's actual origin is Schlag et
      al. 2021 (2102.11174), predating "the DeltaNet paper" (Yang et al. 2024) by 3+ years — flag
      as disputed priority, don't just cite the 2024 paper; (2) NTK-aware scaling has no arXiv
      paper — it's a 2023 Reddit post (u/bloc97), confirmed via YaRN's own bibliography; date is
      approximate (~2023-06), verify the exact post timestamp before treating it as load-bearing.
      **Correction (same day, user-caught):** my first pass wrongly concluded DroPE had no public
      source — I hadn't checked `../../resources/lightninglm.md`, which documents that
      `resources/s8-session.md`'s "V4" is a real published model (LightningLM, arxiv `2606.07404`,
      Rohan Shravan). That paper's own bibliography names DroPE's real source: Gelberg, Eguchi,
      Akiba, Cetin (Sakana AI), arxiv `2512.12167`, 2025-12-13 — verified two ways (RAG hit +
      independent `pdftotext` dump of the citing paper's reference-list pages) then re-confirmed
      against arXiv's own metadata for 2512.12167 directly. Lesson: always check the project's own
      `resources/` docs before concluding "not found" on arXiv. Bibliography extraction going
      forward uses `pdftotext` raw dumps as a second check alongside RAG hits, not RAG alone.
- [x] Bonus not-covered-in-class mechanism (2026-08-18): **FlashAttention** (Dao et al., arxiv
      `2205.14135`, 2022-05-27) — chosen over Multi-Token Prediction/Mamba because it's a genuinely
      distinct *third* answer to the two-bills problem (attacks memory bandwidth via IO-aware
      kernel fusion, not FLOPs or KV-cache size — stays exact, no accuracy trade-off). Chronologically
      it fills the ALiBi(2021)→GQA(2023) gap already flagged in the timeline. Full pros/cons and
      the "why this earns a separate slot" argument are written up in
      `resources/chronology-research.md`.
- [x] Decide the app's structure (2026-08-18, **user-approved** via Lavish review of
      `.lavish/s8-app-plan.html`). Locked: **one page, five acts, no router**; a **proportional
      horizontal timeline with four lane bands** (lanes = which bill each mechanism pays: position/
      length, KV-cache memory, quadratic compute, systems/kernels) as the organizing spine — real
      date spacing, because the *gaps* carry a third of the Q2 answer; **five shared widgets**
      (attention pipeline, bill meter, position lab, KV-head sharing, read-pattern gallery) covering
      13 of 20 entries, everything else static SVG; a uniform six-block dossier template whose last
      block is **"how this date was checked", shown on every dossier**; and one
      `data/mechanisms.js` as the single source of truth feeding timeline + dossiers + sources table
      + README (so the four copies of each date cannot drift). LightningLM/V4 appears as a *context
      marker*, visually distinct from the 20 mechanisms. Mobile flips the axis to vertical, keeping
      proportional gaps. Non-negotiable **caption rule**: every widget's conclusion is written into
      its static caption, so a JS failure still leaves a finished explainer.
      Draft of the seven Q2 timeline observations is in the artifact's Act 4 section.
- [x] Pin the last approximate date (2026-08-18): **NTK-aware scaling = 2023-06-29**, no longer
      "~2023-06". User supplied the original permalink (r/LocalLLaMA post `14lz7j5`); reddit.com is
      bot-walled to both `curl` and a real browser ("Prove your humanity"), so the date came from the
      Wayback Machine two independent ways — a snapshot carries Reddit's own
      `created-timestamp="2023-06-29T08:21:29.413+0000"` / `author="bloc97"`, and the CDX index's
      earliest capture of the URL is 08:21:50 UTC the same day (an archive cannot predate what it
      archived). **All 20 rows now carry exact, sourced dates.** Also found: bloc97 published *two*
      methods — NTK-Aware and NTK-By-Parts ("Add NTK-Aware interpolation by parts correction") — and
      YaRN builds on **NTK-By-Parts**, not plain NTK-Aware (both cited separately in arXiv
      2401.07004). Open build detail: one timeline marker or two (leaning one, distinction stated in
      the dossier). See `resources/chronology-research.md` row 12.
- [x] Hosting decision relaxed (2026-08-18, user): not Netlify-specific. Any host that is shareable,
      works in incognito, and renders correctly — Netlify, Vercel, **or ht-ml.app via
      `npx -y lavish-axi share`** (public by default, so it passes the incognito check; inlines local
      assets automatically). App is a static folder with no build step, so the choice is deferred to
      deployment day and nothing in the plan depends on it.
- [x] Phase 1 — date audit artifact (2026-08-18): `.lavish/s8-date-audit.html`, 21 rows (the 20
      mechanisms + the LightningLM/V4 context marker), chronological. Each row carries the date, the
      primary source, **how the date was fixed**, and a confidence tier: **A = direct arXiv v1
      metadata (18 rows)**, **B = citation chain, i.e. the paper was identified from another paper's
      bibliography and verified two independent ways (2 rows — #1 ConvS2S, #20 DroPE)**, **C =
      archival, no paper exists (1 row — #13 NTK)**. Separately, six rows carry an `attribution`
      badge: the *date* is solid but "who was first" is a judgment call (#1, #4, #6, #7, #8, #17) —
      all five calls written out in full below the table, including two challenger papers I have
      **not** verified and will only chase if the user flags the row. Per-row flag checkboxes keep
      local state and queue **one** prompt on submit (not one per tick). Awaiting review — nothing
      goes into `data/mechanisms.js` until it's signed off.
- [x] Reference sheet (2026-08-18, user-requested, for their own understanding — not a deliverable):
      `resources/mechanism-reference.html`. Standalone, opens from disk with no server. Two-bills
      framing + the unnamed third bill (bandwidth, which is why FlashAttention is the bonus pick), an
      at-a-glance table, then one card per mechanism with **the idea / buys / costs / pick it when**,
      plus four "patterns only visible in date order" cards that seed the Act 4 Q2 answer. Its data
      array is the seed for `data/mechanisms.js`.
- [ ] Build the app: standard attention first (Q×K → scores → scale → mask → softmax → weighted
      sum of V), then each subsequent mechanism framed as a response to a specific prior limitation.
  - [x] **Act 0 — attention, built once** (2026-08-19). `site/` scaffold + `site/assets/app.css`
        (shared token system, lane colours, light/dark with a manual toggle) + `site/index.html` +
        `site/assets/act0-pipeline.js`. Six tokens ("the cat sat on the mat"), `d_k = 4` so `√d_k = 2`
        is mental arithmetic. **Q/K/V are authored, not random** — keys advertise grammatical role,
        queries advertise what the token wants — so the attention pattern is readable instead of
        noise; every displayed number is computed from those vectors at render time, none hard-coded.
        Six steps (Inputs · Score · Scale · Mask · Softmax · Weighted sum), a running formula with the
        active term lit, a clickable cell inspector showing the exact arithmetic per cell, output
        vectors as stacked bars, and the causal-mask toggle.
        **Verified numbers** (recomputed independently, not read off the widget): `sat` → `cat` =
        69.1% and its output is 0.691 *animate*; masked row 0 attends 1.0000 to itself; all rows sum
        to exactly 1; mask off → token 0 sends **89.4%** of its attention forward in time, **28.8%**
        of it onto `cat`. Swept all 6 steps × both toggle states and clicked all 36 cells: no console
        errors, no `undefined`/`NaN`, no horizontal overflow at 1440px or 390px.
        Two things caught in review and fixed: the masked triangle lost its hatch after softmax (it
        holds exact zeros, but it is still the same region — hatch restored, legend now reads
        "exactly 0 — was −∞"), and the step-5/step-3 notes asserted mask-on numbers and instructions
        regardless of the toggle — both bodies are now functions of state.
  - [x] **Act 1 — the two bills / the bill meter** (2026-08-19). `site/assets/act1-bills.js`.
        Context slider (1K–256K, powers of 2) + concurrency slider (1–64) + config (layers, head_dim,
        precision) + KV-head presets MHA/GQA-16/GQA-8/GQA-2/MQA, driving both bills at once. Shows
        the substituted formula with the byte total, a "double the context" line (compute ×4, cache
        ×2), and how many users fit in 80 GB.
        **Asserted against the lesson's published figures, not eyeballed:** yardstick = exactly
        `6,442,450,944` bytes = **6.44 GB**; 8 users = **51.54 GB**; GQA-2 = **1.61 GB**; MQA =
        **805 MB**. Swept all **326** control combinations — no NaN/undefined/Infinity anywhere, reset
        returns to the yardstick, no overflow at 1440px or 390px, dark mode checked.
        **Units are decimal GB throughout**, because that is what the lesson quotes — 6.44 GB is
        exactly 6.00 GiB, and mixing the conventions would be a silent 7% error on every comparison
        the app makes. The widget prints both so the difference is visible rather than hidden.
        Two things caught in review: (1) the first chart was linear-x/log-y, where a linear-in-T
        quantity *curves and appears to flatten* — it read as "growth is slowing", the opposite of the
        lesson's claim. Switched to **log–log**, where linear growth is a straight line of slope 1 and
        the four configs are visibly parallel climbs. (2) The GB formatter rounded 51.539… to "51.5",
        disagreeing with the lesson's own 51.54 — now two decimals through the 1–100 GB band.
  - [ ] Acts 2–5 (timeline, dossiers, Q2 answer, sources) — **blocked on the date audit being signed
        off**, since all four render from `data/mechanisms.js`.
- [ ] Write pros/cons for every mechanism: what it buys, what it costs, when you'd choose it.
- [ ] Answer Question 2 directly: what does the completed timeline show that a list wouldn't?
- [ ] Deploy (Netlify/Vercel/equivalent); verify the live link opens in an incognito window.
- [ ] Write README: live link, GitHub repo link, and the chronology's sources.
- [ ] Push to GitHub — decide standalone-repo vs. branch (see S7's incognito-link precedent below).
- [ ] Submit via the assignment form (Q1 link+repo, Q2 timeline-insight writeup, optional Q3 share).

## Resuming the plan review from a different session

The Lavish review is **not** tied to the Claude Code session that opened it. State lives on disk in
`~/.lavish-axi/state.json`, keyed by the artifact's absolute path, and holds the queued `prompts`
list — so feedback the user queues survives with nothing listening.

Whoever picks this up next, from any session:

```
cd /u/Vardhan/Courses/TSAI/ERA/V5/S8/assignment
npx -y lavish-axi .lavish/s8-date-audit.html        # the live one — phase 1, awaiting sign-off
npx -y lavish-axi poll .lavish/s8-date-audit.html   # collects everything queued since the last poll

npx -y lavish-axi .lavish/s8-app-plan.html          # phase 0, already approved (add --reopen if ended in-browser)
npx -y lavish-axi poll .lavish/s8-app-plan.html
```

Each file gets its own session key, so the two reviews are independent and both survive a session
end. The audit's buttons queue exactly one prompt: either "re-verify these rows" (with the ticked
rows + note) or "signed off — proceed to Act 0".

Notes: the server self-stops when idle, which kills the `127.0.0.1:4387` URL but not the state —
reopening restores the same session key. Review is local to this machine only. If the user wants to
read the plan away from this box, `npx -y lavish-axi export .lavish/s8-app-plan.html` writes a
standalone HTML (feedback then comes back as plain text, not queued prompts).

## Push destination — subtree split (confirmed convention for S8+)

Per the standing decision after comparing S6 vs S7 (2026-08-14): work happens only inside this
TSAI branch with normal commits. At submission time, split it out to a standalone public repo:

```
git subtree split --prefix=ERA/V5/S8/assignment -b s8-standalone
git push https://github.com/vpw/era-v5-s8.git s8-standalone:main
```

Keep `CLAUDE.md`/`AGENTS.md`/`TODO.md` in the split (matches S7, user-confirmed). No `gh` CLI or
credential helper on this machine — the user pastes a PAT at the push password prompt, so hand
over the push command rather than running it directly.
