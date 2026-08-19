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
  - [x] **Date audit SIGNED OFF** (2026-08-19, via the artifact's sign-off button): all 21 mechanism
        dates accepted as verified.
  - [x] **`site/data/mechanisms.js` — the single source of truth** (2026-08-19). 21 mechanisms + the
        LightningLM context marker, each with date/lane/source/url/paper/authors, the one-liner, the
        idea, buys/costs/pick, `verified` (how the date was checked) and an optional lineage `note`.
        Plain `window.MECHANISMS` — no modules, so the site still opens from `file://` with no build
        step. **Validated by script, not by eye:** chronological order, required fields on every
        entry, unique ids, valid lanes, and a cross-check of all 21 dates *and* sources against the
        signed-off audit artifact by parsing it directly — **zero mismatches**.
  - [x] **Act 2 — the timeline** (2026-08-19). `site/assets/act2-timeline.js`. Five lane bands
        (baseline / position / memory / compute / systems), markers at their **true proportional**
        position from 2016-11-07 to 2026-06-08, year gridlines, two lineage arcs (delta rule→DeltaNet
        "3 years dormant"; NTK-By-Parts→YaRN "YaRN builds on this one"). Circles = papers, diamonds =
        no paper (the Reddit post and the GitHub PR), dashed = the bonus mechanism, hollow = the
        context model. Deep links (`#rope`) open a dossier.
        **Labels move, markers never do:** a de-overlap pass pushes colliding labels apart and draws a
        leader line back to the dot — necessary because NTK-Aware and NTK-By-Parts are 9 days apart,
        roughly 2px at this scale. Readability must not be bought with positional dishonesty.
        **The "even spacing" toggle is the bonus question made interactive** — it collapses the
        timeline into what a plain ordered list shows, and the note underneath rewrites itself to name
        exactly what was lost. This is the strongest single asset for answering Q2.
        Two bugs caught: context marker stroked **white on white** (the `.dot` class rule beats an SVG
        presentation attribute), and a lane sublabel rendering at `x=-15`, off-canvas.
  - [x] **Act 3 — the dossier template** (2026-08-19, delivered with Act 2 since dossiers open in
        place below the axis). Uniform six blocks; the last is **"how this date was checked", present
        on every one** — asserted by opening all 22 dossiers and checking for the block. Prev/next
        walk the chronology.
  - [ ] Act 4 — the Q2 bonus answer written out, each claim linking back to a visible feature of the
        axis.
  - [ ] Act 5 — the sources table, generated from `data/mechanisms.js` (and the README's table from
        the same place).
- [x] **Date audit round 1 applied** (2026-08-19). The user flagged rows and asked for two specific
      things; the flags were not cosmetic. **The count is now 21 mechanisms, not 20.**
  - **Row 1 date CORRECTED: 2017-05-08 → 2016-11-07.** Was credited to ConvS2S (`1705.03122`) because
    that is what *Attention Is All You Need* cites — but AIAYN cites the **later** of two papers by the
    same group. The earlier one (`1611.02344`, Gehring/Auli/Grangier/Dauphin) states the mechanism
    outright: *"we add position embeddings to encode the absolute position of each source word"*, form
    `e_j = w_j + l_j`. Also chased its own antecedent, Sukhbaatar et al. 2015 (`1503.08895`): its
    "position encoding" is *fixed*, but its separate "Temporal Encoding" matrix is *"learned during
    training"* — a learned position embedding that indexes **memory slots, not token positions**.
    Different granularity, so the marker stays at `1611.02344` with the antecedent stated.
    **General lesson for the README: "the paper the famous paper cites" is a heuristic, not a proof.**
  - **Row 13 SPLIT into 13 + 13b** (user: "Use two"). NTK-By-Parts = **2023-07-07**, and it is not a
    Reddit post — YaRN's bibliography points at a **GitHub PR** (`jquesnelle/scaled-rope#1`, now
    `jquesnelle/yarn#1`). Dated from GitHub's own API, two independent timestamps: PR opened
    `2023-07-07T20:40:33Z` by `bloc97`, first commit authored `2023-07-07T20:24:12Z`, merged 07-09.
    Stronger evidence than row 13's Wayback route. **YaRN builds on 13b, not 13** (2309.00071 §3.2).
  - **Rows 4, 6, 7 stand — caveats are now quotations from the papers, not my characterisations.**
    (4) Sparse Transformers' §2 Related Work credits Image Transformer (`1802.05751`, **2018-02-15**)
    for "blocks of local attention" and claims breadth, not primacy — this was the user's explicit
    "synopsis of Related Work" request. (6) Longformer says Sparse Transformer "uses a form of dilated
    sliding window"; its real novelty is **global attention tokens**. (7) **Strongest flag** — Shen et
    al. `1812.01243` (**2018-12-04**) reached linear complexity **19 months earlier** and Katharopoulos
    et al. call it "concurrent". Row holds only because the lesson's argument needs the *causal*
    fixed-state recurrence, and a full-text scan of `1812.01243` finds exactly **one** hit for
    "causal"/"autoregressive" in the whole PDF — inside a bibliography entry.
  - **Open question put back to the user:** a *third* community method exists, "Dynamic NTK" by
    **emozilla** (different author), cited separately in YaRN. Not plotted — the instruction was "use
    two", and Wayback's earliest capture of its URL is 2023-10-11, months late, so it cannot be dated
    to the standard used everywhere else here. Chase it and add 13c, or leave it in dossier text only?
  - New papers pulled into the library this pass: `1611.02344`, `1812.01243`, `1503.08895`.
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
