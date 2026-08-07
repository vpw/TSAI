# S5 TODO — Data Mixtures and Curriculum

See `NEW-SESSION.md` for full state and the prompt to resume with.
`S5-assignment.md` = the task, `resources/s5-session.md` = lesson, `resources/s5-transcript.md` =
live-class transcript, `resources/s5-widget-data.md` = extracted widget content.

- [x] Extract real widget data from the session page — all 9 widgets in
      `resources/s5-widget-data.md`. Widgets turned out to be standalone pages at
      `/widgets/widget_N_*.html`; driving those directly beats screenshotting the lesson page.
- [x] Draft budget share per capability slot — in `scripts/ledger.py`, computed into
      `SUPPLY_LEDGER.md`. Budget 2.4T is *derived* from the candidate-pool constraint (max
      fundable is 2.72T), not chosen.
- [x] Indic split across verified / unverified / translated / synthetic — derived from real
      supply, because the widget's own default split (20% translated) is unsatisfiable against
      a 5B translated tier.
- [x] Name agentic, reasoning, long-context slots against real datasets from the inventory —
      `data/inventory.json` holds all 32 datasets with samples, tokens, licence, tier.
- [x] Fix the protected always-on floor — 10% of every batch (Indic 7 / agentic 2 / reasoning 1)
      bypassing OPUS, plus mixture floors Indic ≥12%, agentic ≥2%.
- [x] Declare the anneal reserve — 2% of budget = 48B tokens, composer's V5 anneal preset.
- [x] Difficulty and reasoning-length bands — B0–B5 × short/medium/long/ultra (a 6×4 grid, not
      two independent ladders), with the widget's measured token counts and solve rates.
- [x] **Set up AWS access** — profile `AWS-ESS`, instance `i-025fb7b65d7e3460e`
      (g4dn.2xlarge, one T4). Lifecycle in `proxy/scripts/aws_gpu.sh`.
- [x] **Run the proxy ablation** — 8 arms, 3.9 GPU-hours, per-domain bits-per-byte with the
      decision rules committed beforehand (`proxy/HYPOTHESES.md`, commit `920006f`).
      Results in `proxy/RESULTS.md`. Instance stopped and verified `stopped`.
      Floor confirmed; mixture-vs-web-heavy refuted on its general-web clause; four-tier
      Indic no signal; transition stability not reproduced.
- [x] **Cleaning top-up** — 176.9M clean tokens at 92.01% retention across 9 Indic
      languages; cumulative 220.4M against the 320B target. Found and fixed the
      stop-word fallback that was discarding ~95% of kan/guj/mal/pan/ory.
      See `topup/TOPUP_REPORT.md`.
- [x] **Write `README.md`** — the deliverable, committed to `s5-mixture-curriculum`.

Remaining: **push** — the submission is a public GitHub link, so it needs explicit
confirmation. Decide whether `NEW-SESSION.md` (internal working notes) belongs in a
public repo before pushing.
