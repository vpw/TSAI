# S7 TODO — Embeddings and Model Internals

`S7-assignment.md` = the task (5 open embedding-design problems, pick one), `resources/s7-session.md`
= lesson writeup, `resources/s7-transcript.md` = live-class transcript.

- [x] Set up session scaffolding: `CLAUDE.md`, `AGENTS.md`, `resources/s7-session.md`,
      `resources/s7-transcript.md`.
- [ ] Pick which of the 5 problems to pursue (see CLAUDE.md for the list). Needs a decision —
      surface tradeoffs to the user rather than picking unilaterally, since the instructor is
      evaluating these as candidate paper ideas.
- [ ] (Optional, only if the chosen problem needs it) Extract exact widget defaults via the
      `extract-widget-data` skill — e.g. the byte-budget lab's default collision examples, or the
      Kronecker microscope's exact grid/projection shapes — if the session prose's numbers aren't
      sufficient to ground the chosen problem's baseline comparison.
- [ ] Design the proof-of-concept: what does a "small transformer trained on this" look like for
      the chosen problem, what's the toy task, what's the success metric.
- [ ] Implement the embedding scheme + small transformer + training loop.
- [ ] Run the training/eval and capture the evidence (loss curves, worked examples, ablation vs.
      a baseline — likely vs. standard Kronecker or a dense table, matching the session's own
      "control arm" framing).
- [ ] Write the README (state which problem, how the solution works, how it's proven). Optional:
      a small webapp for graphs/animations.
- [ ] Push to GitHub, confirm the link is publicly accessible in an incognito window (per the
      submission form's explicit requirement), and share the link.
