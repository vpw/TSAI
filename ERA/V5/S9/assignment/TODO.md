# S9 TODO — Loss Functions & Output Heads

## ▶ STATUS (2026-08-28): notebook built and executed; README + push remaining

Due **Sat, Aug 29, 2026, 7:00 AM** · 1000 pts · resubmission allowed. Deliverable is a **GitHub
README.md link** (incognito-accessible) with the notebook and/or training logs in the same repo.
No deployment step this session — S7/S8's live-link half does not apply.

- [x] **Session verified** (2026-08-25). Lesson page's own heading reads "Session 9: Loss Functions
      & Output Heads" — matches this folder. **The link supplied was one character short**
      (`…x3t6b`, real id `…x3t6bd`), which renders a valid-looking but empty "Content coming soon"
      page rather than a 404. Real lesson URL:
      `https://axiom.theschoolofai.in/courses/cmq97i5kn032208o8xu5dab4q/sessions/cms9mhq4k7p2v9x3t6bd/lesson`
- [x] **Assignment captured** → `S9-assignment.md`, verbatim from the assignment page. Rubric tab is
      **empty** (no criterion rows, only the 1000-pt total), so there is no per-criterion checklist
      to optimise against — the brief's seven-plus-two numbers are the whole spec.
- [x] **Lesson captured** → `resources/s9-session.md`, all 24 sections verbatim (~55KB). Extraction
      note for future sessions: `get_page_text` truncates at 50,000 chars and the page carries a
      duplicate copy of the article plus 26 mermaid `<style>` blocks that inflate it further. What
      worked: strip the styles, remove the duplicate node, then `navigator.clipboard.writeText()` the
      article text and pull it with `xclip -selection clipboard -o` — verbatim, no truncation, and it
      never routes 55KB through the agent's context. Clipboard writes need the document focused, so
      click the page once first.
- [x] **Transcript captured** (2026-08-25) → `resources/s9-transcript.md`, 134KB, downloaded with
      the user's approval from Google Doc `1kyuLUjj19VaCthgABSzQu-1LnU-iADj_C-RY8XZN1Cg` via
      `export?format=txt`. Verified as the right session two ways: the file is dated **2026/08/22**
      (Session 9's availability date) and its first line is *"today's session is going to be on loss
      function and output heads"*. Page note: the studio recording was corrupted, so a cropped GMeet
      version was uploaded as the studio version — expect rougher audio artifacts than S8's.
- [x] **Widget extraction judged optional** (2026-08-25). The lesson's widgets (vocabulary/dot-product
      explorer, drag-a-logit gradient demo, SFT masking toggle) demonstrate formulas already given in
      full in the prose, and every number this assignment grades comes from the student's own run.
      Not a blocking prerequisite; revisit only if a specific widget value becomes load-bearing.
- [x] **Working set written** — `CLAUDE.md`, `AGENTS.md`, this file.

## ▶ DECIDED (2026-08-25) — settled with the user, don't re-open

- [x] **D1. Proxy configuration for now.** Measure on a small model; compute the real-config
      (`V=131,072`, `D=4,096`) numbers analytically alongside and label them as such. Revisit if a
      GPU becomes available. **Proxy:** `V = 10,000`, `D = 256`, 4 layers, 4 heads, `T = 512`.
      Untrained anchor at this `V` is `ln(10,000) =` **9.2103 nats**, perplexity **10,000**.
- [x] **D2. Tokenizer = the course's own S2 BPE**, `mr` variant (Marathi as 4th language, trained on
      en/hi/te/mr). 10,000 merges-based vocab, NFKC + Metaspace, round-trip verified. Copied into
      `assets/tokenizer.json` so the repo is self-contained on Colab. Only special token is `[UNK]`
      (id 0) — there is no `[PAD]`, so padding uses `[UNK]` as filler and is masked via label
      `-100`, which keeps `V` exactly 10,000 and the `ln V` anchor clean.
- [x] **D3. Skip the phantom Part 3.** Build Parts 1 and 2 as the assignment page specifies.
      Resubmission is open through Aug 29 if the instructor clarifies otherwise.

**Local environment (no GPU on this machine):** `.venv` with torch 2.13.0+**cpu**. Item 1g therefore
reports (i) exact **bytes retained for backward**, counted with `saved_tensors_hooks` — device-
independent, and the quantity the lesson's "16 GiB" refers to — and (ii) peak RSS sampled around the
region, as corroboration. On CUDA the same probe uses `torch.cuda.max_memory_allocated()`.

## ▶ Build pipeline (2026-08-28)

The notebook is generated, not hand-edited. Source of truth is `notebook_src.py` (`# %%` cells):

```
python tools/py2nb.py notebook_src.py S9_loss_functions_and_output_heads.ipynb
python tools/run_nb.py S9_loss_functions_and_output_heads.ipynb   # allow_errors=False -> writes results.json
python tools/dump_log.py S9_loss_functions_and_output_heads.ipynb logs/run.log
python tools/build_readme.py                                      # README.tmpl.md + results.json -> README.md
```

`build_readme.py` **exits non-zero on an unresolved placeholder**, so the write-up cannot contain a
number that the last run did not produce. Edit `README.tmpl.md`, never `README.md`.

A full CPU execution is ~15 minutes, most of it §1b's three training runs and Part 2's 300 steps.

## Part 1 — the loss harness (seven numbers)

Each item must be printed by a cell that actually ran. A number in the README with no cell behind it
is exactly the failure this assignment is testing for.

- [x] **1a. Shapes.** Print every tensor shape and name what each dimension is, one line each.
- [x] **1b. Verify the shift with token *strings*.** Inputs beside targets, decoded — not ids. The
      instructor's stated reason: an off-by-one is invisible in a wall of integers, and the wrong
      shift direction still produces a beautiful loss curve. Deliberately run the *wrong* shift once
      and show it looking healthy — that is the demonstration the warning is asking for.
- [x] **1c. Mask padding**, and show the count of contributing tokens changing.
- [x] **1d. Pack two documents** into one sequence, mask the boundary, show loss **before and after**,
      and explain the difference. (Unmasked, the model is trained to predict document 2's first token
      from document 1's last — a real cross-document bug.)
- [x] **1e. Perplexity**, showing an untrained model sits near vocabulary size. Expected anchor:
      loss `= ln(131,072) =` **11.784 nats**, perplexity **131,072** at the real `V`. State the
      prediction first, then show the run agreeing — or find the bug before going further.
- [x] **1f. Tied vs untied** head parameter counts on this configuration. Note in the write-up that
      **tying is unavailable to V5 in practice** — S7's input side is a byte codec plus one
      projection, so there is no `[V, D]` table to tie to. Report it as the counterfactual it is.
- [x] **1g. Peak memory: ordinary vs chunked cross-entropy**, both numbers and the ratio. The chunked
      version must be **written by hand** (lesson §10: compute logits for a block, take its loss,
      throw the logits away, next block). Measure with `torch.cuda.max_memory_allocated()` around a
      reset, not by estimating.

## Part 2 — one extra head

- [x] **2a.** Add a second output head predicting token `t+2`.
- [x] **2b.** Report both losses separately and their sum.
- [x] **2c.** Explain what happens to head 2's loss relative to head 1 over training. Expect it to
      sit **higher** and stay there — `t+2` is genuinely more uncertain than `t+1` — and say why that
      is the correct behaviour rather than a bug. Lesson §13 is the framing: MTP densifies the
      training signal and, at inference, the extra heads are drafts, i.e. speculative decoding where
      the draft model is the model.

## Ship

- [x] **S1. Run the notebook top to bottom in a fresh runtime.** "Runs top to bottom" is stated in
      the brief, so a notebook that only works with out-of-order cell execution fails the spec.
      Keep the executed outputs in the committed `.ipynb`.
- [ ] **S2. Write `README.md`** — the seven numbers from Part 1 and the two losses from Part 2, each
      next to the explanation the brief asks for, plus the configuration and tokenizer they were
      measured on. This is the graded artifact; the notebook is its evidence.
- [ ] **S3. Push** via subtree split (below), with the `.ipynb` and/or training logs included.
- [ ] **S4. Verify the README link in an incognito window**, then tick the form's checkbox honestly.
- [ ] **S5. Submit** the GitHub README.md link and record it back in this file.

## Standing conventions (carried from S6-S8, don't re-decide)

**Push destination — subtree split, not a nested `.git`** (settled 2026-08-14 after comparing S6 vs
S7). Work happens as normal commits inside this TSAI branch; at submission time:

```
git subtree split --prefix=ERA/V5/S9/assignment -b s9-standalone
git push https://github.com/vpw/era-v5-s9.git s9-standalone:main
```

Keep `CLAUDE.md` / `AGENTS.md` / `TODO.md` in the split (user-confirmed at S7). There's no `gh` CLI
or credential helper on this machine — the user pastes a PAT at the push password prompt, so hand
over the command rather than running it.

**Branch:** still on `s8-attention-variants` at scaffolding time. Cut `s9-loss-functions` before the
first S9 commit — S8 started on S7's branch name by accident and had to be renamed at submission.
