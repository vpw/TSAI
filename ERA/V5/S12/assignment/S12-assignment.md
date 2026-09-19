# Session 12 - Assignment

Source: Section 14 of the Session 12 lesson page ("Distributed Training I, Data Parallel
and ZeRO"), captured verbatim. The Axiom Assignments tab entry ("Session 12 Assignment
QnA") carries the identical text — unlike S11, the graded entry already exists at scaffold
time.

> Work with your agents and create a simple 32 virtual GPUs (can be your CPU threads or
> Colab GPU). Then write a demo model that runs on top of these. Simulate ZeRO1, ZeRO2,
> and ZeRO3. Show how the memory and computation chanages.
>
> Submit your ipynb notebook, and GitHub Repo link with a detailed README that explains
> that YOU have understood these concepts (and not your agent).

(The typo "chanages" is the assignment's own.)

## Submission block

Captured from the Axiom Assignments tab (2026-09-17):
`https://axiom.theschoolofai.in/courses/cmq97i5kn032208o8xu5dab4q/assignments/cmu0lfitf0kaw0fn09qpvj3s1`

- **Status:** Available — **Not submitted**
- **Due:** Sat, Sep 19, 2026, 7:00 AM
- **Points:** 1000
- **Resubmission:** allowed
- **Submission fields:** one — **GitHub Link** (1000 pts), with the checkbox *"I tested
  this link in an incognito window — it's publicly accessible (not private)."* The form
  shows "0/1 answered · 1 will be left blank", so the single GitHub link is the whole
  submission; the notebook is delivered inside that repo, not uploaded separately.
- **Rubric tab:** present but renders no content beyond the brief (checked 2026-09-17) —
  no separate rubric criteria published.

## The instructor's own framing (live class, 2026-09-12)

From `resources/s12-transcript.md`, closing minutes — this expands the two-sentence brief
in ways the written assignment does not:

- *"Work with the agent and create a simple 32 virtual GPUs. you can easily do it on your
  own computer. It's a small program that works can be the GPU on the collab also."* — CPU
  threads on this machine are explicitly sanctioned; no real multi-GPU hardware needed.
- *"Then write a demo model that runs on top of these 32 virtual GPUs. Simulate 01 02 03
  and show how the memory and computation changes anyways your cloud is going to do it.
  But I want you to see the results yourself. I want you to see those results and **ask it
  to make sure that it matches what zero does**."* — the simulation is to be *validated*
  against real ZeRO's arithmetic, not just built.
- *"submit your IPYB notebook and your GitHub rep and explain in detail that you do
  understand zero stages and **you understand the pros and cons of each of the state** that
  you have."* — the README owes a pros/cons treatment per stage, not only measurements.
