# Task

This directory is part of the assignments for the ERA V5 course of The School of AI (TSAI).
Specifically this is for the thirteenth session (S13).

`S13-assignment.md` has the exercise in full, the Axiom submission block (due **Sat 2026-09-26
07:00**, 1000 pts, one GitHub-README link field with a public-accessibility checkbox), and the
instructor's framing from class.

# Details

The session is **Distributed Training II: Model and Pipeline Parallel**. It covers tensor,
sequence, pipeline and context parallelism, and then reversibility (§16–§17), which is what the
assignment is about. A reversible residual rule such as midpoint,
p_{ℓ+1} = p_{ℓ−1} + 2h·f(p_ℓ), can be run backwards. The backward pass then rebuilds each layer's
input from its output instead of storing activations. The effects:
- activation memory becomes independent of depth
- compute rises ~30–50%
- the paper fits ~10× larger batches
- dropout must be 0, and the instructor adds no weight decay

Paper: Gal et al., *Reversing Large Language Models for Efficient Training and Fine-Tuning*,
arXiv 2512.02056. The exact equations for every variant are in `resources/s13-session.md` §16
Addendum.

**What the assignment asks for:**

1. Train a ~20M-parameter LLM for 50M tokens at a fixed batch size that fits (baseline).
2. Retrain with reversibility at the same batch size. Test at least midpoint and "Euler", and
   report which worked, judged from the loss trajectory.
3. Retrain with the chosen reversible variant at the maximum batch size that fits.
4. Report final loss, tokens/s and peak memory for every run, plus other findings (cost with vs
   without reversibility was agreed in class).
5. Submit a detailed README on GitHub. The repo must contain the ipynb notebooks.

As a capable agent, plan to:
1. Read `resources/s13-session.md` §16–17 and the Addendum, plus the transcript's assignment
   segment.
2. Settle the open decisions in `TODO.md`: GPU lane, dataset/tokenizer, model shape, which
   variant "Euler" means.
3. Build a nanoGPT-style model with a pluggable residual rule and a custom `autograd.Function`
   for the reversible stack.
4. Prove the reversible backward matches plain autograd before spending GPU time.
5. Run the training arms on a GPU through the S12 `tools/` pipeline so every README number comes
   from a cell that ran.
6. Write the README.
7. Ship the repo via subtree split, verify it anonymously, and submit before the deadline.

`TODO.md` tracks progress on these steps.

## References
Refer CLAUDE.md if it exists
