# Session 9 - Assignment

**Due:** Sat, Aug 29, 2026, 7:00 AM · **1000 points** · Resubmission allowed
**Source:** https://axiom.theschoolofai.in/courses/cmq97i5kn032208o8xu5dab4q/assignments/cmt3xb6yo0j9e09s1cp2vbydr
**Captured:** 2026-08-25. Verbatim from the assignment page (Brief + submission block).
The Rubric tab exists but is empty — no criterion rows, only the 1000-point total.

---

Session 9 - Assignment QnA
Brief
Rubric
History
Available
Due Sat, Aug 29, 2026, 7:00 AM
1000 points
Resubmission allowed
24. The assignment

One notebook, one loss harness, and one thing you have to get right by reading rather than by guessing.

Part 1: the harness
hidden = model(tokens)
logits = output_head(hidden)
loss = cross_entropy(
    logits[:, :-1].reshape(-1, vocab_size),
    tokens[:, 1:].reshape(-1),
)

Take that and make it correct and observable. You must:

Print every tensor shape and say in one line what each dimension is.
Verify the shift by printing the actual token strings, inputs beside targets. Not the ids. The strings. You will not catch an off-by-one in a wall of integers.
Mask padding and confirm the count of contributing tokens changes.
Pack two documents into one sequence and mask the boundary. Show the loss before and after masking it, and explain the difference.
Compute perplexity, and show that an untrained model sits near your vocabulary size. If it does not, find the bug before you go further.
Compare tied against untied head parameter counts on your configuration.
Measure peak memory for ordinary cross-entropy against a chunked version you write yourself. Report both numbers and the ratio.
Part 2: one extra head

Add a second output head predicting token t+2. Report both losses separately and their sum, and say what happens to the second head's loss over training compared with the first. Explain what you see.

What to submit

A Google Colab notebook moved to GitHub that runs top to bottom, and a short write-up with the seven numbers from Part 1, the two losses from Part 2

One warning, the same one as last time. A target shift in the incorrect direction can produce a beautiful loss curve. Print the strings. Many serious training bugs live in the few lines between the model output and the scalar, and they do not always raise an exception.

What are you submitting? GitHub README.md link.

Your submission
Not submitted
GitHub README.md link

Github repo should have the ipynb file or training logs as well to backyo your readme

1000 pts
I tested this link in an incognito window — it's publicly accessible (not private).
Add another link

0/1 answered · 1 will be left blank

Submit