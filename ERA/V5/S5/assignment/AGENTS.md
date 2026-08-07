# Task
This directory is a part of the assignments for the ERA V5 course of The school of AI (TSAI).
Specifically this is for the fifth session (S5).

The S5-assignment.md file lists the exercise and its evaluation rubric in full — please refer
to that for the details. Unlike prior sessions, the final deliverable here is a **GitHub repo
README.md** with the written plan — not a Netlify widget/app.

# Details
The session covers data mixtures and curriculum design: how to split a fixed pre-training
token budget across capability slots (general web, code, reasoning, agentic, long-context,
Indic, etc.), how to further split the Indic slot across verified/unverified/translated/
synthetic tiers, protected floors and anneal reserves, difficulty/reasoning-length curriculum
bands, and validating the resulting mixture with small proxy runs before trusting it at full
scale. The session writeup is in resources/s5-session.md and the live-class transcript is in
resources/s5-transcript.md.

The session page has interactive widgets (mixture composer, benchmark explainer, dataset
inventory, OPUS live view) whose real live content (actual token/sample counts, defaults,
worked examples) is not captured in the session writeup prose — extract that data first
(the extract-widget-data skill, same approach used in S4) before drafting numbers, so the
plan's figures are grounded rather than invented.

The plan should also connect to the ongoing cleaning work from S4 (../../S4/assignment/),
since the assignment says the cleaning work continues toward the slots this mixture shows to
be starved, and the evaluation rubric penalizes handing a large budget share to a lane with
no real supply behind it.

As a capable agent, plan to: (1) extract real widget data, (2) draft the budget-share numbers
per slot grounded in the dataset inventory, (3) work through the Indic tier split, protected
floor, anneal reserve, and difficulty/reasoning-length bands, (4) propose (and ideally run) a
1B/3B proxy experiment, (5) write it all up as a tight README.md and push/share the repo link.
TODO.md tracks progress on these steps.

## References
Refer CLAUDE.md if it exists
