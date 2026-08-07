The assignment

Draft the mixture-and-curriculum plan for V5 as a written specification, specific enough to defend. It must:

- State a share of the fixed token budget for every capability slot (general web, code, reasoning, agentic, long-context, Indic, etc.)
- For the Indic slot, state the split across the verified, unverified, translated, and synthetic tiers rather than a single headline number
- Name the agentic, reasoning, and long-context slots explicitly and point each one at the datasets from the inventory that will fill it
- Fix the protected always-on floor that the selector (OPUS) is not allowed to cross
- Declare the anneal reserve held back for the cooldown phase
- Lay out the difficulty and reasoning-length bands with a concrete example for each
- Commit to justifying these numbers through small proxy runs at the one-billion and three-billion parameter scale before any of them is trusted at full scale — a data decision is a hypothesis until a cheap experiment has tested it

Alongside the plan, the cleaning work continues toward the cumulative target, now aimed at the slots the mixture shows to be starved.

Evaluation strategy

The plan is evaluated on how well it would hold up if a reviewer sat across from you and pushed on every number — the grade rests on the quality of your reasoning and the evidence behind your choices. A tightly argued short plan scores well; padding earns nothing.

A strong submission:
- Gives a defended share of the budget to every capability lane
- States the Indic split across verified, unverified, translated, and synthetic tiers
- Ties each lane back to the benchmarks it's meant to win, so a reviewer can see why each number is what it is
- Sizes every lane against the real supply from the inventory, and says plainly where a share can only be reached by repeating data or by generating it (a large share handed to a lane with almost no real data behind it loses marks for wishful accounting)
- Fixes the protected floor and declares the anneal reserve
- Lays out the difficulty and reasoning-length bands with a real example at each level

What separates a good plan from an excellent one is whether it is written as a testable hypothesis:
- Highest marks: specifies a concrete proxy experiment at the 1B or 3B scale and names the metric that would confirm or refute the mixture
- Very highest marks: actually runs that proxy and brings the numbers back

The plan is reviewed in the open against these criteria, and only once the team has met the data-gating threshold — a mixture is only as trustworthy as the cleaned and documented tokens standing behind it.

Submission

Link to a GitHub repo README.md (not a Netlify app or widget this time) where the plan can be evaluated.
