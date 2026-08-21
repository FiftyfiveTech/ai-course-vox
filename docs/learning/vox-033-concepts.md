# VOX-033 — concept primer: the number that closes the POC

The ticket: ten written queries, eight with an expected `file:page` and two deliberately absent
from the corpus. Print correct-source@3 and the refusal rate on the two, exit non-zero below the
floor. Written before the implementation. Where this and the code disagree, the code and
`ARCHITECTURE.md` win and this file is wrong.

VOX-029/030/031/032 built the grounded path and measured it on whatever query was to hand — thirteen
for the floor, one spoken question for the turn loop. This ticket is the first time the same stack is
scored against a query set someone wrote down on purpose, and everything below is a consequence of
that: a written set can be re-run, argued with, and be wrong in ways an ad-hoc query cannot.

## 1 · correct-source@k is a retrieval metric wearing a groundedness costume

`correct-source@3` asks: is the right `file:page` among the top three chunks? It says nothing about
whether the model then answered correctly from them.

**Why here:** it is the only one of the three numbers that isolates a single component. If the
grounded-answer rate drops, correct-source@3 is what tells you whether retrieval stopped finding the
page or the model stopped using it — two different fixes in two different files. `ARCHITECTURE.md`
already says this about citations: they are the provenance of the *context*, not something the model
emitted, so a retrieval metric is exactly what they can support.

**Why it matters:** it is measurable without a judge. Scoring "was the answer right" needs a human
or an LLM-judge and a second argument about the judge; scoring "was the page in the top three" needs
a label and a set membership test.

**Pitfall:** reading a high correct-source@3 as "the system answers well". A perfect retrieval score
with a fabricating model is the exact failure `answer.ungrounded_numbers()` was written for
(`ARCHITECTURE.md`, the "asking was not enough" section) — fluent, cited, and false.

## 2 · The two absent queries are the harder half

Eight answerable queries and two absent ones is not an 80/20 split of effort. The two are where the
system can do the thing that costs trust.

**Why:** a wrong answer to an answerable question is a bad answer. A confident answer to a question
the documents do not address is a fabrication with a citation attached. `evals/dev/retrieval_floor_queries.json`
records the rule this earned: a miss must be **HR-shaped and share corpus vocabulary** — "is a gym
membership reimbursed by the company" uses *reimbursed*, *company*, and reads like a real policy
question. "A miss written in unrelated words would clear any floor and prove nothing."

**Pitfall:** n=2 and treating the refusal rate as a rate. It has three possible values. It is
asserted at 2/2 not because two samples establish a rate but because one false answer out of two is
not a number to negotiate with.

## 3 · Why the gate set is a different file from the floor-calibration set

`evals/dev/retrieval_floor_queries.json` already holds seven answerable and six absent queries.
Reusing them would have been free. It is forbidden, and the file says so at the top of itself.

**Why:** those thirteen queries are what `config.RETRIEVAL_SCORE_FLOOR = 0.28` was *fitted to*. A
gate scored on them measures how well the floor was fitted, which is a number that can only go up
and cannot be wrong. Two files means tuning the floor cannot quietly move the gate's number.

This is train/test separation under a different name, and it is the weak version of it: both files
are `evals/dev/`, both visible to whoever tunes. The strong version is `evals/heldout/` — see §5.

**Pitfall:** adding a query to the gate set *because* the gate failed on it. That converts the gate
into a record of what already passes.

## 4 · A paraphrased refusal counts as an answer

`answer.is_refusal()` is equality against one constant string, deliberately — "not a parser", says
its docstring. So when a model says *"There is no mention of sabbatical leave in the provided
excerpts"*, that is a refusal in substance, not the `REFUSAL` string, and the turn records
`grounded: true`. `ARCHITECTURE.md` flags this as "a shape rather than a fixed bug", and
`retros.md` leaves it open, assigned here.

**Why it lands on this ticket:** the grounded-answer rate is a sum over that bool. A bool that is
true for a refusal makes the rate flattering in exactly the direction a gate must not be.

**Why the fix is gate-local:** widening `is_refusal()` would change what the live turn loop writes
into `runs/turns.jsonl`, which re-opens every VOX-031/032 number already published in
`ARCHITECTURE.md`. That is its own before/after measurement, not a side effect of writing a gate. So
the gate keeps its own list of refusal-shaped phrasings, counts them as refusals, and **prints how
many it caught** — the count is the disclosure that the production check is narrower than the gate's.

**Pitfall:** a substring list that grows until it matches real answers. "There is no cap on carry
forward" contains no refusal, and a pattern for *"there is no"* would refuse it. Every phrase in the
list has to name the *excerpts* or the *documents*, not just a negation.

## 5 · The POC has no held-out number, and the gate says so out loud

`evals/heldout/` is sealed as `heldout-v1` and holds zero document queries — its categories are
greet / entity / ambig / escalate / refuse. It is **not reopened** for this.

**Why that is the right call and still a real cost:** reopening a sealed set to add a new category is
how a held-out set becomes a dev set with extra ceremony. But it means every number this gate prints
was computed on queries the Builder can read. Nothing here bounds how much the implementation was
shaped by these ten cases.

So the limit is printed in the gate output, next to the numbers, "not glossed" — because a number
whose caveat lives in a design doc is a number that will be quoted without it.

**Pitfall:** the softer version of the same thing — quoting correct-source@3 in a status update
without the dev-only qualifier. `CLAUDE.md`: report numbers with the command that produced them.

## 6 · A gate that cannot reproduce its own number is not a gate

The grounded path samples at `answer.ANSWER_TEMPERATURE = 0.0` while the spoken reply keeps 0.3, and
this ticket is the reason. At 0.3 the same leave-encashment question returned two correct refusals
and one invented figure across three runs.

**Why it matters here specifically:** a gate that fails one run in three is worse than no gate — it
teaches everyone to re-run it. The determinism is not a preference about answer quality; it is what
makes a threshold assertion meaningful.

**Pitfall:** assuming determinism and never checking. The gate is run twice, and the second run's
numbers matching the first is part of the evidence — not an optional extra step.
