# VOX-034 — concept primer: memory, arithmetic, and questions asked in bad faith

The ticket, as scoped on board task 1887: a turn should understand a follow-up that refers to an
earlier one; it should compute a figure when the excerpt gives a formula rather than an answer; and
it should not be led by a question built to extract a wrong answer. Written before the
implementation. Where this and the code disagree, the code and `ARCHITECTURE.md` win and this file
is wrong.

Everything before this ticket treated a turn as the unit of work. `nlu.messages()` is
`[system, transcript]`, `answer.messages()` is one user message, `state.build()` sees one
transcript. That was not an oversight — it is what made VOX-031's numeric guard and VOX-033's gate
reproducible, because a stateless turn has no history to be wrong about. This ticket spends that
property deliberately, in three directions at once, and the sections that matter most below are
about paying for it rather than about the features.

## 1 · A follow-up fails at retrieval first, and generation second

*"And for paternity?"* after a question about casual leave is the case. It never reaches a model
that could have understood it.

**Why here:** `turn_reply()` hands the bare transcript to `retrieve()`. `tokenize()` drops
stopwords, so the query is `["paternity"]` — one term for BM25, a two-word fragment for the
encoder. It will usually miss `RETRIEVAL_SCORE_FLOOR`, drop to the plain-reply path, and *that*
path has no history either. Two failures, one cause.

**Why it matters:** it decides where the fix goes. Conversation memory in the answer prompt would
not have helped — the excerpts were never fetched. The state has to reach **retrieval**, which is
upstream of every prompt in this repo.

**Pitfall:** measuring this as a generation problem. A dev case whose follow-up happens to clear
the floor on its own words proves nothing about the mechanism, and it is the easy case to write
by accident.

## 2 · Query rewriting is a retrieval-time repair, not a memory feature

Two shapes, and the cheap one is cheap because of a measurement.

**Why here:** retrieve on the raw transcript; only on a floor miss, retry once with the previous
question prepended. Dense search over the whole corpus is 0.009 ms/query measured
(215×384 float32 in `runs/embeddings.npz`, 100-iteration warm mean) and BM25 is arithmetic, so the
retry costs nothing on the common path. The alternative — an LLM rewrite every turn — is +1 call at
the 420–480 ms `t_llm` the NIM arm shows in `runs/turns.jsonl`, against a stage whose median
`t_retrieval_ms` is 41.5 ms over 131 records.

**Why it matters:** the miss-triggered retry is *self-limiting*. It can only rescue a turn that was
already going to be plain, so it cannot change the answer to a question that already worked. That
property is what lets `make gate-poc` stand as a no-regression check rather than needing
re-measurement.

**Pitfall:** rewriting unconditionally because it is simpler to reason about. It puts prior policy
text next to a new question and re-opens every VOX-031/032/033 number.

## 3 · History belongs on the plain path, not in the grounded prompt

`answer.messages()` stays single-shot. This is the one design line the ticket must not cross
casually.

**Why:** the numeric guard is a set difference against *the excerpts retrieved for this question*.
Put the previous turn's answer in the prompt and every figure in it becomes ambient context — the
model can restate a number retrieved for a different question, and `ungrounded_numbers()` cannot
tell, because it only ever sees `hits`.

**Why it matters:** the guard is the only mechanism in this repo that catches a fabricated figure
in code rather than by asking. Widening its blind spot to buy conversational polish trades the
strong protection for the weak one.

**Pitfall:** assuming "it is only the last turn" is a small amount of context. One prior grounded
answer is exactly the payload most likely to contain a plausible number that is wrong for *this*
question.

## 4 · Grounding constrained the facts and never constrained the arithmetic

This is the lesson the ticket inherits, and it is already written down in
`prompts/answer_from_source_v2.md`'s front matter.

**Why:** asked *"base pay of 10,000 and PL balance of 12, how much would my leave encashment be"*,
v1 answered *"you will be paid 12,000"* — cited, fluent, and not a number any reading of the
formula produces. Nothing in v1 forbade arithmetic. Every *fact* it used came from a document; the
*number it produced from them* did not, and grounding had nothing to say about the difference.

**Why it matters:** it is why this ticket cannot be a prompt edit. v2 added *"Never calculate a
figure"* and that was measured to be insufficient on its own — `ungrounded_numbers()` exists
because asking was not enough. Reversing the prohibition means replacing the enforcement, not
removing it.

**Pitfall:** reading v2's rule as timidity to be edited away. It is the residue of a measurement.

## 5 · The model must not do the arithmetic it is asked for

The mechanism: the model emits the formula and its operands as JSON, Python evaluates it.

**Why:** an LLM's arithmetic is *sampled*, not computed. The same question at temperature 0.3
returned two correct refusals and one invented figure across three runs, which is why
`ANSWER_TEMPERATURE` is 0.0. Determinism makes a wrong answer reproducible; it does not make
arithmetic correct. Extraction, by contrast, is a task these arms are reliable at, and
`state.build()` already does it with `response_format: json_object`.

**Why it matters:** it keeps the guard's guarantee intact instead of trading it. No free-floating
number ever reaches TTS — the only numbers that do are ones with a traced derivation.

**Pitfall:** `eval()` on a model-supplied string. A whitelisted AST walk over a fixed operator set
is the honest version: the model is a remote service, and its output is untrusted input.

## 6 · Provenance becomes a graph, and that is where the guard can be lost

`ungrounded_numbers()` is a set difference today: in the excerpts, or refuse. It has to learn a
third state — *the result of a checked computation whose every operand traces to an excerpt or to
the person's own words*.

**Why the direction of the widening matters:** `src/answer.py` currently holds that a number the
person supplied is **not** grounded by having been asked, because "a figure the person themselves
supplied is exactly what a fabricated calculation is built out of". That rule gets **narrowed** to
"legitimate only as an operand inside a checked derivation" — never dropped. If a user-supplied
number legitimises a result on its own, then any number legitimises any result, and the guard is
decorative.

**Why it matters:** the existing asymmetry in `numbers_in(parts=True)` — generous about what a
document contains, strict about what a reply asserts — is the same principle. Keep the direction.

**Pitfall:** letting the model report both the operands *and* the result, then checking only that
the operands trace. The arithmetic is still the model's and the check is theatre. Python must
produce the result, and a mismatch against a model-stated result is a refusal, not a correction.

## 7 · A false premise is a question you must not answer as asked

*"How do I claim my 30 days of paternity leave?"* when the policy states five. Retrieval finds the
right chunk. Nothing then requires the model to contradict the question.

**Why here:** the answer prompt says *"Answer the question that was asked"* and *"Do not summarise
the excerpts"* — both correct instructions that point the wrong way on this input.

**And the numeric guard does not help, which is the finding that makes this category urgent.** The
first draft of this section assumed "30" appeared in no excerpt, so a reply repeating it would
refuse. Checked against `runs/chunks.jsonl`: `30` appears in **seven** chunks, and one of them is
`leave-policy:p12` #15 — *"Paternity leave should be planned in advance at least 30 calendar days
prior"* — the chunk immediately adjacent to the entitlement chunk that says **five** calendar days.
So a paternity question retrieves both, and *"you may claim your 30 days of paternity leave"* passes
`ungrounded_numbers()` with every number traced. There is currently **no** protection on this case.

**Why it matters:** it generalises past false premises. `ungrounded_numbers()` asks whether a number
appears *anywhere in the bundle*, not whether it is attached to *the fact being asserted* — the same
coarseness §6 has to fix for computed figures. A number can be perfectly grounded and still be the
answer to a different question. Note that the contradiction trap in §8 exploits this identically:
`24` is in the corpus as the privilege-leave accumulation cap (`leave-policy:p7` #6), so agreeing
that twelve days becomes twenty-four over two years also passes the guard.

**Why the response shape is new:** it is the one trick category where the correct behaviour is
neither answering nor refusing — a refusal here is indistinguishable from "not in the documents",
which is a different and misleading claim. It needs a third shape: correct the premise from the
excerpt.

**Pitfall:** over-correcting into an assistant that argues with the premise of every question. The
dev set needs ordinary questions alongside the loaded ones, and a regression on those is a failure.

## 8 · Sycophancy: history makes the contradiction trap more dangerous, not less

*"You just said twelve days, so over two years that is twenty-four?"* is only visible once history
exists — and history is also what makes agreeing easy.

**Why:** a model that can see it said "twelve days" is *more* likely to accept an inference built on
it. The trap smuggles in arithmetic (§5) and a leading confirmation at once. Note that
`confirm.YES_WORDS` already contains `"correct"`, so a question ending *"…, correct?"* has an
adjacent failure mode in the confirmation leg.

**Why it matters:** this is the intersection of all three asks in the ticket, so it is the case that
fails if any one of the three mechanisms is weak. It belongs in the dev set for that reason, not
because contradiction traps are common.

**Pitfall:** expecting the fix to be a prompt clause. The reply has to be **re-grounded** —
retrieve for the new claim and answer from that — rather than reasoned about in context.

## 9 · Three numbers, and no average

Cross-queries, computed figures and trick questions get separate gate outputs, plus `make gate-poc`
unmoved as a no-regression check.

**Why:** an aggregate hides a regression in one component behind a gain in another, and these three
have different costs when wrong. A missed follow-up is a bad turn. A wrong computed figure is
someone told the wrong number about their own pay. Those do not belong in one mean.

**Why the figure gate is scored twice:** accuracy on the computable cases *and* refusal rate on
cases where an operand is missing. A system that computes eagerly scores well on the first and
badly on the second, and only the pair distinguishes it from one that is actually careful.

**Pitfall:** a percentage on n=8. Report the fraction and the cases, the way VOX-033 refused to call
2/2 a rate.

## 10 · Eval-before-implementation, because blind labelling is gone

`CLAUDE.md` seals `evals/heldout/` for the Evaluator so that whoever is tuning cannot see the
target. Both sides of this ticket are being written by the same agent, so that protection does not
exist here.

**Why:** if the implementation lands first and the cases second, the cases get written to fit what
already works, and the gate can only pass. Committing the dev cases first, in their own commit, is
the cheap substitute — the implementation is then fitted to cases that already existed, and
`git log` shows which came first.

**Why it matters:** it is the same reasoning that keeps `retrieval_floor_queries.json` and
`pdf_queries.json` in separate files, one step weaker. `pdf_queries.json` already states this limit
in its own `_note`; these new sets must too.

**Pitfall:** adding a case *because* the gate failed on it. That converts the gate into a record of
what already passes.
