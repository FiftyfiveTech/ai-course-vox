# VOX-032 — concept primer: retrieval inside a live turn

The ticket: after STT, retrieve; if chunks come back, answer from them (VOX-031); if not,
reply as before. Written before the implementation. Where this and the code disagree, the
code and `ARCHITECTURE.md` win and this file is wrong.

VOX-029/030/031 built the RAG pieces and drove them from `scripts/ask.py` — a script, one
question, typed. This ticket is the only one that puts them inside a turn that a person
*speaks*, and everything below is a consequence of that move.

## 1 · Routing on retrieval, not on intent

The turn asks retrieval first and decides from what comes back: hits → grounded answer, `[]`
→ plain `nlu.reply`. No classifier decides "is this an HR question".

**Why here:** the classifier already exists and it is BM25 plus `RETRIEVAL_SCORE_FLOOR`.
That floor was *measured* (`make ask` with `--calibrate`, thirteen queries, the number sits
in the gap between answerable and absent), so it is a threshold someone can re-run and
argue with. An intent model would be a second, unmeasured decision in front of a measured
one — and it would be wrong in the expensive direction: a misrouted "hello" costs a plain
reply, a misrouted "how much casual leave" costs a hallucinated policy number.

**Pitfall:** treating the floor as a hyperparameter to nudge until the demo looks good. It
is calibrated on `evals/dev/retrieval_floor_queries.json` and re-measured, never nudged —
and the Builder never sees `evals/heldout/`.

## 2 · Retrieval is timed, but it is not the sixth field

`t_retrieval_ms` goes on the turn record next to the five VOX-003 fields, not inside them.

**Why:** `telemetry.STAGES` is what `TURN_FIELDS`, `stage_sum_ms` and `ok` are derived from,
and the phase gates read exactly those. Adding a sixth stage would (a) redefine "the
five-field split" that VOX-003's gate and tests assert, and (b) make `ok` false for any turn
that ran without an index — a turn that spoke fine. So retrieval is timed like a stage and
recorded like a fact about the turn.

It has to be timed *outside* `turn.stage("llm")`, because BM25 over 215 chunks is
milliseconds and the LLM call is seconds — folding one into the other is how a cheap stage
disappears and a model gets blamed for latency it did not spend.

**Pitfall:** building the index inside the turn. `retrieval.index()` is per-process work;
the loop warms it at startup beside silero and Kokoro, for the same reason — a load inside
the turn lands in a stage number and makes the split a lie.

## 3 · Two fallbacks, and they are different words

- **Retrieval miss** → the plain reply path. VOX has nothing grounded to say, so it behaves
  as it did before this ticket. The listener hears a normal answer to a normal question.
- **No index at all** (a clean clone: `sources/` is gitignored, so there are no chunks)
  → also the plain reply path, but announced once at startup, not per turn. `make demo`
  must still run for a person who has not put the corpus on their machine.
- **Model refusal** (chunks cleared the floor, they do not contain the answer) → the refusal
  sentence, spoken. Not a fallback. That is the answer.

**Pitfall:** collapsing the first two into silence. A demo that quietly stopped being
grounded because `runs/chunks.jsonl` was never built looks exactly like a demo where
retrieval found nothing — and only one of those is a working system.

## 4 · Provenance out loud, and what it does not claim

The turn record carries `sources: ["leave-policy:p4", ...]` and the console prints them
under the answer. The citation is the provenance of the *context*, not something the model
emitted — see the `src/answer.py` docstring. So it says what the answer was grounded in, up
to five chunks; it does not say which sentence the number came from.

**Why it matters in a voice turn:** nothing is read aloud except the answer. The sources are
for the person reading the terminal and for `runs/turns.jsonl` — a spoken "according to
leave-policy page four, chunk seven" is not how a colleague talks, and the prompt asks for
the document *named in prose* instead.

**Pitfall:** parsing the model's prose for the document name. Nothing does, and nothing
should — the local fallback arm is a 3B and a parse failure would land on the turn that was
already going badly.

## 5 · One stage sequence, not two

`src/loop.py` (live mic) and `src/harness.fixture_turn` (recording) already share the stage
sequence deliberately — the harness docstring says why: a third copy is how a comparison
ends up measuring a pipeline the loop does not run. The grounded path is one more thing that
could be copied, so it is not: both call the same function, and the harness only takes the
KB when it is handed one.

**Pitfall:** wiring the loop and leaving `scripts/compare_arms.py` on the old path.
VOX-013's per-arm table would then be timing a pipeline `make demo` no longer runs.

## 6 · What "grounded" is worth measuring as

`grounded` on the turn record is a bool per turn: did a model answer from documents. Summed
over a run it is the grounded-answer rate — which is VOX-033's gate, and it can only be
computed if this ticket writes the field. That is the reason the field exists here rather
than in the next ticket: a rate needs a denominator that was recorded while the turns
happened.

**Pitfall:** reporting the rate off a handful of turns and calling it a number. n=3 is a
sanity check; the gate says what n it needs.
