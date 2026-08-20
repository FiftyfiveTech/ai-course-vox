# Retro ledger

One entry per ticket, appended at ticket close — after the board comment, so the two say the
same thing and the in-repo copy survives the course.

Format:

```
## VOX-NNN — <title>  (YYYY-MM-DD, <name> as Builder|Evaluator)
Executed:   what actually landed
Deviations: where it differs from the ticket spec, and why
Numbers:    measured values + the command that produced them
Lessons:    what the next ticket should do differently
```

Rule, same as `STANDUP.md`: a number goes in here only if it appeared in a terminal.

## VOX-032 — Wire it into the turn loop  (2026-08-20, Vimal as Builder)

Executed:   Retrieval moved inside the turn. `answer.turn_reply()` is the one routing both turn
            loops call — `src/loop.py` (mic) and `src/harness.fixture_turn` (recording): retrieve
            after STT, chunks above the floor go through the VOX-031 grounded prompt, an empty list
            through the plain reply prompt. `t_retrieval_ms`, `retrieved`, `grounded`, `sources` and
            `top_score` on the turn record; the doc:page printed under the answer. `--no-kb` forces
            the pre-RAG path; a missing index is announced at startup and is not fatal. New:
            `make ground` and `tests/fixtures/casual_leave_question.mp3`, so a grounded turn is
            reproducible without a microphone. 16 new tests in `tests/unit/test_grounded_turn.py`.

Deviations: (1) Retrieval is *not* a sixth entry in `telemetry.STAGES`. `TURN_FIELDS`,
            `stage_sum_ms` and `ok` derive from it and the phase gates read exactly those, and
            `ok` would go false for a turn that ran without an index — a turn that spoke fine. It
            is timed like a stage and recorded like a fact about the turn.
            (2) The routing lives in `src/answer.py`, not in `src/loop.py` where the ticket puts
            it. Two loops need it and the harness docstring already says what a second copy costs.
            (3) The harness defaults `idx=None` rather than to the process-wide index, so
            `scripts/compare_arms.py` keeps timing the plain path — a grounded turn carries 5.1x
            the prompt and that cost belongs to the KB, not to the arm on the row.
            (4) Added a synthesised fixture to a folder whose README is about human recordings. Its
            provenance and the caveat (clean TTS audio says nothing about STT on real speech) are
            written into that README.

Numbers:    `uv run pytest tests/unit -q` -> 254 passed.
            `make ground` (5 grounded turns, 3 plain, real turns.jsonl/calls.jsonl):
              prompt_tokens 1381 grounded vs 269-283 plain = 5.1x
              llm call 367-844 ms grounded vs 383-509 ms plain (ranges overlap)
              t_retrieval_ms 0.5-1.0 ms; t_tts_ms 4048-5894; ttfa 5568-7825 ms
              grounded turns answered "12 working days of Casual leave" from
              leave-policy:p4,p3,p7 — top score 0.402, 3 chunks in context
            `uv run python scripts/ask.py "is there a canteen subsidy…"` -> plain reply path, 0
            chunks, no grounded claim.

Lessons:    Two findings the wiring surfaced, both written into ARCHITECTURE.md and neither fixed
            here, because both change a calibrated number or a versioned prompt:
            (1) A user's own numbers can push a covered question under the floor. Same chunk
            (`separation-policy:p13`), same raw BM25 15.41, score 0.331 without the numbers and
            0.250 with them — `10`, `000`, `12` are charged near-maximum IDF in the query ceiling
            and contribute nothing to the numerator. Spoken questions contain numbers; this needs
            re-calibrating on dev, not nudging.
            (2) Grounding does not constrain arithmetic. With the floor dropped, the same question
            retrieved the right chunk — which gives no per-day rate — and the arm answered "you
            will be paid 12,000", cited. `answer.py` guards claims that were never retrieved, not
            a calculation invented on top of what was.
            (3) STT is the third failure on that question: "leave encashment" came back as
            "leaving cashment", which loses the one high-information term before retrieval ever
            runs. That is VOX-021's ticket (vocabulary biasing, measured before and after) and it
            now has a concrete case to be measured on.

## Hybrid retrieval + answer prompt v2  (2026-08-20, Vimal as Builder)

Not a ticket of its own — done on the VOX-032 branch, by decision, after a live `make demo` turn
returned a wrong answer. Recorded here because it changed a calibrated number and a versioned
prompt, and both need to be reproducible.

Executed:   `src/embeddings.py` (a stage module like stt/tts: BACKENDS + LOADERS, two encoder arms
            named by HF repo id, `--embed` to pick one), `arms.embed()`, an `embed` stage in the
            registry, and a dense half in `src/retrieval.py` fused with BM25 by reciprocal rank.
            Chunk vectors cached to `runs/embeddings.npz`, keyed by a fingerprint of the chunk text.
            `make index` builds them, `make floors` re-measures both floors, `make setup` fetches
            the encoder. New prompt `answer_from_source_v2.md`. 21 tests in
            `tests/unit/test_hybrid.py`, plus an autouse conftest fixture that makes real weights
            unreachable from the suite.

Deviations: (1) Fused by rank and not by score — a BM25 fraction and a cosine are different units.
            (2) The lexical half **abstains from the ordering** when its own floor says it found
            nothing. Without that, the answering chunk was dense rank 1 and still missed the top
            five: two lukewarm ranks beat one excellent one plus one terrible one. The dense half
            does not abstain, because no dense signal separates (below).
            (3) No fallback arm for `embed`. A second encoder answers in a different vector space
            from the cached vectors, and every cosine would be meaningless — silently.
            (4) `TurnTimer.arms()` now accepts a registered arm stage that is not one of the five
            timed ones, so `embed_model` lands on the turn record without becoming a sixth field.

Numbers:    `uv run pytest tests/unit -q` -> 286 passed.
            `make index` -> 215 chunks encoded in 25.1s (117 ms/chunk), 215x384, norms min 1.0000
            max 1.0000.
            `make floors` (13 dev queries):
              lexical  answerable min 0.320 | absent max 0.234  -> separable, floor 0.277
              dense    answerable min 0.676 | absent max 0.738  -> NOT separable
              top-1 in an expected document: lexical 7/7, dense 7/7
              union rule at 0.277/0.65 -> 9/13 routed correctly by retrieval alone
            End to end through the grounded prompt: 13/14 (the 14th is the paternity question,
            which now answers "5 calendar days ... for your new-born" from leave-policy:p12).
            The failing query before this: answering chunk at lexical rank 110 of 137; after,
            rank 1. `t_retrieval_ms` 1 ms -> 83-117 ms (one encoder pass on CPU), against
            t_stt 271-370 ms and t_tts 4.0-5.8 s.

Lessons:    (1) The floor was never the safety mechanism — the prompt is. Three dense signals were
            tried as refusal detectors (cosine, z-score, margin, gap to 6th) and none separates on
            this dev set, so more absent questions now reach the model. That is affordable only
            because `answer_from_source_v2.md` refuses them, which is measured, not assumed.
            (2) A negative measurement is the useful one. "NOT separable" printed by `make floors`
            is what stopped a 0.674 floor with 0.002 of margin from being called a threshold.
            (3) Grounding does not constrain arithmetic. v1 answered "you will be paid 12,000" from
            an excerpt containing a formula and no such number. v2 forbids computing a figure.
            (4) A paraphrased refusal is counted as an answer: the model said "There is no mention
            of sabbatical leave in the provided excerpts", which `is_refusal()` does not match, so
            the turn logged `grounded: true`. VOX-033's rate has to handle that — still open.
            (5) The dev set is 13 queries written before any of this was known. Every question in
            this retro would have caught something and none of them is in it.

## Numeric guard + deterministic grounded answers  (2026-08-20, Vimal as Builder)

Same branch, same session, one more live-turn failure. `answer_from_source_v2.md` forbade computing
a figure; the model did it anyway.

Executed:   Temperature became a per-call option (`arms.llm(temperature=...)` threaded to the
            backend through a new `options` channel in `_call`/`_dispatch`), and the grounded path
            asks for 0 while a spoken reply keeps `nlu.TEMPERATURE = 0.3`. Then the prompt's own
            rule — "numbers you may say are the ones written in the excerpts" — enforced in code:
            `answer.numbers_in()` / `ungrounded_numbers()`, and a reply stating a figure that
            appears in no excerpt becomes the refusal, with the suppressed text on stderr. 14 tests.

Deviations: The guard refuses rather than repairing. Stating the rule instead of the number would
            be the nicer answer and is what the prompt asks for; a refusal is what can be
            guaranteed. Chose the guarantee.

Numbers:    `uv run python scripts/ask.py`-driven probe, "if I have a base pay of 5000 rupees and 20
            privileged leave..." asked 3x at temperature 0.3 -> 2 correct refusals, 1 "You will get
            12 rupees in leave encashment." At temperature 0 -> consistently wrong instead:
            "since you have 20 privileged leave, which is 16 days more than the 24-day cap, you
            will get 16 days". With the guard -> refuses, 3/3, caught on the caller's own 20.
            Dev set end to end: 13/14 -> **14/14**, no answerable query refused by the guard.
            `uv run pytest tests/unit -q` -> 300 passed.

Lessons:    (1) A prompt instruction is a request, not a constraint. The two runs where v2 worked
            were not evidence that v2 worked — they were the same distribution, sampled twice.
            (2) Temperature 0 makes a wrong answer reproducible, which is progress and not a fix.
            It is worth having anyway: a gate that cannot reproduce its own number is not a gate.
            (3) Where a rule can be checked mechanically, check it. The guard is ~40 lines and
            catches every variant of an invented figure the model produced across six runs.
            (4) The guard checks presence, not meaning: "12 rupees" would have survived on its own,
            because some excerpt did contain a 12. It died on the caller's own 20. Still open.
