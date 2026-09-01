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
            the turn logged `grounded: true`. VOX-033's rate has to handle that — **closed by VOX-033**, see its retro lesson (4).
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


## VOX-033 — POC gate: grounded-answer rate printed (2026-08-21, Ritika as Evaluator)

Executed:   `evals/dev/pdf_queries.json` — 10 queries, 8 answerable with an expected source
            `file:page`, 2 deliberately absent. `tests/gates/gate_poc_pdf.py` scores them through
            the same `retrieval.retrieve()` / `answer.answer()` path the turn loop runs, prints
            correct-source@3, the refusal rate, the grounded-answer rate and the intersection of
            the last two, and exits non-zero below the floors. `make gate-poc`. Concepts doc first,
            per the learning loop; ARCHITECTURE.md section with the numbers.

Deviations: (1) Built both halves of the ticket. The spec splits it — Evaluator writes the queries,
            Builder writes the gate — and the card was already a day past due with the demo on the
            same day, so it went on one branch for the other developer to review rather than two.
            (2) Nothing under `src/` was touched, deliberately, so a failing gate could not degrade
            the VOX-027 demo. That is what pushed the paraphrased-refusal fix into the gate.
            (3) Added a fourth number the spec does not ask for — grounded AND correct-source@3 —
            after the first run showed why. See lesson (1).
            (4) Left `make gate` alone. It is `pytest tests/gates`, which collects nothing because
            these gates expose `main()` rather than `test_*` functions. All three are run directly.
            Worth fixing; not on a branch whose job is to not move the demo.

Numbers:    `uv run python tests/gates/gate_poc_pdf.py`, meta-llama/Llama-3.1-8B-Instruct on the
            NVIDIA NIM free tier, identical on two consecutive runs:
                correct-source@3            7/8 = 0.875   floor 0.875   PASS
                refusal rate                2/2 = 1.000   floor 1.000   PASS
                grounded-answer             8/8 = 1.000   printed, not asserted
                ...on an expected source    7/8 = 0.875
                paraphrased refusals        0
                19 model calls, cost_usd=0.0 on every one
            `uv run pytest tests/unit -q` -> 313 passed.

Lessons:    (1) The grounded-answer rate is the ticket's title and the weakest of its numbers.
            8/8 grounded next to 7/8 correct-source is not rounding: `grounded` says a model
            answered from the excerpts it was handed, and stays true when those came off the wrong
            pages. q03 answered "any excess beyond 24 days will be subject to encashment" out of
            leave-accrual arithmetic while the rule it was asked about sits in separation-policy.
            One number cannot be read alone, so the gate prints the intersection under it.
            (2) Two of ten candidate labels were wrong, and both were caught by reading the chunk
            rather than trusting retrieval. "Gift from a vendor" looked absent because retrieval
            missed it — `grep -ic gift runs/chunks.jsonl` returns 5. "Hotel stay on official
            travel" looked answerable — travel-policy has no accommodation section. Writing a
            label from what retrieval returns measures nothing; the corpus is the authority.
            (3) Spending the gate's slack in advance, in writing, is what makes 7/8 an assertion
            instead of a shrug. q03 was documented as expected-to-fail in the query set's `_note`
            before the gate first ran, so 7/8 means "seven clean queries pass and the known
            STT-damaged one does not" rather than "one of them flaked".
            (4) Closes lesson (4) of the hybrid-retrieval retro. A paraphrased refusal is now
            counted — `REFUSAL_SHAPES` in the gate, every phrase naming the source rather than
            being a bare negation, each catch printed with the reply that produced it because
            over-counting refusals is generous in the direction that makes the gate pass.
            `answer.is_refusal()` is deliberately unchanged: it decides what the live loop writes
            to `runs/turns.jsonl`, so widening it is its own before/after measurement. It caught 0
            on this set, which is a measurement of Llama-3.1-8B at temperature 0 and not evidence
            the shape is gone.
            (5) Still open: n=2 on the refusal side. Two absent queries cover both refusal paths,
            which is the right design, but a refusal "rate" over two samples has three possible
            values. Widening it needs more written queries, not a different metric.

## VOX-028 — MVP freeze (2026-08-21, Ritika as Builder)

Executed:   The documentation half of the freeze. README rewritten from the template scaffold it
            still was into the repo's front door: a developer note naming the four things about
            this clone that surprise people, a "what to expect" section that leads with the missed
            latency target, system requirements, a required / degrades-a-stage / blocks-one-command
            split for software, getting-started for both the plain and the grounded path, the full
            env-var sample, the HF repo-id arm table, the gating table and the latency-budget table
            — each table naming the `ARCHITECTURE.md` section it was copied from. `.env.example`
            extended from 3 names to every knob in `src/config.py`, with the tunables commented out
            so `config.py` stays the single source of defaults. Week report written to
            `notes/build-log/VOX/week-report.md`: the numbers with their commands, three findings,
            six items of gate debt, and the retro. Concept primer first, as the learning loop asks.

Deviations: (1) The ticket says "README with HF repo ids, the gating table and the latency-budget
            table". The delivered README is much larger than that, because the ticket's own Done
            condition is "a clean clone reproduces the demo" — and a clean clone *cannot* reproduce
            the grounded half at all (gitignored corpus), needs a system package for three of the
            four gates, and re-resolves its own dependency graph. Those are facts a stranger has to
            read before the three tables mean anything, so they go above them.
            (2) `make gate` was left broken rather than fixed. It exits 5 with "no tests ran"
            because all four gates expose `main()` and pytest collects nothing. Fixing it means
            deciding what a gate does when it has no key, no audio or no corpus — a decision, at
            the moment of freezing, about what the word "gate" asserts. It is documented in the
            README's gating table and is item 3 of the gate debt instead.
            (3) `gate_phase1b.py` was on `origin/dev` (PR #24) and not on the branch this ticket
            started from, so the first draft of the gating table said "not written". Merged `dev`
            in and corrected it: the script exists, prints its numbers, and asserts **no**
            threshold — VOX-024 is the ticket that sets one and is still open. That is a different
            debt from a missing file and is written up as one.
            (4) The tag is not pushed by this ticket. `mvp-v1` has to point at a reviewed commit on
            `dev`, and this work is on `feat/vox-028` awaiting review — tagging a branch tip is a
            self-merge with extra steps, which is exactly what finding (2) below is about.

Numbers:    `uv run pytest tests/unit -q` -> **313 passed in 6.63s** (2026-08-21).
            `make gate` -> `no tests ran in 0.01s`, `make: *** [Makefile:69: gate] Error 5`.
            `gh pr list --state merged --limit 40 --json number,author,mergedBy` -> **17 of 23
            merged PRs have author == mergedBy.** 6 of the first 6 were reviewed by a third person;
            every PR from #8 onward except #14 was self-merged.
            `git ls-files uv.lock` -> not tracked. `ls tests/gates/` -> no `test_no_leakage.py`.
            Everything else in the README is a citation, dated and attributed to the command in
            `ARCHITECTURE.md` that produced it; nothing was re-measured for this ticket except the
            unit suite and the two checks above.

Lessons:    (1) A freeze measures the *repo*, not the code, and the measurement is embarrassing on
            purpose. Four separate things this repo deliberately does not track — corpus, lock
            file, dev WAVs, runs — are each individually correct and together mean a stranger
            cannot reproduce most of the week's numbers. None of that was visible until someone had
            to write down what a clean clone can do.
            (2) The self-merge count is the finding of the week and it is a process finding, not a
            technical one. The rule held for two days, and it held only because a third person was
            pressing merge. It decayed silently the moment the pair were merging their own work,
            and the Friday `git log` check found it five days too late to change a single PR.
            Enforcement that depends on remembering to check is not enforcement — branch
            protection on `dev` requiring one approving review is the fix, and it is next week's
            first task, not a resolution.
            (3) The cheapest gate is the one that never gets written. `test_no_leakage.py` needs no
            key, no audio and no corpus, and guards the rule the whole evaluation contract rests
            on. It was never blocking anything, which is exactly why it is still absent on day 5.
            Write the free gate first.
            (4) Two documents beat one, but only with a direction of authority. README says what to
            run and what to expect; `ARCHITECTURE.md` says why, with the measurement. Stating in
            the README that `ARCHITECTURE.md` wins, and having every README number name its source
            section, is what stops the two drifting into disagreement with neither marked wrong.

---

## VOX-026 — end-to-end execution run (2026-08-21, Vimal as Builder)

Executed:   The demo, rehearsed. `evals/demo/session_v1.json` is the running order as data — ten
            turns, one barge-in whose interrupting utterance becomes the next turn's input, two
            confirmations (one confirmed, one cancelled), two refusals of different kinds — and
            `scripts/dry_run.py` (`make dry-run`) runs it end to end against the real stages with no
            microphone. Each turn carries an `expect` block asserted against the *turn record*, so
            "clean" is an exit code rather than an impression, and an `expect` key the checker does
            not implement is itself a failure. Three seams were opened rather than copied:
            `vad.drive` is the endpointing loop `listen()` and a recording now share (the barge hook
            lives inside it), `loop.speak_and_watch` and the new `loop.confirmation_leg` take their
            listener as an argument, and `vad.paced` feeds frames at one every 32 ms so the
            VAD_SILENCE_MS hangover is paid in real time. Full report, findings and the demo runbook:
            `notes/build-log/VOX/vox-026-dry-run.md`.

Deviations: (1) The ticket says "recorded". The user's ten lines are **synthesised** locally by
            Kokoro (a different voice from the reply's) and cached, so the session runs on any
            machine with one command — CLAUDE.md's developer-agnostic rule — and a turn can name a
            `recording` instead when a real voice is wanted. What that buys is reproducibility; what
            it costs is acoustics, and the report says so: every transcript in it is an upper bound.
            (2) Three fixes landed that are not in the ticket, all found by running it. The
            barge-in print used a box-drawing character cp1252 cannot encode, so on this Windows
            console the interruption worked and *then* the turn died on the line announcing it. The
            confirmation leg timed its second STT and TTS into `t_stt_ms` / `t_tts_ms`, overwriting
            the turn's own numbers. And the unit suite was reading `.env`'s demo profile —
            `VOX_SESSION_QUIET_LIMIT=1` failed three tests in `test_session.py`.
            (3) Two changes to the demo *script* rather than the code, both recorded in the JSON: q02
            (leave encashment) was cut because Whisper returns "leaving CashMint" and the turn then
            answers fluently off the wrong document, and t05 asserts `code-of-ethics:p12` where
            `pdf_queries` q06 labels p13 — the dress code is on p12 and p13 is the enforcement note,
            read off `runs/chunks.jsonl`. The label disagreement is handed to the Evaluator, not
            fixed: q06 is scored by VOX-033's gate.
            (4) `VOX_STT_FIXUPS=1` is set in `.env` and read by nothing on `dev` — the mechanism is
            on an unmerged branch. Reported by the preflight on every run rather than fixed.

Numbers:    `make dry-run` twice, consecutively, no edit between them, 2026-08-21:
            **CLEAN 10/10, exit 0** both times (`runs/rehearsal/run-20260821T164822.json`,
            `run-20260821T165040.json`).
            `time_to_first_audio` median **2310 ms** (band 2106-2464) and **2357 ms** (band
            2025-3317) — paced, so these include the ~1120 ms hangover a fixture run collapses to
            ~4 ms. `t_llm` median 474 / 525 ms, band 348-741 then 364-**1456**. `t_vad` 1119-1121 ms
            on all twenty turns. 0/10 fallbacks in both.
            Barge-in: stopped **226.2 ms** and **227.6 ms** after speech began, cutting 5.3 s of a
            7.8 s reply, 183 ms of output buffer behind it.
            Confirmation: t09 `t_tts_ms` **170.5 ms** (the read-back) with `t_confirm_tts_ms`
            **50.6 ms** (the cancel line) — before the fix the record would have said 50.6.
            **86 model calls across the two runs, `max(cost_usd) = 0.0`**, every one on a HF repo id
            at groq / nvidia-nim free tier or local weights.
            `uv run pytest tests/unit -q` -> **337 passed in 9.09s**, from 3 failed / 310 passed
            before the conftest fix.

Lessons:    (1) A rehearsal measures the *run*, and the run includes the console. Three of the five
            things this ticket fixed were invisible from inside a passing unit suite: a print that
            kills a turn, a timing field written twice, and a gitignored env file changing what the
            tests assert. None of them are reachable by testing components harder.
            (2) Two consecutive runs with no edit between them is the whole discipline. The medians
            agreed to 2% and the *tails* disagreed by 850 ms — one run would have shown either the
            comfortable number or the alarming one, and reported it as the truth.
            (3) Inject the seam, never copy the loop. A scripted barge-in built with `sleep()` and
            `abort()` would have measured the abort path — a few ms — and printed it in VOX-011's
            field, where silero's detection delay and BARGE_MIN_SPEECH_MS are most of the 226 ms.
            Passing `listen` into the real function was smaller *and* the only version that measures
            anything.
            (4) A knob nobody reads is worse than a missing one, because it is a setting in a file
            someone will trust under pressure. The cheapest guard is mechanical: compare the names
            assigned in `.env` against the names appearing in the code, and print the difference
            before the first turn.

## VOX-034 - Cross-questions, computed figures, trick questions (2026-08-24, Vimal as Builder)

Executed:   Three dev sets committed BEFORE any implementation, in their own commit (912a82b), then
            three mechanisms and three gates.
            `evals/dev/followup_queries.json` (10) - referential follow-ups plus self-sufficient and
            absent-follow-up controls, scored correct-source@3 on the follow-up turn, history off vs
            on. `src/history.py` is new: a bounded per-session window, an ellipsis test (opener or
            anaphor), and a query rewrite built from previous QUESTIONS and never previous ANSWERS.
            `retrieval.fuse()` combines the fragment's ranking with the resolved question's by
            reciprocal rank.
            `evals/dev/figure_queries.json` (10) - `src/figures.py` and
            `prompts/compute_figure_v1.md`. The model names the operands as JSON, Python evaluates
            them behind an AST whitelist. No arithmetic is done by a model and no constant is ever
            inferred.
            `evals/dev/trick_queries.json` (12) - `prompts/answer_from_source_v3.md` adds
            premise-correction, the anti-sycophancy clause and the worked-example rule.
            `--no-history` restores the pre-ticket loop, the way `--no-kb` restores pre-VOX-032.

Deviations: (1) Scope grew mid-ticket. 1887 was written as follow-ups; the requester added figure
            calculation and trick questions to the same ticket. Kept as one ticket with three
            separate gate numbers rather than split, because averaging them would hide a regression
            in one behind a gain in another. The card's title still says only cross-questions.
            (2) Built both sides again, as VOX-033 did. Blind labelling does not exist when one agent
            writes the implementation and the eval, so the substitute is commit ordering: cases
            first, in their own commit, with the limit printed in each gate's output and stated in
            each file's `_note`.
            (3) g09 was left UNDECIDED in the committed eval, decided by the requester as "refuse",
            and then REVERSED by the requester to "assume 365". Both decisions are recorded in
            figure_queries.json in order, because a threshold whose history is invisible cannot be
            told apart from one moved to make a gate pass. The reversal is implemented as ONE named
            constant in config.DAYS_IN_YEAR, gated by figures.allowed_constant() which matches on the
            operand's NAME — so 365 traces as a days-in-year and nowhere else, and no other constant
            a model might supply gets in. What it costs is written down rather than hidden: the figure
            carries an assumption the listener is never told about, and it is wrong one year in four.
            g03 and g09 are kept as a pair to keep the boundary testable — g03's days-in-year comes
            from the person, g09's is assumed, so removing the allowlist makes g09 refuse again while
            g03 still computes. Both are stable across repeats; see Numbers.
            (4) TWO GATE CRITERIA WERE CORRECTED after a failing run, with the reasons written into
            the gate docstrings because a moved goalpost and a fixed one look identical otherwise.
            `self_sufficient` was asserting non-interference AND retrieval coverage; f07 fails the
            second for a pre-existing reason ("resign" is not in the corpus vocabulary - the
            documents say "resignation"), so it now asserts only what this ticket owns.
            `absent_followup` was asserting that retrieval return nothing, which it cannot: the
            rewrite admits chunks via the DENSE half at lexical 0.098, and src/answer.py already
            names the layer that owns the case. Now scored on the answer refusing.
            (5) Part C DID NOT PASS. Escalated rather than tuned a fourth time - see Numbers.

Numbers:    `uv run python tests/gates/gate_followup.py` (2026-08-24), retrieval-only for the first
            two groups:
                referential          2/6 0.333 -> 4/6 0.667   PASS (need on > off and on >= 0.667)
                self_sufficient      1/2 0.500 -> 1/2 0.500   PASS (non-interference)
                absent_followup      2/2 1.000 -> 2/2 1.000   PASS (answer refuses both columns)
            `uv run python tests/gates/gate_figures.py`, meta-llama/Llama-3.1-8B-Instruct on
            nvidia-nim, `cost_usd 0.000000`:
                accuracy   4/4 = 1.000   floor 1.000   (computable)   ** NOT REPRODUCIBLE **
                refusal    4/4 = 1.000   floor 1.000   (missing_operand)
                no_arithmetic 1/1, trap 1/1
            CORRECTED, same day. That 4/4 was ONE RUN and does not reproduce. Three consecutive
            no-edit runs afterwards scored 2/5, 3/5 and 3/5 (five computable cases by then, g09
            having moved — see below), with individual cases flipping between runs. The gate now
            takes `--repeat` (default 2) and a case passes only if every repeat passes, because
            CLAUDE.md's own standard is that a gate which cannot reproduce its own number is not a
            gate. At `--repeat 3`, 60 model calls, `cost_usd 0.000000`:
                accuracy   2/5 = 0.400   floor 1.000   **FAILED**
                refusal    3/3 = 1.000   floor 1.000
                no_arithmetic 1/1, trap 1/1
                g03  [10000.0, 10000.0, 10000.0]      stable
                g09  [16438.36, 16438.36, 16438.36]   stable
                g01  [None, None, 3.0]                UNSTABLE
                g02  [None, 6.0, None]                UNSTABLE
                g04  [None, None, None]               fails outright
            The split is not random. Encashment is stable because leave-policy:p7 writes the formula
            out verbatim, so extraction is a copy. Accrual is unstable because the corpus states no
            accrual formula at all — leave-policy:p10 gives a worked example ("Ram joined 1st January
            and avails 5 leaves ... 1*9 = 9") and the extractor has to invent the formula's shape
            every run, which it does differently. So the figure path is reliable exactly where the
            document states a formula and unreliable where the rule is only implied by an example.
            That boundary is the real result of part B.
            `uv run python tests/gates/gate_trick.py`, 27 model calls, `cost_usd 0.000000`:
                false_premise 1/2, leading 1/2, contradiction_across_turns 0/2
                traps        2/6 = 0.333   floor 0.600   **FAILED**
                out_of_scope 3/3 = 1.000
                control      3/3 = 1.000   floor 1.000
            No regression, `uv run python tests/gates/gate_poc_pdf.py`: correct-source@3 7/8 = 0.875
            (floor 0.875), refusal 2/2 = 1.000, grounded-answer 8/8 - identical to before the branch,
            re-run after switching the answer prompt to v3.
            `uv run pytest tests/ -q` -> **421 passed** (392 after part A, 363 before the branch).

Lessons:    (1) A gate rejected the design twice before it passed, and both rejections were about
            the mechanism rather than a threshold. The first rewrite fired only on a retrieval MISS,
            which is provably safe and was nearly useless: three of four failures never triggered it
            because they did not miss. A fragment does not fail by finding nothing - "what about
            during a performance improvement plan" returned performance-management at **0.742** when
            the answer is one clause of leave-policy:p5. The second attempt replaced the fragment
            with the concatenation and scored the same 3/6 with a *different* three, because the
            antecedent's terms swamp the follow-up's. Only fusing both rankings worked. A single
            number moving 2/6 -> 3/6 would have looked like progress twice.
            (2) VOX-031's retro left this open: "the guard checks presence, not meaning ... Still
            open." It is now measured and it is worse than it looked. `30` appears in SEVEN chunks,
            one of them the advance-notice clause sitting beside the five-day paternity entitlement,
            so "your 30 days of paternity leave" has every number traced and `ungrounded_numbers()`
            stays silent. `24` is the privilege-leave cap, so the contradiction traps pass too. A
            number can be perfectly grounded and still be the answer to a different question. This
            closes that open item as *confirmed*, not fixed.
            (3) Writing the eval found a bug in shipped code that no test had. `numbers_in()` treated
            "hundred" as a scale that closes a group, so "six hundred thousand" parsed as
            (6*100)+(1*1000) = **1600** and "three hundred and sixty five thousand" as 65300. The
            guard was checking spoken currency figures against numbers nobody said - in both
            directions. Fixed in the shared function with tests; it was reachable only by writing a
            case whose operand a person would speak aloud.
            (4) Tracing an operand by value-membership is not tracing. The extractor returned
            `months_accrued = 5, source "person"` for a question that never said five - it traced
            because 5 is in the excerpts as the leaves Ram takes in the worked example - and computed
            12*5-3 = **57** with every operand "verified". The fix is to check the model's own claim
            about where each number came from, which is cheap and was measurably necessary.
            (5) Prompt strengthening has a ceiling and hits it visibly. Attempt 3 on the trick traps
            produced byte-identical replies to attempt 2 - the additions were ignored - and one of
            them regressed `absent_followup` from 2/2 to 1/2 by making the model keener to answer.
            An addition with no measured benefit and a measured cost was removed rather than kept for
            being well-argued. t01 and t06 are the same root cause: the 8B arm is led by the
            question's framing, and it cannot be instructed out of it. That wants the mechanism the
            figure path got - find what the question's number is attached to in the excerpts and
            compare - which is a ticket, not an attempt.

            (6) The worst number in this ticket is one I reported before checking it twice. The
            figure gate's first accuracy was 4/4 and I put it in a commit message, on the board and
            in this file before running it again. It was a single draw from an unstable extractor and
            three consecutive re-runs scored 2/5, 3/5, 3/5. VOX-026's retro already recorded the
            discipline that catches this — "two consecutive runs with no edit between them is the
            whole discipline" — and it was there to be read. The gate now enforces agreement across
            repeats so the next person cannot make the same mistake by being in a hurry.
            (7) Tracing every operand does not make the arithmetic right. g02 once returned the
            formula `1 - 2` — monthly entitlement minus leaves taken, with the month multiplier
            dropped — and computed -1 leave remaining. Both operands traced, the guard was silent,
            and the sentence read plausibly. The derivation check validates where the NUMBERS came
            from and says nothing about whether the FORMULA is the right one, which is the next thing
            a figure path would need and is not what this one bought.
