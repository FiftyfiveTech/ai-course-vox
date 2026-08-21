# VOX-026 — concept primer: what a rehearsal measures

The ticket: *a scripted 10-turn session on the demo machine including a barge-in and a
confirmation, recorded and dry-run twice. Done when the run completes clean twice with timings
captured. No live debugging during the demo.*

Written before the implementation. Where this file and the code disagree, the code and
`ARCHITECTURE.md` win and this file is wrong.

Every ticket before this one measured a **component**: a stage's latency, a floor, a capture rate.
This one measures the **run** — the thing an audience actually sees, which is the whole chain plus
the operator plus the machine plus whatever is in the environment that day. Nothing below is about
making VOX better. It is all about making a *run of VOX* repeatable, and about being honest when it
is not.

---

## 1 · A rehearsal is a measurement of variance, not of quality

One clean run proves the code can work. Two clean runs start to say something about whether it
*will*. The ticket asks for two on purpose: the second run is the measurement, the first is the
setup.

**Why here:** every latency number in this repo is n=1, n=3 or n=4, on stages the week report already
records as varying 2–4× (`make compare`: `t_llm` 1726–15428 ms on the same arm and the same clip).
A demo is one sample from that distribution. The rehearsal is where you find out whether the sample
you are about to show an audience is a typical one.

**Pitfall:** treating run 2 as a re-test of run 1. If run 1 fails, you fix something and the pair of
runs restarts — a "second run" after a fix is run 1 again. Two *consecutive* clean runs, with no
edit between them, is the only pair that means anything.

---

## 2 · The clock you cannot fake: real-time pacing

`src/harness.fixture_turn` already drives the whole chain from a recording, and its own docstring
carries the caveat: frames arrive as fast as the CPU can push them, so the `VAD_SILENCE_MS`
hangover a person at a mic actually waits out collapses to silero compute — ~4 ms instead of
~1.1 s. Every fixture-driven `time_to_first_audio` in the repo is therefore ~1.1 s optimistic
against what the audience will feel.

The fix is not a correction factor, it is **pacing**: feed the endpointer one 32 ms frame every
32 ms of wall clock. Then the hangover is paid in real time, by the same code, and the number is
the live number with the microphone removed rather than a number with a caveat attached.

**Why it matters here specifically:** a rehearsal exists to predict the demo. A rehearsal that
systematically under-reports the one metric the audience experiences (silence between "…" and the
first syllable) is measuring a pipeline nobody will run.

**Pitfall:** pacing makes the run *slower* than real time if you are not careful — sleeping 32 ms per
frame *plus* silero's own compute per frame means the clip plays back at 1.0×+ε. Pace against an
absolute deadline (t0 + n·32 ms), never by sleeping a fixed interval per frame, or the drift lands
in `t_vad` and you have invented latency instead of measuring it.

---

## 3 · Barge-in without a microphone: inject the listener, not a second copy

`src/loop.speak_and_watch` is the whole of VOX-011: start playback non-blocking, run the ordinary
endpointer across the reply and past it, and on the `on_speech` callback call `playback.abort()`.
There is deliberately **one** listener — an interruption and an ordinary next utterance are the same
code path.

A scripted barge-in must not be a second copy of that function with a `time.sleep()` and an
`abort()`. That would measure the abort path (a few ms of `sd.abort` plus device buffer) and print it
in the field where VOX-011's number lives, which is *speech onset → stop sending*. Silero's
detection delay and `BARGE_MIN_SPEECH_MS` are the majority of that interval, and a scripted abort
skips both.

So the seam is the **listener**, not the barge. `speak_and_watch` takes its listen function as an
argument; the rehearsal passes one that reads paced frames from a clip instead of from the mic.
Every other line — the threshold, the confirm window, the mark the latency is measured from, the
carry-forward of the interrupting utterance into the next turn — is the code that runs live.

**Pitfall:** `Playback.abort()` returns `None` when the reply had already played out, and both
`loop.speak_and_watch` and `telemetry.TurnTimer.barge` treat that as "not a barge-in". A scripted
barge whose clip starts too late measures nothing and says nothing — the run has to *assert* the
turn record carries `barged_in`, or a silent non-event passes as a rehearsed feature.

---

## 4 · One turn, two legs: the confirmation gate

VOX-020's confirmation is a second exchange inside one turn: read back, listen for yes/no, then
either proceed or cancel. It runs `arms.stt` again on the yes/no and `arms.tts` again on the cancel
reply.

In `src/loop.one_turn` both of those are timed with `turn.stage("stt")` and `turn.stage("tts")` —
the same two slots the *first* leg used. `TurnTimer.stage` overwrites, so on a confirmation turn
`t_stt_ms` is the latency of transcribing the word "yes" (a 0.6 s clip) and not of the utterance
that started the turn, and `t_tts_ms` is the cancel sentence and not the read-back.

**Why it matters:** those are the five VOX-003 fields, they are what the phase gates percentile, and
a confirmation turn is exactly the turn where a demo is most likely to feel slow. The one turn whose
latency you most want is the one whose latency the record currently overwrites with a shorter,
flattering number.

**Pitfall — the general shape:** a metric slot written twice per turn does not error, it just quietly
reports the last writer. Every "why is this number better than it feels" bug in this repo has had
this shape: the reused `Capture` clock (VOX-013), `grounded` = 8/8 next to correct-source@3 = 7/8
(VOX-033). A second leg gets its own field names or it corrupts the first leg's.

---

## 5 · The environment is part of the run — and a knob no code reads is worse than no knob

`.env` on this machine carries a **demo profile**: `VOX_TTS_MODEL=piper`,
`VOX_REMOTE_TIMEOUT_S=5`, `VOX_SESSION_QUIET_LIMIT=1`, `VOX_STT_MODEL=large-v3`. Each one is a
deliberate, documented choice for demo smoothness. Two consequences neither of them intended:

1. `VOX_SESSION_QUIET_LIMIT=1` **fails three unit tests**, because `tests/unit/test_session.py`
   asserts against the code default of 2 (six watched turns, `0 turn(s) completed in 1:00.`).
   `.env` is gitignored and per-machine, so the suite's result now depends on who is running it.
   `conftest.py` already guards exactly this class of problem for `VOX_*_MODEL` — the fixture is
   called `no_env_override` — and the guard simply does not cover the session tunables.
2. `VOX_STT_FIXUPS=1` is set with a nine-line comment explaining what it protects the demo from.
   Nothing on this branch reads it. The mechanism lives on `experimental/demo-followup-retrieval`,
   which is not merged. The knob is a **belief**, not a setting, and it is exactly the kind of thing
   that gets discovered live.

**Why here:** "no live debugging during the demo" is not a discipline, it is a property of the
preflight. A run that prints its own effective configuration — and flags any `VOX_*` name in `.env`
that no module reads — cannot be surprised by either of these.

**Pitfall:** fixing (1) by editing `.env`. The demo profile is correct; the *test* is what must not
depend on ambient environment. A test that only passes on a machine with no `.env` is not a test of
the code.

---

## 6 · "Clean" has to be a set of assertions, or it means "looked fine"

CLAUDE.md: *a phase is done when its number is computed and printed, not when it looks right.* A
10-turn rehearsal produces ten turns of console output that any reasonable person will read as
success. So each scripted turn carries what it must do — grounded in a named `document:page`,
refused, confirmed, cancelled, interrupted — and the run compares outcome against script and prints
a per-turn PASS/FAIL plus an exit code.

**Why it matters:** the failures that ruin a demo are the quiet ones. A grounded turn that silently
takes the plain path still answers fluently (`loop.grounding()` exists for precisely this reason). A
barge-in that arrived 200 ms late still prints a reply. A confirmation that never fired still says
something agreeable. None of those are visible in a transcript; all of them are visible against a
script.

**Pitfall:** asserting on the *wording* of a reply. The reply text is a model output at
temperature > 0 on the plain path — pinning it makes the rehearsal fail for the wrong reason. Assert
the structural facts: the record's `grounded`, `sources`, `barged_in`, and the confirmation
classification. Those are decisions the code made, not sentences the model chose.

---

## 7 · Preflight: the failures that only happen at demo time

Things that cannot fail during development because development already warmed them, and that all
fail in front of an audience:

- **`uv run` needs the network to re-resolve the environment.** `en-core-web-sm` is pinned to a
  GitHub release URL; a 504 from GitHub is a failed `uv run` before a line of VOX executes. This
  happened once while writing this ticket. `uv run --no-sync` skips resolution — that is the demo
  invocation, and it belongs in the runbook rather than in someone's memory.
- **Cold weights inside a timed turn.** `src/loop.py` and `src/harness.py` both already load silero,
  Kokoro/piper and the encoder *before* the first turn, for latency-honesty reasons. The rehearsal
  inherits that and adds the same rule for its own audio: the user clips are synthesised in a
  preparation phase, never inside a measured turn.
- **No index.** `answer_mod.knowledge_base()` returning `None` degrades every grounded turn to a
  plain reply, announced once at startup — which is exactly the announcement nobody reads.
- **No output device / the wrong output device.** `time_to_first_audio` is unmeasurable without a
  speaker actually pulling samples, and on open speakers silero hears the reply and it interrupts
  itself. Headphones are a *hard* requirement of the demo, not a preference.
- **A fallback that is not pulled.** The local arm is what a rate limit costs you; if ollama has no
  model, a 429 costs the turn instead of the quality.

**Pitfall:** a preflight that warns and continues. A warning printed 40 lines above the first turn
has the same effect as no warning. Anything that would make a scripted turn fail its assertion
should stop the run before turn 1.

---

## 8 · What this rehearsal still cannot tell you

Written down here so the report can point at it instead of re-deriving it:

- **Acoustics.** The user audio is synthesised locally and handed to STT as samples. It never
  crosses a room, a microphone, or the speaker/mic coupling that makes echo cancellation a problem.
  Whisper finds a piper voice easier than a person in an open-plan office, so per-turn transcript
  accuracy in the rehearsal is an **upper bound**.
- **A person's timing.** A scripted barge-in interrupts at the same offset every time. A human
  interrupts when they lose patience, which is a distribution, and the interesting tail of it is
  "just as the reply ends", where `abort()` returns `None`.
- **The free tier's mood.** Two clean runs at 15:40 say nothing about 15:40 tomorrow. The fallback
  path is what covers that, and the rehearsal should record which turns took it — a run where three
  stages fell back is a *passing* run that would have felt terrible.
