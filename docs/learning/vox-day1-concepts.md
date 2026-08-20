# VOX Day 1 — concept primer

Covers what the turn loop as it stands actually exercises: VOX-001/002/003 (loop, latency
split, telemetry), VOX-006 (arms), VOX-007/enhancement (fallback + cooldown), VOX-011
(barge-in). Written from `ARCHITECTURE.md`; when the two disagree, `ARCHITECTURE.md` wins
and this file is wrong.

Read this before the coach session — the lesson page derives from it, not the other way
round.

## 1 · The turn loop and where each stage runs

One turn = one VAD segment → STT → NLU → (confirm?) → action → TTS. VAD and TTS are local,
STT and NLU are remote on a free tier. The placement lives in `config.PIPELINE` and
`tests/unit/test_fallback.py` asserts each stage's default arm against it.

**Why:** VAD runs per 32 ms frame, so a network hop per frame is not a design; TTS needs no
key and Kokoro is good enough; the two stages worth spending a free tier on are the two with
the widest quality gap.

**Pitfall:** reordering the arm tables in `config.py` silently moves a stage across the
network boundary. That is the test's whole job.

## 2 · Arms: models are a data table, not a branch

`src/arms.py` is the only interface (`stt`, `llm`, `tts`), `src/config.py` the only table.
Arms are named by **Hugging Face repo id**, with `repo/id@provider` when two providers serve
the same weights. The provider is only *where it runs*.

**Pitfall:** `if provider == "groq"` leaking into feature code — the moment that exists, a
model swap stops being a config change.

## 3 · Telemetry: two logs joined by `turn_id`

`runs/calls.jsonl` is one line per model call (cost, provider latency). `runs/turns.jsonl` is
one line per turn (`t_vad`, `t_stt`, `t_llm`, `t_tts`, `time_to_first_audio`). They answer
different questions, which is why they are not one file.

`time_to_first_audio` is measured from the **last frame silero called speech**, not from the
endpoint decision, through to the output device pulling its first block — the user has been
waiting since they stopped talking, so the VAD hangover belongs inside the number.

**Pitfall:** logging only calls. The 0.25–0.84 s spent opening the output device belongs to
no model call and is invisible unless turns are logged separately.

## 4 · The budget misses, and the miss is kept visible

Target was < 2 s end to end; measured is 5.6–20.0 s (n=4, fixture-driven, 2026-08-17). TTS is
the largest stage in every one of the four turns. The budget column stays as written so the
size of the miss stays readable.

**Pitfall:** treating n=4 as a distribution. Every stage varies 2–4× across identical input,
so the first job is a stable measurement, not an optimisation. Fixture runs also understate
live by roughly the 1.1 s VAD hangover.

## 5 · Fallback: what does *not* fall back is the design

429, timeout and 5xx degrade to the local arm rather than lose the turn. 401/other 4xx, a
missing credential, and an empty reply from a reasoning arm deliberately do **not** — a local
arm would return a plausible transcript and leave a broken credential to be found days later
in a WER table.

Both attempts are logged: the refusal as `ok:false`, the local call carrying
`fallback_for: <failed arm id>`. The turn is marked `fell_back` and `<stage>_model` is
rewritten to the arm that actually ran.

**Pitfall:** a fallback turn compared against a clean one. A fallback is a way to make a
failure invisible; the accounting is what stops it.

## 6 · Cooldown

A 429 carries `Retry-After`; `src/cooldown.py` parks that arm for exactly that window (60 s
when the provider does not say), and later turns skip it entirely. In-memory and
process-local on purpose — a persisted cooldown would open a demo with an arm parked an hour
ago.

**Pitfall:** without it, a rate-limited session pays a doomed round-trip every turn — the
pattern that turns a pace limit into a shut-off.

## 7 · Barge-in

Nothing is killed and no second listener exists. Playback already runs on the output device's
callback thread; `Playback.abort()` stops it being handed more samples — `abort()` and not
`stop()`, because stop drains the buffer first. The same `listen()` call that was watching for
speech captures the interrupting utterance and hands it to the next turn, so an interruption
and a polite next utterance are one code path.

Two knobs, provisional until VOX-012 tunes them: `BARGE_MIN_SPEECH_MS` (200) and
`BARGE_SPEECH_THRESHOLD` (0.7, stricter than the ordinary VAD threshold because the decision
fires while the speaker is running). Stop latency is measured from the first speech frame, so
the min-speech window is visible inside the number rather than hidden behind it.

**Pitfall:** open speakers. There is no acoustic echo cancellation and none is in scope —
silero hears Kokoro and the reply interrupts itself. The demo machine runs on headphones.

## 8 · Confirmation

Every write action needs a read-back plus an affirmative labelled `intent=confirm`. Silence,
ambiguity, or the user repeating the original command do not satisfy it; maximum two re-asks,
then abort and log.

**Pitfall:** treating "no objection" as consent. The turn that creates data is the one turn
where guessing is not allowed.
