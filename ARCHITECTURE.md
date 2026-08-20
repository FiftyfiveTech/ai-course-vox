# VOX — Architecture

**Track:** AI Engineering Course, Week 3
**Status:** Draft — awaiting sign-off from Vimal (Evaluator)
**Last updated:** 2026-08-20

---

## Purpose

VOX is an internal, employee-facing voice assistant for workplace commands (e.g. "book a meeting",
"log my hours"). It is **not customer-facing**. No real account writes happen without an explicit
confirmation from the user.

---

## Constraints (hard)

| Constraint | Rule |
|---|---|
| Models | Hugging Face repo ids only (`openai/whisper-large-v3-turbo`, not "Groq Whisper") |
| Spend | Zero. Any paid API call is STOP-and-ask |
| Realtime speech-to-speech APIs | Out of scope — they are paid; stop and ask |
| PII / recordings | No customer PII; no recording without a visible consent notice |
| Account writes | Confirmation required before every write action |
| Held-out set | Sealed Wednesday, tagged `heldout-v1`; Builder never reads it |

---

## Turn loop

```
┌─────────────────────────────────────────────────────────────────────┐
│  User speaks  →  [VAD] silero-vad                    LOCAL          │
│       ↓                                                             │
│  [STT]  whisper-large-v3-turbo (Groq free tier)      REMOTE         │
│       ↓                            └─ on failure ──> faster-whisper-base   LOCAL
│  [NLU / entity extraction]  LLM (NVIDIA NIM)         REMOTE         │
│       structured output validated against schemas/                  │
│       ↓                            └─ on failure ──> Llama-3.2-3B (ollama) LOCAL
│  [Confirmation check]  — required for every write action            │
│       if needed → TTS response asking "Did you mean …?"             │
│       ↓ (confirmed, or read-only)                                   │
│  [Action]  internal tool / API call                                 │
│       ↓                                                             │
│  [TTS response]  Kokoro-82M                          LOCAL          │
│       ↓                            └─ on failure ──> speecht5, then text   LOCAL
│  Wait for next utterance (barge-in allowed — see below)             │
└─────────────────────────────────────────────────────────────────────┘
```

One turn = one VAD segment → STT → NLU → (confirm?) → action → TTS.

**Why the stages sit where they do.** The placement is in `config.PIPELINE` and
`tests/unit/test_fallback.py` asserts each stage's default arm against it, so a table reorder in
`config.py` can no longer move a stage across the network boundary unnoticed.

| Stage | Placement | Why |
|---|---|---|
| VAD | local | runs per 32 ms frame; a network hop per frame is not a design |
| STT | remote | both local `base` arms drop the first word of the fixture — measured, see below |
| LLM | remote | the widest quality gap of the four, and the least tolerable to lose |
| TTS | local | no key, no quota, and Kokoro is already good enough to ship |

---

## Latency budget (target: < 2 s end-to-end)

**The signed-off budget does not hold.** VOX-002 flagged it and deferred the revision to VOX-003;
these are the first measured numbers. The budget column is kept as written so the size of the
miss stays visible rather than being edited away.

| Stage | Budget | Measured (n=4) | Notes |
|---|---|---|---|
| VAD (end-of-speech detection) | ~100 ms | **~1100 ms live** (by construction) | `VAD_SILENCE_MS` — the loop holds the turn open this long to see whether the user is done. Not yet measured on a live mic; the constant is the floor. Silero compute is 0.15–0.83 s on top. |
| STT (`openai/whisper-large-v3-turbo` on Groq) | ~300 ms | **1.0 – 15.6 s** | free tier, wildly variable. The 15.6 s is in the call log too, so it is provider time, not ours. |
| NLU / entity extraction (NVIDIA NIM) | ~500 ms | **0.53 – 1.22 s** | reply only. Structured extraction (VOX-019) is not built, so this will grow. |
| Confirmation TTS + user response | not counted | not built | VOX-020 |
| Action (internal API) | ~200 ms | not built | — |
| TTS (response) | ~200 ms | **3.1 – 12.4 s** | warm `hexgrad/Kokoro-82M`, weights preloaded. **15–60× over budget** and the largest single stage in every one of the four turns. |
| unattributed | — | **0.25 – 0.84 s** | opening the output device before the first block. Real, felt, and belongs to no model call — visible only because turns and calls are logged separately. |
| **Total (no confirmation)** | **~1.3 s** | **5.6 – 20.0 s** | `time_to_first_audio`, fixture-driven |
| **Total (with confirmation)** | **2–4 s** | not built | — |

Measured on 2026-08-17: four turns driven from `tests/fixtures/hello_testing_voice.mp3` via
`scripts/turn_from_fixture.py`, same clip and same machine every time. Two warnings about reading
this table:

- **n=4 is a range, not a distribution.** Every stage varies by 2–4× across four runs of identical
  input, so no single number here is a target to tune against. The variance is the finding. VOX-012
  re-tunes endpointing on VOX-004's 45 utterances and is the first chance to quote a real spread.
- **Fixture runs understate the live figure.** Frames are pushed as fast as the CPU allows, so the
  ~1.1 s VAD hangover a person actually waits out collapses to 3–26 ms. Read a live turn as roughly
  *measured + 1.1 s*. `source` on each turn record says which kind of run produced it.

Where the 2 s target has to come from, on this evidence: TTS is the largest stage in all four turns,
so streaming Kokoro's first chunk to the speaker instead of synthesising the whole reply first, and
then the ~1.1 s VAD hangover. The three model calls are not the problem — and the variance means
the first job is a stable measurement, not an optimisation.

### Telemetry

Two logs, joined by `turn_id`, because they answer different questions:

| | one line per | answers |
|---|---|---|
| `runs/calls.jsonl` | model call | what it cost, how long the provider took |
| `runs/turns.jsonl` | turn | where the turn's wall clock went — `t_vad`, `t_stt`, `t_llm`, `t_tts`, `time_to_first_audio` |

`time_to_first_audio` is measured from the **last frame silero called speech** — not from the
endpoint decision — through to the moment the output device pulls its first block. The user has
been waiting since they stopped talking, so the VAD hangover is inside the number, and the
callback stamp is the first audio rather than the moment playback was queued.

**A `Capture` therefore owns a clock, and reusing one across turns breaks this field.** `speech_end_t`
is a `perf_counter` stamp taken when that capture's speech ended, so a second turn driven from the
same capture is measured from the first turn's origin and inflates by everything in between. VOX-013
hit this: three identical turns read 6.5 s, 39.4 s and 63.6 s. Anything replaying a fixture must
endpoint per turn — `src/harness.py` documents it, `scripts/compare_arms.py` endpoints per turn and
asserts the segment came out identical, and `tests/unit/test_compare.py` pins the mechanism.

---

## Fallback and cooldown

The two remote stages degrade to a local arm rather than losing the turn. `config.FALLBACKS` names
one **local** arm per stage; `src/arms.py` applies it at the single dispatch point every model call
already crosses, so the rule is written once rather than per stage.

### What falls back, and what deliberately does not

`errors.is_transient` is the whole rule. It is narrow on purpose — what is excluded matters as much
as what is included, because a fallback is a way to make a failure invisible.

| Failure | Falls back | Why |
|---|---|---|
| 429 rate limit | yes | the expected free-tier failure; a pace limit, not a bug |
| Timeout / connection error | yes | the offline case, and the one most likely to happen in a live demo |
| 5xx | yes | the provider's side, not the request |
| **401 / any 4xx** | **no** | a bad key or a model off the catalogue. A local arm would return a plausible transcript and leave the broken credential to be found days later in a WER table |
| **Missing credential** | **no** | STOP-and-ask under CLAUDE.md. Running locally instead is how you never notice the free tier was never configured |
| **Empty reply from a reasoning arm** | **no** | a misconfigured arm — it would fail again next turn, so routing around it hides a permanent problem as a flaky one |

Both attempts are logged. The refusal is a `calls.jsonl` line with `ok:false`, and the local call is
a second line carrying `fallback_for: <failed arm id>`. If the fallback also fails, the **original**
exception propagates with the local one chained: the free tier refusing is the cause, and the local
arm failing behind it is a second symptom of the same turn.

### Cooldown

A 429 carries a `Retry-After`, and `src/cooldown.py` parks that arm for exactly that window
(`DEFAULT_COOLDOWN_S`, 60 s, when the provider does not say). Subsequent turns skip the parked arm
entirely and go straight to local, then return to remote on their own when the window lifts.

Without this, a rate-limited session pays a doomed round-trip on every single turn — latency the
user waits for nothing, and the pattern that turns a pace limit into a shut-off. In-memory and
process-local by design: a persisted cooldown would open a demo with an arm parked an hour ago.

`REMOTE_TIMEOUT_S` dropped to 10 s (was 30 s for STT, 60 s for the LLM) as part of this. Those
values were fine when a timeout meant the turn was over anyway; now that a timeout has somewhere to
go, the wait is pure added latency in front of an arm that would have answered.

### What it costs when it fires

A fallback turn is **not comparable to a clean one** and the record says so, in three places: the
human-readable line ends `[fell back: stt]`, the turn record carries `fell_back`, and the gate
prints a `FELL BACK` block. `<stage>_model` is rewritten to the arm that actually ran — otherwise
VOX-013's per-turn comparison would credit the wrong model with the latency.

`t_<stage>_ms` spans the dead round-trip *plus* the local call. That total is honest about what the
user waited for, so `<stage>_failed_ms` breaks out the failed attempt separately; without it a
provider timeout reads as slow local inference.

**The STT fallback is a worse transcript, not a cheaper identical one.** Both local `base` arms
return `'So this is testing.'` where the Groq `large-v3` arms return `'Hello. So this is testing.'`
— they drop the word that starts the turn, identically across both measured runs (see the arms table
below). `faster-base` is the fallback rather than `whisper-base` because it is 2.4–5.7× faster for a
character-identical result, but the accuracy cost is real and is the reason STT is remote by default.

### Measured, n=1, 2026-08-19

One turn on `tests/fixtures/hello_testing_voice.mp3` on each path, same clip and same machine.
The clean run is `uv run python scripts/turn_from_fixture.py --silent`; the fallback run is the
same turn with both remote `api_base` values pointed at `http://127.0.0.1:1/v1`.

| | clean | both free tiers unreachable |
|---|---|---|
| STT | 345 ms — `whisper-large-v3-turbo` @ groq | 3430 ms = **2064 ms** dead round-trip + 1357 ms `faster-whisper-base` |
| LLM | 684 ms — `Llama-3.1-8B` @ nvidia-nim | 7513 ms = **2087 ms** dead round-trip + 5415 ms `Llama-3.2-3B` @ ollama |
| TTS | 3709 ms — `Kokoro-82M` | 12545 ms — `Kokoro-82M`, no fallback needed |
| transcript | `'Hello. So this is testing.'` | `'So this is testing.'` — the first word is gone |

n=1, and every stage in this repo varies by 2–4× across runs, so read these as orders of magnitude.
Two things they do establish:

1. **The turn survives, and the reply is real.** Both free tiers refusing produced a spoken answer
   rather than a dead turn, which is the whole point of the change.
2. **A fallback is not free.** The local LLM is ~8× the remote one's call time here, and on top of
   that the turn pays the failed attempt. A connection *refused* returns in ~2 s; a connection that
   hangs instead costs `REMOTE_TIMEOUT_S` (10 s) before the local arm starts. Cooldown is what stops
   that being paid once per turn.

The ollama arm's ~5 s load is warmed at startup by `nlu.load_ollama` and `arms.warm_fallbacks`, so
it is not in the 5415 ms above. Without that warm-up it would land inside `t_llm`, on the one turn
least able to afford it.

---

## Barge-in

Barge-in = user speaks while TTS is still playing.

- VAD runs continuously, not only after TTS finishes.
- When VAD detects speech during TTS playback, TTS is **immediately interrupted**.
- The new utterance is queued and processed as the next turn.

Barge-in interrupt point: **between TTS playback start and TTS playback end**.
No partial transcriptions are discarded; the full new utterance is captured before STT runs.

### How it is actually built (VOX-011)

The draft said *"the TTS playback thread is killed"*. Nothing is killed, and no thread is started:
playback already runs on the output device's own callback thread, so the mic loop keeps the main
thread and `Playback.abort()` stops the device from being handed any more samples. `abort()` and not
`stop()` — stop drains the buffer first, which is the opposite of interrupting.

There is also no second listener. The ordinary endpointer runs across the whole reply and past it,
so the utterance that interrupts a reply is captured by the same `listen()` call that was watching
for it, and is handed to the next turn as its input. An interruption and a polite next utterance are
therefore the same code path; they differ only in whether anything was still playing when the speech
arrived, which is what `abort()` returning `None` reports.

Two streams, not one duplex stream: the mic is 16 kHz for silero and whisper, Kokoro emits 24 kHz,
and a duplex stream takes a single sample rate — so one stream would mean resampling the reply to
match the microphone. Confirmed working on this machine before anything was written.

**Two knobs, both PROVISIONAL until VOX-012 tunes them on the dev set:**

| | | why it exists |
|---|---|---|
| `BARGE_MIN_SPEECH_MS` | 200 | speech that must accumulate before a reply is cut. Cutting on the first speech frame is faster and lets a cough kill every reply |
| `BARGE_SPEECH_THRESHOLD` | 0.7 | stricter than `VAD_SPEECH_THRESHOLD`, because this decision fires while the speaker is running |

Stop latency is measured **from the first speech frame**, not from the moment the decision was made,
so `BARGE_MIN_SPEECH_MS` is visible inside the printed number instead of hidden behind it.

**No acoustic echo cancellation, and none is in scope.** On open speakers silero hears Kokoro and
the reply interrupts itself, every time. The threshold and the confirmation window reduce how often
that happens; neither fixes it, and no value fixes it, because speaker bleed is real speech as far as
a VAD is concerned. Real AEC means a webrtc/speexdsp dependency and a separate ticket. **The demo
machine runs on headphones**, and VOX-026's dry-run has to be done on the demo hardware for exactly
this reason.

What `abort()` cannot recall is the output device's own buffer — 0.182 s on this machine's MME
device, which is larger than the stop latency itself. So the printed number is when VOX stopped
*sending*, and `out_latency_s` is logged beside it as the tail that can still be heard.

Barge-in needs a turn after the one being interrupted, so `--turns 2` or more switches it on. The
last turn of a run is played blocking, which keeps `make demo`'s default single turn exactly as
VOX-002 and VOX-003 measured it. One consequence worth knowing when reading the logs: a watched
turn's record closes only once the *next* utterance has been endpointed, because one listener spans
both — so its `ts` and its printed latency line land after the user has spoken again.

---

## What counts as a confirmation

A confirmation is required when the action has a write side-effect (creates, modifies, or deletes
data). Read-only queries (e.g. "what meetings do I have today?") do not require confirmation.

**Confirmation is satisfied when:**
1. The system reads back a summary of the intended action ("Book a 1-hour meeting with Priya at
   3 pm tomorrow?").
2. The user responds with an affirmative: "yes", "confirm", "go ahead", "do it", or equivalent.
3. The NLU labels the response `intent=confirm`.

**Confirmation is NOT satisfied by:**
- Silence / timeout — the system re-asks or aborts.
- A partial or ambiguous response — the system asks again.
- The user repeating the original command — that restarts the confirmation flow.

Maximum 2 re-asks before the action is aborted and logged.

---

## Module map

```
src/
  config.py        — the arm table (HF repo id -> provider) + resolve() + endpointing constants
  arms.py          — the one interface: stt(audio, id) / llm(msgs, id) / tts(text, id); resolves,
                     logs and dispatches. Every model call goes through here.
  telemetry.py     — shared cost/latency logger; arms.py is its only caller for model calls
  cooldown.py      — which arms are parked after a 429, and until when
  errors.py        — where a provider response becomes a named failure; `is_transient` is the
                     one place the fallback rule is written
  vad.py           — endpointing via snakers4/silero-vad (local)
  stt.py           — STT backends: openai-audio (Groq), transformers-whisper, faster-whisper
  nlu.py           — the openai-chat backend + the message assembly arms.llm() takes;
                     structured extraction lands in VOX-019
  audio.py         — speaker playback, kept apart from synthesis so VOX-011 can interrupt it
  loop.py          — one chained turn; `make demo`
  confirm.py       — confirmation flow logic (VOX-020, not yet written)
  actions.py       — tool/API calls (read and write) (not yet written)
  tts.py           — TTS backends: kokoro, speecht5, piper
  harness.py       — one chained turn driven from a recording, shared by
                     scripts/turn_from_fixture.py and scripts/compare_arms.py
  sources.py       — PDF corpus -> text chunks with (doc_id, page) provenance (VOX-029);
                     retrieval over those chunks lands in VOX-030

sources/           — the PDF corpus. Gitignored: internal HR policies (see below)

prompts/
  reply_v1.md      — spoken-reply prompt (versioned; never inline)
  extract_v1.md    — entity-extraction prompt (VOX-019, not yet written)

schemas/
  intent.py        — Pydantic models for structured NLU output

evals/
  dev/             — Builder tunes here (15 dev cases)
  heldout/         — Evaluator only; sealed Wednesday as heldout-v1

tests/
  gates/           — one script per phase gate; prints the number
  test_no_leakage.py — asserts dev ∩ heldout = ∅ by content hash
```

---

## Models (HF repo ids)

Every stage is an **arm** chosen at run time (VOX-006). Provider = where it runs. The model id is
the HF repo id, not the provider name; the mapping from repo id to each provider's own model string
lives only in `src/config.py`.

| Stage | HF repo id | Runs on | Backend | Alias |
|---|---|---|---|---|
| VAD | `snakers4/silero-vad` | local | — | not yet an arm |
| STT | `openai/whisper-large-v3-turbo` | Groq free tier | `openai-audio` | `turbo` **(default)** |
| STT | `openai/whisper-large-v3` | Groq free tier | `openai-audio` | `large-v3` |
| STT | `openai/whisper-base` | local | `transformers-whisper` | `whisper-base` |
| STT | `Systran/faster-whisper-base` | local | `faster-whisper` | `faster-base` **(fallback)** |
| LLM | `meta-llama/Llama-3.1-8B-Instruct` | NVIDIA NIM free tier | `openai-chat` | `llama-8b` **(default)** |
| LLM | `openai/gpt-oss-120b` | Groq free tier | `openai-chat` | `gpt-oss` |
| LLM | `meta-llama/Llama-3.1-70B-Instruct` | NVIDIA NIM free tier | `openai-chat` | `llama-70b` |
| LLM | `hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF` | local, via ollama | `ollama-chat` | `llama-3.2-3b` **(fallback)** |
| TTS | `hexgrad/Kokoro-82M` | local | `kokoro` | `kokoro` **(default)** |
| TTS | `microsoft/speecht5_tts` | local | `speecht5` | `speecht5` **(fallback)** |
| TTS | `rhasspy/piper-voices` | local | `piper` | `piper` |

`ollama` is a provider name of its own rather than `local` because the arm runs on this machine and
still speaks HTTP, to a daemon on `localhost:11434`. `Arm.local` is the attribute that answers "are
the weights here", and it is what `PIPELINE` and `FALLBACKS` are checked against — `provider` alone
stopped being sufficient the moment a local arm needed an `api_base`.

The defaults are the arms VOX-002 and VOX-003 measured, so an unflagged run still reproduces those
numbers. Selection:

```bash
make arms                                    # call every arm once, print the calls.jsonl lines
uv run python scripts/check_arms.py --list   # the table, no calls
uv run python -m src.loop --stt openai/whisper-base --tts microsoft/speecht5_tts
uv run python scripts/turn_from_fixture.py --llm gpt-oss     # alias, repo id, or repo/id@provider
VOX_STT_MODEL=faster-base make turn                          # env sets the default; the flag wins
make compare                                 # two whole architectures, five stages each (VOX-013)
```

`stt_model` / `llm_model` / `tts_model` on every `runs/turns.jsonl` line name the arms that produced
that latency split, so two runs with different arms cannot be quietly averaged.

Two arms sharing a `backend` share an adapter, so a new arm on an existing runtime is a table row in
`src/config.py` and no new code. `resolve()` refuses a bare repo id that two providers serve — the
same Llama can be a NIM arm or a Groq arm, and a silently chosen provider attaches the wrong latency
to the right name.

### What choosing an arm cost, measured

`make arms`, 2026-08-18, **two runs**, one call per arm per run on the same 3.30 s segment / same
transcript / same sentence. `load` is excluded — weights are warmed before the call, as in the loop.
Both runs are shown because one of them alone would misrepresent the hosted arms: the spread below
is free-tier queueing, not model speed.

| Stage | Arm | Call, run 1 | Call, run 2 | Output |
|---|---|---|---|---|
| stt | `openai/whisper-large-v3-turbo` @ groq | 364 ms | 397 ms | `'Hello. So this is testing.'` |
| stt | `openai/whisper-large-v3` @ groq | 297 ms | 280 ms | `'Hello. So this is testing.'` |
| stt | `openai/whisper-base` @ local | 4685 ms | 2438 ms | `'So this is testing.'` |
| stt | `Systran/faster-whisper-base` @ local | 827 ms | 1018 ms | `'So this is testing.'` |
| llm | `meta-llama/Llama-3.1-8B-Instruct` @ nvidia-nim | 1503 ms | 384 ms | 17 completion tokens |
| llm | `openai/gpt-oss-120b` @ groq | 633 ms | 733 ms | 61-62 completion tokens |
| llm | `meta-llama/Llama-3.1-70B-Instruct` @ nvidia-nim | 39286 ms | 8139 ms | 16-19 completion tokens |
| tts | `hexgrad/Kokoro-82M` @ local | 1532 ms | 1839 ms | 2.25 s at 24 kHz |
| tts | `microsoft/speecht5_tts` @ local | 1944 ms | 2092 ms | 1.92 s at 16 kHz |

Three findings. n=2 per arm, so the transcripts are a result and the timings are a range:

1. **Both `base` arms drop the first word — the only finding here that is not about latency.**
   Given identical audio the Groq `large-v3` arms return `'Hello. So this is testing.'` and both
   local `base` arms return `'So this is testing.'`, identically in both runs. The local arms are
   not a cheaper version of the same transcript, they are a worse one, and the word they lose is
   the one that starts the turn. WER on `evals/dev` is the measurement that should decide this, not
   this one clip.
2. **Same weights, 2.4-5.7x apart on runtime.** `openai/whisper-base` through transformers took
   4685 ms then 2438 ms; the CTranslate2 int8 conversion of the same model took 827 ms then
   1018 ms, for a character-identical transcript both times. The direction is consistent across
   runs even though the ratio is not; for latency the arm that matters here is the runtime, not the
   model.
3. **70B on the NIM free tier is the slowest arm by an order of magnitude, and the least
   predictable.** 39.3 s then 8.1 s for ~17 tokens, against the 8B's 1.5 s then 0.4 s on the same
   provider. `meta-llama/Llama-3.3-70B-Instruct` — what open question 1 actually asked for — is
   worse still: not on this Groq key's catalogue (404), and no answer from NIM inside 120 s on two
   attempts, so 3.1-70B stands in for it. Nothing here says 70B is slow *as a model*; it says this
   free tier does not serve it at conversational latency.

Also worth recording, because it constrains arm choice rather than tuning: **every chat model Groq's
free tier now serves is a reasoning model.** At default effort `openai/gpt-oss-120b` spent the whole
120-token budget thinking and returned an empty string, so that arm carries
`request={"reasoning_effort": "low"}` in `src/config.py` — not a tuning knob, a precondition for the
arm answering at all. `src/nlu.py` raises a named error on an empty reply rather than handing
silence to TTS.


### Architecture comparison, n=3, 2026-08-20 (VOX-013)

`make compare` runs two *whole* pipelines as real turns and reads the five-field split off
`runs/turns.jsonl`. Different question from `make arms`: that times stages, this times architectures,
and the gaps between the calls belong to no call. The arm sets are `config.ARCHITECTURES`.

| | arm `fast` | arm `quality` |
|---|---|---|
| stt | `Systran/faster-whisper-base` @ local | `openai/whisper-large-v3` @ groq |
| llm | `hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF` @ ollama | `meta-llama/Llama-3.1-70B-Instruct` @ nvidia-nim |
| tts | `rhasspy/piper-voices` @ local | `hexgrad/Kokoro-82M` @ local |
| credentials | none | `GROQ_API_KEY` + `NVIDIA_API_KEY` |

`make compare`, 3 turns per arm, interleaved A/B/A/B, on `tests/fixtures/hello_testing_voice.mp3`.
Fallback **off** throughout, so a refused stage would read FAILED rather than borrowing the local
arm's latency. Weights warmed before anything was timed. Both arms played to the speaker, which is
what makes the fifth stage measurable at all.

| stage (ms) | `fast` min | med | max | `quality` min | med | max |
|---|---|---|---|---|---|---|
| `t_vad` | 4 | 4 | 6 | 4 | 4 | 6 |
| `t_stt` | 1047 | 1062 | 1116 | 299 | 321 | 324 |
| `t_llm` | 4267 | 4321 | 4436 | 1726 | 3794 | 15428 |
| `t_tts` | 552 | 580 | 616 | 4619 | 4932 | 6290 |
| `time_to_first_audio` | 6202 | **6695** | 6717 | 8914 | **9291** | 21269 |
| `stage_sum` | 5886 | 5987 | 6135 | 8343 | 8718 | 20685 |

| | `fast` | `quality` |
|---|---|---|
| transcript | `'So this is testing.'` | `'Hello. So this is testing.'` |
| reply | 65-77 chars | 45-72 chars |
| speech produced | 3.74-4.26 s at 22.05 kHz | 3.12-4.28 s at 24 kHz |
| tts per reply char | **7.53-9.47 ms** | **87.36-102.65 ms** |
| turns completed | 3 of 3 | 3 of 3 |

Six findings. n=3, so treat every timing as a range and the transcripts as results:

1. **The all-local arm wins on predictability, not by a landslide on speed.** Median
   `time_to_first_audio` 6.7 s against 9.3 s, so `fast` is ~28% quicker at the median. The real gap is
   the spread: `fast` lands in a 515 ms band across three turns, `quality` in a **12.4 s** band. For a
   voice agent the second number is the one that decides whether a demo is watchable.
2. **piper is 9-13x cheaper per character than Kokoro, and that is the single biggest lever here.**
   7.53-9.47 ms/char against 87.36-102.65. It turns TTS from the largest stage in the turn (VOX-003's
   finding, still true for `quality` at 4.6-6.3 s) into the smallest model call in the pipeline
   (0.55-0.62 s). **This partly supersedes the plan above of streaming Kokoro's first chunk** —
   switching arms gets most of that win for a config change instead of a rework. Whether piper's voice
   is acceptable is a quality question this table does not answer.
3. **The local LLM spends what the local STT and TTS save.** `t_llm` 4.27-4.44 s on the 3B via ollama
   against a *median* 3.79 s for the 70B on NIM. The small local model is slower than the large hosted
   one half the time — it is simply never surprising, which is finding 1 again.
4. **The first word is still lost, now confirmed end to end.** `faster-whisper-base` returned
   `'So this is testing.'` on all three turns; `whisper-large-v3` returned `'Hello. So this is
   testing.'` on all three. This is the `quality` arm's only non-latency advantage in the table and it
   is the reason `PIPELINE` puts STT remote. WER on `evals/dev` is what should settle it, not one clip.
5. **The `quality` arm is not deployable at the shipped timeout.** This run gave hosted arms 120 s.
   One of three 70B calls took **15.4 s**, and `REMOTE_TIMEOUT_S` is **10 s** — so on the shipped
   configuration that turn times out and falls back, meaning 1 in 3 turns would not have run the arm
   the table credits. `--remote-timeout` exists so the arm can be measured at all; it is not a
   proposal to widen the budget.
6. **Neither arm is near the 2 s target.** The fastest single turn of six was 6202 ms. The budget
   table above stands as unmet, and `t_vad` here is fixture-collapsed — add ~1.1 s for a live mic.

Reading caveats, both of which are in the script rather than left to the reader:

- **`t_tts` is not held equal across arms**, because the two LLMs write different-length replies. That
  is inherent to comparing architectures end to end rather than stages, which is why the per-char
  number is printed alongside it. `make arms` is where TTS text is held constant.
- **Load is excluded from every cell**: 3.5 s faster-whisper, 4.8 s ollama, 3.7 s piper, 10.2 s
  Kokoro, all warmed before timing. A cold Kokoro inside `t_tts` would have doubled that row.

---

## Document source folder (POC, VOX-029)

A folder of PDFs in, one JSON lines file of chunks out — `make index`, no network, no model call,
no key. `pypdf` parses; the tokenizer only counts. This is the front half of the PDF-question POC
(VOX-029 -> 030 -> 031 -> 032 -> 033); retrieval and the grounded answer come next.

```
sources/*.pdf  ──pypdf──>  text per page  ──300-token window, 50 overlap──>  runs/chunks.jsonl
                                                {doc_id, page, chunk_idx, text}
```

**The corpus is gitignored, and so is the chunk file.** These are internal FiftyFive HR policies.
The PDFs and the text extracted from them are the same disclosure, so neither belongs in a repo
someone else clones. A clean clone therefore has nothing to index until the corpus is put in
`sources/`; `make index` says so rather than producing an empty index.

**Chunks never span a page**, so `(doc_id, page)` is exact rather than approximate — which is the
point, because VOX-031 has to *say* where an answer came from. The cost is that a sentence
continuing over a page break is split, and the report prints how many pages were long enough to
split at all so the size of that trade is visible.

**`doc_id` is the filename stem**, and the filenames are the documents' own titles
(`leave-policy.pdf`), not the export hashes they arrived as. `scripts/rename_sources.py` derives
the name from the first title-like line in each PDF and is re-runnable after a fresh export; two
titles are overridden by hand there. Spoken provenance is the reason: "leave-policy page 4" is an
answer, "8f0a7775e8b149cf8de3528d379c9a1e page 4" is a hash.

**Tokens are counted with the tokenizer of the model that will read the chunks**, so "300 tokens"
means what VOX-031's prompt budget means by it. `meta-llama/Llama-3.1-8B-Instruct` is gated and
401s without a token, so `config.TOKENIZER_REPO` pins a mirror of the same Llama-3.1 tokenizer
files (`NousResearch/Meta-Llama-3.1-8B-Instruct`, 128k vocab). It is loaded `local_files_only`, so
an index build either uses the local cache or fails saying `make tokenizer` — that is what makes
"no network calls" true of the *first* build and not only of the second.

### Measured, `make index`, 2026-08-20

| | |
|---|---|
| files | 15 |
| pages | 184 — 163 with text, **21 with none** |
| chunks | 215 |
| tokens | 40,750 (250 per page with text) |
| pages long enough to split | 52 of 163 |

The 21 empty pages are named individually in the output. They are cover and end pages on this
corpus — but an image-only scan looks identical at this stage and would make the POC answer
nothing at query time, so they are reported at load rather than discovered three tickets later.
**This corpus is not scanned:** every one of the 15 PDFs yields real text, which settles the
OCR STOP-and-ask in `notes/build-log/VOX/poc-pdf-query-tickets.md`.

Two facts checked rather than assumed, because both are claims the chunker makes about its own
output: all 215 chunks are verbatim substrings of the page they came from (the decode round-trip
is lossless), and the shared text between consecutive chunks on a page re-tokenizes to 49-52
tokens, mean 50, never 0.

---

## Open questions (for sign-off session)

1. ~~Which NLU model on NVIDIA NIM free tier?~~ **Settled by the VOX-002 ticket:**
   `meta-llama/Llama-3.1-8B-Instruct`. ~~Revisit against 3.3-70B under VOX-013 with measurements.~~
   **The 70B revisit is now an arm, and the first measurements are in:** 3.3-70B is unreachable
   (404 on Groq's catalogue; no answer from NIM inside 120 s, twice), and 3.1-70B on NIM answered in
   39.3 s and 8.1 s against the 8B's 1.5 s and 0.4 s. It stays registered as `llama-70b` so VOX-013
   can measure quality against that cost, but it is not a candidate default. See the table above.
   **VOX-013 has now measured it end to end and it is still not a candidate:** 1.7 s / 3.8 s / 15.4 s
   across three turns, and the 15.4 s one exceeds `REMOTE_TIMEOUT_S` (10 s), so a third of turns would
   fall back on the shipped configuration. `Llama-3.1-8B` stays the default.
2. ~~TTS: local `espnet` or NIM?~~ **Settled by the VOX-002 ticket:** `hexgrad/Kokoro-82M`,
   local. (This doc originally proposed `espnet/kan-bayashi_ljspeech_vits`; the ticket names
   Kokoro, so the ticket wins.)
3. ~~VAD library: `silero-vad` vs WebRTC VAD?~~ **Settled by the VOX-002 ticket:** `silero-vad`.
   Its thresholds become a config file under VOX-012.
4. Confirmation intent labels: finalise the affirmative / negative word list. **Still open** —
   needed for VOX-020, not for the Phase-0 loop.

---

## Setup

```bash
git clone <repo>
cd ai-course-vox
make setup   # creates venv, installs deps, writes .env from .env.example
make test    # unit tests (gates run separately with make gate)
```

Secrets go in `~/.config/`, never in the repo. `.env.example` lists key names only.
