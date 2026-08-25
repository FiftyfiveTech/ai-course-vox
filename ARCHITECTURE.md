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

What the self-interrupt hides is the expensive half: the captured reply is carried into the next turn
as `pending`, and a turn with a `pending` does not open the mic — it transcribes VOX's own reply and
answers it, so the session talks to itself until the clock runs out. VOX-035 (`src/echo.py`,
`VOX_ECHO_GUARD=1`, off by default) stops that without cancelling anything: it correlates the mic
capture's amplitude envelope against what the speaker has actually played and rejects a match. It is
**detection, not cancellation** — it does not restore barge-in on speakers, and it holds the barge-in
decision open for `echo.MIN_DECISION_MS` because the measurement says the answer is not available any
sooner, which costs stop latency and is marked `echo_guard` on the turn record so those runs are not
averaged in with these. The numbers and their limits: `scripts/measure_echo_guard.py`,
`docs/learning/vox-035-concepts.md`.

What `abort()` cannot recall is the output device's own buffer — 0.182 s on this machine's MME
device, which is larger than the stop latency itself. So the printed number is when VOX stopped
*sending*, and `out_latency_s` is logged beside it as the tail that can still be heard.

Barge-in needs a turn after the one being interrupted, so it is on for every reply that has one.
In a timed session — `--minutes`, which is what `make demo` runs — that is every reply until the
clock runs out; with `--turns N` it is every reply but the last, and a bare single turn is played
blocking, exactly as VOX-002 and VOX-003 measured it. One consequence worth knowing when reading
the logs: a watched turn's record closes only once the *next* utterance has been endpointed,
because one listener spans both — so its `ts` and its printed latency line land after the user has
spoken again.

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
  nlu.py           — the openai-chat backend + the message assembly arms.llm() takes, and
                     load_prompt(), shared by both prompt files; extraction lands in VOX-019
  audio.py         — speaker playback, kept apart from synthesis so VOX-011 can interrupt it
  loop.py          — chained turns, bounded by a count or by the session clock; `make demo`.
                     Routes through retrieval after STT (VOX-032)
  confirm.py       — confirmation flow logic (VOX-020, not yet written)
  actions.py       — tool/API calls (read and write) (not yet written)
  tts.py           — TTS backends: kokoro, speecht5, piper
  harness.py       — one chained turn driven from a recording, shared by
                     scripts/turn_from_fixture.py and scripts/compare_arms.py
  sources.py       — PDF corpus -> text chunks with (doc_id, page) provenance (VOX-029)
  retrieval.py     — BM25 over those chunks fused with a dense encoder; top-k with provenance and
                     two floors (VOX-030 + hybrid). No longer the one stage without a cost line:
                     the lexical half is arithmetic, the dense half is a model call
  embeddings.py    — the sentence encoder behind that dense half. A stage module like stt/tts:
                     BACKENDS + LOADERS, arms named by HF repo id, `--embed` picks one
  answer.py        — those chunks -> a spoken answer with its doc:page, through arms.llm
                     (VOX-031). A floor miss refuses here without calling any model.
                     `turn_reply()` is the grounded-or-plain routing both turn loops run (VOX-032)

sources/           — the PDF corpus. Gitignored: internal HR policies (see below)

prompts/
  reply_v1.md      — spoken-reply prompt (versioned; never inline)
  answer_from_source_v1.md — answer only from the retrieved excerpts, or refuse (VOX-031)
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
(VOX-029 -> 030 -> 031 -> 032 -> 033); the grounded answer comes next.

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

## Lexical retrieval (POC, VOX-030)

> This describes the lexical half on its own, which is what VOX-030 built and what
> `VOX_HYBRID_RETRIEVAL=0` still runs. A dense half was added after a live turn showed what
> term matching cannot do — see **§ Hybrid retrieval** below. The floor and the numbers here
> are unchanged and still the ones the lexical half is calibrated on.

Question in, top 5 chunks out with the provenance VOX-029 wrote — `make ask Q="..."`. Okapi BM25
via `rank_bm25`, no embeddings, no model, no network, no key.

```
query ──stopwords──> terms ──BM25 over 215 chunks──> ÷ query ceiling ──> score in [0,1)
                                                                            │
                                              score > RETRIEVAL_SCORE_FLOOR ┤──> top 5 hits
                                                                            └──> [] "not in the documents"
```

**This is the only stage in the pipeline that writes no line to `runs/calls.jsonl`.** Not an
exception to "every model call goes through the cost logger" — there is no model and no provider,
`telemetry.log_call` would fail its own `FREE_TIERS` check, and a `cost_usd: 0.0` row against a
provider that does not exist would be a lie in the ledger. Retrieval still costs time, and
`t_retrieval_ms` goes on the *turn* record in VOX-032. Measured here: **0.6-0.7 ms** per query
over 215 chunks, against a 23-28 ms one-off index build at startup.

**The score is BM25 divided by the query's own ceiling**, i.e. by the score a chunk containing
every query term to saturation would get. Raw BM25 is a *sum* over query terms, so it grows with
query length, and the first attempt at a floor failed on exactly that: the absent question "is
there a canteen subsidy for lunch on working days" scored **9.64** raw while the answerable "am I
responsible for the laptop assigned to me" scored **7.38**. No threshold separates those. After
normalising, the same two are 0.232 and 0.320. Dividing by a per-query constant leaves the ranking
untouched — it only makes the *threshold* mean the same thing for a three-word question and a
ten-word one.

A query term the corpus has never seen is charged at the IDF of a term appearing in exactly one
chunk, so it counts in full in the denominator and not at all in the numerator. BM25 treats an
out-of-vocabulary term as zero-information, which is backwards here: "sabbatical" and "canteen"
are the entire reason those questions are unanswerable.

**The stopword list in `src/retrieval.py` is load-bearing, not hygiene.** `BM25Okapi` floors the
IDF of a term appearing in over half the corpus at `epsilon * average_idf` — positive, not zero —
so without dropping function words, "what is the policy on ..." scores every chunk in the corpus
and no floor can separate anything.

### Measured, `scripts/ask.py --calibrate`, 2026-08-20

215 chunks over 15 documents; 13 dev queries in `evals/dev/retrieval_floor_queries.json`, 7
answerable and 6 deliberately absent. The absent ones are written HR-shaped and share vocabulary
with the corpus — a miss phrased in unrelated words would clear any floor and prove nothing.

| | top-1 score, min | max |
|---|---|---|
| answerable (7) | **0.320** | 0.615 |
| absent from the corpus (6) | 0.126 | **0.234** |

Separable: every answerable query outscored every absent one. `RETRIEVAL_SCORE_FLOOR = 0.28`, the
midpoint of the [0.234, 0.320] gap — a margin of ~0.04 on the tight side. Top-1 landed in a
document that answers the question on **7 of 7**.

Two limits stated rather than glossed. It is a **13-query dev measurement**: a starting point, and
VOX-033's gate over a written query set is what tests it at scale. And it has **no held-out
number** — `heldout-v1` is sealed and contains zero document queries (its categories are greet /
entity / ambig / escalate / refuse), and it is not reopened for this.

---

## Grounded answer (POC, VOX-031)

The retrieved chunks become a spoken answer — `make answer Q="..."`, `src/answer.py`,
`prompts/answer_from_source_v1.md`. **No new arm and no new provider.** The call goes out through
`arms.llm()` like every other LLM call here, so the cost logger, the `--llm` flag, the rate-limit
cooldown and the local ollama fallback all apply without a line of code. That is the whole reason
the POC routes through `arms.llm` rather than calling `httpx` itself.

```
retrieve(question) ──> [] ──────────────────────────────> REFUSAL, sources []   no model call
                   └─> hits ──> answer prompt + excerpts ──arms.llm──> answer, sources doc:page
                                                                    └─> REFUSAL, sources []
```

**A floor miss never reaches the model.** No chunk cleared `RETRIEVAL_SCORE_FLOOR`, so there is
nothing to be grounded *in* and nothing for a model to do but invent. That path costs no tokens, no
round trip, and admits no hallucination — and it makes the refusal testable with no key, which is
what lets VOX-033 measure a refusal rate offline.

The model still has to refuse on the case the floor cannot catch: chunks that score well because
they come from the document that *ought* to answer the question, and then stop short of the answer.
Measured below, both paths fire.

**One refusal sentence, not two.** `answer.REFUSAL` is the literal string in the prompt file, and
`tests/unit/test_answer.py` asserts it. Two wordings would let a listener hear which path ran, and
"the score floor rejected this" is an implementation detail of retrieval, not information about
someone's leave. A model refusal has its `sources` emptied — citing four documents next to "I could
not find that" would claim they support an answer that was never given.

**Citations are the provenance of the context, not a choice the model made.** `sources` is the
deduped `doc_id`/`page` of the chunks passed in, best first. The model is never asked to emit a
citation token, so there is no format for it to get wrong — which matters because the fallback arm
is a 3B and a parse failure there would land on the turn that had already gone wrong. The cost is
stated rather than hidden: `sources` says what the answer was **grounded in**, up to five chunks,
not which sentence it came from. VOX-033's correct-source@3 is a retrieval measurement and is
unaffected; per-sentence attribution needs a different mechanism than this ticket bought.

### Measured, `scripts/ask.py --answer`, 2026-08-20

Four queries against the 215-chunk corpus. `prompt_tokens` and `latency_ms` are the real
`runs/calls.jsonl` fields, not estimates.

| question | chunks | prompt tok | arm | latency | outcome |
|---|---|---|---|---|---|
| how many casual leaves am I entitled to in a year | 3 | 1380 | Llama-3.1-8B @ nim | **942 ms** | answered, `leave-policy:p4,p3,p7` |
| what is the dress code on casual fridays | 2 | 1069 | Llama-3.1-8B @ nim | **519 ms** | answered, `code-of-ethics…:p12` |
| how much is the casual leave encashment paid per day | 4 | 1463 | Llama-3.1-8B @ nim | **691 ms** | **model refused** — chunks cleared the floor, answer not in them |
| is there a canteen subsidy for lunch on working days | 0 | — | none called | **0 ms** | **refused before any call** — top-1 0.232 < floor 0.280 |
| how many casual leaves … (same, `LLM=llama-3.2-3b`) | 3 | 1380 | Llama-3.2-3B @ ollama | **72 650 ms** | answered, same text and same sources |

Two things to carry into VOX-032. Five 300-token chunks is ~2 000 prompt tokens, so `k=5` costs
roughly **6x the plain-reply prompt** and the remote arm still answers inside a second. The local
fallback does not: **72.6 s** against 0.94 s remote, on the same question and the same chunks. The
plain-reply fallback was tolerable because its prompt was small; on the grounded path a rate limit
does not cost the turn's quality, it costs the turn. That belongs in the latency table VOX-032
adds, and it is an argument for `k` being the knob that moves first.

---

## Grounded turn (VOX-032)

The two POC pieces move inside the turn. After STT the transcript goes to retrieval, and what comes
back decides which prompt writes the reply — `make demo`, `make ground` for the same thing from a
recording.

```
STT ──> retrieve ──> hits ──> answer prompt + excerpts ──arms.llm──> spoken answer + doc:page
                  └─> []   ──> reply prompt ────────────arms.llm──> ordinary spoken reply
```

**The router is the measured floor, not a classifier.** `RETRIEVAL_SCORE_FLOOR` was calibrated
against `evals/dev/retrieval_floor_queries.json` and can be re-run; an intent model in front of it
would be a second, unmeasured decision, and it fails in the expensive direction — a misrouted
greeting costs a plain reply, a misrouted policy question costs an invented policy number.

**`t_retrieval_ms` sits beside the five VOX-003 fields, not inside them.** `TURN_FIELDS`,
`stage_sum_ms` and `ok` all derive from `telemetry.STAGES`, and the phase gates read exactly those,
so a sixth stage would redefine the split — and would make `ok` false for a turn that ran without an
index, which is a turn that spoke perfectly well. It is timed *outside* `stage("llm")` for the
opposite reason: retrieval is ~1 ms against an LLM call of ~500 ms, and folding it in charges the
model for time it never spent. `tests/unit/test_grounded_turn.py` asserts both.

**Three states, three different sentences.** No chunk cleared the floor → the plain reply path, as
before this ticket. No index at all (a clean clone: the corpus is gitignored) → also the plain path,
announced once at startup and never per turn, because a demo that quietly stopped being grounded
looks identical to a run of uncovered questions. Chunks that cleared the floor without containing
the answer → the model's refusal, spoken. `--no-kb` forces the pre-RAG path for a whole run.

**`grounded` is on every turn line that reached a reply**, not only the grounded ones — VOX-033 sums
it into a rate, and a rate needs its denominator recorded while the turns were happening. Beside it:
`retrieved`, `sources` (Hit.source spelling, so the printed line and the JSONL cannot drift) and
`top_score`.

**One routing, two loops.** `src/loop.py` and `src/harness.fixture_turn` both call
`answer.turn_reply()`. The harness defaults `idx=None` rather than to the process-wide index, so
`scripts/compare_arms.py` still times the plain path — a grounded turn carries ~5x the prompt, and
that cost belongs to the knowledge base, not to the arm being compared.

### Measured, `make ground`, 2026-08-20

Five grounded turns and three plain ones, same 3.48 s recording for the grounded rows, real
`runs/turns.jsonl` and `runs/calls.jsonl` fields.

| | grounded (n=5) | plain (n=3) |
|---|---|---|
| `t_retrieval_ms` | 0.5-1.0 | not written (no index) / 0.7-0.9 (miss) |
| `prompt_tokens` | **1381** | 269-283 |
| llm call | 367-844 ms | 383-509 ms |
| `t_stt_ms` | 303-317 | 303-408 |
| `t_tts_ms` | 4048-5894 | 3093-4155 |
| `time_to_first_audio_ms` | 5568-7825 | 4780-5683 |

So the grounded prompt is **5.1x** the plain one and the remote arm still answers inside a second —
the two llm ranges overlap, so at k=5 and 215 chunks the context is not what the user waits for. TTS
remains the largest stage in every turn, as it has been since VOX-003. Retrieval is ~1 ms and is
invisible at this corpus size; what it buys is the 12-working-days answer coming out of
`leave-policy:p4` instead of out of the model's memory.

The local fallback is the caveat carried over from VOX-031 and it has not been re-measured here:
1381 prompt tokens took the 3B **72.6 s** in that ticket's table. On the grounded path a rate limit
does not cost the turn's quality, it costs the turn, and `k` is the knob that moves first.

### Two findings this wiring surfaced — both since addressed, see § Hybrid retrieval

**A user's own numbers can push a covered question under the floor.** Measured, `scripts/ask.py`:

Both rows are the same chunk, `separation-policy:p13` — the one that states the PL encashment rule.

| query | terms | its raw BM25 | its score | outcome |
|---|---|---|---|---|
| `if I have base pay of 10,000 and PL balance of 12, how much my leave encashment would be` | 9 | 15.41 | **0.250** | nothing cleared the floor 0.280 — refused with no model call |
| `if I have base pay and PL balance how much leave encashment would I get` | 7 | 15.41 | **0.331** | returned top-1, answered |

Identical raw BM25 sum, different normalised score. `10`, `000` and `12` contribute nothing to the
numerator and are charged near-maximum IDF in the query ceiling (see the OOV rule above), so they
dilute a question the corpus does answer. This is a VOX-030 scoring decision meeting real spoken
questions for the first time. **Since fixed** by the dense half: the same question now retrieves
`separation-policy:p13` and is answered. The lexical floor was not nudged.

**Grounding does not constrain arithmetic.** With the floor dropped so the same question retrieves,
`separation-policy:p13` states that PL encashment is based on last-drawn basic salary and the number
of eligible days, and gives **no per-day rate**. The arm answered *"You will be paid 12,000 for your
leave encashment"* — a computed number the excerpts do not support, carrying four citations.
`answer.py` guards against claims that were never retrieved; it does not guard against a
calculation invented on top of what was. **Since fixed** by
`prompts/answer_from_source_v2.md`, which forbids computing a figure — see § Hybrid retrieval.

---

## Hybrid retrieval (BM25 + a sentence encoder)

Written after a live `make demo` turn got a wrong answer to a question the corpus answers.

**The failure.** "How many paternal leaves am I entitled to according to policy". The corpus says it
plainly — `leave-policy:p12`, *"Every married male employee will be allowed to take 5 calendar days
leave in one go for his new-born"* — and BM25 put that chunk at **rank 110 of 137**. `paternal` is
not `paternity` and `leaves` is not `leave`, so neither term matched anything at all. The turn took
the plain-reply path and answered from the model's general knowledge.

Two things were wrong and only one of them was the wording:

| query | terms | lexical top-1 | rank of the answering chunk |
|---|---|---|---|
| `paternity leave` | 2 | 0.800 | **1** |
| `paternity leave entitlement` | 3 | 0.605 | **1** |
| `how many paternity leaves am I entitled to` | 4 | 0.276 | 11 |
| `How many paternal leaves am I entitled to according to policy` | 6 | 0.140 | **110** |

The score is BM25 divided by the query's own information ceiling, so every filler word a person
actually says — `many`, `entitled`, `according`, `policy` — is charged into the denominator. A
spoken question is 4-8 words and mathematically cannot reach a floor calibrated on 2-3 word queries.
Note row three: **with the exact corpus word**, "how many paternity leaves am I entitled to" scores
0.276 against a floor of 0.280. It was never only about "paternal".

### The two halves

```
query ──> BM25 (terms)     ──> lexical rank + score in [0,1)  ─┐
      └─> bge-small (meaning) ──> cosine rank + cosine        ─┴─> RRF ──> top k
```

`BAAI/bge-small-en-v1.5`, local, 384-dim, CLS-pooled with the query instruction the model was
trained with. It is an arm (`config.EMBED_ARMS`, `--embed`), so `sentence-transformers/all-MiniLM-L6-v2`
is a flag away and the two are comparable the way VOX-013 compares arms.

**Fused by rank, never by score.** A normalised BM25 fraction and a cosine are different units;
adding them with weights means inventing an exchange rate and then tuning it, which is a knob with
no measurement behind it. RRF asks each half only where it put the chunk.

**A chunk is kept if either half vouches for it**, and a refusal needs both to miss. Union and not
intersection: the halves fail on different questions, and requiring both would keep only the
questions BM25 could already answer.

**The half with no evidence abstains from the ordering.** This is the part that was not obvious and
it cost an afternoon. With equal-weight RRF the paternity chunk was *dense rank 1* and still missed
the top five, because two lukewarm ranks (lex 5 + dense 3, on chunks that answer nothing) outscore
one excellent rank plus one terrible one. So when BM25's own best chunk is under `floor`, BM25 has
not found this question and its ordering is noise: it stops voting. That is the floor applied to the
query rather than to the chunk — the same measured number, used for what it actually measures.

The dense half gets no such courtesy, and the asymmetry is measured rather than assumed: on the dev
queries the lexical score separates answerable from absent and **no dense signal does** — not the
cosine, not its z-score against the corpus, not its margin over the mean, not the gap to the 6th
best chunk. An encoder that cannot tell when it is lost cannot be asked to abstain.

**The encoder is a model call**, so it goes through `arms.embed()` and the cost logger like
everything else, and `turn_id` joins the query embedding to its turn. The sentence in `retrieval.py`
about being the one stage with no cost line is gone, because it stopped being true.

**No fallback arm, deliberately.** A second encoder answers in a different vector space from the
cached chunk vectors, so every cosine would be arithmetic between unrelated bases — silently, since
a meaningless cosine is still a number in [-1, 1]. If the encoder cannot load, the dense half is
skipped and BM25 answers alone: a worse ranking, not a wrong one.

### Measured, 2026-08-20

Index build, `make index`: 215 chunks encoded in **25.1 s** (117 ms/chunk, once per re-index),
cached to `runs/embeddings.npz` keyed by a fingerprint of the chunk text. All 215 vectors unit norm
(min 1.0000, max 1.0000), which is what makes `DENSE_SCORE_FLOOR` a cosine at all.

Per turn, `t_retrieval_ms` went from **~1 ms to 83-117 ms** — that is one encoder forward pass on
CPU, and it is the price of the ranking. Against `t_stt_ms` 271-370 ms and `t_tts_ms` 4-5.8 s it is
not what the user waits for.

Floors, `make floors` (13 dev queries, 7 answerable / 6 absent):

| | answerable | absent | separable? |
|---|---|---|---|
| lexical (BM25 ÷ ceiling) | min **0.320** | max **0.234** | yes — floor 0.277 |
| dense (cosine) | min **0.676** | max **0.738** | **no** |

Top-1 landed in an expected document 7/7 for *both* halves — the encoder ranks as well as BM25 on
the queries BM25 can do, and better on the ones it cannot.

The dense column does not separate, and `config.DENSE_SCORE_FLOOR` says so rather than implying a
gap. 0.65 is chosen, not derived: 0.026 under the weakest answerable query, above two of six absent
ones. The alternative was 0.674 — under the weakest answerable by 0.002, which is a coincidence and
not a threshold. A false refusal costs a real employee their answer; a false hit costs one free-tier
call that ends in the right refusal. Not symmetric, and the floor is set accordingly.

So retrieval alone routes **9/13**. End to end, with the prompt and the numeric guard below doing
the job the floor cannot, it routes **14/14** — including the paternity question, which now
answers:

```
Q  How many paternal leaves am I entitled to according to policy
A  You are entitled to 5 calendar days of paternity leave in one go for your newborn.
   grounded=True  sources=['leave-policy:p12', ...]
```

At temperature 0.3 one query routed wrong here — "how many days of sabbatical leave can I take"
came back as *"There is no mention of sabbatical leave in the provided excerpts"*, which is a
refusal in substance but not the `REFUSAL` string, so `is_refusal()` read it as an answer and the
turn logged `grounded: true`. It went away with deterministic sampling, below. It is worth
remembering as a shape rather than a fixed bug: a paraphrased refusal counts as an answer.

VOX-033's gate survives it by counting refusal-shaped phrasings itself — a list in
`gate_poc_pdf.REFUSAL_SHAPES`, every entry naming the *source* ("no mention of", "does not contain",
"not stated in") rather than being a bare negation, because "there is no cap on carry forward" is an
answer. `answer.is_refusal()` is deliberately **not** widened: it decides what the live turn loop
writes into `runs/turns.jsonl`, so changing it re-opens every number in this section, which is its
own before/after measurement rather than a side effect of writing a gate. The gate prints how many
it caught, and the reply that produced each — being generous about what counts as a refusal is
generous in the direction that makes the gate easier to pass, so it is auditable rather than
trusted. Caught on the ten-query set as measured below: **0**.

### The prompt had to move too: `answer_from_source_v2.md`

Grounding constrained which facts the model used and not what it did with them. Asked *"if I have
base pay of 10,000 and PL balance of 12, how much my leave encashment would be"*, v1 answered **"you
will be paid 12,000"** — cited, fluent, and not in any excerpt. The chunk it had gives the formula
`(last drawn basic salary / days in the year) * eligible balance`, which is not 12,000 for any
reading of those inputs.

v2 forbids computing a figure and says what to do instead — state the rule, let the person apply it:

```
A  Leave encashment will be paid along with the final settlement of salary. Payment will be
   calculated based on the basic salary and the number of PL days eligible for encashment.
```

Versioned as a new file rather than edited in place (VOX-018): the numbers in the VOX-031 table
above were measured against v1, and a prompt you can no longer read is a measurement you can no
longer reproduce.

### Asking was not enough: deterministic sampling and a numeric guard

`answer_from_source_v2.md` forbids computing a figure. A live turn then asked *"if I have a base pay
of 5000 rupees and 20 privileged leave, then how much will I get in leave encashment?"* and heard:

```
vox says : 'You will get 12 rupees in leave encashment.'
  grounded in leave-policy:p10, p11, p7, p8, p9 (top score 0.103, 5 chunks in context)
```

Asked three times, the same question returned two correct refusals and that. **Nothing was wrong
with the prompt on the two runs where it worked** — the answer was being sampled from a distribution
that contains the bad one, at `nlu.TEMPERATURE = 0.3`.

**So the grounded path samples at 0** (`answer.ANSWER_TEMPERATURE`), while a spoken reply keeps 0.3.
Temperature became a per-*call* option rather than an arm field, so both still run on the same arm
and stay comparable. Reading five policy excerpts is not a task where variety is a feature, and a
gate that cannot reproduce its own number is not a gate.

That made it reproducible and still wrong — consistently, now:

> You are entitled to 12 working days of Privilege Leaves every year... Since you have 20 privileged
> leave, which is 16 days more than the 24-day cap, you will get 16 days in leave encashment.

Fluent, cited, and false twice over: 20 is not more than 24, and no excerpt contains 16.

**So the prompt's own rule is enforced in code.** `answer.ungrounded_numbers()` extracts every figure
a reply states — digits and words alike, so "twenty-five thousand" and "25,000" are the same number —
and checks each against the excerpts. A figure that appears in none makes the reply ungrounded, and
it becomes the same refusal a listener would have heard if retrieval had missed. The suppressed
reply goes to stderr with the offending number so the refusal can be explained.

Two asymmetries make it usable rather than merely strict:

- **A phrase asserts its value, not its pieces.** "twenty-five thousand" claims 25000; an answer
  held to 5 and 20 as well would be refused for saying a number correctly. On the *excerpt* side the
  pieces do count, so a document written "25 thousand" still matches. Generous about what a document
  contains, strict about what a reply claims.
- **A number the person supplied is not grounded by having been asked.** The excerpts are the only
  source. Echoing the caller's own figure back inside an answer is the shape of the fabricated
  calculation, not an innocent restatement — every wrong variant above was caught on the `20`.

Measured, `make floors` queries end to end: **14/14**, with no answerable query refused by the
guard — including the ones whose answers really do contain numbers ("twelve working days",
"25,000", "the eighth of November", "three months", "5 calendar days"). The encashment question now
refuses. That is blunter than the ideal answer (state the rule, let the person apply it) and it is
the right failure: a refusal costs a question, a wrong rupee figure costs trust.

### What this does not fix

- **STT is upstream of all of it.** Spoken, the same encashment question came back as *"how much my
  leaving cashment would be"* — the one high-information term destroyed before retrieval ran. That
  is VOX-021 (vocabulary biasing, measured before and after) and it now has a concrete case.
- **A 33M encoder does not know when it is lost.** Every question now reaches the model unless both
  floors reject it, so the refusal budget has shifted from arithmetic to tokens. A larger encoder is
  the obvious next arm to measure, and it is a row in `config.EMBED_ARMS`.
- **The numeric guard checks presence, not meaning.** "You will get 12 rupees" would survive it if
  some excerpt happened to contain a 12 — as one did. What killed that reply was the caller's own
  `20`, not the wrong unit on the `12`. Catching a number that is real but means something else
  needs a different mechanism than this one.
- **The floor-calibration set is 13 queries**, all written before any of this was known. It is a
  starting point, not a distribution. The queries in this section were not in it; VOX-033 added them
  to a second set, `evals/dev/pdf_queries.json`, scored below. Two files rather than one, because
  those thirteen are what `RETRIEVAL_SCORE_FLOOR` was fitted to and a gate scored on them measures
  how well the floor was fitted.

---

## The POC gate: what the grounded path actually scores (VOX-033)

`make gate-poc` -> `tests/gates/gate_poc_pdf.py` over `evals/dev/pdf_queries.json`: ten written
queries, eight with an expected source `file:page` and two the corpus does not cover. Measured
2026-08-21, `meta-llama/Llama-3.1-8B-Instruct` on the NVIDIA NIM free tier, identical across two
consecutive runs:

| number | measured | floor | asserted? |
|---|---|---|---|
| correct-source@3 | **7/8 = 0.875** | 0.875 | yes |
| refusal rate on the 2 absent | **2/2 = 1.000** | 1.000 | yes |
| grounded-answer rate | **8/8 = 1.000** | — | no |
| ...grounded *on an expected source* | **7/8 = 0.875** | — | no |
| paraphrased refusals caught | **0** | — | no |

19 model calls, `cost_usd=0.0` on every one (10 local encoder passes, 9 remote LLM — q10 is refused
by the floor with no call at all).

**The last two rows are the finding.** A grounded-answer rate of 8/8 next to a correct-source@3 of
7/8 is not a rounding difference: `grounded` only says a model answered from the excerpts it was
handed, and it is true even when those excerpts came off the wrong pages. q03 — *"how much my
leaving cashment would be"*, the STT-damaged form of q02 — retrieved `leave-policy` p7/p10 (leave
accrual arithmetic, dense-half hits at lexical score 0.000) and answered *"any excess beyond 24 days
will be subject to encashment"* from them. Fluent, grounded, cited, and off the wrong document; the
rule it was asked about is on `separation-policy` p13 and p26. So the rate is printed with the
intersection under it, and the intersection is the honest reading.

That is also why the grounded-answer rate is not asserted. It is bounded above by correct-source@3
and below by the numeric guard's correct refusals, so a floor on it would fail this gate twice for
one cause.

**The floor is 7/8 and the measurement is 7/8, by design rather than by luck.** The one query of
slack is spent in advance on q03, named in the query set's `_note` before the gate first ran: it is
a measurement of STT damage, not of retrieval, and it is the concrete case VOX-021's vocabulary
biasing exists for. Seven clean queries pass; a regression on any of them takes this to 6/8 and
fails. Two of the seven are hits at rank 3 and not rank 1 — the dress-code question, where
`code-of-ethics` p12 and `annual-event-policy` p3 both outrank the right page — which is what makes
@3 rather than @1 the number worth printing.

**Two candidate labels were cut rather than relabelled**, and they are the reason every label was
read off `runs/chunks.jsonl` before it was written down. *"Can I accept a gift from a vendor"* looked
like an absent query because retrieval missed it — but `grep -ic gift runs/chunks.jsonl` returns 5,
so the corpus does cover it and the label would have been false. *"Who pays for my hotel stay on
official travel"* looked answerable — but `travel-policy` has no accommodation section and "hotel"
appears only incidentally, in a clause about travel *from* the hotel to the office. Label error is
the one failure mode that fails a gate for nothing.

**The two absent queries take the two different refusal paths on purpose**, because `src/answer.py`
distinguishes them and only one involves a model: q09 (paid menstrual leave) clears the floor on
leave-policy chunks about other leave types and is refused *by the model* from excerpts that look
relevant; q10 (health insurance coverage) is rejected by the floor and refused with no model call.
Both are HR-shaped and share corpus vocabulary, and neither subject appears anywhere in the corpus.

**Limit, printed in the gate's own output rather than left here.** These queries are dev-only.
`evals/heldout/` is sealed as `heldout-v1` and holds zero document queries — its categories are
greet / entity / ambig / escalate / refuse — and it was not reopened for this. So none of the numbers
above has a held-out counterpart, and nothing bounds how much the implementation was shaped by these
ten cases. Reopening a sealed set to add a category is how a held-out set becomes a dev set with
extra ceremony; the cost of not doing it is this paragraph.

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
