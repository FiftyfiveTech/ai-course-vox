# VOX — Architecture

**Track:** AI Engineering Course, Week 3
**Status:** Draft — awaiting sign-off from Vimal (Evaluator)
**Last updated:** 2026-08-17

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
│  User speaks  →  VAD detects end-of-utterance                       │
│       ↓                                                             │
│  [STT]  whisper-large-v3-turbo  (via Groq free tier)                │
│       ↓                                                             │
│  [NLU / entity extraction]  LLM (NVIDIA NIM free tier)              │
│       structured output validated against schemas/                  │
│       ↓                                                             │
│  [Confirmation check]  — required for every write action            │
│       if needed → TTS response asking "Did you mean …?"             │
│       ↓ (confirmed, or read-only)                                   │
│  [Action]  internal tool / API call                                 │
│       ↓                                                             │
│  [TTS response]  read result back to user                           │
│       ↓                                                             │
│  Wait for next utterance (barge-in allowed — see below)             │
└─────────────────────────────────────────────────────────────────────┘
```

One turn = one VAD segment → STT → NLU → (confirm?) → action → TTS.

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

---

## Barge-in

Barge-in = user speaks while TTS is still playing.

- VAD runs continuously, not only after TTS finishes.
- When VAD detects speech during TTS playback, TTS is **immediately interrupted**.
- The new utterance is queued and processed as the next turn.
- Implementation: the TTS playback thread is killed; VAD segment is handed off to the turn loop.

Barge-in interrupt point: **between TTS playback start and TTS playback end**.
No partial transcriptions are discarded; the full new utterance is captured before STT runs.

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
  vad.py           — endpointing via snakers4/silero-vad (local)
  stt.py           — STT backends: openai-audio (Groq), transformers-whisper, faster-whisper
  nlu.py           — the openai-chat backend + the message assembly arms.llm() takes;
                     structured extraction lands in VOX-019
  audio.py         — speaker playback, kept apart from synthesis so VOX-011 can interrupt it
  loop.py          — one chained turn; `make demo`
  confirm.py       — confirmation flow logic (VOX-020, not yet written)
  actions.py       — tool/API calls (read and write) (not yet written)
  tts.py           — TTS backends: kokoro, speecht5

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
| STT | `Systran/faster-whisper-base` | local | `faster-whisper` | `faster-base` |
| LLM | `meta-llama/Llama-3.1-8B-Instruct` | NVIDIA NIM free tier | `openai-chat` | `llama-8b` **(default)** |
| LLM | `openai/gpt-oss-120b` | Groq free tier | `openai-chat` | `gpt-oss` |
| LLM | `meta-llama/Llama-3.1-70B-Instruct` | NVIDIA NIM free tier | `openai-chat` | `llama-70b` |
| TTS | `hexgrad/Kokoro-82M` | local | `kokoro` | `kokoro` **(default)** |
| TTS | `microsoft/speecht5_tts` | local | `speecht5` | `speecht5` |

The defaults are the arms VOX-002 and VOX-003 measured, so an unflagged run still reproduces those
numbers. Selection:

```bash
make arms                                    # call every arm once, print the calls.jsonl lines
uv run python scripts/check_arms.py --list   # the table, no calls
uv run python -m src.loop --stt openai/whisper-base --tts microsoft/speecht5_tts
uv run python scripts/turn_from_fixture.py --llm gpt-oss     # alias, repo id, or repo/id@provider
VOX_STT_MODEL=faster-base make turn                          # env sets the default; the flag wins
```

`stt_model` / `llm_model` / `tts_model` on every `runs/turns.jsonl` line name the arms that produced
that latency split, so two runs with different arms cannot be quietly averaged.

Two arms sharing a `backend` share an adapter, so a new arm on an existing runtime is a table row in
`src/config.py` and no new code. `resolve()` refuses a bare repo id that two providers serve — the
same Llama can be a NIM arm or a Groq arm, and a silently chosen provider attaches the wrong latency
to the right name.

### What choosing an arm cost, measured

`make arms`, 2026-08-18, one call each on the same 3.30 s segment / same transcript / same sentence.
`load` is excluded — weights are warmed before the call, as in the loop.

| Stage | Arm | Call | Output |
|---|---|---|---|
| stt | `openai/whisper-large-v3-turbo` @ groq | 364 ms | `'Hello. So this is testing.'` |
| stt | `openai/whisper-large-v3` @ groq | 297 ms | `'Hello. So this is testing.'` |
| stt | `openai/whisper-base` @ local | 4685 ms | `'So this is testing.'` |
| stt | `Systran/faster-whisper-base` @ local | 827 ms | `'So this is testing.'` |
| llm | `meta-llama/Llama-3.1-8B-Instruct` @ nvidia-nim | 1503 ms | 17 completion tokens |
| llm | `openai/gpt-oss-120b` @ groq | 633 ms | 61 completion tokens |
| llm | `meta-llama/Llama-3.1-70B-Instruct` @ nvidia-nim | 39286 ms | 16 completion tokens |
| tts | `hexgrad/Kokoro-82M` @ local | 1532 ms | 2.25 s at 24 kHz |
| tts | `microsoft/speecht5_tts` @ local | 1944 ms | 1.92 s at 16 kHz |

Three findings, each n=1 and none of them a distribution:

1. **Both `base` arms drop the first word.** Given identical audio the Groq `large-v3` arms return
   `'Hello. So this is testing.'` and both local `base` arms return `'So this is testing.'`. The
   local arms are not a cheaper version of the same transcript, they are a worse one, and the word
   they lose is the one at the start of the turn. WER on `evals/dev` is the measurement that should
   decide this, not this one clip.
2. **Same weights, 5.7x apart on runtime.** `openai/whisper-base` through transformers took 4685 ms;
   the CTranslate2 int8 conversion of the same model took 827 ms for a character-identical
   transcript. The arm that matters for latency here is the runtime, not the model.
3. **70B on the NIM free tier is not a real-time arm.** 39 s for 16 tokens, against 1.5 s for the 8B
   on the same provider. `meta-llama/Llama-3.3-70B-Instruct` — what open question 1 actually asked
   for — is worse: not on this Groq key's catalogue (404), and no answer from NIM inside 120 s on
   two attempts, so 3.1-70B stands in for it.

Also worth recording, because it constrains arm choice rather than tuning: **every chat model Groq's
free tier now serves is a reasoning model.** At default effort `openai/gpt-oss-120b` spent the whole
120-token budget thinking and returned an empty string, so that arm carries
`request={"reasoning_effort": "low"}` in `src/config.py` — not a tuning knob, a precondition for the
arm answering at all. `src/nlu.py` raises a named error on an empty reply rather than handing
silence to TTS.

---

## Open questions (for sign-off session)

1. ~~Which NLU model on NVIDIA NIM free tier?~~ **Settled by the VOX-002 ticket:**
   `meta-llama/Llama-3.1-8B-Instruct`. ~~Revisit against 3.3-70B under VOX-013 with measurements.~~
   **The 70B revisit is now an arm, and the first measurement is in:** 3.3-70B is unreachable
   (404 on Groq's catalogue; no answer from NIM inside 120 s, twice), and 3.1-70B on NIM answered in
   39 s against the 8B's 1.5 s. It stays registered as `llama-70b` so VOX-013 can measure quality
   against that cost, but it is not a candidate default. See the measured table above.
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
