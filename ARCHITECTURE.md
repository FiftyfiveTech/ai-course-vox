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
  config.py        — model registry (HF repo id -> provider) + endpointing constants
  telemetry.py     — shared cost/latency logger; every model call goes through this
  vad.py           — endpointing via snakers4/silero-vad (local)
  stt.py           — speech-to-text via openai/whisper-large-v3-turbo (Groq)
  nlu.py           — transcript -> spoken reply; structured extraction lands in VOX-019
  audio.py         — speaker playback, kept apart from synthesis so VOX-011 can interrupt it
  loop.py          — one chained turn; `make demo`
  confirm.py       — confirmation flow logic (VOX-020, not yet written)
  actions.py       — tool/API calls (read and write) (not yet written)
  tts.py           — text-to-speech via hexgrad/Kokoro-82M (local)

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

| Role | HF repo id | Runs on |
|---|---|---|
| VAD / endpointing | `snakers4/silero-vad` | local |
| STT | `openai/whisper-large-v3-turbo` | Groq free tier |
| NLU / reply | `meta-llama/Llama-3.1-8B-Instruct` | NVIDIA NIM free tier |
| TTS | `hexgrad/Kokoro-82M` | local |

Provider = where it runs. The model id is the HF repo id, not the provider name. The mapping from
repo id to each provider's own model string lives only in `src/config.py`.

VOX-006 replaces this fixed table with arms selectable by flag; until then Phase 0 pins one arm
per stage so a measured number always has an unambiguous model behind it.

---

## Open questions (for sign-off session)

1. ~~Which NLU model on NVIDIA NIM free tier?~~ **Settled by the VOX-002 ticket:**
   `meta-llama/Llama-3.1-8B-Instruct`. Revisit against 3.3-70B under VOX-013 with measurements.
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
