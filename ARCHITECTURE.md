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

| Stage | Budget | Notes |
|---|---|---|
| VAD (end-of-speech detection) | ~100 ms | local; no network hop |
| STT (`openai/whisper-large-v3-turbo` on Groq) | ~300 ms | Groq free tier; streaming not used |
| NLU / entity extraction (NVIDIA NIM) | ~500 ms | one structured-output call |
| Confirmation TTS + user response | not counted | adds one extra turn |
| Action (internal API) | ~200 ms | internal network |
| TTS (response) | ~200 ms | local or NIM |
| **Total (no confirmation)** | **~1.3 s** | well within budget |
| **Total (with confirmation)** | **2–4 s** | acceptable for write actions |

Telemetry for every model call is logged via `src/telemetry.py` (cost + latency).

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
  telemetry.py     — shared cost/latency logger; every model call goes through this
  vad.py           — voice activity detection (local)
  stt.py           — speech-to-text via openai/whisper-large-v3-turbo (Groq)
  nlu.py           — entity extraction; calls LLM with structured output
  confirm.py       — confirmation flow logic
  actions.py       — tool/API calls (read and write)
  tts.py           — text-to-speech

prompts/
  extract_v1.md    — NLU prompt (versioned; never inline)

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
| STT | `openai/whisper-large-v3-turbo` | Groq free tier |
| NLU / entity extraction | TBD (e.g. `meta-llama/Llama-3.1-8B-Instruct`) | NVIDIA NIM free tier |
| TTS | TBD (e.g. local `espnet/kan-bayashi_ljspeech_vits`) | local / Ollama |

Provider = where it runs. The model id is the HF repo id, not the provider name.

---

## Open questions (for sign-off session)

1. Which NLU model on NVIDIA NIM free tier? (Llama-3.1-8B vs 3.3-70B — latency vs accuracy)
2. TTS: local `espnet` or NIM? Local is zero-cost but voice quality is lower.
3. VAD library: `silero-vad` (local) vs WebRTC VAD?
4. Confirmation intent labels: finalise the affirmative / negative word list.

---

## Setup

```bash
git clone <repo>
cd ai-course-vox
make setup   # creates venv, installs deps, writes .env from .env.example
make test    # unit tests (gates run separately with make gate)
```

Secrets go in `~/.config/`, never in the repo. `.env.example` lists key names only.
