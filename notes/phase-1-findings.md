# Phase 1 findings

Gate command: `.venv/bin/python tests/gates/gate_phase1.py`
Gate result: PASS (turns=10, arms=9, barge-in NOT_RUN — VOX-011 pending)
Measured: 2026-08-20, 50 turns from `runs/turns.jsonl`

---

## Which stage dominates time-to-first-audio

| Stage | Avg (ms) | Min | Max | Share of STT+LLM+TTS |
|---|---|---|---|---|
| STT | 302 | 217 | 478 | **10.5%** |
| LLM | 644 | 344 | 2773 | **21.1%** |
| TTS | 4249 | 1628 | 7639 | **68.4%** |
| **time_to_first_audio** | **4397** | 3793 | 5114 | — |

**TTS dominates.** It accounts for 68% of the STT+LLM+TTS budget and is the single largest contributor to time-to-first-audio. The user waits for Kokoro to synthesise the full reply before hearing a single word.

---

## What optimising the LLM would have bought

LLM averages 644 ms. Even eliminating the LLM entirely would save at most 21% of the pipeline — roughly 900 ms off a 4.4 s wait. That does not move the experience from "slow" to "fast".

The 70B arm (meta-llama/Llama-3.1-70B-Instruct on NIM) averaged 6805 ms — about 10× the default 8B arm. Switching up to 70B costs more than the entire STT+TTS pipeline combined. The 8B arm (451 ms) and gpt-oss-120b on Groq (585 ms) are both acceptable; the size difference between them does not change time-to-first-audio meaningfully.

---

## Arm comparison (from gate_phase1.py)

### STT

| HF repo id | Provider | call_ms | Notes |
|---|---|---|---|
| openai/whisper-large-v3-turbo | groq | 253 | default; 4 decoder layers |
| openai/whisper-large-v3 | groq | 296 | 32 decoder layers, +43 ms |
| Systran/faster-whisper-base | local | 635 | int8, 2.5× slower than Groq |
| openai/whisper-base | local | 2229 | transformers, 8.8× slower than Groq |

Remote Groq is 2.5–8.8× faster than local for STT. The turbo vs full-size difference (253 vs 296 ms) is small enough that accuracy should decide, not latency.

### LLM

| HF repo id | Provider | call_ms | Notes |
|---|---|---|---|
| meta-llama/Llama-3.1-8B-Instruct | nvidia-nim | 451 | default |
| openai/gpt-oss-120b | groq | 585 | reasoning model, reasoning_effort=low |
| meta-llama/Llama-3.1-70B-Instruct | nvidia-nim | 6805 | 15× slower than 8B |

The 8B arm is the right default. The 70B arm's latency cost is not justified for short spoken replies.

### TTS

| HF repo id | Provider | call_ms | Audio length | Notes |
|---|---|---|---|---|
| hexgrad/Kokoro-82M | local | 2333 | 5.08 s at 24 kHz | default |
| microsoft/speecht5_tts | local | 4028 | 4.93 s at 16 kHz | 1.7× slower |

Kokoro is the right default. SpeechT5 is slower and lower quality.

---

## Where to optimise next

1. **TTS — streaming synthesis.** Kokoro synthesises the whole reply before playback starts. Streaming phoneme-by-phoneme would cut perceived latency from ~4 s to ~200 ms (time-to-first-syllable). This is the highest-leverage change in the pipeline.

2. **STT — keep Groq turbo.** At 253 ms it is not a bottleneck. Local fallback (635 ms faster-whisper-base) adds 380 ms — acceptable for a rate-limited turn.

3. **LLM — keep 8B on NIM.** 451 ms. The next meaningful gain (eliminating LLM) saves only 21% of time-to-first-audio and is not on the roadmap.

---

## Barge-in (VOX-011)

Measured stop latency: NOT_RUN (stub in gate_phase1.py). VOX-011 is in Review. Once merged, the stop latency will be printed on the turn record as `barge_stop_ms`.
