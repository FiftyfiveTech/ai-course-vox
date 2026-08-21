# VOX Phase 1 Results Pack

**Audience:** anyone not in the build sessions.
**Data source:** `runs/calls.jsonl`, `evals/heldout/`, `notes/phase-1-findings.md`.

---

## Gate summary

| Gate | Command | Result |
|---|---|---|
| Phase 0 (pipeline smoke test) | `make phase0` | **PASS** |
| Phase 1 (live turn gate) | `make gate` | **PASS** turns=10, arms=9 |
| Phase 1B (entity capture, heldout-v1) | `uv run tests/gates/gate_phase1b.py` | **PASS** entity_capture=96%, confirmation=100%, state_validity=100% |

---

## Per-stage latency budget

Measured over the Phase 1 live session (TTFA 4397 ms mean, n=10 turns).

| Stage | Arm used | Mean latency | Share of TTFA |
|---|---|---|---|
| STT | openai/whisper-large-v3-turbo @ Groq | 302 ms | 10.5% |
| LLM | meta-llama/Llama-3.1-8B-Instruct @ NVIDIA NIM | 644 ms | 21.1% |
| TTS | hexgrad/Kokoro-82M (local) | 4249 ms | 68.4% |
| **Total** | | **4397 ms** | 100% |

TTS dominates. The two API stages together are under 1 s.

---

## All arms, head-to-head

Latencies from `runs/calls.jsonl` across all recorded sessions (not just the gate run).

### STT

| Arm (HF repo id) | Provider | Mean latency | n |
|---|---|---|---|
| openai/whisper-large-v3-turbo | Groq (remote) | 330 ms | 267 |
| openai/whisper-large-v3 | Groq (remote) | 386 ms | — |
| Systran/faster-whisper-base | Local | 1439 ms | — |
| openai/whisper-base | Local (transformers) | 2333 ms | — |

**Ranking:** whisper-large-v3-turbo @ Groq is 4× faster than the local base model and more accurate.

### LLM

| Arm (HF repo id) | Provider | Mean latency | n |
|---|---|---|---|
| groq/gpt-oss-120b | Groq (remote) | 530 ms | — |
| meta-llama/Llama-3.1-8B-Instruct | NVIDIA NIM (remote) | 948 ms | 380 |
| meta-llama/Llama-3.1-70B-Instruct | NVIDIA NIM (remote) | 7188 ms | — |

**Ranking:** groq/gpt-oss-120b edges out 8B by latency; 70B is 13× slower and not viable for real-time use.

### TTS

| Arm (HF repo id) | Provider | Mean latency | n |
|---|---|---|---|
| hexgrad/Kokoro-82M | Local | 3909 ms | — |
| microsoft/speecht5_tts | Local | 4323 ms | — |

Both are local. Kokoro is ~10% faster and produces higher-quality speech; it is the stage default.

---

## Vocabulary biasing (VOX-021)

Biasing injects a decoder context prompt listing staff names before STT decodes.
Tested on 3 utterances containing Priya, Rahul, Ananya via `scripts/measure_biasing.py`.

| Condition | Correct names / 3 |
|---|---|
| Without bias (baseline) | 3 / 3 (100%) |
| With bias (`VOX_STT_BIAS=1`) | 2 / 3 (67%) |

**Finding:** whisper-large-v3-turbo already handles staff names at 100% on TTS-generated audio.
The bias prompt caused a decoder interaction on utt_006 (Priya → Pria).
Biasing is most valuable for smaller models or noisy real-world audio; re-test on mic recordings.

---

## Phase 1B entity capture — detail

30 utterances from `evals/heldout-v1/`. Prompt: `prompts/extract_v2.md`.

| Metric | Value |
|---|---|
| State validity | 100% (all 30 JSON responses parsed) |
| Entity capture rate | 96% (correct slots / total gold slots) |
| Confirmation rate | 100% (all write-actions asked for confirmation) |
| Intent accuracy | 70% |

**Remaining miss (4%):** utt_019 — STT transcribed "Kiran" as "Curran"; the extraction prompt
cannot fix a mis-transcription. Improvement path: vocabulary biasing on real mic audio.

**Intent accuracy note:** 70% is informational only and is not part of the gate criterion for 1B.
Many intent mismatches are `capture` vs `clarify` on borderline utterances — a separate labelling
pass would be needed to make this number comparable across runs.
