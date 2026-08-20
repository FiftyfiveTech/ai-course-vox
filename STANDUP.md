# Standup log

Append at the end of each working session. Two minutes. Newest at the bottom.

Format — one block per session:

```
## YYYY-MM-DD — <name> (Builder|Evaluator)
Did:      what actually landed, with the task id
Number:   any measured number + the command that produced it (or "none today")
Blocked:  what is in the way, or "nothing"
Next:     the single next task id
```

Rule: a number goes in this file only if it appeared in your terminal. No number is better than a
remembered one.

## 2026-08-20 — Vimal (Builder)
Did:      VOX-013. `make compare` runs two whole architectures as real turns and prints the
          five-field VOX-003 split for both. Added the piper TTS arm (`rhasspy/piper-voices`,
          22.05 kHz) because the ticket names it and it is the only TTS arm faster than Kokoro.
          Extracted `src/harness.py` so the comparison and `make turn` run the same turn code
          instead of two copies. Fallback is off in compare, so a refused stage reads FAILED
          rather than borrowing the local arm's latency.
Number:   `make compare` — n=3 per arm, interleaved, hello_testing_voice.mp3, exit 0.
          median time_to_first_audio: fast 6695 ms (band 6202-6717), quality 9291 ms (band
          8914-21269). tts per reply char: piper 7.53-9.47 ms, Kokoro 87.36-102.65 ms.
          t_llm: 3B/ollama 4267-4436 ms, 70B/NIM 1726-15428 ms.
          One of three 70B calls took 15428 ms against REMOTE_TIMEOUT_S of 10 s, so a third of
          turns would fall back on the shipped config. Full table in ARCHITECTURE.md.
Blocked:  nothing. Found and fixed a real bug on the way: reusing one endpointed Capture across
          turns freezes speech_end_t, so time_to_first_audio read 6.5 s / 39.4 s / 63.6 s on three
          identical turns. compare now endpoints per turn and asserts the segment is unchanged;
          tests/unit/test_compare.py pins it.
Next:     VOX-014 — write the phase-1 finding.
