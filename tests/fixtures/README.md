# tests/fixtures

Small recordings used to check that the turn loop works, **not** eval cases.

Eval audio belongs in `evals/dev/` or `evals/heldout/`, is listed in a manifest, and is scored.
Nothing here is scored, nothing here is labelled, and nothing here may be added to a manifest —
`tests/gates/test_no_leakage.py` compares the dev and held-out splits by content hash, and a
fixture that drifted into either split would corrupt that check.

## hello_testing_voice.mp3

Spoken by Vimal, recorded 2026-08-17 for VOX-002. 4.30 s, mono, 48 kHz.
Transcript: *"Hello. So this is testing."*

Kept because it is the reason two bugs were found, and both are regressions worth being able to
reproduce:

1. It contains a **1.05 s mid-sentence pause** after "Hello". At the original
   `VAD_SILENCE_MS = 700` the endpointer cut the turn there and STT returned `"Bye."`. This file
   is the evidence behind the provisional 1100 ms in `src/config.py`.
2. The resulting 1.0 s clip made `whisper-large-v3-turbo` auto-detect **French** and return
   `"Salut !"`, after which the LLM replied in French. STT now pins `language=en`.

`make demo` does not read this file — it is live-mic only. To drive the whole chain from it,
including telemetry and playback:

```
uv run python scripts/turn_from_fixture.py tests/fixtures/hello_testing_voice.mp3
```

That script goes through `Endpointer`, the same decision logic the live loop runs, and appends a
turn line to `runs/turns.jsonl`. Read its `t_vad` and `time_to_first_audio` with the caveat in the
script's docstring: frames arrive as fast as the CPU allows, so the ~1.1 s hangover a live turn
waits out is missing from them.

## casual_leave_question.mp3

**Synthesised, not recorded.** `hexgrad/Kokoro-82M` reading *"how many casual leaves am I entitled
to in a year"*, written 2026-08-20 for VOX-032. 3.48 s, mono, 16 kHz, 18 KB.

It exists because VOX-032 put retrieval inside the turn, and the only way to demonstrate that
without standing at a microphone is to hand the loop a spoken question the corpus can answer:

```
make ground        # scripts/turn_from_fixture.py tests/fixtures/casual_leave_question.mp3 --kb
```

That prints the doc:page the answer was grounded in and appends `grounded`, `sources` and
`t_retrieval_ms` to `runs/turns.jsonl`. Needs `make index` to have run — the corpus itself is
gitignored, so this file is a *question* about internal documents and contains none of their
content.

Being synthetic is the point and also the caveat: it is clean, evenly paced audio with no room
tone, so it says nothing about STT on real speech. It is here to exercise the routing, not the
recogniser — which is why the human recording above is still the one the endpointing regressions
are checked against.

For just the endpointing decision:

```python
import soundfile as sf, torch, torchaudio
from src import vad, stt
from src.config import SAMPLE_RATE
from src.telemetry import new_turn_id

audio, sr = sf.read("tests/fixtures/hello_testing_voice.mp3", dtype="float32", always_2d=True)
a16 = torchaudio.functional.resample(torch.from_numpy(audio.mean(axis=1)), sr, SAMPLE_RATE).numpy()
cap, state = vad.endpoint_frames(vad.frames_from(a16))
print(state, cap.t_vad_ms, stt.transcribe(cap.segment, new_turn_id()))
```
