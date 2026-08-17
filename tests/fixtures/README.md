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

`make demo` does not read this file — it is live-mic only. Drive the chain from it with the
`Endpointer` directly, which is the same decision logic the live loop runs:

```python
import soundfile as sf, torch, torchaudio
from src import vad, stt
from src.config import SAMPLE_RATE
from src.telemetry import new_turn_id

audio, sr = sf.read("tests/fixtures/hello_testing_voice.mp3", dtype="float32", always_2d=True)
a16 = torchaudio.functional.resample(torch.from_numpy(audio.mean(axis=1)), sr, SAMPLE_RATE).numpy()
segment, state = vad.endpoint_frames(vad.frames_from(a16))
print(state, stt.transcribe(segment, new_turn_id()))
```
