"""Text to speech: hexgrad/Kokoro-82M, running locally. No network, no spend.

Returns samples rather than playing them. Playback is src/audio.py's job, because VOX-011
(barge-in) has to be able to interrupt playback without touching synthesis.
"""
import numpy as np

from src.config import TTS, TTS_VOICE
from src.telemetry import log_call

_pipeline = None


def _kokoro():
    global _pipeline
    if _pipeline is None:
        from kokoro import KPipeline
        # lang_code 'a' = American English. The weights come from the HF repo id in config.
        _pipeline = KPipeline(lang_code="a", repo_id=TTS.repo_id)
    return _pipeline


def _samples(chunk):
    """Kokoro's yielded chunk shape has moved between releases — take audio from either form."""
    audio = chunk[2] if isinstance(chunk, tuple) else getattr(chunk, "audio", None)
    if audio is None:
        raise RuntimeError(f"cannot find audio in Kokoro chunk of type {type(chunk).__name__}")
    return audio.detach().cpu().numpy() if hasattr(audio, "detach") else np.asarray(audio)


def synthesize(text, turn_id):
    """-> float32 mono array at TTS_SAMPLE_RATE (24 kHz)."""
    with log_call("tts", TTS, turn_id, voice=TTS_VOICE, chars=len(text)) as rec:
        pipeline = _kokoro()
        parts = [_samples(c) for c in pipeline(text, voice=TTS_VOICE)]
        if not parts:
            raise RuntimeError(f"Kokoro produced no audio for {text!r}")
        audio = np.concatenate(parts).astype(np.float32)
        rec["audio_s"] = round(len(audio) / 24_000, 3)

    return audio
