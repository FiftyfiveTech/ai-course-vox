"""Speech-to-text backends. One adapter per way of running whisper; the arms are in config.py.

Nothing here resolves an arm, logs a call, or knows which arm is the default — `src/arms.py` owns
all three. A backend takes the arm it was told to run, the audio, and the mutable call record it
should add its measured facts to.

The hosted adapter uploads a WAV; the local ones take the float32 array as it comes off the
endpointer. That asymmetry is deliberate — encoding to WAV for a local model would add latency
that only exists to satisfy an HTTP boundary that isn't there.
"""
import io

import httpx
import numpy as np
import soundfile as sf

from src.config import SAMPLE_RATE, STT_LANGUAGE

_loaded = {}   # arm.id -> the loaded local model, so weights load once per process


def _wav_bytes(segment):
    """Encode the float32 segment as a 16-bit WAV in memory — nothing touches disk."""
    buf = io.BytesIO()
    sf.write(buf, segment, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def openai_audio(arm, segment, rec, timeout=30):
    """OpenAI-compatible /audio/transcriptions — Groq's free tier serves both whisper arms here."""
    r = httpx.post(
        f"{arm.api_base}/audio/transcriptions",
        headers={"Authorization": f"Bearer {arm.key()}"},
        files={"file": ("turn.wav", _wav_bytes(segment), "audio/wav")},
        data={"model": arm.provider_model, "response_format": "json",
              "temperature": "0", "language": STT_LANGUAGE},
        timeout=timeout,
    )
    r.raise_for_status()
    text = (r.json().get("text") or "").strip()
    rec["chars"] = len(text)
    return text


def load_transformers_whisper(arm):
    """Load the weights outside a timed turn. Called by arms.warm(); idempotent."""
    if arm.id not in _loaded:
        from transformers import pipeline
        _loaded[arm.id] = pipeline("automatic-speech-recognition", model=arm.provider_model)
    return _loaded[arm.id]


def transformers_whisper(arm, segment, rec):
    """openai/whisper-base on the CPU through transformers. No network, no key."""
    asr = load_transformers_whisper(arm)
    out = asr(
        {"raw": np.asarray(segment, dtype=np.float32), "sampling_rate": SAMPLE_RATE},
        generate_kwargs={"language": STT_LANGUAGE, "task": "transcribe"},
    )
    text = (out.get("text") or "").strip()
    rec["chars"] = len(text)
    return text


def load_faster_whisper(arm):
    if arm.id not in _loaded:
        from faster_whisper import WhisperModel
        _loaded[arm.id] = WhisperModel(arm.provider_model, device="cpu",
                                       compute_type=arm.extra["compute_type"])
    return _loaded[arm.id]


def faster_whisper(arm, segment, rec):
    """The same weights through CTranslate2. `transcribe` is lazy — consuming it does the work."""
    model = load_faster_whisper(arm)
    segments, info = model.transcribe(np.asarray(segment, dtype=np.float32),
                                      language=STT_LANGUAGE, beam_size=5)
    text = " ".join(s.text.strip() for s in segments).strip()
    rec["chars"] = len(text)
    rec["language_prob"] = round(info.language_probability, 3)
    return text


BACKENDS = {
    "openai-audio": openai_audio,
    "transformers-whisper": transformers_whisper,
    "faster-whisper": faster_whisper,
}

LOADERS = {
    "transformers-whisper": load_transformers_whisper,
    "faster-whisper": load_faster_whisper,
}
