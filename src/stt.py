"""Speech to text: openai/whisper-large-v3-turbo, served on the Groq free tier."""
import io

import httpx
import soundfile as sf

from src.config import SAMPLE_RATE, STT, STT_LANGUAGE
from src.telemetry import log_call


def _wav_bytes(segment):
    """Encode the float32 segment as a 16-bit WAV in memory — nothing touches disk."""
    buf = io.BytesIO()
    sf.write(buf, segment, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def transcribe(segment, turn_id, timeout=30):
    """-> transcript text. Raises on a provider error rather than returning a plausible blank."""
    audio = _wav_bytes(segment)
    audio_s = round(len(segment) / SAMPLE_RATE, 3)

    with log_call("stt", STT, turn_id, audio_s=audio_s, language=STT_LANGUAGE) as rec:
        r = httpx.post(
            f"{STT.api_base}/audio/transcriptions",
            headers={"Authorization": f"Bearer {STT.key()}"},
            files={"file": ("turn.wav", audio, "audio/wav")},
            data={"model": STT.provider_model, "response_format": "json",
                  "temperature": "0", "language": STT_LANGUAGE},
            timeout=timeout,
        )
        r.raise_for_status()
        text = (r.json().get("text") or "").strip()
        rec["chars"] = len(text)

    return text
