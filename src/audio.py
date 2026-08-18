"""Speaker playback, kept separate from synthesis so VOX-011 can interrupt it."""
import sounddevice as sd

from src.config import TTS_SAMPLE_RATE


def play(audio, sample_rate=TTS_SAMPLE_RATE, block=True):
    """Play float32 mono samples through the default output device."""
    sd.play(audio, samplerate=sample_rate)
    if block:
        sd.wait()
