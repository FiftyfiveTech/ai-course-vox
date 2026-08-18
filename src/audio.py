"""Speaker playback, kept separate from synthesis so VOX-011 can interrupt it."""
import sys
import threading

import numpy as np
import sounddevice as sd

from src.config import TTS_SAMPLE_RATE


def play(audio, sample_rate=TTS_SAMPLE_RATE, block=True, on_first_audio=None):
    """Play float32 mono samples through the default output device.

    Driven by a stream callback rather than `sd.play()` so that `on_first_audio` can be stamped
    the moment the device pulls its first block — that instant is the "first audio" in
    time_to_first_audio. `sd.play()` returns as soon as playback is *queued*, which is a
    different and flatteringly smaller number, and the one metric VOX-003 exists to measure is
    the one worth refusing to guess at.

    With block=False the started stream is returned instead of waited on; VOX-011 needs that
    handle to stop a reply the user talked over.
    """
    samples = np.ascontiguousarray(audio, dtype="float32").reshape(-1, 1)
    finished = threading.Event()
    pos = 0

    def callback(outdata, frames, time_info, status):
        nonlocal pos
        if pos == 0 and on_first_audio is not None:
            on_first_audio()
        chunk = samples[pos:pos + frames]
        n = len(chunk)
        outdata[:n] = chunk
        pos += n
        if n < frames:
            outdata[n:] = 0
            raise sd.CallbackStop

    stream = sd.OutputStream(samplerate=sample_rate, channels=1, dtype="float32",
                             callback=callback, finished_callback=finished.set)
    stream.start()
    if not block:
        return stream

    try:
        # Bounded so a wedged output device fails loudly instead of hanging the turn loop.
        if not finished.wait(len(samples) / sample_rate + 5):
            print("  (playback did not finish — output device stalled)", file=sys.stderr)
    finally:
        stream.close()
    return None
