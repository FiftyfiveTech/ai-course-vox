"""Speaker playback, kept separate from synthesis so VOX-011 can interrupt it."""
import sys
import threading
import time

import numpy as np
import sounddevice as sd

from src.config import TTS_SAMPLE_RATE


class Playback:
    """A reply the speaker is working through, and the handle that cuts it mid-sentence.

    Driven by a stream callback rather than `sd.play()` so that `on_first_audio` can be stamped the
    moment the device pulls its first block — that instant is the "first audio" in
    time_to_first_audio. `sd.play()` returns as soon as playback is *queued*, which is a different
    and flatteringly smaller number, and the one metric VOX-003 exists to measure is the one worth
    refusing to guess at.

    `pos` is how many samples the device has actually pulled, which is what makes `abort()` able to
    say whether it cut anything at all rather than only that it was called.
    """

    def __init__(self, samples, sample_rate, on_first_audio=None):
        self.samples = np.ascontiguousarray(samples, dtype="float32").reshape(-1, 1)
        self.sample_rate = sample_rate
        self._on_first_audio = on_first_audio
        self.pos = 0
        self.finished = threading.Event()
        self.stopped_t = None            # when abort() returned, or None if the reply played out
        self.stream = sd.OutputStream(samplerate=sample_rate, channels=1, dtype="float32",
                                      callback=self._callback,
                                      finished_callback=self.finished.set)

    def _callback(self, outdata, frames, time_info, status):
        if self.pos == 0 and self._on_first_audio is not None:
            self._on_first_audio()
        chunk = self.samples[self.pos:self.pos + frames]
        n = len(chunk)
        outdata[:n] = chunk
        self.pos += n
        if n < frames:
            outdata[n:] = 0
            raise sd.CallbackStop

    @property
    def reply_s(self):
        """How long the whole reply would take to speak."""
        return round(len(self.samples) / self.sample_rate, 3)

    @property
    def played_s(self):
        """How much of it the device has pulled so far."""
        return round(self.pos / self.sample_rate, 3)

    @property
    def out_latency_s(self):
        """The output stream's own buffer, in seconds.

        Logged next to a stop latency rather than ignored: `abort()` discards what this process has
        not handed over yet, but samples already inside the device buffer are past recall. On MME
        this measured 0.182 s, which is larger than the stop latency itself — so the printed number
        is when VOX stopped *sending*, and this is the tail that can still be heard after it.
        """
        return round(float(self.stream.latency), 3)

    def start(self):
        self.stream.start()
        return self

    def abort(self):
        """Stop mid-sentence, discarding what has not been played. -> the stop mark, or None.

        `abort()` and not `stop()`: stop drains the buffer first, which is exactly the opposite of
        what an interruption wants.

        None means there was nothing left to cut — the reply had already finished — so a caller can
        tell a real barge-in from a user who simply started speaking after the reply ended. The two
        are otherwise indistinguishable, and counting the second as an interruption would inflate
        every barge-in number on the board.
        """
        if self.finished.is_set() or self.pos >= len(self.samples):
            return None
        self.stream.abort()
        self.stopped_t = time.perf_counter()
        return self.stopped_t

    def wait(self):
        """Block until the reply has played out. -> True, or False if the device stalled."""
        # Bounded so a wedged output device fails loudly instead of hanging the turn loop.
        if self.finished.wait(len(self.samples) / self.sample_rate + 5):
            return True
        print("  (playback did not finish — output device stalled)", file=sys.stderr)
        return False

    def close(self):
        self.stream.close()


def play(audio, sample_rate=TTS_SAMPLE_RATE, block=True, on_first_audio=None):
    """Play float32 mono samples through the default output device.

    With block=False the started `Playback` is returned instead of waited on; VOX-011 needs that
    handle to stop a reply the user talked over, and closing it is then the caller's job.
    """
    playback = Playback(audio, sample_rate, on_first_audio).start()
    if not block:
        return playback

    try:
        playback.wait()
    finally:
        playback.close()
    return None
