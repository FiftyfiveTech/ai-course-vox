"""Endpointing: mic frames in, one finished utterance out. snakers4/silero-vad, local.

This is the stage that decides where a turn ends, so it owns the whole decision: it does not
hand raw audio upward and let STT guess.

The decision is split from the audio source on purpose. `Endpointer` sees one 32 ms frame at a
time and knows nothing about microphones, so the same logic that runs live can be driven from a
recording — which is how it gets tested without a person in the room. `listen()` is the only
part that touches the mic, and it stays the loop's single entry point.

That one entry point is also what makes barge-in cheap (VOX-011). `listen()` runs across a reply and
past it, reporting speech through `on_speech` as soon as there is enough of it to act on, so the
utterance that interrupted a reply is captured here like any other and there is no second listener to
keep in step with this one.

Not logged through telemetry.log_call: silero runs locally per frame, so a record per inference
would be thousands of lines a turn. VOX-003 times this stage as t_vad at the turn level, from
the marks `Capture` carries out of here.
"""
import sys
import time

import numpy as np
import sounddevice as sd
import torch
from silero_vad import load_silero_vad

from src.config import (BARGE_MIN_SPEECH_MS, SAMPLE_RATE, VAD_FRAME, VAD_MAX_UTTERANCE_MS,
                        VAD_MIN_SPEECH_MS, VAD_SILENCE_MS, VAD_SPEECH_THRESHOLD)

MS_PER_FRAME = VAD_FRAME / SAMPLE_RATE * 1000  # 32 ms

# What Endpointer.push returns, so a caller can react without reading private state.
WAITING = "waiting"    # no speech yet
SPEAKING = "speaking"  # inside an utterance
DONE = "done"          # utterance complete — call .capture()
TOO_SHORT = "short"    # a cough, not a turn; the endpointer has rearmed itself

_model = None


def _vad_model():
    global _model
    if _model is None:
        _model = load_silero_vad()
    return _model


class Capture:
    """One endpointed utterance, plus the two marks the turn timer needs to place it in time.

    `speech_end_t` is when the last frame that silero called speech was processed — as close as
    this stage gets to "the user stopped talking". `endpointed_t` is when the endpointer said so.
    Everything downstream is waiting on the second mark, but the user has been waiting since the
    first, which is why time_to_first_audio is measured from `speech_end_t` and not from DONE.
    """

    def __init__(self, segment, speech_end_t, endpointed_t, spoken_s, infer_ms):
        self.segment = segment
        self.speech_end_t = speech_end_t
        self.endpointed_t = endpointed_t
        self.spoken_s = spoken_s
        self.infer_ms = infer_ms          # silero compute only, no waiting

    @property
    def t_vad_ms(self):
        """The delay endpointing adds before STT may start.

        Live, this is dominated by the VAD_SILENCE_MS hangover — the loop is holding the turn
        open waiting to see whether the user is finished. Driven from a recording, frames arrive
        as fast as the CPU can push them, so the same subtraction yields silero compute instead.
        Both are true of their run; `source` on the turn record says which one you are reading.
        """
        return round((self.endpointed_t - self.speech_end_t) * 1000, 1)

    def __len__(self):
        return len(self.segment)


class Endpointer:
    """Frame-by-frame end-of-utterance detection. One instance per turn.

    `threshold` overrides VAD_SPEECH_THRESHOLD for this instance. Barge-in wants a stricter one
    than ordinary endpointing does (see BARGE_SPEECH_THRESHOLD), and a module-level constant read
    directly could not differ between the two.
    """

    def __init__(self, model=None, threshold=None):
        self.model = model or _vad_model()
        self.model.reset_states()
        self.threshold = VAD_SPEECH_THRESHOLD if threshold is None else threshold
        self._reset()
        self.waited_ms = 0.0
        self.infer_ms = 0.0

    def _reset(self):
        self.frames = []
        self.speech_frames = 0
        self.silence_ms = 0.0
        self.started = False
        self.first_speech_t = None
        self.last_speech_t = None
        self.done_t = None

    @property
    def speech_ms(self):
        """How much speech is in the utterance so far. Cleared with everything else on a rearm."""
        return self.speech_frames * MS_PER_FRAME

    def push(self, frame):
        """Feed exactly VAD_FRAME samples of float32 mono at 16 kHz. -> one of the states above."""
        if len(frame) != VAD_FRAME:
            raise ValueError(f"silero needs exactly {VAD_FRAME} samples, got {len(frame)}")

        t0 = time.perf_counter()
        prob = self.model(torch.from_numpy(frame), SAMPLE_RATE).item()
        now = time.perf_counter()
        self.infer_ms += (now - t0) * 1000
        is_speech = prob >= self.threshold

        if not self.started:
            self.waited_ms += MS_PER_FRAME
            if not is_speech:
                return WAITING
            self.started = True
            self.frames.append(frame)
            self.speech_frames = 1
            # The mark barge-in stop latency is measured from: the user has been talking since
            # here, whatever a caller later decides is enough speech to act on.
            self.first_speech_t = now
            self.last_speech_t = now
            return SPEAKING

        self.frames.append(frame)
        if is_speech:
            self.speech_frames += 1
            self.silence_ms = 0.0
            self.last_speech_t = now
        else:
            self.silence_ms += MS_PER_FRAME

        if self.silence_ms >= VAD_SILENCE_MS:
            if self.speech_ms >= VAD_MIN_SPEECH_MS:
                self.done_t = now
                return DONE
            # Too short to be a turn — a cough or a door. Rearm rather than transcribe it.
            self._reset()
            self.model.reset_states()
            return TOO_SHORT

        if len(self.frames) * MS_PER_FRAME >= VAD_MAX_UTTERANCE_MS:
            self.done_t = now
            return DONE

        return SPEAKING

    def flush(self):
        """End the utterance at end-of-audio. -> DONE if enough speech was collected."""
        if self.started and self.speech_ms >= VAD_MIN_SPEECH_MS:
            self.done_t = time.perf_counter()
            return DONE
        return WAITING

    def segment(self):
        """The endpointed utterance as one float32 array."""
        if not self.frames:
            raise RuntimeError("no audio collected — push frames until push() returns DONE")
        return np.concatenate(self.frames)

    def spoken_s(self):
        return self.speech_ms / 1000

    def capture(self):
        """The finished utterance with its timing marks. Call once push() returned DONE."""
        return Capture(self.segment(), self.last_speech_t, self.done_t or time.perf_counter(),
                       self.spoken_s(), round(self.infer_ms, 1))


def drive(frames, on_speech=None, confirm_ms=BARGE_MIN_SPEECH_MS, threshold=None, model=None,
          max_wait_ms=None, echo=None):
    """The endpointing decision over any source of frames. -> (Capture or None, state).

    `listen()` is this function over a microphone and `endpoint_frames()` is it over a recording,
    which is the whole reason it exists: the `on_speech` hook is where barge-in is decided, and a
    second copy of this loop is exactly how a scripted run (VOX-026) ends up measuring an
    `abort()` call instead of an interruption. Silero's detection delay and `confirm_ms` are the
    majority of a VOX-011 stop latency, so a rehearsal that skips them is not rehearsing the
    feature.

    `max_wait_ms` bounds the silence before speech and is what makes a muted mic return rather
    than hang; None means the frames themselves are the bound, so a recording ends by running out.
    `echo` receives the three lines a person at a mic needs to see, and is None for a driven run
    that is printing its own.
    """
    ep = Endpointer(model=model, threshold=threshold)
    say = echo if echo is not None else (lambda *a, **kw: None)
    fired = False        # on_speech is once per utterance, not once per frame above the threshold

    for frame in frames:
        state = ep.push(frame)
        if state == SPEAKING and len(ep.frames) == 1:
            fired = False                     # a new utterance after a rearm gets its own hook call
            say("  speech detected…")
        elif state == TOO_SHORT:
            say("  (too short — still listening)")
        elif state == DONE:
            return ep.capture(), DONE
        elif state == WAITING and max_wait_ms is not None and ep.waited_ms >= max_wait_ms:
            return None, WAITING

        if on_speech is not None and not fired and ep.speech_ms >= confirm_ms:
            fired = True
            on_speech(ep.first_speech_t)

    # The frames ran out. Live this is unreachable — a microphone does not end — so it is the
    # end-of-recording case, and `flush()` decides whether what was collected is a turn.
    state = ep.flush()
    return (ep.capture() if state == DONE else None), state


def listen(max_wait_s=30, on_speech=None, confirm_ms=BARGE_MIN_SPEECH_MS, threshold=None,
           announce=True):
    """Block until the user speaks and stops. -> Capture, or None.

    Returns None if nothing was said within max_wait_s, so the caller can exit cleanly instead
    of hanging on a muted mic.

    `on_speech(first_speech_t)` is called once per utterance, as soon as `confirm_ms` of speech has
    accumulated, and is handed the moment speech *started* rather than the moment it was confirmed —
    a caller measuring a reaction has to measure it from when the user began, not from when this
    function became sure. It fires and then endpointing carries on unchanged, which is what makes
    barge-in cheap: the utterance that interrupted a reply is captured by this same call, so it needs
    no second listener and no separate capture path (VOX-011).

    A rearm after TOO_SHORT clears the mark, so a cough that never reaches `confirm_ms` does not
    fire the hook and the real utterance behind it still can.

    The decision itself is `drive()`. This function is the microphone: opening the device, saying so,
    and reporting the finished utterance.
    """
    def mic_frames(stream):
        while True:
            block, overflowed = stream.read(VAD_FRAME)
            if overflowed:
                # A dropped frame shifts the endpoint decision, so say so rather than hide it.
                print("  (audio overflow — a frame was dropped)", file=sys.stderr)
            yield block[:, 0].copy()

    with sd.InputStream(channels=1, samplerate=SAMPLE_RATE, dtype="float32",
                        blocksize=VAD_FRAME) as stream:
        if announce:
            print("listening… speak now.", flush=True)
        cap, _ = drive(mic_frames(stream), on_speech=on_speech, confirm_ms=confirm_ms,
                       threshold=threshold, max_wait_ms=max_wait_s * 1000,
                       echo=lambda line: print(line, flush=True))

    if cap is None:
        return None
    print(f"  endpointed: {len(cap) / SAMPLE_RATE:.2f}s of audio "
          f"({cap.spoken_s:.2f}s of speech, t_vad {cap.t_vad_ms:.0f}ms)", flush=True)
    return cap


def endpoint_frames(frames, model=None, **kw):
    """Drive the same Endpointer from an iterable of frames. Used to test the decision offline.

    -> (Capture, state). Not part of the live path; `listen()` is what `make demo` calls. Keyword
    arguments are `drive()`'s — `on_speech` and `threshold` are what a scripted barge-in needs.
    """
    return drive(frames, model=model, **kw)


def paced(frames, echo=None):
    """Yield frames at real time: one VAD_FRAME every 32 ms of wall clock (VOX-026).

    Unpaced, a recording collapses the `VAD_SILENCE_MS` hangover to silero compute — ~4 ms against
    the ~1.1 s a person at a mic actually waits out — so every fixture-driven
    `time_to_first_audio` in this repo is optimistic by about a second. Pacing pays that second in
    the same code, which is what makes a rehearsal number comparable with a live one.

    Timed against an absolute deadline rather than by sleeping a fixed interval per frame: silero's
    own compute is ~1-4 ms a frame, and sleeping 32 ms *plus* that drifts slower than real time,
    which would land in `t_vad` as latency nobody spent.
    """
    period = VAD_FRAME / SAMPLE_RATE
    t0 = time.perf_counter()
    behind_ms = 0.0
    for i, frame in enumerate(frames):
        wait = (t0 + i * period) - time.perf_counter()
        if wait > 0:
            time.sleep(wait)
        else:
            behind_ms = max(behind_ms, -wait * 1000)
        yield frame
    if behind_ms > period * 1000 and echo is not None:
        # The machine could not keep up with real time, so the pacing is not what was measured.
        echo(f"  (pacing fell {behind_ms:.0f}ms behind real time — t_vad is not a live number)")


def frames_from(audio):
    """Split a float32 16 kHz array into exact VAD_FRAME chunks, dropping any short tail."""
    n = len(audio) // VAD_FRAME * VAD_FRAME
    return [audio[i:i + VAD_FRAME] for i in range(0, n, VAD_FRAME)]
