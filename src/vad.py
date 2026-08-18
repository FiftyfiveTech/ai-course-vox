"""Endpointing: mic frames in, one finished utterance out. snakers4/silero-vad, local.

This is the stage that decides where a turn ends, so it owns the whole decision: it does not
hand raw audio upward and let STT guess.

The decision is split from the audio source on purpose. `Endpointer` sees one 32 ms frame at a
time and knows nothing about microphones, so the same logic that runs live can be driven from a
recording — which is how it gets tested without a person in the room. `listen()` is the only
part that touches the mic, and it stays the loop's single entry point.

Not logged through telemetry.log_call: silero runs locally per frame, so a record per inference
would be thousands of lines a turn. VOX-003 times this stage as t_vad at the turn level.
"""
import sys

import numpy as np
import sounddevice as sd
import torch
from silero_vad import load_silero_vad

from src.config import (SAMPLE_RATE, VAD_FRAME, VAD_MAX_UTTERANCE_MS, VAD_MIN_SPEECH_MS,
                        VAD_SILENCE_MS, VAD_SPEECH_THRESHOLD)

MS_PER_FRAME = VAD_FRAME / SAMPLE_RATE * 1000  # 32 ms

# What Endpointer.push returns, so a caller can react without reading private state.
WAITING = "waiting"    # no speech yet
SPEAKING = "speaking"  # inside an utterance
DONE = "done"          # utterance complete — call .segment()
TOO_SHORT = "short"    # a cough, not a turn; the endpointer has rearmed itself

_model = None


def _vad_model():
    global _model
    if _model is None:
        _model = load_silero_vad()
    return _model


class Endpointer:
    """Frame-by-frame end-of-utterance detection. One instance per turn."""

    def __init__(self, model=None):
        self.model = model or _vad_model()
        self.model.reset_states()
        self._reset()
        self.waited_ms = 0.0

    def _reset(self):
        self.frames = []
        self.speech_frames = 0
        self.silence_ms = 0.0
        self.started = False

    def push(self, frame):
        """Feed exactly VAD_FRAME samples of float32 mono at 16 kHz. -> one of the states above."""
        if len(frame) != VAD_FRAME:
            raise ValueError(f"silero needs exactly {VAD_FRAME} samples, got {len(frame)}")

        prob = self.model(torch.from_numpy(frame), SAMPLE_RATE).item()
        is_speech = prob >= VAD_SPEECH_THRESHOLD

        if not self.started:
            self.waited_ms += MS_PER_FRAME
            if not is_speech:
                return WAITING
            self.started = True
            self.frames.append(frame)
            self.speech_frames = 1
            return SPEAKING

        self.frames.append(frame)
        if is_speech:
            self.speech_frames += 1
            self.silence_ms = 0.0
        else:
            self.silence_ms += MS_PER_FRAME

        if self.silence_ms >= VAD_SILENCE_MS:
            if self.speech_frames * MS_PER_FRAME >= VAD_MIN_SPEECH_MS:
                return DONE
            # Too short to be a turn — a cough or a door. Rearm rather than transcribe it.
            self._reset()
            self.model.reset_states()
            return TOO_SHORT

        if len(self.frames) * MS_PER_FRAME >= VAD_MAX_UTTERANCE_MS:
            return DONE

        return SPEAKING

    def flush(self):
        """End the utterance at end-of-audio. -> DONE if enough speech was collected."""
        if self.started and self.speech_frames * MS_PER_FRAME >= VAD_MIN_SPEECH_MS:
            return DONE
        return WAITING

    def segment(self):
        """The endpointed utterance as one float32 array."""
        if not self.frames:
            raise RuntimeError("no audio collected — push frames until push() returns DONE")
        return np.concatenate(self.frames)

    def spoken_s(self):
        return self.speech_frames * MS_PER_FRAME / 1000


def listen(max_wait_s=30):
    """Block until the user speaks and stops. -> float32 mono array at 16 kHz, or None.

    Returns None if nothing was said within max_wait_s, so the caller can exit cleanly instead
    of hanging on a muted mic.
    """
    ep = Endpointer()
    max_wait_ms = max_wait_s * 1000

    with sd.InputStream(channels=1, samplerate=SAMPLE_RATE, dtype="float32",
                        blocksize=VAD_FRAME) as stream:
        print("listening… speak now.", flush=True)
        while True:
            block, overflowed = stream.read(VAD_FRAME)
            if overflowed:
                # A dropped frame shifts the endpoint decision, so say so rather than hide it.
                print("  (audio overflow — a frame was dropped)", file=sys.stderr)

            state = ep.push(block[:, 0].copy())
            if state == SPEAKING and len(ep.frames) == 1:
                print("  speech detected…", flush=True)
            elif state == TOO_SHORT:
                print("  (too short — still listening)", flush=True)
            elif state == DONE:
                break
            elif state == WAITING and ep.waited_ms >= max_wait_ms:
                return None

    segment = ep.segment()
    print(f"  endpointed: {len(segment) / SAMPLE_RATE:.2f}s of audio "
          f"({ep.spoken_s():.2f}s of speech)", flush=True)
    return segment


def endpoint_frames(frames, model=None):
    """Drive the same Endpointer from an iterable of frames. Used to test the decision offline.

    -> (segment, state). Not part of the live path; `listen()` is what `make demo` calls.
    """
    ep = Endpointer(model)
    for frame in frames:
        if ep.push(frame) == DONE:
            return ep.segment(), DONE
    state = ep.flush()
    return (ep.segment() if state == DONE else None), state


def frames_from(audio):
    """Split a float32 16 kHz array into exact VAD_FRAME chunks, dropping any short tail."""
    n = len(audio) // VAD_FRAME * VAD_FRAME
    return [audio[i:i + VAD_FRAME] for i in range(0, n, VAD_FRAME)]
