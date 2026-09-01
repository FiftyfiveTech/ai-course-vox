"""VOX-007, failure mode 1: silence in produces no turn.

Three places can wrongly turn silence into a turn, and each is tested at its own level:

  Endpointer      a frame silero calls silence must not start an utterance, and a burst too short
                  to be speech must rearm rather than be transcribed.
  listen()        a muted mic must return None instead of blocking forever.
  one_turn()      None from listen() must end the turn before STT is called, and write no line —
                  a turn record for a turn nobody took would put a null into every latency gate.

No mic and no silero download: silero is replaced by a model that calls any nonzero sample speech,
so each test states its audio as frames of zeros and frames of tone. That keeps these tests about
the endpointing decision rather than about silero's probabilities, which have their own arm tests.
"""
import types

import numpy as np
import pytest
import torch

from src import loop, nlu, vad
from src.config import (LLM_ARMS, SAMPLE_RATE, STT_ARMS, TTS_ARMS, VAD_FRAME, VAD_MIN_SPEECH_MS,
                        VAD_SILENCE_MS)
from src.vad import DONE, MS_PER_FRAME, TOO_SHORT, WAITING, Endpointer, endpoint_frames


class FakeVad:
    """silero's call interface, decided by the audio itself: any nonzero sample is speech."""

    def __init__(self):
        self.resets = 0

    def reset_states(self):
        self.resets += 1

    def __call__(self, frame, sample_rate):
        assert sample_rate == SAMPLE_RATE, f"silero was given {sample_rate} Hz"
        return torch.tensor(1.0 if float(frame.abs().max()) > 0.0 else 0.0)


def _samples(ms):
    """Whole frames only — push() refuses anything that is not exactly VAD_FRAME long."""
    return round(ms / MS_PER_FRAME) * VAD_FRAME


def silence(ms):
    return np.zeros(_samples(ms), dtype=np.float32)


def speech(ms):
    return np.full(_samples(ms), 0.5, dtype=np.float32)


# A cough: audible, but under the minimum that counts as a turn. Plus enough trailing silence to
# take the endpointer past its hangover, so the short-segment branch is the one that fires.
COUGH_MS = VAD_MIN_SPEECH_MS - 100
HANGOVER_MS = VAD_SILENCE_MS + 100


# --- the endpointing decision ----------------------------------------------------------------

def test_silence_never_starts_an_utterance():
    ep = Endpointer(FakeVad())
    states = [ep.push(f) for f in vad.frames_from(silence(2000))]
    assert set(states) == {WAITING}
    assert not ep.started
    assert ep.frames == []


def test_silence_yields_no_capture():
    cap, state = endpoint_frames(vad.frames_from(silence(2000)), FakeVad())
    assert cap is None
    assert state == WAITING


def test_a_cough_rearms_instead_of_becoming_a_turn():
    """Under VAD_MIN_SPEECH_MS is a door or a throat, and transcribing it invents a turn."""
    ep = Endpointer(FakeVad())
    audio = np.concatenate([speech(COUGH_MS), silence(HANGOVER_MS)])
    states = [ep.push(f) for f in vad.frames_from(audio)]

    assert states.count(TOO_SHORT) == 1
    assert DONE not in states
    # Rearmed, not merely stopped: the next utterance must not inherit the cough's frames.
    assert not ep.started and ep.frames == [] and ep.speech_frames == 0


def test_a_cough_before_a_real_utterance_is_not_in_the_segment():
    """The positive half of rearming — the turn that follows a cough is the turn, whole and alone."""
    audio = np.concatenate([speech(COUGH_MS), silence(HANGOVER_MS),
                            speech(600), silence(HANGOVER_MS)])
    cap, state = endpoint_frames(vad.frames_from(audio), FakeVad())

    assert state == DONE
    assert cap.spoken_s == pytest.approx(0.6, abs=MS_PER_FRAME / 1000)
    assert len(cap) < _samples(COUGH_MS + HANGOVER_MS + 600)


def test_flush_at_end_of_audio_refuses_a_short_segment():
    """End of a recording is not an endpoint. A cough at the tail must not become the last turn."""
    ep = Endpointer(FakeVad())
    for frame in vad.frames_from(speech(COUGH_MS)):
        ep.push(frame)
    assert ep.flush() == WAITING


def test_a_real_utterance_still_endpoints():
    """The control. Without this, every test above would pass against a VAD that never fires."""
    audio = np.concatenate([silence(200), speech(700), silence(HANGOVER_MS)])
    cap, state = endpoint_frames(vad.frames_from(audio), FakeVad())

    assert state == DONE
    assert cap.spoken_s * 1000 >= VAD_MIN_SPEECH_MS
    assert cap.t_vad_ms >= 0


# --- the mic loop ----------------------------------------------------------------------------

class FakeStream:
    """sounddevice's read/context interface over a fixed block, counting what was asked of it."""

    def __init__(self, block_for):
        self.block_for = block_for      # call index -> mono samples
        self.reads = 0
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.closed = True
        return False

    def read(self, n):
        block = self.block_for(self.reads)[:n]
        self.reads += 1
        return block.reshape(-1, 1), False


def fake_mic(monkeypatch, block_for):
    """Put `vad` on a scripted mic and a fake silero. -> the stream, to assert against."""
    stream = FakeStream(block_for)
    monkeypatch.setattr(vad, "_vad_model", FakeVad)
    monkeypatch.setattr(vad, "sd", types.SimpleNamespace(InputStream=lambda **kw: stream))
    return stream


@pytest.fixture
def quiet_mic(monkeypatch):
    """A mic that only ever hands back silence."""
    return fake_mic(monkeypatch, lambda i: silence(MS_PER_FRAME))


def test_listen_returns_none_when_the_mic_stays_quiet(quiet_mic):
    """So the caller can exit cleanly instead of hanging on a muted mic."""
    assert vad.listen(max_wait_s=1) is None
    assert quiet_mic.closed
    # Bounded by max_wait_s, not by luck: 1 s of 32 ms frames, give or take the frame it decides on.
    assert quiet_mic.reads == pytest.approx(1000 / MS_PER_FRAME, abs=2)


def test_the_timeout_counts_silent_frames_not_wall_clock(monkeypatch):
    """A stream of coughs keeps listen() alive past max_wait_s. Undocumented, so pinned here.

    `waited_ms` only advances on a frame arriving before an utterance has started, and rearming
    after a cough deliberately does not clear it. So somebody clearing their throat into the mic
    reads as "a person is there", not as an idle mic — which is the behaviour we want, but it means
    max_wait_s is a silence budget and not a deadline. A test that assumed otherwise would hang.
    """
    cycle = vad.frames_from(np.concatenate([speech(COUGH_MS), silence(HANGOVER_MS)]))

    class Stop(Exception):
        pass

    def block_for(i):
        if i >= 300:                      # ~9.6 s of frames, well past max_wait_s=1
            raise Stop
        return cycle[i % len(cycle)]

    fake_mic(monkeypatch, block_for)

    with pytest.raises(Stop):             # i.e. it did not return None, and did not endpoint
        vad.listen(max_wait_s=1)


# --- the turn ---------------------------------------------------------------------------------

DEFAULTS = {"stt": STT_ARMS[0], "llm": LLM_ARMS[0], "tts": TTS_ARMS[0]}


@pytest.fixture
def silent_mic_turn(monkeypatch):
    """`vad.listen` hears nothing, and every stage after it explodes if it is reached."""
    def unreachable(*a, **kw):
        raise AssertionError("a stage ran on a turn that had no audio")

    calls = []
    monkeypatch.setattr(loop.vad, "listen", lambda *a, **kw: calls.append(1) or None)
    monkeypatch.setattr(loop.arms, "stt", unreachable)
    monkeypatch.setattr(nlu, "reply", unreachable)
    monkeypatch.setattr(loop.arms, "tts", unreachable)
    monkeypatch.setattr(loop.audio, "play", unreachable)
    return calls


def test_a_quiet_turn_writes_no_line(silent_mic_turn, turns_log, calls_log, capsys):
    """No turn happened, so there is nothing to describe — a null-latency line is worse than none.

    Every gate reads these logs. A record here would put a null into t_stt/t_llm/t_tts and drag
    down a percentile that no user ever waited through.
    """
    loop.one_turn(DEFAULTS)

    assert not turns_log.exists()
    assert not calls_log.exists()
    assert "nothing heard" in capsys.readouterr().out


def test_a_quiet_mic_ends_the_run_without_a_turn(monkeypatch, turns_log, capsys):
    """End to end through main(): one attempt, no turn counted, and a non-zero exit."""
    monkeypatch.setattr("sys.argv", ["vox", "--turns", "3"])
    monkeypatch.setattr(loop.vad, "_vad_model", FakeVad)
    # Warming is arms.warm's job and is tested there; here it would download the Kokoro weights.
    monkeypatch.setattr(loop.arms, "select", lambda args: DEFAULTS)
    heard = []
    monkeypatch.setattr(loop.vad, "listen", lambda *a, **kw: heard.append(1) or None)

    assert loop.main() == 1
    assert len(heard) == 1, "a silent mic should stop the run, not be retried twice more"
    assert "0 turn(s) completed" in capsys.readouterr().out
    assert not turns_log.exists()
