"""VOX-011: talking over VOX stops it, and the words that stopped it become the next turn's input.

Barge-in is one feature spread across three levels, and each is tested where it lives:

  Playback    aborting a reply that is still playing cuts it; aborting one that already finished
              cuts nothing, and has to say so — otherwise a user who simply waited for the reply to
              end is counted as an interruption and every barge-in number on the board is inflated.
  listen()    the hook fires once per utterance, only after enough speech to be sure, and is handed
              the moment speech *started* so a stop latency is measured from the user and not from
              the endpointer's confidence.
  one_turn()  an interrupted reply is on the turn record with its stop latency, and the utterance
              that interrupted it reaches the next turn's STT instead of the mic being reopened.

No mic, no speaker, no weights. The output device is a fake whose blocks this file pulls by hand,
which is also the only way to assert what `played_s` was at the moment of the cut — a real device
pulls on its own schedule and the number would be different on every run.
"""
import json
import time
import types

import numpy as np
import pytest
import sounddevice as real_sd
import torch

from conftest import fake_state             # same directory; pytest puts tests/unit on sys.path
from src import audio, loop, vad
from src.config import (BARGE_MIN_SPEECH_MS, BARGE_SPEECH_THRESHOLD, LLM_ARMS, SAMPLE_RATE,
                        STT_ARMS, TTS_ARMS, VAD_FRAME, VAD_SILENCE_MS, VAD_SPEECH_THRESHOLD)
from src.vad import MS_PER_FRAME, SPEAKING, WAITING, Endpointer

DEFAULTS = {"stt": STT_ARMS[0], "llm": LLM_ARMS[0], "tts": TTS_ARMS[0]}


# --- the speaker ------------------------------------------------------------------------------

class FakeOutputStream:
    """sounddevice's OutputStream interface, pulled by the test instead of by a device."""

    def __init__(self, samplerate, channels, dtype, callback, finished_callback):
        self.samplerate = samplerate
        self.callback = callback
        self.finished_callback = finished_callback
        self.started = self.aborted = self.closed = False
        self.latency = 0.182          # what MME reported for this machine in the step-0 spike

    def start(self):
        self.started = True

    def pull(self, frames):
        """Be the device for one block. -> False once the reply has played out."""
        out = np.zeros((frames, 1), dtype="float32")
        try:
            self.callback(out, frames, None, None)
        except real_sd.CallbackStop:
            self.finished_callback()
            return False
        return True

    def abort(self):
        self.aborted = True
        self.finished_callback()

    def close(self):
        self.closed = True


def fake_speaker(monkeypatch):
    """Put audio.py on a fake output device. -> the list of streams it opens."""
    made = []

    def OutputStream(**kw):
        made.append(FakeOutputStream(**kw))
        return made[-1]

    # CallbackStop is the real class: audio.py raises it, and swapping it for another type would
    # make end-of-reply look like a crash rather than a finished stream.
    monkeypatch.setattr(audio, "sd", types.SimpleNamespace(
        OutputStream=OutputStream, CallbackStop=real_sd.CallbackStop))
    return made


def reply(seconds, sample_rate=24_000):
    return np.zeros(int(seconds * sample_rate), dtype="float32")


def test_aborting_a_reply_that_is_still_playing_cuts_it(monkeypatch):
    made = fake_speaker(monkeypatch)
    playback = audio.play(reply(1.0), sample_rate=24_000, block=False)
    stream = made[0]
    assert stream.started

    stream.pull(1200)                                  # 50 ms of a 1 s reply
    assert playback.played_s == 0.05

    stopped = playback.abort()

    assert stopped is not None, "there was a second of reply left, so this was a real interruption"
    assert stream.aborted, "abort, not stop — draining the buffer is the opposite of interrupting"
    assert playback.stopped_t == stopped
    assert (playback.played_s, playback.reply_s) == (0.05, 1.0)


def test_aborting_a_reply_that_already_finished_cuts_nothing(monkeypatch):
    """The control that keeps a patient user from being recorded as an interruption."""
    made = fake_speaker(monkeypatch)
    playback = audio.play(reply(0.1), sample_rate=24_000, block=False)
    while made[0].pull(1200):
        pass

    assert playback.played_s == playback.reply_s
    assert playback.abort() is None
    assert not made[0].aborted
    assert playback.stopped_t is None


def test_the_first_pulled_block_is_the_first_audio(monkeypatch):
    """Kept from before the refactor: the stamp is the device pulling, not playback being queued."""
    made = fake_speaker(monkeypatch)
    stamps = []
    audio.play(reply(1.0), sample_rate=24_000, block=False, on_first_audio=lambda: stamps.append(1))

    made[0].pull(1200)
    made[0].pull(1200)

    assert len(stamps) == 1, "one first audio per reply, however many blocks the device pulls"


def test_the_output_latency_is_reported_not_assumed(monkeypatch):
    """It is larger than the stop latency on this machine, so it cannot be left out of the record."""
    made = fake_speaker(monkeypatch)
    playback = audio.play(reply(1.0), sample_rate=24_000, block=False)
    assert playback.out_latency_s == made[0].latency


# --- the listener -----------------------------------------------------------------------------

class FakeVad:
    """silero's call interface, decided by the audio: any nonzero sample is speech.

    The same shape as the fake in test_silence.py, kept local so each file states its own audio.
    Returns 1.0 rather than a borderline value so it reads as speech at either threshold.
    """

    def reset_states(self):
        pass

    def __call__(self, frame, sample_rate):
        return torch.tensor(1.0 if float(frame.abs().max()) > 0.0 else 0.0)


class ProbVad:
    """A silero whose probability the test chooses, for the threshold boundary."""

    def __init__(self, prob):
        self.prob = prob

    def reset_states(self):
        pass

    def __call__(self, frame, sample_rate):
        return torch.tensor(self.prob if float(frame.abs().max()) > 0.0 else 0.0)


class FakeStream:
    """sounddevice's read/context interface over scripted blocks."""

    def __init__(self, block_for):
        self.block_for = block_for
        self.reads = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, n):
        block = self.block_for(self.reads)[:n]
        self.reads += 1
        return block.reshape(-1, 1), False


def fake_mic(monkeypatch, frames):
    """Drive vad.listen from a fixed list of frames, repeating silence once they run out."""
    quiet = np.zeros(VAD_FRAME, dtype=np.float32)
    stream = FakeStream(lambda i: frames[i] if i < len(frames) else quiet)
    monkeypatch.setattr(vad, "_vad_model", FakeVad)
    monkeypatch.setattr(vad, "sd", types.SimpleNamespace(InputStream=lambda **kw: stream))
    return stream


def _samples(ms):
    return round(ms / MS_PER_FRAME) * VAD_FRAME


def silence(ms):
    return np.zeros(_samples(ms), dtype=np.float32)


def speech(ms):
    return np.full(_samples(ms), 0.5, dtype=np.float32)


HANGOVER_MS = VAD_SILENCE_MS + 100
COUGH_MS = BARGE_MIN_SPEECH_MS - 100      # audible, but under the bar for cutting a reply


def test_the_hook_fires_once_and_is_handed_the_start_of_speech(monkeypatch):
    """Once per utterance, not once per frame — a hook per frame would abort an aborted stream."""
    fake_mic(monkeypatch, vad.frames_from(np.concatenate([
        silence(100), speech(600), silence(HANGOVER_MS)])))
    marks = []

    cap = vad.listen(on_speech=marks.append)

    assert len(marks) == 1
    assert cap is not None
    # The mark is when the user began, so it sits before the last speech frame of the same utterance.
    # Measuring from the confirmation instead would hide BARGE_MIN_SPEECH_MS inside the stop latency.
    assert marks[0] < cap.speech_end_t


def test_a_cough_does_not_fire_the_hook_but_the_utterance_behind_it_does(monkeypatch):
    """Cutting on the first speech frame would let a door or a chair kill every reply."""
    fake_mic(monkeypatch, vad.frames_from(np.concatenate([
        speech(COUGH_MS), silence(HANGOVER_MS), speech(600), silence(HANGOVER_MS)])))
    marks = []

    cap = vad.listen(on_speech=marks.append)

    assert len(marks) == 1, "the cough must not interrupt; the real utterance after it must"
    assert cap.spoken_s == pytest.approx(0.6, abs=MS_PER_FRAME / 1000)


def test_the_hook_waits_for_the_confirmation_window(monkeypatch):
    """The whole point of confirm_ms: speech under it never reaches the hook at all."""
    fake_mic(monkeypatch, vad.frames_from(np.concatenate([
        speech(300), silence(HANGOVER_MS)])))
    marks = []

    vad.listen(on_speech=marks.append, confirm_ms=1_000)

    assert marks == []


def test_the_barge_threshold_is_stricter_than_the_endpointing_one():
    """Not a tuning detail: barge-in is the one decision that fires while the speaker is running."""
    assert BARGE_SPEECH_THRESHOLD > VAD_SPEECH_THRESHOLD


def test_a_borderline_frame_is_speech_for_endpointing_but_not_for_barge_in():
    """The threshold is per instance, so the two decisions can genuinely differ on the same frame."""
    frame = np.full(VAD_FRAME, 0.5, dtype=np.float32)
    borderline = (VAD_SPEECH_THRESHOLD + BARGE_SPEECH_THRESHOLD) / 2

    assert Endpointer(ProbVad(borderline)).push(frame) == SPEAKING
    assert Endpointer(ProbVad(borderline), threshold=BARGE_SPEECH_THRESHOLD).push(frame) == WAITING


# --- the turn ---------------------------------------------------------------------------------

class FakeCapture:
    """What listen() hands back, with the marks the turn timer reads.

    `segment` is built per instance, not shared on the class, so `is` can tell two captures apart —
    which is the only way to assert that the audio STT ran on is the audio that interrupted the
    reply rather than something else of the same shape.
    """

    speech_end_t = 100.0
    endpointed_t = 101.1
    spoken_s = 1.2
    infer_ms = 9.0

    def __init__(self):
        self.segment = [0.0] * 32_000

    @property
    def t_vad_ms(self):
        return round((self.endpointed_t - self.speech_end_t) * 1000, 1)

    def __len__(self):
        return len(self.segment)


class FakePlayback:
    """audio.play(block=False)'s handle, with the interruption scripted by the test."""

    def __init__(self, cut=True):
        self.cut = cut                 # whether there was still reply left when abort() was called
        self.reply_s = 3.4
        self.played_s = 1.2
        self.out_latency_s = 0.182
        self.stopped_t = None
        self.aborts = 0
        self.closed = False

    def abort(self):
        self.aborts += 1
        if not self.cut:
            return None
        self.stopped_t = time.perf_counter()
        return self.stopped_t

    def close(self):
        self.closed = True


SPOKE_FOR_MS = 300.0        # how long the fake user had been talking when the hook fired


def watched_turn(monkeypatch, playback=None, interrupting=None):
    """A turn whose stages all work, playing to `playback` and hearing `interrupting`.

    -> the playback, and the list of audio arrays STT was given.
    """
    playback = playback if playback is not None else FakePlayback()
    heard = []

    def play(samples, **kw):
        kw["on_first_audio"]()                  # the device pulled its first block
        return playback if kw.get("block") is False else None

    def listen(*a, **kw):
        on_speech = kw.get("on_speech")
        if on_speech is None:
            # The turn's own input, listened for before any reply exists. Only the watching call
            # gets a hook, which is what tells the two apart here.
            return FakeCapture()
        on_speech(time.perf_counter() - SPOKE_FOR_MS / 1000)
        return interrupting

    monkeypatch.setattr(loop.audio, "play", play)
    monkeypatch.setattr(loop.vad, "listen", listen)
    monkeypatch.setattr(loop.arms, "stt", lambda a, *rest, **kw: heard.append(a) or "stop talking")
    # Since the VOX-019/020 merge the un-retrieved path is the structured extractor, so this
    # is the call that writes the spoken reply on a turn with no index behind it.
    monkeypatch.setattr(loop.state, "build", lambda *a, **kw: fake_state("Your last payslip was on the fourth."))
    monkeypatch.setattr(loop.arms, "tts", lambda *a, **kw: types.SimpleNamespace(
        audio=[0.0] * 240, sample_rate=24_000))
    return playback, heard


def test_an_interrupted_turn_records_its_stop_latency(monkeypatch, turns_log):
    """The criterion: the number exists, on the record, in ms, next to what it cut."""
    playback, _ = watched_turn(monkeypatch, interrupting=FakeCapture())

    loop.one_turn(DEFAULTS, watch=True)
    rec = json.loads(turns_log.read_text(encoding="utf-8").strip())

    assert rec["barged_in"] is True
    # Measured from when the fake user started speaking, so it contains the confirmation window.
    assert rec["barge_stop_ms"] == pytest.approx(SPOKE_FOR_MS, abs=150)
    assert (rec["played_s"], rec["reply_s"], rec["cut_s"]) == (1.2, 3.4, 2.2)
    assert rec["out_latency_s"] == 0.182, "the tail abort() cannot recall is on the record too"
    assert playback.closed, "the stream is closed even though it was aborted"


def test_an_interrupted_turn_stays_in_the_latency_percentiles(monkeypatch, turns_log):
    """Being talked over is a fact about the reply, not a failed turn — the gates still count it."""
    watched_turn(monkeypatch, interrupting=FakeCapture())

    loop.one_turn(DEFAULTS, watch=True)
    rec = json.loads(turns_log.read_text(encoding="utf-8").strip())

    assert rec["ok"] is True
    assert rec["time_to_first_audio_ms"] is not None, "audio was reached before it was cut"
    assert all(rec[f"t_{stage}_ms"] is not None for stage in ("vad", "stt", "llm", "tts"))


def test_the_printed_line_carries_the_stop_latency(monkeypatch, capsys):
    """`request_review` needs a number a human can read off a recording, not only a JSONL field."""
    watched_turn(monkeypatch, interrupting=FakeCapture())

    loop.one_turn(DEFAULTS, watch=True)
    out = capsys.readouterr().out

    assert "barge-in" in out
    assert "after you started speaking" in out       # printed the instant it happens
    assert "of output buffer behind it" in out       # and again on the turn's own report line


def test_the_interrupting_words_become_the_next_turns_input(monkeypatch, turns_log):
    """The second half of the ticket. The mic is not reopened; the captured audio is reused."""
    cap = FakeCapture()
    _, heard = watched_turn(monkeypatch, interrupting=cap)

    first = loop.one_turn(DEFAULTS, watch=True)
    assert first.pending is cap

    def unreachable(*a, **kw):
        raise AssertionError("the mic was reopened for audio the loop already had")

    monkeypatch.setattr(loop.vad, "listen", unreachable)
    second = loop.one_turn(DEFAULTS, pending=first.pending)

    assert second.spoken is True
    assert heard[-1] is cap.segment, "STT ran on exactly the audio that interrupted the reply"
    records = [json.loads(line) for line in turns_log.read_text(encoding="utf-8").splitlines()]
    assert records[1]["input"] == "carried-in"
    assert "input" not in records[0], "the interrupted turn heard itself; only the next one was fed"


def test_speech_after_the_reply_finished_is_carried_in_but_is_not_a_barge_in(monkeypatch,
                                                                            turns_log):
    """The control. A user who waits politely produces the same Capture and no interruption."""
    cap = FakeCapture()
    playback, _ = watched_turn(monkeypatch, playback=FakePlayback(cut=False), interrupting=cap)

    result = loop.one_turn(DEFAULTS, watch=True)
    rec = json.loads(turns_log.read_text(encoding="utf-8").strip())

    assert playback.aborts == 1, "the loop still asked; the playback said there was nothing to cut"
    assert "barged_in" not in rec
    assert "barge_stop_ms" not in rec
    assert result.pending is cap, "it is still the next turn's input either way"


def test_an_unwatched_turn_plays_to_the_end_and_carries_nothing(monkeypatch, turns_log):
    """`make demo`'s default single turn must behave exactly as VOX-002 and VOX-003 measured it."""
    blocks = []

    def play(samples, **kw):
        blocks.append(kw.get("block", True))
        kw["on_first_audio"]()
        return None

    watched_turn(monkeypatch, interrupting=FakeCapture())
    monkeypatch.setattr(loop.audio, "play", play)

    result = loop.one_turn(DEFAULTS)
    rec = json.loads(turns_log.read_text(encoding="utf-8").strip())

    assert blocks == [True], "an unwatched reply is played blocking, as it always was"
    assert result.pending is None
    assert "barged_in" not in rec
    assert rec["ok"] is True


def test_a_watched_turn_that_hears_nothing_after_its_reply_ends_the_session(monkeypatch):
    """Silence after a reply is the end of the conversation, not a turn to keep waiting for."""
    watched_turn(monkeypatch, interrupting=None)

    result = loop.one_turn(DEFAULTS, watch=True)

    assert result.spoken is True, "the reply was still spoken"
    assert result.keep_going is False
    assert result.pending is None


def test_only_the_last_turn_of_a_run_is_unwatched(monkeypatch, capsys):
    """Where barge-in is switched on: every reply that has a turn after it is interruptible."""
    watches = []

    def one_turn(chosen, pending=None, watch=False, idx=None, **kw):
        # **kw: these tests are about watching and carry-forward, not about VOX-034 history.
        watches.append(watch)
        return loop.TurnResult(True, True, None)

    monkeypatch.setattr("sys.argv", ["vox", "--turns", "3"])
    monkeypatch.setattr(loop.vad, "_vad_model", lambda: None)
    monkeypatch.setattr(loop.arms, "select", lambda args: DEFAULTS)
    monkeypatch.setattr(loop, "one_turn", one_turn)

    assert loop.main() == 0
    assert watches == [True, True, False]


def test_a_carried_capture_is_passed_to_the_next_turn_by_main(monkeypatch):
    """main() is what threads the pending audio from one turn to the next."""
    cap = FakeCapture()
    seen = []

    def one_turn(chosen, pending=None, watch=False, idx=None, **kw):
        # **kw: these tests are about watching and carry-forward, not about VOX-034 history.
        seen.append(pending)
        return loop.TurnResult(True, True, cap if pending is None else None)

    monkeypatch.setattr("sys.argv", ["vox", "--turns", "3"])
    monkeypatch.setattr(loop.vad, "_vad_model", lambda: None)
    monkeypatch.setattr(loop.arms, "select", lambda args: DEFAULTS)
    monkeypatch.setattr(loop, "one_turn", one_turn)

    loop.main()

    assert seen == [None, cap, None]


def test_the_carried_capture_is_reported_in_seconds_of_audio(monkeypatch, capsys):
    """A demo needs to see that the words it just heard cut off are the ones going in next."""
    cap = FakeCapture()
    watched_turn(monkeypatch, interrupting=cap)

    loop.one_turn(DEFAULTS, pending=cap)

    assert f"carried in: {len(cap) / SAMPLE_RATE:.2f}s" in capsys.readouterr().out
