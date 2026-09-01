"""VOX-007, failure mode 3: a TTS failure degrades to text, loudly.

Synthesis is the last stage, so by the time it fails the turn has already spent a mic, an STT call
and an LLM call. Losing all of that because a vocoder raised is the wrong trade — the reply is
good, only the voice is missing. So the turn degrades: the answer goes to the user as text and the
session continues.

"Loudly" is the whole point and is what is asserted here. A degrade that only printed the reply
would be indistinguishable from a normal turn, and a degrade that wrote nothing to the turn record
would be worse than a crash: the turn silently leaves the latency percentiles, so the gate numbers
*improve* every time synthesis breaks. Three things therefore have to be true at once — stderr says
it failed, the turn line carries `degraded`, and the call line still shows the failure.

No weights: the TTS backend is replaced through the same BACKENDS seam test_arms.py uses.
"""
import json

import pytest

from conftest import fake_state          # same directory; pytest puts tests/unit on sys.path
from src import arms, loop
from src.config import LLM_ARMS, STT_ARMS, TTS_ARMS

DEFAULTS = {"stt": STT_ARMS[0], "llm": LLM_ARMS[0], "tts": TTS_ARMS[0]}
ANSWER = "Your last payslip was issued on the fourth."


class FakeCapture:
    """What vad.listen() hands back, with the two marks the turn timer reads."""
    segment = [0.0] * 16
    speech_end_t = 100.0
    endpointed_t = 100.4
    spoken_s = 1.2
    infer_ms = 9.0

    @property
    def t_vad_ms(self):
        return round((self.endpointed_t - self.speech_end_t) * 1000, 1)


@pytest.fixture
def turn(monkeypatch):
    """A turn where everything up to TTS works and the speaker records what it was given.

    -> the list of audio.play calls, so a test can assert nothing reached the speaker.
    """
    played = []
    monkeypatch.setattr(loop.vad, "listen", lambda *a, **kw: FakeCapture())
    monkeypatch.setattr(loop.arms, "stt", lambda *a, **kw: "when was I paid")
    # The un-retrieved path is VOX-019's extractor since the merge — one_turn is called here with
    # no index, so this is the call that produces the reply, and `nlu.reply` is no longer on it.
    monkeypatch.setattr(loop.state, "build", lambda *a, **kw: fake_state(ANSWER))
    monkeypatch.setattr(loop.audio, "play", lambda audio, **kw: played.append(kw) or None)
    return played


def break_tts(monkeypatch, exc=None):
    """Make the default TTS arm's backend raise, the way a dead vocoder does."""
    exc = exc or RuntimeError("Kokoro produced no audio for '…'")

    def boom(arm, text, rec):
        raise exc

    monkeypatch.setitem(arms._MODULES["tts"].BACKENDS, TTS_ARMS[0].backend, boom)
    return exc


def working_tts(monkeypatch):
    monkeypatch.setitem(arms._MODULES["tts"].BACKENDS, TTS_ARMS[0].backend,
                        lambda arm, text, rec: [0.0] * 240)


# --- the criterion --------------------------------------------------------------------------

def test_a_tts_failure_degrades_to_text_and_keeps_the_session(turn, monkeypatch, capsys):
    break_tts(monkeypatch)

    result = loop.one_turn(DEFAULTS)

    assert result.spoken is False, "nothing reached the speaker, so the turn was not spoken"
    assert result.keep_going is True, "a broken voice is no reason to end the conversation"
    assert result.pending is None, "nothing played, so there was nothing to be interrupted"


def test_the_degrade_is_loud_and_carries_the_reply(turn, monkeypatch, capsys):
    """On stderr, as readable prose, marked in a way a normal turn never is."""
    break_tts(monkeypatch, RuntimeError("SpeechT5 produced no audio"))

    loop.one_turn(DEFAULTS)
    err = capsys.readouterr().err

    assert "TTS FAILED" in err
    assert "SpeechT5 produced no audio" in err       # which failure, not just that one did
    assert ANSWER in err                             # the reply itself, so the user still gets it
    assert "text only" in err


def test_the_degraded_reply_is_readable_not_a_repr(turn, monkeypatch, capsys):
    """`vox says : '…'` is repr-quoted debug output. What replaces speech has to be prose."""
    break_tts(monkeypatch)

    loop.one_turn(DEFAULTS)

    assert ANSWER in capsys.readouterr().err.splitlines()[-1]


def test_a_tts_failure_does_not_reach_the_speaker(turn, monkeypatch):
    """Degrading must not half-play a buffer that synthesis never finished."""
    break_tts(monkeypatch)

    loop.one_turn(DEFAULTS)

    assert turn == []


def test_the_degrade_is_on_the_turn_record(turn, monkeypatch, turns_log):
    """Not only on the terminal. Every gate reads this file, so the file has to say it happened."""
    break_tts(monkeypatch)

    loop.one_turn(DEFAULTS)
    rec = json.loads(turns_log.read_text(encoding="utf-8").strip())

    assert rec["degraded"] == "tts"
    assert "RuntimeError" in rec["degrade_reason"]
    assert rec["ok"] is False
    assert rec["t_tts_ms"] is not None                # the failure took time; it is not a free turn
    assert rec["time_to_first_audio_ms"] is None      # no audio, so no time to it — not a fast turn


def test_the_call_record_still_shows_the_failed_synthesis(turn, monkeypatch, calls_log):
    """Degrading at the loop must not hide the failure from the per-call log."""
    break_tts(monkeypatch)

    loop.one_turn(DEFAULTS)
    tts_calls = [json.loads(line) for line in calls_log.read_text(encoding="utf-8").splitlines()
                 if json.loads(line)["stage"] == "tts"]

    assert len(tts_calls) == 1
    assert tts_calls[0]["ok"] is False
    assert "RuntimeError" in tts_calls[0]["error"]
    assert tts_calls[0]["model_id"] == TTS_ARMS[0].repo_id
    assert tts_calls[0]["cost_usd"] == 0.0


# --- the controls ---------------------------------------------------------------------------

def test_a_successful_turn_carries_no_degrade_marker(turn, monkeypatch, turns_log, capsys):
    """Without this, every test above would pass against a loop that degrades unconditionally."""
    working_tts(monkeypatch)

    result = loop.one_turn(DEFAULTS)
    rec = json.loads(turns_log.read_text(encoding="utf-8").strip())

    assert (result.spoken, result.keep_going) == (True, True)
    assert "degraded" not in rec
    assert "TTS FAILED" not in capsys.readouterr().err
    assert len(turn) == 1, "a working turn plays exactly once"


def test_a_tts_failure_does_not_stop_a_multi_turn_run(turn, monkeypatch, capsys):
    """The point of degrading rather than raising: turn two still happens, and is still counted."""
    monkeypatch.setattr("sys.argv", ["vox", "--turns", "2"])
    monkeypatch.setattr(loop.vad, "_vad_model", lambda: None)
    monkeypatch.setattr(loop.arms, "select", lambda args: DEFAULTS)

    states = iter([break_tts, working_tts])
    real_one_turn = loop.one_turn

    def one_turn(chosen, **kw):
        next(states)(monkeypatch)
        return real_one_turn(chosen, **kw)

    monkeypatch.setattr(loop, "one_turn", one_turn)

    assert loop.main() == 0                          # the second turn spoke, so the run succeeded
    out = capsys.readouterr()
    assert "1 turn(s) completed" in out.out, "the degraded turn must not be counted as spoken"
    assert "TTS FAILED" in out.err
