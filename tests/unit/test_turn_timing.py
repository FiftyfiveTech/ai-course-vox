"""VOX-003: a turn line carries all five timings, or it is not a turn line.

Pure arithmetic and file writing — no model, no mic, no network. The real numbers come from a
run; what is asserted here is the shape that makes those numbers readable, and the one thing
easiest to get quietly wrong: which mark time_to_first_audio is measured from.
"""
import json

import pytest

from src.telemetry import TURN_FIELDS, TurnTimer, turn_timer
from src.vad import Capture

# `turns_log` is autouse in tests/unit/conftest.py.


def capture(speech_end_t, endpointed_t):
    """A Capture with the marks set by hand — no audio, no silero."""
    return Capture(segment=[0.0] * 16, speech_end_t=speech_end_t, endpointed_t=endpointed_t,
                   spoken_s=1.5, infer_ms=12.3)


def test_record_always_carries_all_five_fields():
    t = TurnTimer("abc123")
    t.vad(capture(speech_end_t=100.0, endpointed_t=101.1))
    with t.stage("stt"):
        pass
    rec = t.record()
    assert set(TURN_FIELDS) <= rec.keys()
    # llm and tts never ran: present and null, so a missing measurement cannot read as a fast stage.
    assert rec["t_llm_ms"] is None and rec["t_tts_ms"] is None
    assert rec["ok"] is False


def test_time_to_first_audio_runs_from_the_last_speech_frame():
    """Not from the endpoint decision — the user has been waiting through the hangover too."""
    t = TurnTimer("abc123")
    t.vad(capture(speech_end_t=100.0, endpointed_t=101.1))   # 1100 ms hangover
    t.first_audio_t = 104.0
    rec = t.record()
    assert rec["t_vad_ms"] == 1100.0
    assert rec["time_to_first_audio_ms"] == 4000.0           # 4.0 s, hangover included
    assert rec["time_to_first_audio_ms"] > rec["t_vad_ms"]


def test_first_audio_keeps_the_earliest_stamp():
    """The stream callback fires once per block; only the first one is the first audio."""
    t = TurnTimer("abc123")
    t.first_audio()
    first = t.first_audio_t
    t.first_audio()
    assert t.first_audio_t == first


def test_quiet_mic_writes_no_line(turns_log):
    with turn_timer("abc123"):
        pass                                                  # nothing heard, no vad() call
    assert not turns_log.exists()


def test_failed_turn_still_writes_its_latency(turns_log):
    with pytest.raises(RuntimeError):
        with turn_timer("abc123", source="mic") as t:
            t.vad(capture(speech_end_t=100.0, endpointed_t=101.1))
            with t.stage("stt"):
                raise RuntimeError("groq said no")

    rec = json.loads(turns_log.read_text(encoding="utf-8").strip())
    assert set(TURN_FIELDS) <= rec.keys()
    assert rec["ok"] is False
    assert "groq said no" in rec["error"]
    assert rec["t_stt_ms"] is not None                        # the failure took time; it is logged


def test_source_distinguishes_a_fixture_run_from_a_live_one(turns_log):
    with turn_timer("abc123", source="tests/fixtures/hello_testing_voice.mp3") as t:
        t.vad(capture(speech_end_t=100.0, endpointed_t=100.2))
        t.first_audio_t = 103.0
    rec = json.loads(turns_log.read_text(encoding="utf-8").strip())
    assert rec["source"] == "tests/fixtures/hello_testing_voice.mp3"


def test_unknown_stage_is_refused():
    t = TurnTimer("abc123")
    with pytest.raises(ValueError, match="unknown stage"):
        with t.stage("asr"):
            pass
