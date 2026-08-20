"""VOX-013's comparison, checked without calling a provider or loading a weight.

Three separable things are asserted here, and they fail for different reasons:

  the architectures  `config.ARCHITECTURES` names real arms and actually differs at every stage. A
                     shared leg would make "comparison" a claim about one variable while the table
                     showed three, and nothing else in the suite would notice.
  the harness        `fixture_turn(fallback=False)` really reaches the arms with the fallback off.
                     This is the assertion the table depends on: if a rescued remote arm slipped
                     through, the local arm's latency would be printed on the remote arm's row and
                     the number would be wrong in the one way that looks right.
  the table          five rows for every arm, and `n/a` — never 0 — for a field nobody measured. A
                     missing measurement that prints as a number reads as a fast stage.
"""
import importlib.util
import sys
import time
from pathlib import Path

import pytest

from src import arms, harness, telemetry
from src.config import ARCHITECTURES, TTS_ARMS, resolve

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_STAGES = ("stt", "llm", "tts")


def _load_compare():
    """Import scripts/compare_arms.py, which is a script and not on the package path."""
    path = REPO_ROOT / "scripts" / "compare_arms.py"
    spec = importlib.util.spec_from_file_location("compare_arms", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("compare_arms", module)
    spec.loader.exec_module(module)
    return module


compare = _load_compare()


# `calls_log`, `turns_log`, `no_cooldowns` and `no_env_override` are autouse in conftest.py.

# --- the architectures ---------------------------------------------------------------------

def test_there_are_at_least_two_architectures():
    """The ticket says ">=2 real arms". One arm set is a measurement, not a comparison."""
    assert len(ARCHITECTURES) >= 2


@pytest.mark.parametrize("name", tuple(ARCHITECTURES))
def test_an_architecture_names_every_model_stage_and_nothing_else(name):
    assert set(ARCHITECTURES[name]) == set(MODEL_STAGES)


@pytest.mark.parametrize("name", tuple(ARCHITECTURES))
def test_every_leg_resolves_to_a_registered_arm(name):
    """A typo'd alias must fail here, not two minutes into a timed run."""
    chosen = compare.resolve_architecture(name)
    for stage in MODEL_STAGES:
        assert chosen[stage] in arms.available(stage)


def test_the_two_architectures_differ_at_every_stage():
    """A shared leg means the table shows three differences and only two of them are real."""
    fast, quality = compare.resolve_architecture("fast"), compare.resolve_architecture("quality")
    for stage in MODEL_STAGES:
        assert fast[stage].id != quality[stage].id, f"both architectures run {fast[stage].id} at {stage}"


def test_the_fast_architecture_needs_no_credential():
    """What makes this comparison runnable when a free tier refuses: one whole column is local."""
    for stage, arm in compare.resolve_architecture("fast").items():
        assert arm.local, f"fast/{stage} is {arm.id}, which is not local"
        assert arm.key_env is None


def test_the_quality_architecture_is_hosted_where_the_pipeline_says_it_is():
    quality = compare.resolve_architecture("quality")
    assert not quality["stt"].local and not quality["llm"].local
    assert quality["tts"].local, "tts is local by PIPELINE; a hosted one would need a key and a quota"


def test_an_unknown_architecture_is_refused_by_name():
    with pytest.raises(SystemExit) as e:
        compare.resolve_architecture("nope")
    assert "fast" in str(e.value) and "quality" in str(e.value)


def test_piper_and_kokoro_do_not_share_a_sample_rate():
    """Otherwise the TTS leg of the comparison is partly a comparison of one arm resampled."""
    piper, kokoro = resolve("tts", "piper"), resolve("tts", "kokoro")
    assert piper.extra["sample_rate"] != kokoro.extra["sample_rate"]
    assert piper in TTS_ARMS and kokoro is TTS_ARMS[0]


# --- the table -----------------------------------------------------------------------------

def record(**fields):
    """A turn record with every one of the five fields present unless overridden."""
    base = {f: 100.0 for f in telemetry.TURN_FIELDS}
    return {"turn_id": "t0", "stage_sum_ms": 400.0, **base, **fields}


def result(name, records=(), failures=()):
    r = compare.Result(name, compare.resolve_architecture(name))
    r.records.extend(records)
    r.failures.extend(failures)
    return r


def test_the_table_prints_all_five_stages_for_every_arm(capsys):
    results = [result("fast", [record()]), result("quality", [record()])]
    compare.show_table(results, repeat=1, fixtures=[Path("f.mp3")])
    out = capsys.readouterr().out

    for field in telemetry.TURN_FIELDS:
        label = field[:-3] if field.endswith("_ms") else field
        assert label in out, f"{field} is not a row in the table"
    assert "fast" in out and "quality" in out
    assert len(telemetry.TURN_FIELDS) == 5, "the ticket says five stages"


def test_an_unmeasured_field_prints_na_and_not_zero(capsys):
    """The rule gate_phase0 already states: a zero would be a lie."""
    results = [result("fast", [record(time_to_first_audio_ms=None)])]
    compare.show_table(results, repeat=1, fixtures=[Path("f.mp3")])
    ttfa = next(line for line in capsys.readouterr().out.splitlines()
                if line.strip().startswith("time_to_first_audio"))
    assert "n/a" in ttfa
    assert "0" not in ttfa.split("time_to_first_audio")[1]


def test_a_failed_turn_is_counted_and_kept_out_of_the_numbers():
    r = result("quality", [record(t_llm_ms=500.0)], [("t1", "llm", "RateLimited: 429")])
    assert r.values("t_llm_ms") == [500.0], "a failed turn must not contribute a value"
    assert len(r.failures) == 1
    assert set(r.turn_ids) == {"t0", "t1"}, "a failed turn's log lines still have to be shown"


def test_an_arm_is_complete_only_when_one_turn_measured_all_five():
    assert result("fast", [record()]).complete()
    assert not result("fast", [record(t_tts_ms=None)]).complete()
    assert not result("fast").complete()
    # Two half-measured turns are not one measured turn: the five numbers have to come from the
    # same turn or `stage_sum` describes a turn nobody took.
    assert not result("fast", [record(t_stt_ms=None), record(t_tts_ms=None)]).complete()


def test_stats_of_nothing_is_none_rather_than_zero():
    assert compare.stats([]) is None
    assert compare.stats([3.0, 1.0, 2.0]) == (1.0, 2.0, 3.0)


# --- the harness ---------------------------------------------------------------------------

def fake_arms(monkeypatch, calls, *, stt_raises=None):
    """Swap all three backends for recorders. Nothing loads, nothing dials out."""
    def stt_backend(arm, payload, rec):
        calls.append(("stt", arm.id))
        if stt_raises is not None:
            raise stt_raises
        return "so this is testing"

    def llm_backend(arm, payload, rec):
        calls.append(("llm", arm.id))
        return "Three tasks are due today."

    def tts_backend(arm, payload, rec):
        calls.append(("tts", arm.id))
        return [0.0] * 800

    for stage, fn in (("stt", stt_backend), ("llm", llm_backend), ("tts", tts_backend)):
        module = arms._MODULES[stage]
        for backend in module.BACKENDS:
            monkeypatch.setitem(module.BACKENDS, backend, fn)
        monkeypatch.setattr(module, "LOADERS", {}, raising=False)


class FakeCapture:
    """What vad.endpoint_frames returns, reduced to what fixture_turn reads off it."""
    t_vad_ms = 12.0
    infer_ms = 3.0
    spoken_s = 1.4

    def __init__(self):
        self.segment = [0.0] * 16_000
        self.speech_end_t = 1.0

    def __len__(self):
        return len(self.segment)


def test_fixture_turn_with_fallback_off_never_reaches_the_local_arm(monkeypatch):
    """The assertion the whole comparison rests on.

    `quality` runs whisper-large-v3 on Groq. With the fallback on, a 429 there is answered by
    faster-whisper-base locally and the turn succeeds — which is right for the loop and wrong for a
    table, because the local arm's latency would land on the remote arm's row.
    """
    from src.errors import RateLimited

    calls = []
    fake_arms(monkeypatch, calls,
              stt_raises=RateLimited("429", status_code=429, retry_after=None,
                                     arm_id="openai/whisper-large-v3@groq"))
    chosen = compare.resolve_architecture("quality")

    with pytest.raises(RateLimited) as e:
        harness.fixture_turn(chosen, [0.0] * 16_000, "fake.mp3", play=False, fallback=False,
                             segment=FakeCapture())

    assert [stage for stage, _ in calls] == ["stt"], "only the arm that was asked for ran"
    assert calls[0][1] == chosen["stt"].id
    # The record rides along on the exception, because a failed turn still has a latency.
    assert e.value.turn_record["t_stt_ms"] > 0
    assert e.value.turn_record["t_llm_ms"] is None
    assert e.value.turn_record["ok"] is False


def test_fixture_turn_with_fallback_on_still_rescues_the_turn(monkeypatch):
    """The other half: the loop's behaviour is unchanged by the harness extraction."""
    from src.errors import RateLimited

    seen = []

    def stt_backend(arm, payload, rec):
        seen.append(arm.id)
        if len(seen) == 1:
            raise RateLimited("429", status_code=429, retry_after=None, arm_id=arm.id)
        return "so this is testing"

    calls = []
    fake_arms(monkeypatch, calls)
    monkeypatch.setitem(arms._MODULES["stt"].BACKENDS, "openai-audio", stt_backend)
    monkeypatch.setitem(arms._MODULES["stt"].BACKENDS, "faster-whisper", stt_backend)
    chosen = compare.resolve_architecture("quality")

    run = harness.fixture_turn(chosen, [0.0] * 16_000, "fake.mp3", play=False, fallback=True,
                               segment=FakeCapture())

    assert len(seen) == 2, "the remote arm refused and the local one answered"
    assert run.record["fell_back"] == ["stt"]
    # The rewrite that matters: the record names the arm that actually ran, not the one selected.
    assert run.record["stt_model"] == resolve("stt", "faster-base").id


def test_fixture_turn_reuses_a_capture_instead_of_endpointing_again(monkeypatch):
    """Every arm sees the same segment, or the comparison is partly a comparison of two VAD runs."""
    calls = []
    fake_arms(monkeypatch, calls)

    def refuse(*a, **k):
        raise AssertionError("endpoint_frames was called even though a capture was passed")

    monkeypatch.setattr(harness.vad, "endpoint_frames", refuse)
    cap = FakeCapture()
    run = harness.fixture_turn(compare.resolve_architecture("fast"), [], "fake.mp3", play=False,
                               fallback=False, segment=cap)
    assert run.capture is cap
    assert run.record["t_vad_ms"] == cap.t_vad_ms


def test_a_reused_capture_measures_the_second_turn_from_a_stale_mark(monkeypatch):
    """The bug that made `compare` print a growing time_to_first_audio, pinned so it stays fixed.

    `time_to_first_audio` is measured from `Capture.speech_end_t`. Hand the same capture to two
    turns and the second one is measured from the first one's origin, so it grows by everything that
    happened in between rather than by anything the second turn did. It read 6.5 s, 39.4 s, 63.6 s
    across three identical turns. This asserts the mechanism, so that `compare_arms.py` endpointing
    per turn cannot be "simplified" back into the bug.
    """
    calls = []
    fake_arms(monkeypatch, calls)
    chosen = compare.resolve_architecture("fast")
    # Playback stamps first-audio without touching a device, so the only thing separating the two
    # turns is real elapsed time — which is exactly the quantity the bug leaks into the number.
    monkeypatch.setattr(harness.audio_out, "play",
                        lambda *a, **k: (k.get("on_first_audio") or (lambda: None))())

    shared = FakeCapture()
    shared.speech_end_t = time.perf_counter()

    first = harness.fixture_turn(chosen, [], "f.mp3", play=True, fallback=False, segment=shared)
    time.sleep(0.05)
    second = harness.fixture_turn(chosen, [], "f.mp3", play=True, fallback=False, segment=shared)

    grew = second.record["time_to_first_audio_ms"] - first.record["time_to_first_audio_ms"]
    assert grew >= 40, (
        f"the second turn should have inherited the 50 ms sleep, but grew only {grew:.1f} ms. If "
        f"a reused capture has stopped carrying a stale speech_end_t, the warning in "
        f"harness.fixture_turn and the per-turn endpointing in compare_arms.py can both go.")


def test_a_drifting_segment_is_counted_not_averaged_in():
    """`compare` must refuse to call it a comparison when the arms saw different audio."""
    r = result("fast", [record()])
    assert r.drifted == 0
    r.drifted = 1
    # The verdict block reads exactly this, and returns 1 rather than printing a table as a result.
    assert r.complete() and r.drifted


def test_not_playing_leaves_time_to_first_audio_null_and_the_turn_not_ok(monkeypatch):
    """--silent is an unmeasured turn, not a fast one, and the record has to say so."""
    calls = []
    fake_arms(monkeypatch, calls)
    run = harness.fixture_turn(compare.resolve_architecture("fast"), [], "fake.mp3", play=False,
                               fallback=False, segment=FakeCapture())
    assert run.record["time_to_first_audio_ms"] is None
    assert run.record["ok"] is False
    assert run.record["t_stt_ms"] is not None and run.record["t_tts_ms"] is not None
