"""VOX-026: the scripted session, and the three seams that let it run without a person.

What is asserted here, and why each one is not covered anywhere else:

  vad.drive       the endpointing loop `listen()` and a recording now share. The barge-in hook is
                  inside it, so a second copy of this loop is how a rehearsal ends up measuring an
                  `abort()` call and printing it in VOX-011's field.
  vad.paced       frames at real time. The point of the whole rehearsal: unpaced, the
                  VAD_SILENCE_MS hangover collapses to silero compute and every ttfa reads about a
                  second better than a person hears it.
  confirmation    the yes/no leg, which used to be a block inside `one_turn` and timed its two
                  extra calls into `t_stt_ms` and `t_tts_ms` — the same slots the turn's own
                  utterance and read-back had already written.
  the console     a print that kills a turn. `speak_and_watch` opened its barge-in line with a
                  box-drawing character, which cp1252 cannot encode, so on a Windows console the
                  interruption worked and then the turn died on the line announcing it.
  the script      `evals/demo/session_v1.json` against the checker in `scripts/dry_run.py`: an
                  expectation the checker does not implement asserts nothing and prints PASS.

No mic, no speaker, no weights, no network. The fake silero is the same one test_barge_in.py and
test_silence.py use — any nonzero sample is speech — kept local for the same reason they keep
theirs: a file that states its own audio is readable on its own.
"""
import importlib.util
import io
import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import pytest
import torch

from conftest import SESSION_DEFAULTS, fake_state    # same directory; pytest puts it on sys.path
from src import config, harness, loop, telemetry, vad
from src.config import LLM_ARMS, STT_ARMS, TTS_ARMS, VAD_FRAME, VAD_SILENCE_MS
from src.vad import DONE, MS_PER_FRAME

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULTS = {"stt": STT_ARMS[0], "llm": LLM_ARMS[0], "tts": TTS_ARMS[0]}


def _load_dry_run():
    """Import scripts/dry_run.py, which is a script and not on the package path."""
    path = REPO_ROOT / "scripts" / "dry_run.py"
    spec = importlib.util.spec_from_file_location("dry_run", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("dry_run", module)
    spec.loader.exec_module(module)
    return module


dry_run = _load_dry_run()
SESSION = json.loads((REPO_ROOT / "evals" / "demo" / "session_v1.json").read_text(encoding="utf-8"))


class FakeVad:
    """silero's call interface, decided by the audio: any nonzero sample is speech."""

    def reset_states(self):
        pass

    def __call__(self, frame, sample_rate):
        return torch.tensor(1.0 if float(frame.abs().max()) > 0.0 else 0.0)


def silence(n):
    return [np.zeros(VAD_FRAME, dtype="float32") for _ in range(n)]


def speech(n):
    return [np.full(VAD_FRAME, 0.5, dtype="float32") for _ in range(n)]


def utterance(speech_frames=20, tail_frames=None):
    """Silence, then speech, then enough silence for the endpointer to call it finished."""
    tail = tail_frames if tail_frames is not None else int(VAD_SILENCE_MS / MS_PER_FRAME) + 2
    return silence(3) + speech(speech_frames) + silence(tail)


# --- the endpointing loop, shared by the mic and a recording ------------------------------------

def test_drive_endpoints_a_recording_the_way_the_mic_does():
    cap, state = vad.drive(utterance(), model=FakeVad())

    assert state == DONE
    assert cap is not None
    assert cap.spoken_s == pytest.approx(20 * MS_PER_FRAME / 1000, abs=0.05)
    assert cap.t_vad_ms > 0, "the hangover is a real interval even when the frames are free"


def test_the_barge_hook_fires_once_and_is_handed_the_start_of_speech():
    """The mark VOX-011's stop latency is measured from. Measuring from the confirmation instead
    would hide BARGE_MIN_SPEECH_MS inside the number."""
    marks = []
    before = time.perf_counter()

    cap, _ = vad.drive(utterance(), model=FakeVad(), on_speech=marks.append, confirm_ms=100)

    assert len(marks) == 1, "once per utterance, not once per frame above the threshold"
    assert before <= marks[0] <= cap.endpointed_t
    assert marks[0] < cap.speech_end_t, "handed when speech started, not when it ended"


def test_a_recording_with_no_speech_in_it_is_not_a_turn():
    cap, state = vad.drive(silence(40), model=FakeVad())

    assert cap is None and state != DONE


def test_endpoint_frames_still_answers_the_way_its_callers_expect():
    """`harness.fixture_turn` and `scripts/compare_arms.py` unpack (Capture, state) from it."""
    cap, state = vad.endpoint_frames(utterance(), model=FakeVad())

    assert state == DONE and cap is not None


# --- real-time pacing ---------------------------------------------------------------------------

def test_paced_frames_arrive_at_real_time():
    frames = silence(10)
    t0 = time.perf_counter()

    got = list(vad.paced(frames))

    elapsed = time.perf_counter() - t0
    assert len(got) == len(frames) and all(a is b for a, b in zip(got, frames))
    # Nine gaps of 32 ms between ten frames. The upper bound is loose because a sleep is a floor,
    # not a promise, and this suite runs on whatever CI has.
    assert 0.9 * 9 * MS_PER_FRAME / 1000 < elapsed < 1.0


def test_pacing_does_not_add_the_consumers_own_cost_to_the_clock():
    """The absolute-deadline choice, pinned: silero costs 1-4 ms a frame, and sleeping 32 ms *plus*
    that would drift slower than real time — inventing t_vad nobody waited for."""
    t0 = time.perf_counter()

    for _frame in vad.paced(silence(10)):
        time.sleep(0.010)                      # a slow-ish consumer, as silero is

    elapsed = time.perf_counter() - t0
    assert elapsed < 9 * (MS_PER_FRAME + 10) / 1000, "the consumer's cost was added to the pacing"


def test_pacing_says_so_when_the_machine_could_not_keep_up():
    """A number measured on a machine that fell behind is not a live number, and has to say so."""
    said = []

    for _frame in vad.paced(silence(4), echo=said.append):
        time.sleep(0.200)                      # far slower than 32 ms a frame

    assert said and "behind real time" in said[0]


# --- the confirmation leg -----------------------------------------------------------------------

def confirming_turn(monkeypatch, transcript="yes, go ahead", heard=True):
    """A TurnTimer that has already timed a first leg, plus the fakes the second one needs.

    -> (turn, spoken) where `spoken` collects the text handed to TTS.
    """
    spoken = []
    monkeypatch.setattr(loop.arms, "stt", lambda *a, **kw: transcript)
    monkeypatch.setattr(loop.arms, "tts", lambda text, *a, **kw: spoken.append(text) or
                        types.SimpleNamespace(audio=[0.0] * 240, sample_rate=24_000))

    turn = telemetry.TurnTimer("t", source="rehearsal:test")
    with turn.stage("stt"):
        time.sleep(0.02)                       # the utterance that started the turn
    with turn.stage("tts"):
        time.sleep(0.02)                       # the read-back
    first = (turn.ms["stt"], turn.ms["tts"])

    cap = types.SimpleNamespace(segment=[0.0] * 16_000) if heard else None
    listen = lambda **kw: cap                                                  # noqa: E731
    return turn, spoken, first, listen


def test_the_confirmation_leg_does_not_overwrite_the_turns_own_latency(monkeypatch):
    """The bug this ticket found: timed as stages, the yes/no replaced `t_stt_ms` with the latency
    of transcribing one word, and the cancel reply replaced `t_tts_ms` with its own."""
    turn, _spoken, first, listen = confirming_turn(monkeypatch, "no, cancel that")

    got = loop.confirmation_leg(turn, "t", DEFAULTS, fake_state("Shall I go ahead?",
                                                                next_action="confirm"),
                                listen=listen, play=False, say=lambda *_: None)

    assert got == "no"
    assert (turn.ms["stt"], turn.ms["tts"]) == first, "the first leg's five fields are untouched"
    # >= 0 and not > 0: the fakes here return instantly, so what is asserted is that the second
    # leg got fields of its own — the numbers themselves are measured in a real run.
    assert turn.extra["t_confirm_stt_ms"] >= 0
    assert turn.extra["t_confirm_tts_ms"] >= 0, "the cancel reply was spoken and timed separately"
    assert turn.extra["confirmation"] == "no"


def test_a_confirmed_action_costs_no_second_reply(monkeypatch):
    turn, spoken, _first, listen = confirming_turn(monkeypatch, "yes, go ahead")

    got = loop.confirmation_leg(turn, "t", DEFAULTS, fake_state("Shall I go ahead?",
                                                                next_action="confirm"),
                                listen=listen, play=False, say=lambda *_: None)

    assert got == "yes"
    assert spoken == [], "a yes proceeds; there is nothing to say about it"
    assert "t_confirm_tts_ms" not in turn.extra


def test_silence_after_a_read_back_is_a_cancel(monkeypatch):
    """The safe direction. A confirmation nobody answered must not proceed."""
    turn, spoken, _first, listen = confirming_turn(monkeypatch, heard=False)

    got = loop.confirmation_leg(turn, "t", DEFAULTS, fake_state("Shall I go ahead?",
                                                                next_action="confirm"),
                                listen=listen, play=False, say=lambda *_: None)

    assert got == "no"
    assert turn.extra["confirmation_transcript"] is None
    assert spoken == ["Got it, cancelled."]


def test_a_turn_that_asked_for_nothing_runs_no_second_leg(monkeypatch):
    turn, spoken, first, listen = confirming_turn(monkeypatch)

    got = loop.confirmation_leg(turn, "t", DEFAULTS, fake_state("Your last payslip was Tuesday."),
                                listen=listen, play=False, say=lambda *_: None)

    assert got is None
    assert "confirmation" not in turn.extra and spoken == []
    assert (turn.ms["stt"], turn.ms["tts"]) == first


# --- a print that killed a turn -----------------------------------------------------------------

def cp1252_stdout(monkeypatch):
    """Make sys.stdout what a Windows console is: cp1252, and strict about it. -> the buffer."""
    raw = io.BytesIO()
    monkeypatch.setattr(sys, "stdout", io.TextIOWrapper(raw, encoding="cp1252", newline=""))
    return raw


def test_the_barge_in_line_survives_a_windows_console(monkeypatch):
    """The demo-fatal one. `abort()` had already cut the reply, and then the print announcing it
    raised UnicodeEncodeError out of the endpointer and killed the turn."""
    cp1252_stdout(monkeypatch)
    playback = types.SimpleNamespace(reply_s=3.4, played_s=1.2, out_latency_s=0.182,
                                     abort=lambda: time.perf_counter(), close=lambda: None)
    monkeypatch.setattr(loop.audio, "play", lambda *a, **kw: playback)
    turn = telemetry.TurnTimer("t", source="rehearsal:test")

    def listen(on_speech=None, threshold=None, announce=True, **_):
        on_speech(time.perf_counter() - 0.3)
        return types.SimpleNamespace(segment=[0.0] * 16_000)

    got = loop.speak_and_watch(turn, types.SimpleNamespace(audio=[0.0] * 240, sample_rate=24_000),
                               listen=listen)

    assert got is not None, "the interruption was captured, not lost to a console encoding"
    assert turn.extra["barged_in"] is True
    assert turn.extra["barge_stop_ms"] == pytest.approx(300, abs=150)


def test_utf8_console_makes_a_cp1252_stream_encode_anything(monkeypatch):
    """The general case, which choosing nicer glyphs cannot fix: a turn prints a transcript and a
    reply, and neither is a string this process chose the characters of. The corpus itself carries
    U+25CF bullets, so this is not hypothetical."""
    raw = cp1252_stdout(monkeypatch)

    config.utf8_console()
    print("● 100 % ─ ok", end="")
    sys.stdout.flush()

    assert "100 %" in raw.getvalue().decode("utf-8")


def test_utf8_console_is_silent_about_a_stream_it_cannot_reconfigure(monkeypatch):
    """pytest's capture and a plain StringIO have no reconfigure(); an entry point must not care."""
    monkeypatch.setattr(sys, "stdout", io.StringIO())

    config.utf8_console()          # must not raise


# --- the script and the checker have to agree ---------------------------------------------------

def test_the_session_is_ten_turns_with_a_barge_in_and_two_confirmations():
    """The ticket's own shape: ten turns, one interruption, a confirmation. Asserted here so a later
    edit to the script cannot quietly drop the part the ticket names."""
    turns = SESSION["turns"]

    assert len(turns) == 10
    assert len({t["id"] for t in turns}) == 10, "ids are how a failure is reported"
    assert sum(1 for t in turns if t["kind"] == "barge") == 1
    assert sum(1 for t in turns if "confirm_with" in t) == 2
    assert {t["expect"]["confirmation"] for t in turns if "confirm_with" in t} == {"yes", "no"}, \
        "a gate you can only say yes to is not a gate"


def test_the_carried_turn_follows_the_barge_turn():
    """It has no audio of its own — it runs on what interrupted the previous reply, so the order is
    load-bearing rather than cosmetic."""
    kinds = [t["kind"] for t in SESSION["turns"]]

    for i, kind in enumerate(kinds):
        if kind == "carried":
            assert i > 0 and kinds[i - 1] == "barge"


def test_every_turn_has_something_to_say():
    for turn in SESSION["turns"]:
        has_audio = turn.get("say") or turn.get("recording") or turn["kind"] == "carried"
        assert has_audio, f"{turn['id']} has no line, no recording and nothing carried into it"
        assert turn.get("why"), f"{turn['id']} does not say why it is in the demo"


def test_every_expectation_is_one_the_checker_implements():
    """An `expect` key nobody asserts is worse than no expectation: it prints PASS."""
    for turn in SESSION["turns"]:
        unknown = set(turn.get("expect", {})) - dry_run.CHECKED_KEYS
        assert not unknown, f"{turn['id']} expects {sorted(unknown)}, which check() ignores"


def test_the_checker_fails_an_expectation_it_does_not_implement():
    """The other half: the unknown key is reported, not skipped."""
    bad = dry_run.check({"id": "tX", "kind": "ask", "expect": {"vibes": "good"}},
                        {"ok": True}, None)

    assert any("vibes" in line for line in bad)


def test_a_barge_that_cut_nothing_fails_its_turn():
    """`Playback.abort()` returns None when the reply had already played out, so a scripted barge
    that arrives too late is a silent non-event — which is exactly what an assertion is for."""
    turn = {"id": "t04", "kind": "barge", "expect": {"barged_in": True}}
    run = harness.TurnRun({}, "t", None, "", "", None, None, None)

    bad = dry_run.check(turn, {"ok": True}, run)

    assert any("not interrupted" in line for line in bad)
    assert any("no input to run on" in line for line in bad)


def test_a_clean_turn_reports_nothing():
    turn = {"id": "t02", "kind": "ask",
            "expect": {"path": "grounded", "sources_any": ["leave-policy:p15"]}}
    rec = {"ok": True, "grounded": True, "retrieved": 5,
           "sources": ["leave-policy:p15", "leave-policy:p17"]}

    assert dry_run.check(turn, rec, None) == []


def test_the_env_scan_only_flags_names_no_module_reads():
    """The check that would have caught VOX_STT_FIXUPS: set in .env, read by nothing on this branch.

    Asserted against the repo as it stands — every name the scan returns must genuinely be absent
    from the code, which is the property the preflight prints."""
    for name in dry_run.env_names_no_module_reads():
        for folder in ("src", "scripts"):
            for py in (REPO_ROOT / folder).rglob("*.py"):
                assert name not in py.read_text(encoding="utf-8"), \
                    f"{name} is read in {py} — the scan is over-reporting"


# --- the fix that started the ticket ------------------------------------------------------------

def test_the_pinned_session_defaults_match_config_py():
    """`conftest.SESSION_DEFAULTS` duplicates the defaults in config.py on purpose — a value the
    environment cannot reach. This is the test that fails if the two drift apart."""
    source = (REPO_ROOT / "src" / "config.py").read_text(encoding="utf-8")

    for name, default in SESSION_DEFAULTS.items():
        literal = f'{default:g}' if isinstance(default, float) else str(default)
        assert f'os.environ.get("VOX_{name}", "{literal}")' in source, \
            f"config.py's default for VOX_{name} is no longer {literal!r}"


def test_the_suite_does_not_read_the_operators_demo_profile():
    """`.env` on the demo machine sets VOX_SESSION_QUIET_LIMIT=1 and that failed three tests in
    test_session.py. Whatever is in the environment, a unit test sees the code's defaults."""
    assert config.SESSION_QUIET_LIMIT == SESSION_DEFAULTS["SESSION_QUIET_LIMIT"]
    assert loop.SESSION_QUIET_LIMIT == SESSION_DEFAULTS["SESSION_QUIET_LIMIT"]
