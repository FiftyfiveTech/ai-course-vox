"""A run bounded by the clock instead of by a turn count — what `make demo` now starts.

The unit under test is `loop.Budget` and the `while` loop in `main()` that reads it. Everything a
turn does is faked here: this file is about what ends a session and what keeps one going, and it
would be a poor test of that if it also needed a microphone, a speaker, weights or three real
minutes. `Budget` takes its clock as an argument for exactly that last reason.

Four decisions are pinned:

  the clock      turns keep coming until the deadline, and it is read between turns only — a turn
                 that started in time finishes, so a session overruns by at most one reply.
  barge-in       every reply in a timed run plays with the mic open, because inside a conversation
                 there is always a next turn to carry an interruption into. That is the difference
                 from `--turns N`, whose last reply holds no mic.
  a pause        silence does not end a timed session the way it ends a counted one, but
                 SESSION_QUIET_LIMIT consecutive silences do — otherwise a muted mic is
                 indistinguishable from a person thinking, and the run spins quietly to the end.
  the wind-down  a sentence spoken while the last reply was playing is answered, not dropped on the
                 way out of the loop.

`--turns N` is unchanged and is tested where it always was — tests/unit/test_barge_in.py and
tests/unit/test_silence.py. The two tests here that touch it only assert that this ticket left it
alone.
"""
import pytest

from src import config, loop
from src.config import LLM_ARMS, STT_ARMS, TTS_ARMS

DEFAULTS = {"stt": STT_ARMS[0], "llm": LLM_ARMS[0], "tts": TTS_ARMS[0]}


class FakeClock:
    """time.monotonic, advanced by the test instead of by the world. Seconds."""

    def __init__(self, per_call=0.0):
        self.now = 1_000.0
        self.per_call = per_call        # how much a *turn* costs, added by the fake one_turn

    def __call__(self):
        return self.now

    def tick(self, seconds):
        self.now += seconds


class FakeCapture:
    """Stands in for a vad.Capture carried out of a watched reply. Only its length is read."""

    def __len__(self):
        return 16_000


def run_main(monkeypatch, argv, clock, turn=None):
    """Drive `loop.main()` with every turn faked. -> the list of `watch` values it asked for.

    `turn(i, clock) -> TurnResult` is the whole of a turn: it advances the clock by however long
    that turn is supposed to have taken and says what came of it. The default is a turn that
    speaks, wants to continue, and takes 20 seconds.
    """
    watches = []

    def default_turn(i, clock):
        clock.tick(20)
        return loop.TurnResult(True, True, None)

    body = turn or default_turn
    calls = {"i": 0}

    def one_turn(chosen, pending=None, watch=False, idx=None, **kw):
        # **kw so this double does not have to be edited every time one_turn grows an argument it
        # does not care about. These tests are about what BOUNDS a session — the turn count, the
        # clock, the quiet streak — and `history` (VOX-034) is not one of those.
        watches.append(watch)
        result = body(calls["i"], clock)
        calls["i"] += 1
        return result

    monkeypatch.setattr("sys.argv", ["vox"] + argv)
    monkeypatch.setattr(loop.time, "monotonic", clock)
    monkeypatch.setattr(loop.vad, "_vad_model", lambda: None)
    monkeypatch.setattr(loop.arms, "select", lambda args: DEFAULTS)
    monkeypatch.setattr(loop.answer_mod, "knowledge_base", lambda: None)
    monkeypatch.setattr(loop, "one_turn", one_turn)
    return watches


# --- Budget: the two policies, stated once ------------------------------------------------------

def test_a_run_with_no_flags_is_still_one_turn():
    """VOX-002's default is not disturbed by a session existing: no flag, one turn, then exit."""
    budget = loop.Budget(clock=FakeClock())

    assert budget.timed is False
    assert budget.open() is True
    assert budget.watch() is False, "one turn, so its reply has no next turn to be cut into"
    budget.took(quiet=False)
    assert budget.open() is False


def test_turns_and_minutes_are_not_both_a_bound():
    """Two answers to "what ends this run" is a bug, so it is refused rather than resolved."""
    with pytest.raises(ValueError):
        loop.Budget(turns=3, minutes=3)


def test_the_deadline_is_read_between_turns_and_never_during_one():
    """A turn that started in time finishes: the overrun is one reply, not a cut-off sentence."""
    clock = FakeClock()
    budget = loop.Budget(minutes=1, clock=clock)

    assert budget.open() is True
    clock.tick(59)
    assert budget.open() is True, "a turn may still start with a second left"
    budget.took(quiet=False)
    clock.tick(40)                                   # that turn ran 40 s past the deadline
    assert budget.open() is False
    assert budget.left_s() == 0


def test_a_timed_run_counts_up_and_a_counted_run_counts_down():
    """The line between turns is what tells a person the session is going to end by itself."""
    clock = FakeClock()
    budget = loop.Budget(minutes=3, clock=clock)

    clock.tick(41)
    assert budget.clock_str(budget.left_s()) == "2:19"
    assert budget.clock_str() == "0:41", "no argument means elapsed, for the closing line"


# --- main(): what keeps a session going ---------------------------------------------------------

def test_turns_keep_coming_until_the_clock_runs_out(monkeypatch, capsys):
    """The whole point of the ticket: `make demo` is a conversation, not one turn.

    Ten 20-second turns fit in three minutes and the eleventh does not start — the arithmetic is
    the fake clock's, so this asserts the loop's decision and not a real machine's timing.
    """
    clock = FakeClock()
    watches = run_main(monkeypatch, ["--minutes", "3"], clock)

    assert loop.main() == 0
    assert len(watches) == 9, "180 s of clock, 20 s a turn, checked before each one"
    assert "9 turn(s) completed in 3:00." in capsys.readouterr().out


def test_every_reply_in_a_timed_session_is_interruptible(monkeypatch):
    """Barge-in is not a `--turns 3` feature any more — it is how the demo behaves throughout.

    The last turn of a counted run plays its reply with no mic open, because there is no turn left
    to carry an interruption into. Inside a session there always is, until the clock says otherwise.
    """
    clock = FakeClock()
    watches = run_main(monkeypatch, ["--minutes", "1"], clock)

    loop.main()

    assert watches and all(watches), "a reply you cannot talk over is not a conversation"


def test_the_last_turn_of_a_counted_run_is_still_unwatched(monkeypatch):
    """`--turns N` is untouched by this ticket. Here so a later edit cannot quietly merge the two."""
    clock = FakeClock()
    watches = run_main(monkeypatch, ["--turns", "3"], clock)

    loop.main()

    assert watches == [True, True, False]


def test_a_pause_does_not_end_a_timed_session(monkeypatch, capsys):
    """Silence in the middle of a conversation is a person thinking, not a session that finished.

    A counted run stops on the first quiet listen — with a fixed number of replies left there is
    nothing to wait for. A timed one has the clock as its reason to keep the mic open, and says so
    on stdout rather than exiting on a pause that the user did not intend as an ending.
    """
    clock = FakeClock()

    def turn(i, clock):
        clock.tick(30)                               # vad.listen's max_wait_s, roughly
        if i == 1:
            return loop.TurnResult(False, False, None)      # heard nothing this time
        return loop.TurnResult(True, True, None)

    watches = run_main(monkeypatch, ["--minutes", "3"], clock, turn=turn)

    assert loop.main() == 0
    out = capsys.readouterr().out
    assert len(watches) == 6, "the quiet turn was one turn, not the end of the run"
    assert "still listening" in out
    assert "5 turn(s) completed" in out, "five of the six were spoken; the quiet one was not"


def test_a_dead_mic_gives_up_instead_of_spinning_to_the_deadline(monkeypatch, capsys):
    """A muted mic looks exactly like a thoughtful pause, so the tolerance has to be bounded.

    SESSION_QUIET_LIMIT consecutive silences end the session with 2:00 still on the clock, which is
    what makes "nothing was heard" a result a person reads rather than three minutes of nothing.
    """
    clock = FakeClock()

    def silent(i, clock):
        clock.tick(30)
        return loop.TurnResult(False, False, None)

    watches = run_main(monkeypatch, ["--minutes", "3"], clock, turn=silent)

    assert loop.main() == 1, "no turn was spoken, so the run failed"
    assert len(watches) == config.SESSION_QUIET_LIMIT
    assert "0 turn(s) completed in 1:00." in capsys.readouterr().out


def test_a_spoken_turn_clears_the_quiet_streak(monkeypatch):
    """Two silences a minute apart are two pauses; the limit is about consecutive ones."""
    clock = FakeClock()

    def alternating(i, clock):
        clock.tick(20)
        return loop.TurnResult(True, True, None) if i % 2 else loop.TurnResult(False, False, None)

    watches = run_main(monkeypatch, ["--minutes", "3"], clock, turn=alternating)

    loop.main()

    assert len(watches) == 9, "the session ran to the deadline, not to the second silence"


def test_a_sentence_spoken_as_time_runs_out_is_answered_and_not_dropped(monkeypatch, capsys):
    """The wind-down. The user talked over the last reply; the clock is not a reason to ignore them.

    That final turn is unwatched: it is the one turn in a session with nothing after it, so holding
    the mic open would only make the process sit through max_wait_s on the way out.
    """
    clock = FakeClock()
    cap = FakeCapture()
    seen = []

    def turn(i, clock):
        clock.tick(70)
        seen.append(i)
        # The first turn overruns the one-minute deadline and comes back holding the next
        # utterance; the second is the wind-down and carries nothing further.
        return loop.TurnResult(True, True, cap if i == 0 else None)

    watches = run_main(monkeypatch, ["--minutes", "1"], clock, turn=turn)

    assert loop.main() == 0
    assert seen == [0, 1], "the carried-in sentence got its reply"
    assert watches == [True, False], "and the reply to it opened no mic afterwards"
    assert "time is up" in capsys.readouterr().out


def test_ctrl_c_ends_the_session_with_its_counts(monkeypatch, capsys):
    """The documented way to stop early (CONSENT_NOTICE says so) must not be a traceback.

    What a person needs at that moment is the same three lines the clock would have printed: how
    many turns were spoken, and where the two logs are.
    """
    clock = FakeClock()

    def interrupted(i, clock):
        clock.tick(10)
        if i == 1:
            raise KeyboardInterrupt
        return loop.TurnResult(True, True, None)

    run_main(monkeypatch, ["--minutes", "3"], clock, turn=interrupted)

    assert loop.main() == 0, "a session someone chose to end is not a failed run"
    out = capsys.readouterr().out
    assert "1 turn(s) completed in 0:20." in out
    assert "calls:" in out and "turns:" in out


def test_bare_minutes_takes_the_configured_session_length(monkeypatch):
    """What `make demo` runs. The knob is the env var, so a dev can have half a minute of it.

    The flag with no value is config.SESSION_MINUTES; the flag with a value overrides it for one
    run. Both are the same argparse action, so this pins the `const` that makes the bare form work.
    """
    clock = FakeClock()

    def turn(i, clock):
        clock.tick(20)
        return loop.TurnResult(True, True, None)

    monkeypatch.setattr(loop, "SESSION_MINUTES", 0.5)
    watches = run_main(monkeypatch, ["--minutes"], clock, turn=turn)

    loop.main()

    assert len(watches) == 2, "30 s of session, 20 s a turn — the second one still starts in time"
