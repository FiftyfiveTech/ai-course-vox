"""VOX-035: on open speakers the mic hears the reply, and the reply must not become the next turn.

The bug this guards is not the cosmetic self-interrupt. It is the carry-forward: `speak_and_watch`
holds the mic open, the capture becomes `pending`, and `one_turn` with a `pending` never opens the
mic at all — it transcribes VOX's own reply and answers it, and the loop feeds itself until the
session clock runs out. So the guard is tested at each of the three levels it has to work at:

  echo.py     the envelope test separates a reply that came back through a room from a person
              talking over it, and refuses to answer at all below MIN_DECISION_MS.
  drive()     a rejected capture does not cut the reply *and* does not survive to be endpointed.
              Rejecting without rearming leaves the echo to be carried forward, which is the half
              of the bug that actually feeds itself.
  one_turn()  a transcript that is the reply just spoken, on carried-in audio, drops the turn.

The room here is synthetic — a delay, an attenuation, a one-pole lowpass, two reflections and noise
— and the voices are generated. That is enough to prove the mechanism and it is deliberately not
enough to tune `corr_threshold`: scripts/measure_echo_guard.py is where the numbers come from, and
its docstring says why a real machine needs a real recording. The fixtures here are kept local
rather than imported from that script, the way test_barge_in.py and test_silence.py each state
their own audio.
"""
import types

import numpy as np
import pytest
import torch

from conftest import fake_state
from src import echo, loop, vad
from src.config import LLM_ARMS, SAMPLE_RATE, STT_ARMS, TTS_ARMS, VAD_FRAME, VAD_SILENCE_MS
from src.vad import DONE, MS_PER_FRAME

DEFAULTS = {"stt": STT_ARMS[0], "llm": LLM_ARMS[0], "tts": TTS_ARMS[0]}
REPLY_RATE = 24_000                # Kokoro's: the rate change the real path crosses is crossed here


# --- audio a test can state -----------------------------------------------------------------------

def speechlike(seconds, rate, seed=0):
    """Audio with phone-scale structure: 40-120 ms segments, own level and spectrum, real pauses.

    Not a slow amplitude modulation. An envelope that only moves at a syllable rate is very nearly a
    straight line over any window shorter than a syllable, and straight lines correlate with each
    other — which made an earlier version of this file assert that unrelated speech scores 0.99.
    """
    rng = np.random.default_rng(seed)
    out, total = [], 0
    while total < int(seconds * rate):
        n = int(rng.uniform(0.04, 0.12) * rate)
        total += n
        if rng.random() < 0.18:
            out.append(np.zeros(n, dtype=np.float32))
            continue
        t = np.arange(n) / rate
        f0, harmonics, tilt = rng.uniform(90, 220), int(rng.integers(3, 12)), rng.uniform(0.3, 2.5)
        sig = sum((h ** -tilt) * np.sin(2 * np.pi * f0 * h * t + rng.uniform(0, 6.3))
                  for h in range(1, harmonics + 1))
        if rng.random() < 0.3:
            sig = sig * 0.3 + rng.normal(0, 0.5, n)              # a fricative
        shape = np.hanning(n) if rng.random() < 0.5 else np.ones(n)
        out.append((sig * shape * rng.uniform(0.2, 1.0)).astype(np.float32))
    return np.concatenate(out)[:int(seconds * rate)]


def through_a_room(x, rate, delay_ms=180.0, gain=0.05, seed=1):
    """What the mic hears when the speaker plays `x`: delayed, quiet, dulled, reflected, noisy.

    `delay_ms` is the demo machine's measured MME output-buffer latency (0.182 s) rounded — the
    floor, with the flight time across a desk on top of it. `gain` 0.05 is about 26 dB down.
    """
    rng = np.random.default_rng(seed)
    delayed = np.concatenate([np.zeros(int(delay_ms / 1000 * rate), dtype=np.float32), x])
    out, acc = np.empty_like(delayed), 0.0
    for i, v in enumerate(delayed):
        acc = 0.7 * acc + 0.3 * float(v)                         # the cone and the air
        out[i] = acc
    reflections = np.zeros_like(out)
    for lag, g in ((int(0.013 * rate), 0.4), (int(0.029 * rate), 0.25)):
        reflections[lag:] += out[:-lag] * g
    return ((out + reflections) * gain + rng.normal(0, 2e-4, len(out))).astype(np.float32)


def resample_to(x, src_rate, dst_rate):
    n = int(len(x) * dst_rate / src_rate)
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)


def a_room(seed=0, window_ms=1200, ref_s=2.0):
    """-> (mic audio that is echo, the reference that produced it, an unrelated voice).

    `window_ms` defaults comfortably above `echo.MIN_DECISION_MS`, because that is what the live
    guard is handed: `speak_and_watch` holds the barge decision open until it has that much, and the
    endpoint check gets the whole utterance.
    """
    reply = speechlike(3.0, REPLY_RATE, seed=seed)
    ref = reply[:int(ref_s * REPLY_RATE)]                # what the speaker has played by "now"
    mic = resample_to(through_a_room(reply, REPLY_RATE), REPLY_RATE, SAMPLE_RATE)
    end = int(ref_s * SAMPLE_RATE)                       # the mic's "now" is the same instant
    heard = mic[end - int(window_ms / 1000 * SAMPLE_RATE):end]
    person = speechlike(window_ms / 1000, SAMPLE_RATE, seed=10_000 + seed)
    return heard, ref, person


# --- the envelope test ------------------------------------------------------------------------------

def test_the_reply_coming_back_through_a_room_correlates_with_itself():
    """Across the 24 kHz -> 16 kHz rate change, which is why this is envelopes and not waveforms."""
    heard, ref, _ = a_room()

    r, delay_ms = echo.best_correlation(heard, ref, SAMPLE_RATE, REPLY_RATE)

    assert r > 0.75, f"synthetic echo should be unmistakable, got r={r}"
    assert 120 <= delay_ms <= 260, f"the 180 ms delay should be found, got {delay_ms}ms"


def test_the_lag_is_searched_back_from_the_end_of_the_reference():
    """The bug this pins cost a day: searched forwards, the window is the *oldest* stretch of the
    reference — the one place the echo cannot be — and a 180 ms echo was reported at 350 ms."""
    delays = [echo.best_correlation(*a_room(seed=s)[:2], SAMPLE_RATE, REPLY_RATE)[1]
              for s in range(5)]

    assert all(120 <= d <= 260 for d in delays), f"delays drifted off the true 180 ms: {delays}"


def test_a_person_talking_over_the_reply_does_not(caplog):
    """The false positive that would cost a real utterance — the one failure worse than the bug.

    Checked over several rooms, not one: a single seed passing this is chance, and the measurement
    that set the threshold is a distribution, not a point.
    """
    scores = []
    for seed in range(8):
        _, ref, person = a_room(seed=seed)
        scores.append(echo.best_correlation(person, ref, SAMPLE_RATE, REPLY_RATE)[0])

    assert max(scores) < 0.75, f"an unrelated voice must not read as echo, got {max(scores):.3f}"


def test_below_the_decision_minimum_the_guard_refuses_to_answer():
    """400 ms threw away 18.3% of real speech, which is why MIN_DECISION_MS is 700.

    Refusing is not a gap in the guard: `speak_and_watch` holds the barge decision open until it has
    this much, and the endpoint check is handed the whole utterance.
    """
    heard, ref, _ = a_room(window_ms=400)

    assert echo.best_correlation(heard, ref, SAMPLE_RATE, REPLY_RATE) == (0.0, 0.0)


def test_a_window_of_exactly_the_minimum_is_answered():
    """700 ms is 87 whole 8 ms hops and 700/8 is 87.5 — compared in milliseconds, the minimum-length
    window refused itself, and the guard silently never fired on a short reply."""
    heard, ref, _ = a_room(window_ms=700)

    r, _ = echo.best_correlation(heard, ref, SAMPLE_RATE, REPLY_RATE)

    assert r > 0.0, "a window of exactly MIN_DECISION_MS must get an answer"


def test_silence_correlates_with_nothing():
    _, ref, _ = a_room()
    quiet = np.zeros(int(1.2 * SAMPLE_RATE), dtype=np.float32)

    assert echo.best_correlation(quiet, ref, SAMPLE_RATE, REPLY_RATE)[0] == 0.0


def test_a_reference_shorter_than_the_capture_is_not_echo():
    """The reply had barely started. Nothing to correlate against is not evidence of a person."""
    heard, _, _ = a_room()

    assert echo.best_correlation(heard, np.zeros(1000, dtype=np.float32),
                                 SAMPLE_RATE, REPLY_RATE)[0] == 0.0


def test_the_separation_is_printed_not_asserted(capsys):
    """The number the guard lives or dies on, in the output, per the repo's measured-gate rule."""
    rows = []
    for seed in range(8):
        heard, ref, person = a_room(seed=seed)
        r_echo, delay = echo.best_correlation(heard, ref, SAMPLE_RATE, REPLY_RATE)
        r_person, _ = echo.best_correlation(person, ref, SAMPLE_RATE, REPLY_RATE)
        rows.append((r_echo, r_person, delay))

    echoes = [r for r, _, _ in rows]
    people = [r for _, r, _ in rows]
    with capsys.disabled():
        print(f"\n  self-echo separation (SYNTHETIC room, 24k reply -> 16k mic, 8 rooms):"
              f"\n    echo   min={min(echoes):.3f} mean={np.mean(echoes):.3f} "
              f"at ~{np.median([d for _, _, d in rows]):.0f}ms"
              f"\n    person max={max(people):.3f} mean={np.mean(people):.3f}"
              f"\n    margin {min(echoes) - max(people):+.3f}   "
              f"(config.yaml corr_threshold, full table: scripts/measure_echo_guard.py)")

    assert min(echoes) > max(people), "the two populations must not overlap on this fixture"


# --- the transcript test ------------------------------------------------------------------------------

SPOKEN = "You are entitled to twenty six weeks of paid maternity leave under the policy."


def test_a_fragment_of_the_reply_is_recognised():
    assert echo.echoes_reply("twenty six weeks of paid maternity leave", SPOKEN)


def test_a_real_follow_up_about_the_same_topic_is_not():
    assert not echo.echoes_reply("can I split that leave across two years", SPOKEN)


def test_a_short_confirmation_is_never_judged():
    """VOX-020's gate would lose every 'yes, go ahead' to this function otherwise."""
    assert not echo.echoes_reply("yes go ahead", "Shall I go ahead and book that meeting?")


def test_no_reply_to_compare_against_is_not_echo():
    """The first turn of a session, and every turn of a run with the guard off."""
    assert not echo.echoes_reply("twenty six weeks of paid maternity leave", None)


# --- the endpointer ------------------------------------------------------------------------------------

class FakeVad:
    """silero's interface, decided by the audio: any nonzero sample is speech."""

    def __init__(self):
        self.resets = 0

    def reset_states(self):
        self.resets += 1

    def __call__(self, frame, sample_rate):
        return torch.tensor(1.0 if float(frame.abs().max()) > 0.0 else 0.0)


def frames(*sections):
    """(level, ms) pairs -> a list of exact VAD_FRAME frames at those levels."""
    out = []
    for level, ms in sections:
        out += [np.full(VAD_FRAME, level, dtype=np.float32) for _ in range(round(ms / MS_PER_FRAME))]
    return out


def test_a_rejected_capture_neither_fires_the_hook_nor_survives():
    """Both halves. Rejecting without rearming still hands the echo to the next turn as `pending`,
    and that is the half of the bug that feeds itself."""
    fired, asked = [], []

    def reject(segment):
        asked.append(len(segment))
        return True                                  # everything is echo, in this room

    cap, state = vad.endpoint_frames(
        frames((0.5, 900), (0.0, VAD_SILENCE_MS + 200)),
        model=FakeVad(), on_speech=lambda t: fired.append(t), reject=reject)

    assert asked, "the guard was never asked"
    assert fired == [], "a rejected capture must not cut the reply"
    assert cap is None, "and must not be carried into the next turn either"
    assert state != DONE


def test_the_guard_is_asked_once_per_utterance_and_not_again_at_endpoint():
    """Once, at the confirmation point, and deliberately not a second time when the turn ends.

    A guard that compares mic audio against what the speaker has played needs both to mean the same
    instant. At the confirmation point they do. At DONE they do not: VAD_SILENCE_MS of hangover has
    passed and the reply has played on, so the echo sits further back in the reference than the
    delay search reaches — the check would not fail loudly, it would quietly answer about the wrong
    stretch of reply. What gets past this is caught after STT by `echoes_reply`, which needs no
    alignment at all.
    """
    seen = []

    def reject(segment):
        seen.append(len(segment))
        return False                                 # let it through, and count the askings

    cap, state = vad.endpoint_frames(
        frames((0.5, 900), (0.0, VAD_SILENCE_MS + 200)),
        model=FakeVad(), reject=reject)

    assert len(seen) == 1, f"asked {len(seen)} times; the endpoint asking is misaligned by design"
    assert state == DONE and cap is not None


def test_a_rejected_burst_leaves_the_real_utterance_behind_it_intact():
    """The guard must not deafen the turn: what it drops is the echo, not the person after it."""
    calls = []

    def reject(segment):
        calls.append(1)
        return len(calls) == 1                       # first burst echo, everything after a person

    cap, state = vad.endpoint_frames(
        frames((0.5, 900), (0.0, 200), (0.7, 900), (0.0, VAD_SILENCE_MS + 200)),
        model=FakeVad(), reject=reject)

    assert state == DONE and cap is not None
    assert float(np.abs(cap.segment).max()) == pytest.approx(0.7), \
        "the capture is the person's utterance, not the echo before it"


def test_with_no_guard_nothing_changes():
    """The default path, unchanged — every VOX-011 number on the board was measured on it."""
    fired = []

    cap, state = vad.endpoint_frames(
        frames((0.5, 600), (0.0, VAD_SILENCE_MS + 200)),
        model=FakeVad(), on_speech=lambda t: fired.append(t))

    assert len(fired) == 1 and state == DONE and cap is not None


# --- the turn -------------------------------------------------------------------------------------------

class FakeCapture:
    """A vad.Capture as the previous turn's `speak_and_watch` would have handed it forward."""

    def __init__(self, seconds=1.0):
        self.segment = np.zeros(int(seconds * SAMPLE_RATE), dtype=np.float32)
        self.speech_end_t = self.endpointed_t = 0.0
        self.spoken_s = seconds
        self.infer_ms = 1.0
        self.t_vad_ms = 1.0

    def __len__(self):
        return len(self.segment)


def stub_turn(monkeypatch, transcript, reply="a reply"):
    """Everything but the decision under test. The mic is wired to fail: a carried-in turn that
    opens it has stopped being the path this file is about."""
    monkeypatch.setattr(loop.arms, "stt", lambda *a, **kw: transcript)
    monkeypatch.setattr(loop.state, "build", lambda *a, **kw: fake_state(reply))
    monkeypatch.setattr(loop.arms, "tts", lambda *a, **kw: types.SimpleNamespace(
        audio=np.zeros(2400, dtype=np.float32), sample_rate=REPLY_RATE))
    monkeypatch.setattr(loop.audio, "play", lambda *a, **kw: None)

    def no_mic(*a, **kw):
        raise AssertionError("a turn with carried-in audio must not open the mic")

    monkeypatch.setattr(loop.vad, "listen", no_mic)


def test_a_carried_in_echo_transcript_drops_the_turn(monkeypatch):
    monkeypatch.setattr(loop, "ECHO_GUARD", True)
    stub_turn(monkeypatch, "twenty six weeks of paid maternity leave under the policy")

    result = loop.one_turn(DEFAULTS, pending=FakeCapture(), last_reply=SPOKEN)

    assert result.spoken is False, "nothing was said, so nothing is counted"
    assert result.keep_going is True, "the session continues — the user has not gone away"
    assert result.pending is None, "and the echo is not handed on again"


def test_a_real_carried_in_utterance_is_still_answered(monkeypatch):
    """The control. Barge-in works by carrying audio forward, so the guard must not break it."""
    monkeypatch.setattr(loop, "ECHO_GUARD", True)
    stub_turn(monkeypatch, "what about paternity leave", reply="Two weeks.")

    result = loop.one_turn(DEFAULTS, pending=FakeCapture(), last_reply=SPOKEN)

    assert result.spoken is True and result.reply == "Two weeks."


def test_a_turn_that_opened_its_own_mic_is_never_judged(monkeypatch):
    """Only carried-in audio can be an echo. A turn that opened the mic is listening after the
    speaker went quiet, and a user who repeats VOX's words back at it meant to."""
    monkeypatch.setattr(loop, "ECHO_GUARD", True)
    stub_turn(monkeypatch, "twenty six weeks of paid maternity leave under the policy")
    monkeypatch.setattr(loop.vad, "listen", lambda *a, **kw: FakeCapture())

    result = loop.one_turn(DEFAULTS, last_reply=SPOKEN)

    assert result.spoken is True


def test_with_the_guard_off_the_echo_is_answered_as_before(monkeypatch):
    """What the colleague is seeing today, pinned — so the fix is visibly the thing that changed."""
    monkeypatch.setattr(loop, "ECHO_GUARD", False)
    stub_turn(monkeypatch, "twenty six weeks of paid maternity leave under the policy")

    result = loop.one_turn(DEFAULTS, pending=FakeCapture(), last_reply=SPOKEN)

    assert result.spoken is True, "the loop answers its own reply — this is the bug, not a pass"


# --- the whole chain, with real signal ---------------------------------------------------------------

def test_the_runaway_loop_does_not_start(monkeypatch):
    """The end-to-end claim, with nothing stubbed but the output device and silero.

    Every other test in this file replaces one piece: a `reject` that returns a scripted answer, a
    capture with no audio in it. This one wires the real ones together — `Playback.reference` over
    what a device has actually pulled, `echo.is_self_echo` over a real correlation, `loop.echo_reject`
    holding them together, `vad.drive` acting on the answer — and feeds it the echo of the reply that
    is playing. If any of the four is wrong about where "now" is, the correlation lands on the wrong
    stretch of reply and this fails.
    """
    from test_barge_in import fake_speaker                  # same directory; states the fake device

    monkeypatch.setattr(loop, "ECHO_GUARD", True)
    made = fake_speaker(monkeypatch)

    reply = speechlike(3.0, REPLY_RATE, seed=7)
    playback = loop.audio.play(reply, sample_rate=REPLY_RATE, block=False)
    played_s = 2.0
    made[0].pull(int(played_s * REPLY_RATE))                # the device has reached 2.0 s of reply

    # What the mic is hearing at that same instant: the reply, through a room.
    mic = resample_to(through_a_room(reply, REPLY_RATE), REPLY_RATE, SAMPLE_RATE)
    now = int(played_s * SAMPLE_RATE)
    # Just over MIN_DECISION_MS, ending at the instant the device has reached. Live, the mic and the
    # speaker advance together and that alignment is free; in a test it has to be arranged, and
    # arranging it wrong is what a longer window would do — the guard would be asked at 700 ms about
    # a mic that had already run half a second past the reference.
    heard = mic[now - int(0.75 * SAMPLE_RATE):now]

    monkeypatch.setattr(vad, "_vad_model", FakeVad)
    cap, state = vad.endpoint_frames(
        vad.frames_from(heard) + frames((0.0, VAD_SILENCE_MS + 200)),
        model=FakeVad(), confirm_ms=echo.MIN_DECISION_MS,
        reject=loop.echo_reject(playback, REPLY_RATE, None))

    assert cap is None, "the reply came back through the mic and was accepted as the user's turn"
    assert state != DONE


def test_a_person_over_the_same_reply_is_still_heard(monkeypatch):
    """The control for the test above, and the one that matters more: on the same playback, at the
    same instant, a real voice must survive the guard and reach the next turn."""
    from test_barge_in import fake_speaker

    monkeypatch.setattr(loop, "ECHO_GUARD", True)
    made = fake_speaker(monkeypatch)

    reply = speechlike(3.0, REPLY_RATE, seed=7)
    playback = loop.audio.play(reply, sample_rate=REPLY_RATE, block=False)
    made[0].pull(int(2.0 * REPLY_RATE))

    person = speechlike(0.75, SAMPLE_RATE, seed=4242)

    monkeypatch.setattr(vad, "_vad_model", FakeVad)
    cap, state = vad.endpoint_frames(
        vad.frames_from(person) + frames((0.0, VAD_SILENCE_MS + 200)),
        model=FakeVad(), confirm_ms=echo.MIN_DECISION_MS,
        reject=loop.echo_reject(playback, REPLY_RATE, None))

    assert state == DONE and cap is not None, "the guard ate a real utterance"
