"""The hybrid pipeline: local VAD and TTS, remote STT and LLM, local fallback on a bad provider.

Two kinds of assertion, and they fail for different reasons.

  the placement  every stage runs where PIPELINE says it does. Before that table existed the
                 architecture was a consequence of which row happened to be first in each tuple in
                 config.py, so reordering a table moved a stage across the network boundary and
                 nothing anywhere said so.
  the fallback   a transient provider failure runs the stage's local arm instead of ending the
                 turn — and, just as importantly, a *non*-transient one still does not. Falling back
                 on a bad key would hide a config bug behind a quietly worse transcript, which is
                 strictly worse than failing.

The controls carry the weight here. Almost every test below would pass against code that fell back
on any exception whatsoever; `test_a_bad_key_does_not_fall_back` and its neighbours are what make
the rest mean something.

No network, no mic, no weights: HTTP is faked per module the way test_rate_limit.py does it, and
local arms answer through the BACKENDS seam test_arms.py uses.
"""
import json
import types

import httpx
import pytest

from conftest import fake_state
from src import arms, cooldown, errors, loop, nlu, stt
from src.config import ARMS, FALLBACKS, LLM_ARMS, PIPELINE, STT_ARMS, TTS_ARMS, resolve
from tests.unit.test_rate_limit import FakeResponse, _capture, hosted_arm, rate_limited, serve

MODEL_STAGES = tuple(ARMS)          # stt, llm, tts, embed — vad is not an arm yet

# The stages a fallback is *meaningful* for. Not `embed`: a second encoder answers the query in a
# different vector space from the one the cached chunk vectors live in, so every cosine it produced
# would be arithmetic between unrelated bases — and a meaningless cosine is still a number between
# -1 and 1, so nothing downstream could tell. `src/retrieval.py` skips the dense half instead, which
# is a worse ranking rather than a wrong one.
FALLBACK_STAGES = tuple(s for s in MODEL_STAGES if s != "embed")


# --- the placement ----------------------------------------------------------------------------

@pytest.mark.parametrize("stage", MODEL_STAGES)
def test_the_default_arm_runs_where_the_pipeline_says(stage):
    """The diagram, as a test. A table reorder in config.py now has to be a deliberate one."""
    default = ARMS[stage][0]
    assert PIPELINE[stage] == ("local" if default.local else "remote"), (
        f"{stage} defaults to {default.id}, which is "
        f"{'local' if default.local else 'remote'}, but PIPELINE says {PIPELINE[stage]}")


def test_the_pipeline_covers_every_stage_including_vad():
    """vad is not an arm, but it is a stage, and leaving it out of the table would hide a move."""
    assert set(PIPELINE) == {"vad"} | set(MODEL_STAGES)


def test_the_expensive_stages_are_remote_and_the_per_frame_one_is_not():
    """Spelling out the intent, so a future edit to PIPELINE reads as the change it is."""
    assert PIPELINE["stt"] == "remote" and PIPELINE["llm"] == "remote"
    assert PIPELINE["vad"] == "local" and PIPELINE["tts"] == "local"


# --- the fallback table -----------------------------------------------------------------------

@pytest.mark.parametrize("stage", FALLBACK_STAGES)
def test_every_stage_has_a_local_fallback(stage):
    """A remote fallback would fail for the same reason the primary just did."""
    fb = resolve(stage, FALLBACKS[stage])
    assert fb.local, f"{stage} falls back to {fb.id}, which is not local"
    assert fb in ARMS[stage], "a fallback must be an arm on its own stage"


def test_the_encoder_stage_deliberately_has_no_fallback():
    """Stated as a test because "we forgot" and "we decided" look identical in a config table.

    See FALLBACK_STAGES above: a fallback encoder would not degrade the answer, it would make the
    comparison meaningless.
    """
    assert "embed" not in FALLBACKS
    assert arms.fallback_for("embed", resolve("embed")) is None


def test_the_llm_fallback_is_the_only_local_llm_arm():
    """Guards the reason the ollama arm was added at all — if a second one appears, choose again."""
    local_llms = [a for a in LLM_ARMS if a.local]
    assert [a.alias for a in local_llms] == [FALLBACKS["llm"]]


def test_an_arm_that_is_already_the_fallback_has_nowhere_to_go():
    """Otherwise a failing local arm would be asked to rescue itself, doubling every local failure."""
    fb = resolve("stt", FALLBACKS["stt"])
    assert arms.fallback_for("stt", fb) is None
    assert arms.fallback_for("stt", STT_ARMS[0]) is fb


# --- what falls back --------------------------------------------------------------------------

TRANSIENT = [
    pytest.param(lambda: rate_limited(), id="429"),
    pytest.param(lambda: FakeResponse(503, {"error": "upstream unavailable"}), id="503"),
    pytest.param(lambda: FakeResponse(500, {"error": "internal"}), id="500"),
]


@pytest.mark.parametrize("response", TRANSIENT)
def test_a_transient_provider_failure_runs_the_local_arm(response, monkeypatch):
    serve(monkeypatch, stt, response())
    local_stt(monkeypatch, "when was I paid")

    assert arms.stt([0.0] * 16, hosted_arm("stt").id, turn_id="t") == "when was I paid"


@pytest.mark.parametrize("exc", [
    pytest.param(httpx.TimeoutException("timed out"), id="timeout"),
    pytest.param(httpx.ConnectError("no route to host"), id="connect-error"),
])
def test_an_unreachable_provider_runs_the_local_arm(exc, monkeypatch):
    """The offline case. A dead network is the failure most likely to be seen in a live demo."""
    raising_post(monkeypatch, stt, exc)
    local_stt(monkeypatch, "when was I paid")

    assert arms.stt([0.0] * 16, hosted_arm("stt").id, turn_id="t") == "when was I paid"


def test_the_llm_falls_back_too(monkeypatch):
    """The stage with no local arm before this change, so it is the one worth checking end to end."""
    serve(monkeypatch, nlu, rate_limited())
    fb = resolve("llm", FALLBACKS["llm"])
    monkeypatch.setitem(arms._MODULES["llm"].BACKENDS, fb.backend,
                        lambda arm, msgs, rec: "On the fourth.")

    assert nlu.reply("when was I paid", "t") == "On the fourth."


# --- what does not ----------------------------------------------------------------------------

def test_a_bad_key_does_not_fall_back(monkeypatch, calls_log):
    """The control that makes every test above mean something.

    A 401 is a config bug. Falling back would return a plausible transcript from a worse model and
    leave the broken credential to be discovered days later by someone reading a WER table.
    """
    serve(monkeypatch, stt, FakeResponse(401, {"error": {"message": "invalid api key"}}))
    local_stt(monkeypatch, "should never be reached")

    with pytest.raises(httpx.HTTPStatusError):
        arms.stt([0.0] * 16, hosted_arm("stt").id, turn_id="t")

    assert len(calls_log.read_text(encoding="utf-8").splitlines()) == 1, "no second attempt"


def test_a_missing_credential_does_not_fall_back(monkeypatch):
    """Zero spend is a hard constraint, and an unset key is a STOP-and-ask, not a routing decision.

    Running locally instead would be a reasonable-looking way to never notice that the free tier was
    never configured — and the whole point of the remote stages is that they are better.
    """
    arm = hosted_arm("stt")
    monkeypatch.delenv(arm.key_env, raising=False)
    local_stt(monkeypatch, "should never be reached")

    with pytest.raises(RuntimeError, match=arm.key_env):
        arms.stt([0.0] * 16, arm.id, turn_id="t")


def test_an_empty_reply_from_a_reasoning_arm_does_not_fall_back(monkeypatch):
    """nlu raises this when an arm spends its whole budget thinking. It is a misconfigured arm —
    it would raise again next turn, so routing around it hides a permanent problem as a flaky one."""
    serve(monkeypatch, nlu, FakeResponse(200, {
        "choices": [{"message": {"content": ""}, "finish_reason": "length"}],
        "usage": {"prompt_tokens": 40, "completion_tokens": 120}}))
    monkeypatch.setitem(arms._MODULES["llm"].BACKENDS,
                        resolve("llm", FALLBACKS["llm"]).backend,
                        lambda arm, msgs, rec: "should never be reached")

    with pytest.raises(RuntimeError, match="empty reply"):
        nlu.reply("when was I paid", "t")


@pytest.mark.parametrize("exc,transient", [
    (errors.RateLimited("x", status_code=429, retry_after=1, arm_id="a"), True),
    (errors.ProviderError("x", status_code=503, arm_id="a"), True),
    (httpx.TimeoutException("x"), True),
    (httpx.ConnectError("x"), True),
    (httpx.HTTPStatusError("x", request=None, response=None), False),
    (RuntimeError("GROQ_API_KEY is not set"), False),
    (ValueError("bad frame size"), False),
])
def test_the_transient_rule_is_written_in_one_place(exc, transient):
    assert errors.is_transient(exc) is transient


# --- the log and the turn record ----------------------------------------------------------------

def test_both_attempts_reach_the_call_log(monkeypatch, calls_log):
    """A fallback must cost two lines, not one. One line would read as a turn that just ran local."""
    remote = hosted_arm("stt")
    serve(monkeypatch, stt, rate_limited())
    local_stt(monkeypatch, "when was I paid")

    arms.stt([0.0] * 16, remote.id, turn_id="t-two")
    refused, covered = [json.loads(x) for x in calls_log.read_text(encoding="utf-8").splitlines()]

    assert (refused["model_id"], refused["ok"]) == (remote.repo_id, False)
    assert (covered["model_id"], covered["ok"]) == (resolve("stt", FALLBACKS["stt"]).repo_id, True)
    assert covered["fallback_for"] == remote.id
    assert refused["cost_usd"] == covered["cost_usd"] == 0.0


def test_the_turn_record_names_the_arm_that_actually_ran(monkeypatch, turns_log):
    """`turn.arms()` stamps the *selected* arm before the turn. After a fallback that is a lie, and
    it is the field VOX-013's per-turn comparison reads."""
    remote = hosted_arm("stt")
    fb = resolve("stt", FALLBACKS["stt"])
    serve(monkeypatch, stt, rate_limited())
    local_stt(monkeypatch, "when was I paid")
    run_turn(monkeypatch, stt_arm=remote)

    rec = json.loads(turns_log.read_text(encoding="utf-8").strip())

    assert rec["stt_model"] == fb.id, "the record must name the arm that produced the transcript"
    assert rec["stt_fallback_from"] == remote.id
    assert rec["fell_back"] == ["stt"]
    assert rec["llm_model"] == LLM_ARMS[0].id, "an unaffected stage keeps its selected arm"
    assert "llm_fallback_from" not in rec


def test_the_failed_attempt_s_latency_is_broken_out(monkeypatch, turns_log, calls_log):
    """t_stt_ms now spans the dead round-trip plus the local call. Honest about what the user
    waited for, but without the split the provider's timeout is charged to local inference."""
    serve(monkeypatch, stt, rate_limited())
    local_stt(monkeypatch, "when was I paid")
    run_turn(monkeypatch, stt_arm=hosted_arm("stt"))

    rec = json.loads(turns_log.read_text(encoding="utf-8").strip())
    refused = json.loads(calls_log.read_text(encoding="utf-8").splitlines()[0])

    assert rec["stt_failed_ms"] == refused["latency_ms"], "one measurement, not two clocks"
    assert rec["t_stt_ms"] >= rec["stt_failed_ms"], "the stage contains the attempt it made"


def test_a_clean_turn_carries_no_fallback_marker(monkeypatch, turns_log):
    """Without this, every test above passes against a loop that marks every turn as fallen back."""
    serve(monkeypatch, stt, FakeResponse(200, {"text": "when was I paid"}))
    run_turn(monkeypatch, stt_arm=hosted_arm("stt"))

    rec = json.loads(turns_log.read_text(encoding="utf-8").strip())

    assert "fell_back" not in rec
    assert "stt_fallback_from" not in rec
    assert rec["stt_model"] == hosted_arm("stt").id


# --- the sample rate --------------------------------------------------------------------------

def test_a_tts_fallback_returns_the_rate_the_arm_that_answered_produced(monkeypatch):
    """Kokoro is 24 kHz and SpeechT5 16 kHz. `arms.tts` used to read the rate off the arm it was
    asked for, which after a fallback is not the arm that made the samples — every fallback reply
    would have played 1.5x fast, which sounds like a bug in synthesis rather than in routing."""
    primary, fb = TTS_ARMS[0], resolve("tts", FALLBACKS["tts"])
    assert primary.extra["sample_rate"] != fb.extra["sample_rate"], "the arms must differ here"

    def boom(arm, text, rec):
        raise errors.ProviderError("vocoder gone", status_code=503, arm_id=arm.id)

    monkeypatch.setitem(arms._MODULES["tts"].BACKENDS, primary.backend, boom)
    monkeypatch.setitem(arms._MODULES["tts"].BACKENDS, fb.backend, lambda a, t, rec: [0.0] * 8)

    speech = arms.tts("hello", primary.id, turn_id="t")

    assert speech.sample_rate == fb.extra["sample_rate"]


def test_a_fallback_that_also_fails_reports_the_original_failure(monkeypatch):
    """The free tier refusing is the fact worth reporting; the local arm failing behind it is a
    second symptom of the same turn, and burying the first under it loses the cause."""
    serve(monkeypatch, stt, rate_limited())

    def boom(arm, payload, rec):
        raise RuntimeError("no weights on disk")

    monkeypatch.setitem(arms._MODULES["stt"].BACKENDS,
                        resolve("stt", FALLBACKS["stt"]).backend, boom)

    with pytest.raises(errors.RateLimited) as e:
        arms.stt([0.0] * 16, hosted_arm("stt").id, turn_id="t")

    assert isinstance(e.value.__cause__, RuntimeError), "the local failure is chained, not lost"


# --- the cooldown -----------------------------------------------------------------------------

def test_a_rate_limited_arm_is_not_called_again_inside_its_window(monkeypatch):
    """The reason cooldown exists: a doomed round-trip per turn is latency the user pays for
    nothing, and asking a free tier to refuse repeatedly is how it stops answering at all."""
    remote = hosted_arm("stt")
    posts = serve(monkeypatch, stt, rate_limited())
    local_stt(monkeypatch, "when was I paid")

    for _ in range(4):
        assert arms.stt([0.0] * 16, remote.id, turn_id="t") == "when was I paid"

    assert len(posts) == 1, "one refusal, then the arm is parked"
    assert cooldown.blocked(remote) > 0


def test_the_arm_comes_back_when_the_window_expires(monkeypatch):
    """A cooldown that never lifts is just a broken arm. This is the half that makes it temporary."""
    remote = hosted_arm("stt")
    posts = serve(monkeypatch, stt, rate_limited())
    local_stt(monkeypatch, "local answer")

    arms.stt([0.0] * 16, remote.id, turn_id="t")          # 429 -> parked for 12.5 s
    assert cooldown.blocked(remote) > 0

    cooldown.clear()                                       # stand-in for the window elapsing
    serve(monkeypatch, stt, FakeResponse(200, {"text": "remote answer"}))

    assert arms.stt([0.0] * 16, remote.id, turn_id="t") == "remote answer"
    assert len(posts) == 1, "the second call went to the new response, not the old one"


def test_a_cooldown_without_a_retry_after_still_parks_the_arm(monkeypatch):
    """Providers are not required to send Retry-After. Absent must not read as "retry immediately"."""
    remote = hosted_arm("stt")
    posts = serve(monkeypatch, stt, rate_limited(retry_after=None))
    local_stt(monkeypatch, "when was I paid")

    arms.stt([0.0] * 16, remote.id, turn_id="t")
    arms.stt([0.0] * 16, remote.id, turn_id="t")

    assert len(posts) == 1
    assert cooldown.blocked(remote) > 0


def test_a_cooldown_is_skipped_when_the_stage_has_nowhere_to_fall_back(monkeypatch):
    """Parking the only arm a stage has would turn a 12 s pace limit into a dead session."""
    monkeypatch.delitem(arms.FALLBACKS, "stt")
    remote = hosted_arm("stt")
    posts = serve(monkeypatch, stt, rate_limited())

    for _ in range(2):
        with pytest.raises(errors.RateLimited):
            arms.stt([0.0] * 16, remote.id, turn_id="t")

    assert len(posts) == 2, "with no fallback the arm is all there is — it must still be tried"


# --- helpers ----------------------------------------------------------------------------------

def local_stt(monkeypatch, text):
    """Make the STT stage's local fallback answer, without whisper weights on disk."""
    monkeypatch.setitem(arms._MODULES["stt"].BACKENDS,
                        resolve("stt", FALLBACKS["stt"]).backend,
                        lambda arm, payload, rec: text)


def raising_post(monkeypatch, module, exc):
    """Point one adapter's httpx.post at a raise instead of a response — the offline case.

    Replaces the module's `httpx` reference rather than setting `post` on the real httpx module,
    for the reason `serve()` gives: `stt.httpx` and `nlu.httpx` are the same object, so patching it
    in place leaks into whichever adapter the test is not about.
    """
    def post(url, **kw):
        raise exc

    monkeypatch.setattr(module, "httpx", types.SimpleNamespace(
        post=post, Headers=httpx.Headers, HTTPStatusError=httpx.HTTPStatusError))


def run_turn(monkeypatch, stt_arm):
    """One full loop.one_turn with everything but STT stubbed out. -> what one_turn returned."""
    monkeypatch.setattr(loop.vad, "listen", lambda *a, **kw: _capture())
    # Since the VOX-019/020 merge the un-retrieved path is the structured extractor, so this
    # is the call that writes the spoken reply on a turn with no index behind it.
    monkeypatch.setattr(loop.state, "build", lambda *a, **kw: fake_state("On the fourth."))
    monkeypatch.setattr(loop.audio, "play", lambda audio, **kw: None)
    monkeypatch.setitem(arms._MODULES["tts"].BACKENDS, TTS_ARMS[0].backend,
                        lambda arm, text, rec: [0.0] * 240)
    return loop.one_turn({"stt": stt_arm, "llm": LLM_ARMS[0], "tts": TTS_ARMS[0]})
