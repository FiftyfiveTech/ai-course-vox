"""VOX-007, failure mode 2: a 429 is surfaced, not swallowed.

Every arm here runs on a free tier, so 429 is the expected failure and the one most easily mistaken
for something else. Before `errors.check`, both hosted adapters called `raise_for_status()`, which
raises the same `httpx.HTTPStatusError` for 429 and for 401 — the number lived only inside httpx's
message string, so nothing could branch on it and the call record did not carry it. "Rate limited,
wait" and "your key is wrong" then looked identical in runs/calls.jsonl.

Surfaced means three things, and each has a test: it raises rather than returning a plausible
blank, it reaches runs/calls.jsonl with `ok: false` before it propagates, and it is *distinguishable*
from every other refusal. The last one is why the 401 test is here — without it, the others would
pass against code that called every failure a rate limit.

No network: `httpx.post` is replaced per module. Real `httpx.Headers` is used for the response so
the case-insensitive Retry-After lookup is exercised rather than assumed.
"""
import json
import types

import httpx
import pytest

from conftest import fake_state
from src import arms, errors, loop, nlu, stt
from src.config import LLM_ARMS, STT_ARMS, TTS_ARMS
from src.errors import RateLimited

# `serve` patches httpx per module, so the local fallback arms below are patched at the BACKENDS
# seam instead — they never speak HTTP, and loading real whisper weights in a unit test would make
# this suite need a network and a gigabyte of disk.

RETRY_AFTER = 12.5


class FakeResponse:
    """Just the four things `errors.check` and the two adapters read off a response."""

    def __init__(self, status_code, json_body=None, headers=None):
        self.status_code = status_code
        self.headers = httpx.Headers(headers or {})
        self._json = json_body or {}

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"Client error '{self.status_code}' for url 'https://example.invalid'",
                request=httpx.Request("POST", "https://example.invalid"), response=None,
            )
        return self


TRANSCRIPT_OK = FakeResponse(200, {"text": "  when was I paid  "})
CHAT_OK = FakeResponse(200, {"choices": [{"message": {"content": "On the fourth."},
                                          "finish_reason": "stop"}],
                             "usage": {"prompt_tokens": 40, "completion_tokens": 5}})


def rate_limited(retry_after=RETRY_AFTER):
    headers = {"Retry-After": str(retry_after)} if retry_after is not None else {}
    return FakeResponse(429, {"error": {"message": "rate limit reached"}}, headers)


def serve(monkeypatch, module, response):
    """Point one adapter module's httpx at a canned response. -> the list of calls it made.

    Patched per module, not on the shared httpx module object: `stt.httpx` and `nlu.httpx` are the
    same object, so patching it in place would leak into whichever adapter the test is not about.
    """
    posts = []

    def post(url, **kw):
        posts.append(url)
        return response

    monkeypatch.setattr(module, "httpx", types.SimpleNamespace(
        post=post, Headers=httpx.Headers, HTTPStatusError=httpx.HTTPStatusError))
    return posts


# The two hosted adapters, each with the arm and payload it takes. Both must behave identically on
# a 429 — a rate limit that only one stage reported would be a gap the other stage hides.
HOSTED = [
    pytest.param(stt, stt.openai_audio, "stt", lambda: [0.0] * 16, TRANSCRIPT_OK, id="stt"),
    pytest.param(nlu, nlu.openai_chat, "llm", lambda: nlu.messages("hi"), CHAT_OK, id="llm"),
]


def hosted_arm(stage):
    """The first arm on this stage that actually speaks HTTP — a local one never sees a 429."""
    return next(a for a in arms.available(stage) if a.provider != "local")


# --- the criterion --------------------------------------------------------------------------

@pytest.mark.parametrize("module,backend,stage,payload,ok_response", HOSTED)
def test_a_429_raises_rate_limited(module, backend, stage, payload, ok_response, monkeypatch):
    """Not a blank transcript, not an empty reply — the caller must not be able to miss it."""
    serve(monkeypatch, module, rate_limited())
    arm = hosted_arm(stage)

    with pytest.raises(RateLimited) as e:
        backend(arm, payload(), {})

    assert e.value.status_code == 429
    assert e.value.retry_after == RETRY_AFTER
    assert e.value.arm_id == arm.id


@pytest.mark.parametrize("module,backend,stage,payload,ok_response", HOSTED)
def test_the_message_names_the_arm_and_the_wait(module, backend, stage, payload, ok_response,
                                                monkeypatch):
    """Whoever reads this in a terminal needs to know which arm and how long, without a log dive."""
    serve(monkeypatch, module, rate_limited())
    arm = hosted_arm(stage)

    with pytest.raises(RateLimited) as e:
        backend(arm, payload(), {})

    assert arm.id in str(e.value)
    assert "429" in str(e.value)
    assert "12.5s" in str(e.value)


def test_a_429_is_logged_before_the_fallback_runs(monkeypatch, calls_log):
    """The "not swallowed" assertion, now that the 429 is survivable.

    A stage that falls back returns a transcript, so the 429 no longer reaches the caller — which is
    exactly the shape of change that quietly loses a failure. It must therefore be *more* visible in
    the log, not less: the refusal is line one with ok:false, and the arm that covered for it is
    line two, naming what it stood in for. Fails the day anyone wraps a call in a bare except.
    """
    arm = hosted_arm("stt")
    serve(monkeypatch, stt, rate_limited())

    arms.stt([0.0] * 16, arm.id, turn_id="t429")

    refused, covered = [json.loads(x) for x in calls_log.read_text(encoding="utf-8").splitlines()]

    assert refused["ok"] is False
    assert "RateLimited" in refused["error"]
    assert refused["status_code"] == 429
    assert refused["retry_after_s"] == RETRY_AFTER
    assert refused["model_id"] == arm.repo_id        # the HF repo id, not the provider's string
    assert refused["cost_usd"] == 0.0                # a refused call still cost nothing
    assert refused["turn_id"] == "t429"

    assert covered["ok"] is True
    assert covered["fallback_for"] == arm.id, "the local line must say whose work it took over"
    assert covered["turn_id"] == "t429"              # both attempts belong to the same turn


def test_a_429_still_propagates_when_there_is_nowhere_to_fall_back(monkeypatch, calls_log):
    """The control for the test above: covering a 429 is the fallback's doing, not the 429's.

    Without this, the change from `pytest.raises` to a plain call reads as "a 429 is fine now".
    Remove the stage's fallback and the original contract is exactly as it was.
    """
    monkeypatch.delitem(arms.FALLBACKS, "stt")
    serve(monkeypatch, stt, rate_limited())

    with pytest.raises(RateLimited):
        arms.stt([0.0] * 16, hosted_arm("stt").id, turn_id="t429")

    rec = json.loads(calls_log.read_text(encoding="utf-8").strip())
    assert rec["ok"] is False and rec["status_code"] == 429


def test_a_429_is_distinguishable_from_a_bad_key(monkeypatch, calls_log):
    """The control that makes the tests above mean something.

    Without it they all pass against a check that called every non-2xx a rate limit, which would be
    strictly worse than the raise_for_status() it replaced: "wait a minute" and "fix your key" need
    opposite responses, and only one of them is worth retrying.
    """
    arm = hosted_arm("stt")
    serve(monkeypatch, stt, FakeResponse(401, {"error": {"message": "invalid api key"}}))

    with pytest.raises(httpx.HTTPStatusError):
        arms.stt([0.0] * 16, arm.id, turn_id="t401")

    rec = json.loads(calls_log.read_text(encoding="utf-8").strip())
    assert rec["status_code"] == 401
    assert "RateLimited" not in rec["error"]
    assert "retry_after_s" not in rec


def test_a_429_without_retry_after_still_raises(monkeypatch):
    """Providers are not required to send it. Absent must read as "not said", never as zero."""
    serve(monkeypatch, stt, rate_limited(retry_after=None))

    with pytest.raises(RateLimited) as e:
        stt.openai_audio(hosted_arm("stt"), [0.0] * 16, {})

    assert e.value.retry_after is None
    assert "429" in str(e.value)
    assert "no Retry-After given" in str(e.value)


def test_an_unparseable_retry_after_is_reported_as_absent(monkeypatch):
    """The HTTP-date form is legal. Guessing at it is worse than saying the provider did not say."""
    serve(monkeypatch, stt, rate_limited(retry_after="Wed, 21 Oct 2026 07:28:00 GMT"))

    with pytest.raises(RateLimited) as e:
        stt.openai_audio(hosted_arm("stt"), [0.0] * 16, {})

    assert e.value.retry_after is None


@pytest.mark.parametrize("module,backend,stage,payload,ok_response", HOSTED)
def test_a_success_records_its_status_code(module, backend, stage, payload, ok_response,
                                           monkeypatch):
    """Recorded on every response, not only refusals, so calls.jsonl groups by what was said."""
    serve(monkeypatch, module, ok_response)
    rec = {}

    result = backend(hosted_arm(stage), payload(), rec)

    assert rec["status_code"] == 200
    assert "retry_after_s" not in rec
    assert result.strip() == result and result             # the adapter still returns its text


# --- the turn ------------------------------------------------------------------------------

def test_a_rate_limited_turn_falls_back_and_still_records_the_refusal(monkeypatch, turns_log,
                                                                      capsys):
    """The 429 no longer ends the turn — but it must not become invisible either.

    Three things have to be true at once, and it is the third that stops this being a way to hide a
    rate limit: stderr says it happened, the turn record names the arm that *actually* ran rather
    than the one that was selected, and the failed attempt's own latency is broken out so the local
    arm is not credited with a dead round-trip.
    """
    remote = hosted_arm("stt")
    monkeypatch.setattr("sys.argv", ["vox", "--turns", "1"])
    monkeypatch.setattr(loop.vad, "_vad_model", lambda: None)
    monkeypatch.setattr(loop.arms, "select", lambda args: {
        "stt": remote, "llm": LLM_ARMS[0], "tts": TTS_ARMS[0]})
    monkeypatch.setattr(loop.vad, "listen", lambda *a, **kw: _capture())
    monkeypatch.setattr(loop.state, "build", lambda *a, **kw: fake_state("On the fourth."))
    monkeypatch.setattr(loop.audio, "play", lambda audio, **kw: None)
    serve(monkeypatch, stt, rate_limited())
    local_stt(monkeypatch, "when was I paid")
    working_tts(monkeypatch)

    assert loop.main() == 0, "the fallback answered, so the turn was spoken"

    err = capsys.readouterr().err
    assert "STT FALLBACK" in err
    assert "12.5s" in err                                # which refusal, and for how long
    assert "Traceback" not in err

    rec = json.loads(turns_log.read_text(encoding="utf-8").strip())
    assert rec["fell_back"] == ["stt"]
    assert rec["stt_model"] == arms.resolve("stt", arms.FALLBACKS["stt"]).id
    assert rec["stt_fallback_from"] == remote.id
    assert "RateLimited" in rec["stt_fallback_reason"]
    assert rec["stt_failed_ms"] is not None, "the refusal took time and must not vanish into t_stt"


def test_a_rate_limit_never_retries_the_same_arm(monkeypatch, capsys):
    """Retrying a pace limit twice in the same second is how a free tier gets shut off.

    This used to be spelled "the run stops on the first 429". Falling back changed the remedy but
    not the rule, and made it stricter: the arm goes on cooldown for the window the provider asked
    for, so turn two does not even attempt it. Five turns, one request.
    """
    remote = hosted_arm("stt")
    monkeypatch.setattr("sys.argv", ["vox", "--turns", "5"])
    monkeypatch.setattr(loop.vad, "_vad_model", lambda: None)
    monkeypatch.setattr(loop.arms, "select", lambda args: {
        "stt": remote, "llm": LLM_ARMS[0], "tts": TTS_ARMS[0]})
    monkeypatch.setattr(loop.vad, "listen", lambda *a, **kw: _capture())
    monkeypatch.setattr(loop.state, "build", lambda *a, **kw: fake_state("On the fourth."))
    silent_speaker(monkeypatch)
    posts = serve(monkeypatch, stt, rate_limited())
    local_stt(monkeypatch, "when was I paid")
    working_tts(monkeypatch)

    assert loop.main() == 0
    assert len(posts) == 1, "the rate-limited arm must be asked once, not once per turn"


def test_a_rate_limit_that_the_fallback_cannot_cover_still_stops_the_run(monkeypatch, turns_log,
                                                                        capsys):
    """The last resort, unchanged. When there is nowhere to go, main() still ends the session.

    Without this the two tests above would pass against a loop that had simply stopped handling
    RateLimited at all.
    """
    monkeypatch.setattr("sys.argv", ["vox", "--turns", "5"])
    monkeypatch.setattr(loop.vad, "_vad_model", lambda: None)
    monkeypatch.setattr(loop.arms, "select", lambda args: {
        "stt": STT_ARMS[0], "llm": LLM_ARMS[0], "tts": TTS_ARMS[0]})
    monkeypatch.setattr(loop.vad, "listen", lambda *a, **kw: _capture())

    attempts = []

    def limited(*a, **kw):
        attempts.append(1)
        raise RateLimited("groq rate-limited x (429) — retry after 12.5s",
                          status_code=429, retry_after=RETRY_AFTER, arm_id="x")

    monkeypatch.setattr(loop.arms, "stt", limited)

    assert loop.main() == 1
    assert len(attempts) == 1, "the run must stop on the first 429, not retry inside the loop"

    err = capsys.readouterr().err
    assert "RATE LIMITED" in err
    assert "Traceback" not in err

    rec = json.loads(turns_log.read_text(encoding="utf-8").strip())
    assert rec["ok"] is False
    assert "RateLimited" in rec["error"]
    assert rec["t_stt_ms"] is not None                   # the refusal took time; it is logged


def local_stt(monkeypatch, text):
    """Make the STT stage's local fallback arm answer, without weights on disk."""
    fb = arms.resolve("stt", arms.FALLBACKS["stt"])
    monkeypatch.setitem(arms._MODULES["stt"].BACKENDS, fb.backend,
                        lambda arm, payload, rec: text)


def working_tts(monkeypatch):
    monkeypatch.setitem(arms._MODULES["tts"].BACKENDS, TTS_ARMS[0].backend,
                        lambda arm, text, rec: [0.0] * 240)


class FakePlayback:
    """What audio.play hands back with block=False. Silent, and never interrupted.

    Needed because every turn but the last of a multi-turn run plays its reply interruptibly
    (VOX-011), and the loop closes the handle it was given. Returning None is enough only for a
    single-turn run, where playback still blocks.
    """

    reply_s = 1.0
    played_s = 0.0

    def close(self):
        pass


def silent_speaker(monkeypatch):
    """A speaker that makes no sound, whether the turn plays blocking or watched."""
    monkeypatch.setattr(loop.audio, "play",
                        lambda audio, **kw: FakePlayback() if kw.get("block") is False else None)


def _capture():
    # `__len__` because the real Capture has one and the loop prints it when a capture is carried
    # from one turn into the next (VOX-011).
    return type("Cap", (), {"segment": [0.0] * 16, "speech_end_t": 100.0, "endpointed_t": 100.4,
                            "spoken_s": 1.2, "infer_ms": 9.0, "t_vad_ms": 400.0,
                            "__len__": lambda self: len(self.segment)})()
