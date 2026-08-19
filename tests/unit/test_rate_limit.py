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

from src import arms, errors, loop, nlu, stt
from src.config import LLM_ARMS, STT_ARMS, TTS_ARMS
from src.errors import RateLimited

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


def test_a_429_is_logged_before_it_propagates(monkeypatch, calls_log):
    """The "not swallowed" assertion. Fails the day anyone wraps a call in a bare except."""
    arm = hosted_arm("stt")
    serve(monkeypatch, stt, rate_limited())

    with pytest.raises(RateLimited):
        arms.stt([0.0] * 16, arm.id, turn_id="t429")

    rec = json.loads(calls_log.read_text(encoding="utf-8").strip())
    assert rec["ok"] is False
    assert "RateLimited" in rec["error"]
    assert rec["status_code"] == 429
    assert rec["retry_after_s"] == RETRY_AFTER
    assert rec["model_id"] == arm.repo_id            # the HF repo id, not the provider's string
    assert rec["cost_usd"] == 0.0                    # a refused call still cost nothing
    assert rec["turn_id"] == "t429"


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

def test_a_rate_limited_turn_records_the_error_and_reports_it(monkeypatch, turns_log, capsys):
    """Through main(), because deciding the session is over is main's call, not a turn's.

    A traceback is technically "surfaced" but it is not readable, and it skips the summary that
    tells you where the logs are. This asserts a sentence on stderr and a turn line on disk.
    """
    monkeypatch.setattr("sys.argv", ["vox", "--turns", "3"])
    monkeypatch.setattr(loop.vad, "_vad_model", lambda: None)
    monkeypatch.setattr(loop.arms, "select", lambda args: {
        "stt": hosted_arm("stt"), "llm": LLM_ARMS[0], "tts": TTS_ARMS[0]})
    monkeypatch.setattr(loop.vad, "listen", lambda *a, **kw: _capture())
    serve(monkeypatch, stt, rate_limited())

    assert loop.main() == 1                              # nothing was spoken

    err = capsys.readouterr().err
    assert "RATE LIMITED" in err
    assert "12.5s" in err
    assert "Traceback" not in err

    rec = json.loads(turns_log.read_text(encoding="utf-8").strip())
    assert rec["ok"] is False
    assert "RateLimited" in rec["error"]
    assert rec["t_stt_ms"] is not None                   # the refusal took time; it is logged


def test_a_rate_limit_stops_the_run_rather_than_hammering_the_provider(monkeypatch, capsys):
    """Retrying a pace limit two more times in the same second is how a free tier gets shut off."""
    attempts = []
    monkeypatch.setattr("sys.argv", ["vox", "--turns", "5"])
    monkeypatch.setattr(loop.vad, "_vad_model", lambda: None)
    monkeypatch.setattr(loop.arms, "select", lambda args: {
        "stt": STT_ARMS[0], "llm": LLM_ARMS[0], "tts": TTS_ARMS[0]})
    monkeypatch.setattr(loop.vad, "listen", lambda *a, **kw: _capture())

    def limited(*a, **kw):
        attempts.append(1)
        raise RateLimited("groq rate-limited x (429) — retry after 12.5s",
                          status_code=429, retry_after=RETRY_AFTER, arm_id="x")

    monkeypatch.setattr(loop.arms, "stt", limited)

    assert loop.main() == 1
    assert len(attempts) == 1, "the run must stop on the first 429, not retry inside the loop"


def _capture():
    return type("Cap", (), {"segment": [0.0] * 16, "speech_end_t": 100.0, "endpointed_t": 100.4,
                            "spoken_s": 1.2, "infer_ms": 9.0, "t_vad_ms": 400.0})()
