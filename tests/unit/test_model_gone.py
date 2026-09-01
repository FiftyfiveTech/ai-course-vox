"""A retired model costs the turn one arm, not the session.

On 2026-08-26 NIM retired `meta-llama/Llama-3.1-8B-Instruct`, the default LLM arm at the time. On
2026-09-01 `make demo` reached the LLM stage of a live turn and ended in a traceback:

    httpx.HTTPStatusError: Client error '410 Gone' for url '.../chat/completions'

Nothing was wrong with the key, the quota or the request. `errors.check` sent every 4xx to
`raise_for_status()`, and `is_transient` excluded `HTTPStatusError` on purpose — a bad key or a
malformed body must not be papered over by a local arm. A retired model is the same status class and
the opposite fix: the request is right and the table is stale, so the only repair is an edit nobody
can make mid-turn.

So 404 and 410 became `ModelGone`, and `ModelGone` is transient. These tests pin both halves of
that, and the third one pins what did *not* change — 401 and 400 still stop the run, or this would
have traded a loud break for a silent downgrade.

No network: `httpx.post` is replaced per module, the same seam test_rate_limit.py uses.
"""
import json
import types

import httpx
import pytest

from src import arms, errors, nlu
from src.errors import ModelGone

# The bodies the two providers actually sent, quoted from the failures that prompted this file.
NIM_410 = {"type": "about:blank", "title": "Gone", "status": 410,
           "detail": "The model 'meta/llama-3.1-8b-instruct' has reached its end of life on "
                     "2026-08-26T09:00:00Z and is no longer available."}
NIM_404 = {"detail": "Function 'e3a1': Not found for account '2jDD25'."}
GROQ_404 = {"error": {"message": "The model `x` does not exist", "type": "invalid_request_error"}}


class FakeResponse:
    """Just what `errors.check` and `errors._detail` read off a response."""

    def __init__(self, status_code, json_body=None, text=""):
        self.status_code = status_code
        self.headers = httpx.Headers({})
        self.reason_phrase = "Gone"
        self.text = text
        self._json = json_body

    def json(self):
        if self._json is None:
            raise ValueError("not json")
        return self._json

    def raise_for_status(self):
        raise httpx.HTTPStatusError(
            f"Client error '{self.status_code}' for url 'https://example.invalid'",
            request=httpx.Request("POST", "https://example.invalid"), response=None)


def serve(monkeypatch, module, response):
    monkeypatch.setattr(module, "httpx", types.SimpleNamespace(
        post=lambda url, **kw: response, Headers=httpx.Headers,
        HTTPStatusError=httpx.HTTPStatusError))


def remote_llm():
    return next(a for a in arms.available("llm") if a.provider != "local")


# --- 410 and 404 become ModelGone -----------------------------------------------------------

@pytest.mark.parametrize("status,body", [(410, NIM_410), (404, NIM_404), (404, GROQ_404)],
                         ids=["nim-410-eol", "nim-404-entitlement", "groq-404-catalogue"])
def test_a_retired_model_raises_model_gone(status, body, monkeypatch):
    serve(monkeypatch, nlu, FakeResponse(status, body))
    arm = remote_llm()

    with pytest.raises(ModelGone) as e:
        nlu.openai_chat(arm, nlu.messages("hi"), {})

    assert e.value.status_code == status
    assert e.value.arm_id == arm.id


def test_the_message_quotes_the_provider_and_says_where_to_fix_it(monkeypatch):
    """The end-of-life date is the one fact that says wait-or-edit, so it must survive to the log."""
    serve(monkeypatch, nlu, FakeResponse(410, NIM_410))
    arm = remote_llm()

    with pytest.raises(ModelGone) as e:
        nlu.openai_chat(arm, nlu.messages("hi"), {})

    msg = str(e.value)
    assert "2026-08-26T09:00:00Z" in msg          # the provider's own words, not a summary
    assert arm.provider_model in msg
    assert "config.py" in msg and "preflight" in msg


def test_a_non_json_body_still_produces_a_message(monkeypatch):
    """A proxy can answer 410 in HTML. `_detail` runs on the failure path and must not add one."""
    serve(monkeypatch, nlu, FakeResponse(410, None, text="<html>410 Gone</html>"))

    with pytest.raises(ModelGone) as e:
        nlu.openai_chat(remote_llm(), nlu.messages("hi"), {})

    assert "410 Gone" in str(e.value)


# --- and ModelGone is transient, so the turn survives ---------------------------------------

def test_model_gone_is_transient():
    assert errors.is_transient(ModelGone("x", status_code=410, arm_id="a@b"))


def test_a_retired_arm_falls_back_to_the_local_one_loudly(monkeypatch, calls_log, capsys):
    """The whole point: the session keeps going, and says out loud why it is going differently."""
    serve(monkeypatch, nlu, FakeResponse(410, NIM_410))
    # The fallback arm speaks HTTP through this same module, so `serve` would hand it the 410 too;
    # patched at the backend seam instead, which is where `_backend` looks it up.
    monkeypatch.setitem(nlu.BACKENDS, "ollama-chat", lambda arm, msgs, rec, **kw: "Local reply.")

    reply = arms.llm(nlu.messages("hi"), turn_id="t-1")

    assert reply == "Local reply."
    banner = capsys.readouterr().err
    assert "LLM FALLBACK" in banner and "ModelGone" in banner

    lines = [json.loads(x) for x in calls_log.read_text().splitlines()]
    refused, rescued = lines[0], lines[1]
    assert refused["ok"] is False and refused["status_code"] == 410
    assert rescued["ok"] is True and rescued["fallback_for"] == remote_llm().id


# --- what did not change --------------------------------------------------------------------

@pytest.mark.parametrize("status", [400, 401, 403, 422])
def test_every_other_4xx_still_stops_the_run(status, monkeypatch):
    """A bad key must not be rescued into a plausible answer — that was the original rule."""
    serve(monkeypatch, nlu, FakeResponse(status, {"error": {"message": "nope"}}))

    with pytest.raises(httpx.HTTPStatusError):
        nlu.openai_chat(remote_llm(), nlu.messages("hi"), {})


@pytest.mark.parametrize("status", [400, 401, 403, 422])
def test_and_is_not_transient(status, monkeypatch):
    assert not errors.is_transient(
        httpx.HTTPStatusError("x", request=httpx.Request("POST", "https://e.invalid"),
                              response=None))
