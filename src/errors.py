"""The one place a provider's HTTP response becomes either data or a named failure.

Both hosted adapters — `stt.openai_audio` and `nlu.openai_chat` — called `raise_for_status()`
directly, which raises the same `httpx.HTTPStatusError` for every non-2xx. That made a rate limit
indistinguishable in code from a bad key: the number 429 existed only inside httpx's message
string, so nothing could branch on it and the call record did not carry it.

That matters for this project specifically. Every arm runs on a free tier, so 429 is the *expected*
failure — it means "you are within your budget but going too fast", which is a different fact from
"your credential is wrong" and needs a different response from whoever reads the log.

Two things happen here:

  status_code lands on every call record, success included, so calls.jsonl can be grouped by what
              the provider said without parsing prose.
  429         raises RateLimited, carrying Retry-After when the provider sent one.

Deliberately no retry and no backoff — still true, and worth restating now that a fallback exists.
Retrying *the same arm* inside a timed turn would fold the wait into t_stt or t_llm and make the
VOX-003 latency split describe a turn nobody took. Routing to a *different* arm is a different
thing: it is bounded by one extra call, and both calls are logged separately. `is_transient` below
is the one place that says which failures earn that second call.
"""
import httpx


class RateLimited(RuntimeError):
    """A provider said 429. A RuntimeError subclass so existing broad handlers behave as before."""

    def __init__(self, message, *, status_code, retry_after, arm_id):
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after      # seconds the provider asked for, or None
        self.arm_id = arm_id


class ProviderError(RuntimeError):
    """A provider failed on its own side (5xx).

    Split out from `raise_for_status()` for the same reason 429 was: a 503 means "this provider is
    having a bad minute, ask someone else", and a 401 means "your key is wrong, asking someone else
    hides the bug". Both were the same `httpx.HTTPStatusError` before, and the status code lived
    only inside a message string, so nothing could branch on the difference.
    """

    def __init__(self, message, *, status_code, arm_id):
        super().__init__(message)
        self.status_code = status_code
        self.arm_id = arm_id


class ModelGone(RuntimeError):
    """A provider said 404 or 410 for the model itself: this arm no longer exists there.

    Added 2026-09-01, after NIM retired `meta-llama/Llama-3.1-8B-Instruct` on 2026-08-26 and the
    demo died mid-turn on a 410 whose body said exactly that. The 4xx rule below is right about a
    bad key and a malformed body — routing around those hides a bug — but a retired model is a
    third thing, and it fails the *other* way: the request is correct, the credential is fine, and
    the only fix is a table edit nobody can make while the mic is open.

    So this one is transient. The session degrades to the local arm, loudly — `_run_fallback`
    prints the banner and calls.jsonl carries `status_code` and `fallback_for` — and the staleness
    is caught before the next session by `scripts/preflight.py`, which asks each provider's
    catalogue what it still serves. Falling back is not how you find out the arm is dead; it is how
    the turn survives finding out.

    404 and 410 are both here because providers do not agree: NIM answers 410 with an end-of-life
    date for a retired model and 404 for one this account was never entitled to, and Groq answered
    404 for `Llama-3.3-70B` when it was not on the key's catalogue. All three mean "not from me".
    """

    def __init__(self, message, *, status_code, arm_id):
        super().__init__(message)
        self.status_code = status_code
        self.arm_id = arm_id


def is_transient(exc):
    """-> True if another arm could plausibly survive `exc`. The whole fallback rule, in one place.

    Deliberately narrow. What is *not* here matters as much as what is:

      httpx.HTTPStatusError   4xx — a bad key, a malformed body, a refused parameter. The local
                              arm would succeed and hide a bug that needs fixing, and a
                              silently-degraded transcript is a worse outcome than a loud stop.
                              404/410 used to be in this bucket and are now ModelGone above:
                              same status class, opposite fix, so they get the opposite answer.
      RuntimeError            a missing credential or a provider off the free-tier list. That is a
                              STOP-and-ask under CLAUDE.md, not a runtime condition to route around.
      the empty-reply error   nlu.py raising because a reasoning arm spent its budget thinking. That
                              is a misconfigured arm, and it would raise again next turn.
    """
    return isinstance(exc, (RateLimited, ProviderError, ModelGone,
                            httpx.TimeoutException, httpx.TransportError))


def _retry_after_s(headers):
    """-> the Retry-After wait in seconds, or None.

    Only the delay-seconds form is read. The HTTP-date form is legal but no free tier here sends it,
    and guessing wrong is worse than reporting nothing — None reads as "the provider did not say".
    """
    raw = headers.get("retry-after")
    if raw is None:
        return None
    try:
        return round(float(raw), 2)
    except (TypeError, ValueError):
        return None


def _detail(r):
    """-> the provider's own explanation, or its status line. Never raises: this runs on a failure.

    Three shapes across two providers — NIM sends RFC 9457 `{"detail": ...}`, Groq sends OpenAI's
    `{"error": {"message": ...}}`, and either can send HTML from a proxy — so the body is tried as
    each and falls back to the raw text it actually is.
    """
    try:
        body = r.json()
    except Exception:
        return (r.text or "").strip()[:200] or f"no body ({r.reason_phrase})"
    if isinstance(body, dict):
        for value in (body.get("detail"), (body.get("error") or {}).get("message")
                      if isinstance(body.get("error"), dict) else body.get("error")):
            if isinstance(value, str) and value.strip():
                return value.strip()[:200]
    return str(body)[:200]


def check(r, arm, rec):
    """Record what the provider said, then raise if it was a refusal. -> None on success."""
    rec["status_code"] = r.status_code
    if r.status_code == 429:
        retry_after = _retry_after_s(r.headers)
        rec["retry_after_s"] = retry_after
        wait = f"retry after {retry_after:g}s" if retry_after is not None else "no Retry-After given"
        raise RateLimited(
            f"{arm.provider} rate-limited {arm.id} (429) — {wait}. Every arm here is on a free "
            f"tier, so this is a pace limit, not a spend limit: wait it out or pick another arm.",
            status_code=429, retry_after=retry_after, arm_id=arm.id,
        )
    if r.status_code >= 500:
        raise ProviderError(
            f"{arm.provider} failed serving {arm.id} ({r.status_code}). That is the provider's "
            f"side, not the request — another arm can answer this turn.",
            status_code=r.status_code, arm_id=arm.id,
        )
    if r.status_code in (404, 410):
        # The provider's own words are the useful part here — NIM's 410 body carries the end-of-life
        # date, which is the one fact that says whether to wait or to edit config.py — so the body is
        # quoted rather than summarised.
        raise ModelGone(
            f"{arm.provider} no longer serves {arm.provider_model} for {arm.id} "
            f"({r.status_code}): {_detail(r)}. Re-point the arm in config.py — "
            f"`uv run python scripts/preflight.py` lists what each provider still serves.",
            status_code=r.status_code, arm_id=arm.id,
        )
    # Every other 4xx keeps the original behaviour exactly: one HTTPStatusError, no fallback, fix
    # the request.
    r.raise_for_status()
