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


def is_transient(exc):
    """-> True if another arm could plausibly survive `exc`. The whole fallback rule, in one place.

    Deliberately narrow. What is *not* here matters as much as what is:

      httpx.HTTPStatusError   4xx — a bad key, a model not on the catalogue, a malformed body. The
                              local arm would succeed and hide a bug that needs fixing, and a
                              silently-degraded transcript is a worse outcome than a loud stop.
      RuntimeError            a missing credential or a provider off the free-tier list. That is a
                              STOP-and-ask under CLAUDE.md, not a runtime condition to route around.
      the empty-reply error   nlu.py raising because a reasoning arm spent its budget thinking. That
                              is a misconfigured arm, and it would raise again next turn.
    """
    return isinstance(exc, (RateLimited, ProviderError,
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
    # 4xx keeps the original behaviour exactly: one HTTPStatusError, no fallback, fix the request.
    r.raise_for_status()
