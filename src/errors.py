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

Deliberately no retry and no backoff. A retry inside a timed turn would fold the wait into t_stt or
t_llm and make the VOX-003 latency split describe a turn nobody took. Retry-After is recorded so a
later ticket can act on it above the turn boundary, where the wait is visible.
"""


class RateLimited(RuntimeError):
    """A provider said 429. A RuntimeError subclass so existing broad handlers behave as before."""

    def __init__(self, message, *, status_code, retry_after, arm_id):
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after      # seconds the provider asked for, or None
        self.arm_id = arm_id


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
    r.raise_for_status()
