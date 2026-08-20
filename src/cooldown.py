"""Which arms are temporarily out of rotation, and until when.

When a free tier says 429 it is telling us how long to wait. Before this module we threw that
number away and either stopped the run or, worse, asked again next turn — which is how a pace limit
becomes a shut-off. Now the arm is parked for the window the provider asked for, the stage runs on
its local fallback in the meantime, and the arm comes back on its own.

Deliberately process-local and in memory. Persisting it across runs would mean a demo that opens
with a rate-limited arm from an hour ago, and the window is short enough that it would usually be
stale anyway. A fresh process gets a fresh chance.

`time.monotonic` rather than wall clock: a clock adjustment mid-session must not un-park an arm or
strand it for hours.
"""
import time

_until = {}     # arm.id -> the monotonic instant it may be called again


def block(arm, seconds):
    """Park `arm` for `seconds`. -> the instant it comes back.

    Extends an existing block rather than shortening it: two 429s in a row from the same provider
    mean the second Retry-After is the one to believe, but a *shorter* second window is not a reason
    to go back early.
    """
    until = time.monotonic() + max(0.0, float(seconds))
    _until[arm.id] = max(until, _until.get(arm.id, 0.0))
    return _until[arm.id]


def blocked(arm):
    """-> seconds still to wait before `arm` may be called, or 0.0 if it is free.

    A float rather than a bool so callers can say how long, which is the difference between "the
    arm is unavailable" and a log line someone can act on.
    """
    until = _until.get(arm.id)
    if until is None:
        return 0.0
    remaining = until - time.monotonic()
    if remaining <= 0:
        del _until[arm.id]          # expired — drop it so the table cannot grow without bound
        return 0.0
    return round(remaining, 1)


def clear():
    """Forget every block. For tests — a 429 in one test must not park an arm for the next."""
    _until.clear()
