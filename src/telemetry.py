"""The shared cost/latency logger. Every model call goes through this — no exceptions.

Deliberately minimal: one JSONL line per *model call*. The five-field per-turn breakdown
(t_vad, t_stt, t_llm, t_tts, time_to_first_audio) is VOX-003's deliverable and is built on
top of these records rather than replacing them.

Cost is logged as 0.0 with the tier that justifies it. A non-zero number here means the zero
spend constraint has been broken and the run should stop.
"""
import json
import time
import uuid
from contextlib import contextmanager

from src.config import RUNS_DIR

CALLS_LOG = RUNS_DIR / "calls.jsonl"

# Free-tier endpoints and local weights. Anything not on this list is a STOP-and-ask.
FREE_TIERS = {"groq": "free-tier", "nvidia-nim": "free-tier", "local": "local-weights"}


def new_turn_id():
    return uuid.uuid4().hex[:12]


@contextmanager
def log_call(stage, arm, turn_id, **extra):
    """Time one model call and append a record. Re-raises after logging the failure.

    Yields a dict the caller can add measured facts to (chars, tokens, audio seconds) —
    whatever the stage actually knows.
    """
    if arm.provider not in FREE_TIERS:
        raise RuntimeError(
            f"provider {arm.provider!r} for {arm.repo_id} is not a known free tier. Zero spend "
            f"is a hard constraint — stop and ask before calling it."
        )
    record = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "turn_id": turn_id,
        "stage": stage,
        "model_id": arm.repo_id,          # HF repo id, never the provider's string
        "provider": arm.provider,
        "tier": FREE_TIERS[arm.provider],
        "cost_usd": 0.0,
        **extra,
    }
    t0 = time.perf_counter()
    try:
        yield record
    except Exception as e:
        record["ok"] = False
        record["error"] = f"{type(e).__name__}: {e}"
        raise
    else:
        record["ok"] = True
    finally:
        record["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        _append(record)


def _append(record):
    RUNS_DIR.mkdir(exist_ok=True)
    with CALLS_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
