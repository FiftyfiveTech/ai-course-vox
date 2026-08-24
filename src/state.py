"""Structured turn state extraction (VOX-019).

`build()` is the single entry point: it calls the LLM with the extract prompt using JSON
mode, parses the response, and validates it through the TurnState Pydantic model.

If the LLM returns malformed JSON, or the JSON does not satisfy TurnState's schema, a
ValidationError (or JSONDecodeError) is raised — the turn never silently continues.
"""
import json
import re
import sys

import httpx

from schemas.turn_state import TurnState
from src import errors
from src.config import PROMPTS_DIR
from src.telemetry import log_call

PROMPT_FILE = PROMPTS_DIR / "extract_v2.md"

# Extraction needs precision, not creativity. Temperature 0 gives deterministic output.
TEMPERATURE = 0.0
# Enough for the JSON object + reply text. Intentionally smaller than the reply budget so
# a reasoning arm cannot silently spend the whole budget thinking.
MAX_TOKENS = 512


def _system_prompt() -> str:
    text = PROMPT_FILE.read_text(encoding="utf-8")
    return re.sub(r"\A---\n.*?\n---\n", "", text, flags=re.DOTALL).strip()


def build(transcript: str, turn_id: str, model_id: str | None = None,
          history=None) -> TurnState:
    """Extract structured turn state from a transcript.

    -> TurnState. Raises pydantic.ValidationError when the LLM returns malformed state,
    or json.JSONDecodeError when the response is not valid JSON. The turn must not catch
    these silently — they are the mechanism that ensures a turn never continues with bad state.

    Uses JSON mode (response_format: json_object) so the response is always parseable.
    Falls back to stripping a ```json … ``` fence if the provider ignores the format hint.

    `history` is a `src.history.History` or None. When present, its prior turns are rendered
    as {role, content} messages and inserted between the system prompt and the current user
    message so that the extractor can resolve cross-references ("and what about that?",
    "cancel it") correctly. Capped to nlu.MAX_HISTORY_TURNS exchanges to bound token cost —
    the History window may be wider than what extraction is willing to pay for.
    """
    from src import arms  # late import: arms imports nlu which is a sibling of this module
    from src.nlu import MAX_HISTORY_TURNS

    arm = arms.resolve("llm", model_id)

    msgs = [{"role": "system", "content": _system_prompt()}]
    if history:
        msgs.extend(history.messages_prefix()[-(MAX_HISTORY_TURNS * 2):])
    msgs.append({"role": "user", "content": transcript})
    body = {
        "model": arm.provider_model,
        "messages": msgs,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "response_format": {"type": "json_object"},
    }
    # Arm-level overrides (e.g. reasoning_effort for gpt-oss) must not override the above.
    for k, v in (arm.extra.get("request") or {}).items():
        body.setdefault(k, v)

    with log_call("llm", arm, turn_id,
                  messages=len(msgs), prompt_file=PROMPT_FILE.name,
                  transcript_chars=len(transcript)) as rec:
        r = httpx.post(
            f"{arm.api_base}/chat/completions",
            headers=arm.auth_headers({"Content-Type": "application/json"}),
            json=body,
            timeout=arm.timeout_s,
        )
        errors.check(r, arm, rec)
        payload = r.json()
        raw = (payload["choices"][0]["message"].get("content") or "").strip()
        usage = payload.get("usage") or {}
        rec["completion_tokens"] = usage.get("completion_tokens")
        rec["finish_reason"] = payload["choices"][0].get("finish_reason")

    # Strip a markdown fence if the provider ignored response_format.
    text = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE).strip()

    if not text:
        raise RuntimeError(
            f"{arm.id} returned an empty response for state extraction "
            f"(finish_reason={payload['choices'][0].get('finish_reason')!r})"
        )

    data = json.loads(text)          # JSONDecodeError -> turn fails loudly
    state = TurnState(**data)        # ValidationError -> turn fails loudly
    return state
