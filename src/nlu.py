"""Transcript to spoken reply.

Two things live here: the one LLM backend (an OpenAI-compatible /chat/completions call, which is
what both the NIM arm and the Groq arm speak), and the message assembly that turns a transcript
into the `msgs` list that `arms.llm()` takes. Keeping those apart is what makes the arm swappable —
the prompt is a property of the task, the arm is a property of the run, and VOX-018 will version
the first without touching the second.

Phase 0 asks only for a reply. Structured intent extraction is VOX-019, so nothing here parses
entities — keeping the two apart means the Evaluator can tell which commit satisfied which gate.
"""
import re

import httpx

from src import errors
from src.config import PROMPTS_DIR

# A spoken turn is short; this is a guardrail, not a target. Held equal across arms so a latency
# comparison is not really a comparison of how much each arm was allowed to say.
MAX_TOKENS = 120
TEMPERATURE = 0.3

# Prompt library (VOX-018). One versioned file per intent stage — none inlined in code.
# Keys match the intent labels in ENTITY_SPEC.md; "reply" is the Phase-0 fallback.
PROMPT_FILES = {
    "greet":    PROMPTS_DIR / "greet_v1.md",
    "clarify":  PROMPTS_DIR / "clarify_v1.md",
    "confirm":  PROMPTS_DIR / "confirm_v1.md",
    "capture":  PROMPTS_DIR / "capture_v1.md",
    "escalate": PROMPTS_DIR / "escalate_v1.md",
    "refuse":   PROMPTS_DIR / "refuse_v1.md",
    "reply":    PROMPTS_DIR / "reply_v1.md",   # Phase-0 generic fallback
}

# Default: the generic reply prompt keeps Phase-0 behaviour unchanged.
_DEFAULT_STAGE = "reply"


def load_prompt(stage=None):
    """Load a versioned prompt file and strip its YAML front matter.

    Args:
        stage: one of the keys in PROMPT_FILES, or None for the default.
    Returns the system prompt string ready to pass to the LLM.
    """
    key = stage if stage in PROMPT_FILES else _DEFAULT_STAGE
    path = PROMPT_FILES[key]
    text = path.read_text(encoding="utf-8")
    return re.sub(r"\A---\n.*?\n---\n", "", text, flags=re.DOTALL).strip()


def messages(transcript, stage=None):
    """-> the `msgs` list for arms.llm(). The only place a turn's prompt shape is decided.

    Args:
        transcript: the user's spoken text.
        stage: prompt stage to use (greet, clarify, confirm, capture, escalate, refuse).
               None falls back to the generic reply prompt.
    """
    return [
        {"role": "system", "content": load_prompt(stage)},
        {"role": "user", "content": transcript},
    ]


def openai_chat(arm, msgs, rec, timeout=60):
    """OpenAI-compatible /chat/completions. Serves every LLM arm; NIM and Groq both speak it.

    `arm.extra["request"]` adds the fields an arm cannot be called without — `reasoning_effort` for
    gpt-oss. It is merged after the shared parameters and deliberately cannot override them, so no
    arm can quietly give itself a bigger budget than the ones it is being compared against.
    """
    body = {"model": arm.provider_model, "messages": msgs,
            "temperature": TEMPERATURE, "max_tokens": MAX_TOKENS}
    for k, v in (arm.extra.get("request") or {}).items():
        body.setdefault(k, v)

    r = httpx.post(
        f"{arm.api_base}/chat/completions",
        headers={"Authorization": f"Bearer {arm.key()}",
                 "Content-Type": "application/json"},
        json=body,
        timeout=timeout,
    )
    errors.check(r, arm, rec)
    payload = r.json()
    choice = payload["choices"][0]
    text = (choice["message"].get("content") or "").strip()
    usage = payload.get("usage") or {}
    rec["prompt_tokens"] = usage.get("prompt_tokens")
    rec["completion_tokens"] = usage.get("completion_tokens")
    rec["reply_chars"] = len(text)
    rec["finish_reason"] = choice.get("finish_reason")

    if not text:
        # A reasoning arm can spend the whole budget thinking and return an empty reply, which
        # would reach TTS as "synthesise nothing" and fail somewhere far less informative.
        raise RuntimeError(
            f"{arm.id} returned an empty reply "
            f"(finish_reason={choice.get('finish_reason')!r}, "
            f"{usage.get('completion_tokens')} completion tokens of {MAX_TOKENS}). A reasoning arm "
            f"needs a reasoning_effort in its config.py request options."
        )
    return text


BACKENDS = {"openai-chat": openai_chat}
LOADERS = {}          # both arms are hosted; there is nothing to warm


def reply(transcript, turn_id, model_id=None, stage=None):
    """-> one short reply suitable for reading aloud, from the named arm or the default.

    Args:
        transcript: the user's spoken text.
        turn_id: telemetry join key.
        model_id: HF repo id of the LLM arm, or None for the default.
        stage: prompt stage (greet, clarify, confirm, capture, escalate, refuse).
               None uses the generic reply prompt.
    """
    from src import arms                      # imported here: arms imports this module for BACKENDS
    prompt_key = stage if stage in PROMPT_FILES else _DEFAULT_STAGE
    return arms.llm(messages(transcript, stage), model_id, turn_id=turn_id,
                    prompt_file=PROMPT_FILES[prompt_key].name,
                    transcript_chars=len(transcript))
