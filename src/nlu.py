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
from pathlib import Path

import httpx

from src import errors
from src.config import PROMPTS_DIR

# A spoken turn is short; this is a guardrail, not a target. Held equal across arms so a latency
# comparison is not really a comparison of how much each arm was allowed to say.
MAX_TOKENS = 120

# How many prior exchanges to include as context. Each exchange = 1 user + 1 assistant message, so
# 3 exchanges = 6 messages. Beyond this the benefit flattens while the token cost grows linearly.
MAX_HISTORY_TURNS = 3

# The default for a *spoken reply*, where a little variation reads as human. Not the default for a
# grounded answer, which passes its own — see answer.ANSWER_TEMPERATURE.
TEMPERATURE = 0.3

# How long ollama keeps the fallback model resident after a call. Its default is 5 minutes, which
# is shorter than a demo and would let the model page out between the warm-up and the rate limit
# that needs it — putting the load back inside a turn, which is the thing load_ollama exists to
# prevent. -1 keeps it until the daemon is told otherwise.
OLLAMA_KEEP_ALIVE = -1

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

# YAML front matter is metadata *about* the prompt (version, the arm it was written against,
# what supersedes it). No model ever sees it.
FRONT_MATTER = re.compile(r"\A---\n.*?\n---\n", re.DOTALL)

# Default: the generic reply prompt keeps Phase-0 behaviour unchanged.
_DEFAULT_STAGE = "reply"


def load_prompt(stage=None):
    """-> a versioned prompt file's body, YAML front matter stripped. Never inlined in code.

    Here rather than in each module that has a prompt: VOX-031 added a second prompt file, and two
    copies of this regex is two places for "the front matter leaked into the system message" to
    happen. The front matter is metadata *about* the prompt — version, the arm it was written
    against, what supersedes it — and no model should ever see it.

    Args:
        stage: one of the keys in PROMPT_FILES, a Path to a prompt file (VOX-031's answer prompt
               is not a turn stage, so it is not in the table), or None for the default.
    Returns the system prompt string ready to pass to the LLM.
    """
    path = stage if isinstance(stage, Path) else PROMPT_FILES[
        stage if stage in PROMPT_FILES else _DEFAULT_STAGE]
    return FRONT_MATTER.sub("", path.read_text(encoding="utf-8")).strip()


def system_prompt():
    """The plain-reply prompt (VOX-018's reply_v1). VOX-031's answer prompt is in src/answer.py."""
    return load_prompt()


def messages(transcript, stage=None, history=None):
    """-> the `msgs` list for arms.llm(). The only place a turn's prompt shape is decided.

    `history` (VOX-034) is a `src.history.History` or None, and prior turns are inserted between the
    system message and this transcript. This is the PLAIN path only: the grounded path builds its own
    messages in `src/answer.py` and stays single-shot, because the numeric guard is a set difference
    against the excerpts retrieved for the current question and a previous answer in the prompt would
    be a figure it cannot see. `history=None` reproduces the pre-VOX-034 message list exactly.

    Args:
        transcript: the user's spoken text.
        stage: prompt stage to use (greet, clarify, confirm, capture, escalate, refuse).
               None falls back to the generic reply prompt.
        history: prior turns for conversational coherence, or None.
    """
    msgs = [{"role": "system", "content": load_prompt(stage)}]
    if history:
        msgs.extend(history.messages_prefix())
    msgs.append({"role": "user", "content": transcript})
    return msgs


def openai_chat(arm, msgs, rec, timeout=None, temperature=None, json_mode=False, max_tokens=None):
    """OpenAI-compatible /chat/completions. Serves every LLM arm — NIM, Groq and ollama all speak it.

    `arm.extra["request"]` adds the fields an arm cannot be called without — `reasoning_effort` for
    gpt-oss. It is merged after the shared parameters and deliberately cannot override them, so no
    arm can quietly give itself a bigger budget than the ones it is being compared against.

    `temperature` is per *call*, not per arm, because the two things this repo asks an LLM to do want
    different sampling and neither is a property of the model: a spoken reply is better for a little
    variety, and a grounded answer read out of five policy excerpts is not. It is a call option
    rather than an arm field so that both still run on the same arm and stay comparable — see
    `answer.ANSWER_TEMPERATURE` for the measurement that made this a parameter.
    """
    body = {"model": arm.provider_model, "messages": msgs,
            "temperature": TEMPERATURE if temperature is None else temperature,
            "max_tokens": MAX_TOKENS if max_tokens is None else max_tokens}
    # VOX-034: `json_mode` for the figure extractor, which needs a parseable object rather than a
    # sentence. Same mechanism state.build() already uses; here it is a per-call option because the
    # same arm serves both the spoken reply and the extraction, and neither is a property of the arm.
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    for k, v in (arm.extra.get("request") or {}).items():
        body.setdefault(k, v)

    r = httpx.post(
        f"{arm.api_base}/chat/completions",
        headers=arm.auth_headers({"Content-Type": "application/json"}),
        json=body,
        timeout=arm.timeout_s if timeout is None else timeout,
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


def load_ollama(arm):
    """Get the local model resident in memory before a turn depends on it. Called by arms.warm().

    Two steps, and both matter for a different reason.

    Checking it is *pulled* is deliberately not a pull: `ollama pull` fetches ~2 GB, and doing that
    inside a fallback — which by definition happens when something has already gone wrong — turns a
    rate-limited turn into a several-minute stall with no explanation. Better to say now that the
    fallback is not ready, while the remote arm is still working.

    Loading it is the other half, and the reason this is a LOADER rather than a validator. A cold
    ollama call pages the whole model in first: measured at over 10 s here, which is longer than the
    entire remote budget the fallback exists to escape. An empty-prompt /api/generate is ollama's
    own way to ask for that without generating anything, so the load lands in startup where every
    other local weight already does, instead of inside t_llm.
    """
    root = arm.api_base.rsplit("/v1", 1)[0]
    try:
        r = httpx.get(f"{root}/api/tags", timeout=5)
        r.raise_for_status()
    except Exception as e:
        raise RuntimeError(
            f"no ollama daemon at {root} ({type(e).__name__}), so {arm.repo_id} cannot serve as "
            f"the local LLM fallback. Start ollama, or accept that a rate limit ends the turn."
        ) from e

    pulled = {m.get("name", "") for m in (r.json().get("models") or [])}
    if arm.provider_model not in pulled:
        raise RuntimeError(
            f"{arm.provider_model} is not pulled, so {arm.repo_id} cannot serve as the local LLM "
            f"fallback. Run:\n    ollama pull {arm.provider_model}"
        )

    load = httpx.post(f"{root}/api/generate",
                      json={"model": arm.provider_model, "keep_alive": OLLAMA_KEEP_ALIVE},
                      timeout=arm.timeout_s)
    load.raise_for_status()
    return arm


# `ollama-chat` is the same adapter on the same wire protocol — the separate name exists because
# LOADERS is keyed by backend, and "is the model pulled?" is a question only the ollama arm can be
# asked. Pointing both keys at one function keeps that a registry fact rather than a second adapter.
BACKENDS = {"openai-chat": openai_chat, "ollama-chat": openai_chat}

# The two hosted arms have nothing to warm. The ollama arm is checked rather than loaded: the
# daemon owns the weights, and this process only needs to know they are there.
LOADERS = {"ollama-chat": load_ollama}


def reply(transcript, turn_id, model_id=None, stage=None, on_fallback=None, fallback=True,
          history=None):
    """-> one short reply suitable for reading aloud, from the named arm or the default.

    Args:
        transcript: the user's spoken text.
        turn_id: telemetry join key.
        model_id: HF repo id of the LLM arm, or None for the default.
        stage: prompt stage (greet, clarify, confirm, capture, escalate, refuse).
               None uses the generic reply prompt.
        on_fallback: optional callback when the remote arm falls back to local.
        fallback: set False to disable fallback (e.g. arm comparison scripts).
        history: prior turns (VOX-034), or None for the pre-VOX-034 single-shot prompt.
    """
    from src import arms                      # imported here: arms imports this module for BACKENDS
    prompt_key = stage if stage in PROMPT_FILES else _DEFAULT_STAGE
    msgs = messages(transcript, stage, history=history)
    return arms.llm(msgs, model_id, turn_id=turn_id,
                    on_fallback=on_fallback, fallback=fallback,
                    prompt_file=PROMPT_FILES[prompt_key].name,
                    transcript_chars=len(transcript),
                    history_turns=len(history) if history else 0)
