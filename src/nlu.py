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

PROMPT_FILE = PROMPTS_DIR / "reply_v1.md"

# A spoken turn is short; this is a guardrail, not a target. Held equal across arms so a latency
# comparison is not really a comparison of how much each arm was allowed to say.
MAX_TOKENS = 120
TEMPERATURE = 0.3

# How long ollama keeps the fallback model resident after a call. Its default is 5 minutes, which
# is shorter than a demo and would let the model page out between the warm-up and the rate limit
# that needs it — putting the load back inside a turn, which is the thing load_ollama exists to
# prevent. -1 keeps it until the daemon is told otherwise.
OLLAMA_KEEP_ALIVE = -1


def load_prompt(path):
    """-> a versioned prompt file's body, YAML front matter stripped. Never inlined in code.

    Here rather than in each module that has a prompt: VOX-031 added a second prompt file, and two
    copies of this regex is two places for "the front matter leaked into the system message" to
    happen. The front matter is metadata *about* the prompt — version, the arm it was written
    against, what supersedes it — and no model should ever see it.
    """
    return re.sub(r"\A---\n.*?\n---\n", "", path.read_text(encoding="utf-8"),
                  flags=re.DOTALL).strip()


def system_prompt():
    """The plain-reply prompt (VOX-018's reply_v1). VOX-031's answer prompt is in src/answer.py."""
    return load_prompt(PROMPT_FILE)


def messages(transcript):
    """-> the `msgs` list for arms.llm(). The only place a turn's prompt shape is decided."""
    return [
        {"role": "system", "content": system_prompt()},
        {"role": "user", "content": transcript},
    ]


def openai_chat(arm, msgs, rec, timeout=None):
    """OpenAI-compatible /chat/completions. Serves every LLM arm — NIM, Groq and ollama all speak it.

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


def reply(transcript, turn_id, model_id=None, on_fallback=None, fallback=True):
    """-> one short reply suitable for reading aloud, from the named arm or the default."""
    from src import arms                      # imported here: arms imports this module for BACKENDS
    return arms.llm(messages(transcript), model_id, turn_id=turn_id, on_fallback=on_fallback,
                    fallback=fallback,
                    prompt_file=PROMPT_FILE.name, transcript_chars=len(transcript))
