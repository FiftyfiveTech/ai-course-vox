"""Transcript to spoken reply: meta-llama/Llama-3.1-8B-Instruct on the NVIDIA NIM free tier.

Phase 0 asks only for a reply. Structured intent extraction is VOX-019, so nothing here parses
entities — keeping the two apart means the Evaluator can tell which commit satisfied which gate.
"""
import re

import httpx

from src.config import LLM, PROMPTS_DIR
from src.telemetry import log_call

PROMPT_FILE = PROMPTS_DIR / "reply_v1.md"


def system_prompt():
    """The versioned prompt file with its YAML front matter stripped. Never inlined in code."""
    text = PROMPT_FILE.read_text(encoding="utf-8")
    return re.sub(r"\A---\n.*?\n---\n", "", text, flags=re.DOTALL).strip()


def reply(transcript, turn_id, timeout=30):
    """-> one short reply suitable for reading aloud."""
    with log_call("llm", LLM, turn_id, prompt_file=PROMPT_FILE.name,
                  transcript_chars=len(transcript)) as rec:
        r = httpx.post(
            f"{LLM.api_base}/chat/completions",
            headers={"Authorization": f"Bearer {LLM.key()}",
                     "Content-Type": "application/json"},
            json={
                "model": LLM.provider_model,
                "messages": [
                    {"role": "system", "content": system_prompt()},
                    {"role": "user", "content": transcript},
                ],
                "temperature": 0.3,
                "max_tokens": 120,   # a spoken turn is short; this is a guardrail, not a target
            },
            timeout=timeout,
        )
        r.raise_for_status()
        body = r.json()
        text = body["choices"][0]["message"]["content"].strip()
        usage = body.get("usage") or {}
        rec["prompt_tokens"] = usage.get("prompt_tokens")
        rec["completion_tokens"] = usage.get("completion_tokens")
        rec["reply_chars"] = len(text)

    return text
