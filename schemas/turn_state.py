"""Structured turn state (VOX-019).

Every turn must produce a valid TurnState or raise — a turn never silently continues with
missing or malformed state. The Pydantic validator is the enforcement point.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

Intent = Literal["greet", "capture", "clarify", "confirm", "escalate", "refuse", "unknown"]
NextAction = Literal["reply", "confirm", "clarify", "escalate", "refuse"]


class TurnState(BaseModel):
    """The structured output of one turn's LLM step.

    Fields
    ------
    intent      : what the user wants — one of the six intent labels in ENTITY_SPEC.md,
                  or "unknown" when the utterance cannot be classified.
    entities    : extracted slot values keyed by entity type (empty when none are present).
    confidence  : model's self-reported confidence, 0.0–1.0.
    next_action : what VOX will do — drives confirmation (VOX-020) and escalation logic.
    reply       : the spoken reply ready for TTS. Must be non-blank.
    """

    intent: Intent
    entities: dict[str, str | list[str]] = Field(default_factory=dict)
    confidence: float = Field(ge=0.0, le=1.0)
    next_action: NextAction
    reply: str

    @model_validator(mode="after")
    def _reply_not_blank(self) -> TurnState:
        if not self.reply.strip():
            raise ValueError("reply must not be blank — TTS has nothing to synthesise")
        return self
