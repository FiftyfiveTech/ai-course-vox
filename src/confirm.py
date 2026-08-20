"""Confirmation gate (VOX-020).

When TurnState.next_action == "confirm", VOX reads back the key details
and waits for the user to say yes or no before proceeding. A turn that
skips confirmation for a sensitive action is a miss.

Rule (from prompts/extract_v1.md + evals/confirmation_cases.json):
  Any action that changes data — book_meeting, log_hours, set_reminder —
  must use next_action="confirm" and include key details + "Shall I go
  ahead?" in the reply. The gate verifies this fires on every dev case
  that needs it.
"""
from schemas.turn_state import TurnState

# next_action values that require a spoken yes/no before the turn proceeds.
CONFIRM_NEXT_ACTIONS = {"confirm"}

# These intents always involve writing data. Used by the checker script
# to identify which dev fixtures *should* trigger confirmation.
WRITE_INTENTS = {"book_meeting", "log_hours", "set_reminder", "capture"}

YES_WORDS = {"yes", "yeah", "yep", "yup", "sure", "ok", "okay",
             "go ahead", "confirm", "do it", "proceed", "correct"}
NO_WORDS  = {"no", "nope", "nah", "cancel", "stop", "abort",
             "don't", "do not", "nevermind", "never mind", "negative"}


def needs_confirmation(state: TurnState) -> bool:
    """True when this turn must be confirmed before executing."""
    return state.next_action in CONFIRM_NEXT_ACTIONS


def classify_response(transcript: str) -> str:
    """Classify a short yes/no reply. -> 'yes' | 'no' | 'unclear'."""
    t = transcript.strip().lower()
    if any(w in t for w in YES_WORDS):
        return "yes"
    if any(w in t for w in NO_WORDS):
        return "no"
    return "unclear"


def cancelled_reply() -> str:
    """What VOX says when the user cancels."""
    return "Got it, cancelled."


def unclear_reply() -> str:
    """What VOX says when the yes/no response is unclear."""
    return "Sorry, I did not catch that. Please say yes to confirm or no to cancel."
