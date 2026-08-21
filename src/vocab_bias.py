"""Vocabulary biasing for Whisper (VOX-021).

Whisper's initial_prompt / prompt parameter biases the decoder towards words that appear
in the context string. Passing staff names and product names it would otherwise mis-hear
is the pipeline-beats-model-tuning lesson: one config line fixes what fine-tuning would
take labelled data and a GPU run to address.

The prompt is passed to every STT call when VOX_STT_BIAS=1 is set in the environment,
or when `build_prompt()` is called explicitly. It is invisible to the user — it is decoder
context, not a spoken utterance.

To measure the effect, run:
    .venv/bin/python scripts/measure_biasing.py
"""
import os

# Staff names, product names, and internal project names that Whisper mis-hears without
# biasing. Collected from the dev utterances and confirmed against gate transcripts.
# Add names here as new team members or products appear — no code change required.
HOTWORDS = [
    # Staff names (first name only — what people say out loud)
    "Priya", "Rahul", "Snefer", "Ananya", "Kiran",
    # Product / system name
    "VOX",
    # Internal project names that appear in entity fixtures
    "AI course",
]


def build_prompt() -> str:
    """Build the Whisper initial_prompt string from HOTWORDS.

    Whisper treats this as preceding context — putting the names in a natural sentence
    works better than a bare comma-separated list because the decoder expects prose.
    """
    names = ", ".join(HOTWORDS)
    return (
        f"FiftyFive internal assistant. Staff: {names}. "
        "Internal projects: VOX, AI course."
    )


def enabled() -> bool:
    """True when VOX_STT_BIAS=1 is set in the environment."""
    return os.environ.get("VOX_STT_BIAS", "0").strip() == "1"


def prompt_if_enabled() -> str | None:
    """Return the bias prompt when enabled, else None."""
    return build_prompt() if enabled() else None
