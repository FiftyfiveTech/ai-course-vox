"""Model registry and runtime constants.

Models are named by **Hugging Face repo id** everywhere in this codebase. A provider is only
*where the weights run*, so the provider's own model string is a lookup detail that lives here
and nowhere else. If you find a provider string anywhere outside this file, it is a bug.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO_ROOT / "runs"
PROMPTS_DIR = REPO_ROOT / "prompts"

# Real values live outside the repo; .env is gitignored and holds names the shell can export.
load_dotenv(REPO_ROOT / ".env")
load_dotenv(Path.home() / ".config" / "secrets.env", override=False)


class Arm:
    """One model, pinned to the provider that serves it on a free tier."""

    def __init__(self, repo_id, provider, provider_model, api_base=None, key_env=None):
        self.repo_id = repo_id            # the only name we speak out loud
        self.provider = provider          # where it runs
        self.provider_model = provider_model
        self.api_base = api_base
        self.key_env = key_env

    def key(self):
        """The provider credential, or a STOP-and-ask style failure naming what is missing."""
        if self.key_env is None:
            return None
        k = os.environ.get(self.key_env)
        if not k:
            raise RuntimeError(
                f"{self.key_env} is not set, so {self.repo_id} cannot run on {self.provider}. "
                f"Add it to .env (names are listed in .env.example). Do not substitute a paid "
                f"endpoint — zero spend is a hard constraint."
            )
        return k


# Phase 0 pins one arm per stage. VOX-006 turns this dict into a --flag selection.
STT = Arm(
    repo_id="openai/whisper-large-v3-turbo",
    provider="groq",
    provider_model="whisper-large-v3-turbo",
    api_base="https://api.groq.com/openai/v1",
    key_env="GROQ_API_KEY",
)

LLM = Arm(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    provider="nvidia-nim",
    provider_model="meta/llama-3.1-8b-instruct",
    api_base="https://integrate.api.nvidia.com/v1",
    key_env="NVIDIA_API_KEY",
)

TTS = Arm(
    repo_id="hexgrad/Kokoro-82M",
    provider="local",
    provider_model="hexgrad/Kokoro-82M",
)

# --- audio ---------------------------------------------------------------------------------
# silero-vad and whisper both want 16 kHz mono; Kokoro emits 24 kHz. Resampling happens only
# at the speaker, so no stage silently degrades what the next stage sees.
SAMPLE_RATE = 16_000
VAD_FRAME = 512          # silero requires exactly 512 samples per 16 kHz frame (32 ms)
TTS_SAMPLE_RATE = 24_000
TTS_VOICE = os.environ.get("VOX_TTS_VOICE", "af_heart")

# Pinned, not auto-detected. On a short clip whisper will guess the language from too little
# evidence — a 1.0 s "Hello" came back as French "Salut !" — and the LLM then answers in that
# language. Phase 0 is English only; the multilingual leg is VOX-022 and will set this per case.
STT_LANGUAGE = os.environ.get("VOX_STT_LANGUAGE", "en")

# --- endpointing (VOX-012 lifts these into a config file) ----------------------------------
VAD_SPEECH_THRESHOLD = 0.5     # silero speech probability above which a frame counts as speech
VAD_MIN_SPEECH_MS = 250        # shorter than this is a cough, not a turn
VAD_MAX_UTTERANCE_MS = 15_000  # hard stop so a stuck mic cannot hang the loop

# PROVISIONAL — set from a single recording, not from a dev set. VOX-012 must re-tune this on
# VOX-004's 45 utterances and print the number it chose.
#
# Measured on hello-testing-voice.mp3 ("Hello. So this is testing.", one 1.05 s mid-sentence
# pause), sweeping this value and transcribing what the endpointer produced:
#     700 ms -> 1.02 s segment -> "Bye."                       (endpointed mid-sentence)
#     900 ms -> 1.25 s segment -> "Bye."                       (endpointed mid-sentence)
#    1100 ms -> 3.30 s segment -> "Hello. So this is testing."  (correct)
#    1300 ms -> 3.30 s segment -> "Hello. So this is testing."  (correct, slower)
# 1100 is the smallest value tested that clears a natural pause. It costs ~400 ms of extra
# trailing silence on every turn, which lands in time_to_first_audio when VOX-003 measures it.
VAD_SILENCE_MS = 1_100

CONSENT_NOTICE = (
    "VOX records microphone audio for this turn only. Audio stays on this machine, is sent to "
    "the STT provider for transcription, and is not written to disk. Internal use only — do not "
    "speak customer PII. Ctrl-C to abort."
)
