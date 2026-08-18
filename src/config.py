"""Model registry and runtime constants.

Models are named by **Hugging Face repo id** everywhere in this codebase. A provider is only
*where the weights run*, so the provider's own model string is a lookup detail that lives here
and nowhere else. If you find a provider string anywhere outside this file, it is a bug.

VOX-006 turned the one-arm-per-stage pins into the tables below. A stage's arms are ordered, and
the first entry is that stage's default, so `make demo` with no flags runs exactly what VOX-002 and
VOX-003 measured. `resolve()` is the only way code reaches an arm; `src/arms.py` is the only
caller.
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

    def __init__(self, repo_id, provider, provider_model, backend, alias,
                 api_base=None, key_env=None, **extra):
        self.repo_id = repo_id            # the only name we speak out loud
        self.provider = provider          # where it runs
        self.provider_model = provider_model
        self.backend = backend            # which adapter runs it — the dispatch key in arms.py
        self.alias = alias                # short name for the CLI; the repo id always works too
        self.api_base = api_base
        self.key_env = key_env
        # Backend-specific pins that are still part of *which model this is* — a vocoder repo, a
        # quantisation, a native sample rate. They belong to the registry for the same reason the
        # provider's model string does: so no other file has to name them.
        self.extra = extra

    @property
    def id(self):
        """The unambiguous name: a repo id can be served by more than one provider."""
        return f"{self.repo_id}@{self.provider}"

    def __repr__(self):
        return f"Arm({self.id})"

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


GROQ = "https://api.groq.com/openai/v1"
NIM = "https://integrate.api.nvidia.com/v1"

# --- the arms (VOX-006) --------------------------------------------------------------------
# First entry per stage is the default, and the defaults are the arms VOX-002/VOX-003 measured.
# Two arms sharing a `backend` share an adapter: adding whisper-large-v3 or Llama-3.3-70B was a
# table row, not new code. Every provider here must appear in telemetry.FREE_TIERS or the first
# call raises — zero spend is checked in one place, not remembered in eight.

STT_ARMS = (
    Arm(repo_id="openai/whisper-large-v3-turbo", provider="groq",
        provider_model="whisper-large-v3-turbo", backend="openai-audio", alias="turbo",
        api_base=GROQ, key_env="GROQ_API_KEY"),
    # Same family, 32 decoder layers instead of 4. The turbo-vs-full contrast is the one VOX-013
    # measures; both stay multilingual, which distil-large-v3-en would not for VOX-022.
    Arm(repo_id="openai/whisper-large-v3", provider="groq",
        provider_model="whisper-large-v3", backend="openai-audio", alias="large-v3",
        api_base=GROQ, key_env="GROQ_API_KEY"),
    # Local, so a turn can run with no network and no key. transformers arrives with kokoro.
    Arm(repo_id="openai/whisper-base", provider="local",
        provider_model="openai/whisper-base", backend="transformers-whisper", alias="whisper-base"),
    # The same weights through CTranslate2 — the int8 conversion, hence a Systran repo id rather
    # than an openai/ one. Paired with the arm above it measures runtime, not model.
    Arm(repo_id="Systran/faster-whisper-base", provider="local",
        provider_model="Systran/faster-whisper-base", backend="faster-whisper",
        alias="faster-base", compute_type="int8"),
)

LLM_ARMS = (
    Arm(repo_id="meta-llama/Llama-3.1-8B-Instruct", provider="nvidia-nim",
        provider_model="meta/llama-3.1-8b-instruct", backend="openai-chat", alias="llama-8b",
        api_base=NIM, key_env="NVIDIA_API_KEY"),
    # The provider swap at this stage. gpt-oss is a reasoning model, and every chat model Groq's
    # free tier now serves is: with the default effort it spent all 120 tokens thinking and
    # returned an empty reply, measured 2026-08-18. `reasoning_effort` is therefore not a tuning
    # knob here, it is part of how this arm has to be called at all.
    Arm(repo_id="openai/gpt-oss-120b", provider="groq", provider_model="openai/gpt-oss-120b",
        backend="openai-chat", alias="gpt-oss", api_base=GROQ, key_env="GROQ_API_KEY",
        request={"reasoning_effort": "low"}),
    # The size contrast ARCHITECTURE.md open question 1 asks for. It wanted Llama-3.3-70B, but
    # that model is not on this Groq key's catalogue (404) and NIM did not answer it inside 120 s
    # on two attempts; 3.1-70B on NIM answered in 4.9 s. Slowest arm here by ~7x — see `make arms`.
    Arm(repo_id="meta-llama/Llama-3.1-70B-Instruct", provider="nvidia-nim",
        provider_model="meta/llama-3.1-70b-instruct", backend="openai-chat", alias="llama-70b",
        api_base=NIM, key_env="NVIDIA_API_KEY"),
)

TTS_ARMS = (
    Arm(repo_id="hexgrad/Kokoro-82M", provider="local", provider_model="hexgrad/Kokoro-82M",
        backend="kokoro", alias="kokoro", sample_rate=24_000),
    # SpeechT5 needs a vocoder and a 512-dim speaker embedding; both are model identity, so they
    # are pinned here. The embedding is pinned by *filename*, not by the index 7306 every tutorial
    # uses: an index is a position in whatever order a loader happened to produce, and this arm's
    # voice has to be the same on every run or an A/B against Kokoro compares two things at once.
    # (For the record, sorted-order index 7306 is this file — cmu_us_slt, US female.)
    Arm(repo_id="microsoft/speecht5_tts", provider="local",
        provider_model="microsoft/speecht5_tts", backend="speecht5", alias="speecht5",
        sample_rate=16_000, vocoder="microsoft/speecht5_hifigan",
        xvector_repo="Matthijs/cmu-arctic-xvectors", xvector_zip="spkrec-xvect.zip",
        xvector_file="spkrec-xvect/cmu_us_slt_arctic-wav-arctic_a0508.npy"),
)

ARMS = {"stt": STT_ARMS, "llm": LLM_ARMS, "tts": TTS_ARMS}
STAGE_ENV = {"stt": "VOX_STT_MODEL", "llm": "VOX_LLM_MODEL", "tts": "VOX_TTS_MODEL"}

DEFAULT_STT, DEFAULT_LLM, DEFAULT_TTS = STT_ARMS[0], LLM_ARMS[0], TTS_ARMS[0]


def resolve(stage, model_id=None):
    """-> the Arm named by `model_id`, or the stage default.

    Accepts the unambiguous `repo/id@provider`, a bare `repo/id`, or the short alias. A bare repo
    id served by two providers is refused rather than guessed — the same Llama can be a NIM arm or
    a Groq arm, and a silently chosen provider would attach the wrong latency to the right name.
    """
    if stage not in ARMS:
        raise ValueError(f"unknown stage {stage!r} — expected one of {tuple(ARMS)}")
    arms = ARMS[stage]

    if model_id is None:
        model_id = os.environ.get(STAGE_ENV[stage]) or None
    if model_id is None:
        return arms[0]

    want = model_id.strip()
    for arm in arms:
        if want in (arm.id, arm.alias):
            return arm

    hits = [a for a in arms if a.repo_id == want]
    if len(hits) == 1:
        return hits[0]
    if hits:
        raise RuntimeError(
            f"{want} is served by more than one provider for the {stage} stage. Name one of "
            f"{', '.join(a.id for a in hits)} instead."
        )
    raise RuntimeError(
        f"no {stage} arm called {want!r}. Known {stage} arms:\n" +
        "\n".join(f"    {a.id}  (alias {a.alias})" for a in arms)
    )

# --- audio ---------------------------------------------------------------------------------
# silero-vad and whisper both want 16 kHz mono; Kokoro emits 24 kHz. Resampling happens only
# at the speaker, so no stage silently degrades what the next stage sees.
SAMPLE_RATE = 16_000
VAD_FRAME = 512          # silero requires exactly 512 samples per 16 kHz frame (32 ms)
# Playback default only. Each TTS arm declares its own rate (Kokoro 24 kHz, SpeechT5 16 kHz) and
# synthesis returns it alongside the samples, so nothing has to assume this one.
TTS_SAMPLE_RATE = DEFAULT_TTS.extra["sample_rate"]
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
