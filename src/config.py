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
                 api_base=None, key_env=None, local=None, **extra):
        self.repo_id = repo_id            # the only name we speak out loud
        self.provider = provider          # where it runs
        self.provider_model = provider_model
        self.backend = backend            # which adapter runs it — the dispatch key in arms.py
        self.alias = alias                # short name for the CLI; the repo id always works too
        self.api_base = api_base
        self.key_env = key_env
        # Whether the weights run on this machine — the thing PIPELINE and FALLBACKS are about.
        # Usually the same question as `provider == "local"`, but not always: the ollama arm runs
        # locally and still speaks HTTP to a daemon on localhost, so it has a provider name of its
        # own and says so explicitly. Nothing infers locality from the provider string any more.
        self.local = (provider == "local") if local is None else local
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

    @property
    def timeout_s(self):
        """How long this arm gets to answer. Local arms get far longer — see LOCAL_TIMEOUT_S."""
        return LOCAL_TIMEOUT_S if self.local else REMOTE_TIMEOUT_S

    def auth_headers(self, extra=None):
        """-> request headers, with Authorization only when this arm has a credential.

        The ollama arm speaks HTTP with no key at all. Sending `Bearer None` at it is the kind of
        thing that works until a server decides to validate the header, so the header is omitted
        rather than filled with a placeholder.
        """
        headers = dict(extra or {})
        k = self.key()
        if k is not None:
            headers["Authorization"] = f"Bearer {k}"
        return headers


GROQ = "https://api.groq.com/openai/v1"
NIM = "https://integrate.api.nvidia.com/v1"
# The ollama daemon's OpenAI-compatible endpoint. Local, but over HTTP — hence an api_base and a
# provider of its own rather than provider="local", which means "loaded in this process".
OLLAMA = os.environ.get("VOX_OLLAMA_HOST", "http://localhost:11434").rstrip("/") + "/v1"

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
    # The stage's local fallback, and the only local LLM arm. Ollama serves the same
    # OpenAI-compatible /chat/completions the two hosted arms speak, so it costs no new adapter —
    # `openai_chat` just has to stop sending an Authorization header it has no key for.
    # Same Llama family as the default on purpose: when the free tier refuses, the reply should
    # sound like a smaller version of the usual voice, not a different assistant.
    Arm(repo_id="hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF", provider="ollama",
        provider_model="hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M",
        backend="ollama-chat", alias="llama-3.2-3b", api_base=OLLAMA, local=True),
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

# --- where each stage runs -------------------------------------------------------------------
# The architecture, written down as something that can fail a test. Until this table existed the
# placement was only a consequence of which row happened to be first in each tuple above, so
# reordering a table moved a stage across the network boundary and nothing said so.
#
# The reasoning behind the placement, so a future reorder is a decision and not a slip:
#   vad   local — runs per 32 ms frame; a network hop per frame is not a design, it is a bill
#   stt   remote — the local `base` arms drop the first word of the fixture (see ARCHITECTURE.md)
#   llm   remote — the widest quality gap of the four, and the least tolerable to lose
#   tts   local — no key, no quota, and the arm is already good enough to ship
PIPELINE = {"vad": "local", "stt": "remote", "llm": "remote", "tts": "local"}

# Where a stage goes when its arm fails in a way another arm could survive. Every value must name
# a *local* arm on the same stage; tests/unit/test_fallback.py asserts exactly that, because a
# fallback that is itself remote would fail for the same reason the primary just did.
FALLBACKS = {"stt": "faster-base", "llm": "llama-3.2-3b", "tts": "speecht5"}

# How long a rate-limited arm stays out of rotation when the provider sent no Retry-After. Long
# enough that a free tier is not poked once per turn, short enough that one 429 does not exile the
# good arm for the rest of a demo.
DEFAULT_COOLDOWN_S = 60.0

# How long a hosted arm gets before the turn gives up on it. Was 30 s for STT and 60 s for the LLM,
# which were fine when a timeout meant the turn was over anyway. Now that a timeout has somewhere to
# go, the wait is pure added latency in front of a local arm that would have answered — against a
# 2 s budget, waiting a minute to find out is worse than being wrong quickly.
REMOTE_TIMEOUT_S = float(os.environ.get("VOX_REMOTE_TIMEOUT_S", "10"))

# A local arm over HTTP gets its own, much longer budget. Giving up on it early is not a fallback,
# it is just a lost turn — there is nowhere further to go, and no free tier to be polite to. Ollama
# also pages a 2 GB model into memory on a cold call, which alone exceeds the remote budget; that
# load is what `nlu.load_ollama` moves out of the turn, and this is the belt to its braces.
LOCAL_TIMEOUT_S = float(os.environ.get("VOX_LOCAL_TIMEOUT_S", "120"))


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
