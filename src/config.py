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

import yaml
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO_ROOT / "runs"
PROMPTS_DIR = REPO_ROOT / "prompts"

# VOX-012: tuning values live in config.yaml, not in code.
_CFG_FILE = REPO_ROOT / "config.yaml"
_cfg = yaml.safe_load(_CFG_FILE.read_text()) if _CFG_FILE.exists() else {}

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
    # The fast leg of VOX-013's TTS contrast, and the only arm here that is actually faster than
    # Kokoro — speecht5 is slower, so a "fast arm" built on it would show no contrast at all.
    #
    # One HF repo holds every piper voice as an .onnx / .onnx.json pair, so the repo id alone does
    # not say what will speak: the voice files are pinned here for the same reason the SpeechT5
    # speaker embedding above is. `sample_rate` is asserted against the .onnx.json at load, so this
    # number cannot drift away from the weights and silently pitch-shift playback.
    Arm(repo_id="rhasspy/piper-voices", provider="local",
        provider_model="rhasspy/piper-voices", backend="piper", alias="piper",
        sample_rate=22_050,
        onnx="en/en_US/lessac/medium/en_US-lessac-medium.onnx",
        onnx_config="en/en_US/lessac/medium/en_US-lessac-medium.onnx.json"),
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

# --- named architectures (VOX-013) -----------------------------------------------------------
# Two *complete* pipelines, so `make compare` measures architectures rather than stages. Named by
# alias, the same way the CLI names an arm, so re-pinning a leg is a table edit here and not a code
# change in the script — the rule the arm tables above already follow.
#
# The contrast is all-local against all-hosted, which is the widest one the registry holds and the
# one PIPELINE's placement decisions are actually about. `fast` also needs no credential and no
# quota, so the comparison still produces a column when a free tier refuses.
#
# Every alias here must name a registered arm on that stage, and the two must differ at all three
# stages; tests/unit/test_compare.py asserts both. A shared leg would make "comparison" a claim
# about one variable while the table shows three.
ARCHITECTURES = {
    "fast":    {"stt": "faster-base", "llm": "llama-3.2-3b", "tts": "piper"},
    "quality": {"stt": "large-v3",    "llm": "llama-70b",    "tts": "kokoro"},
}

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

# --- endpointing + barge-in (VOX-012: values now live in config.yaml) ----------------------
_ep = _cfg.get("endpointing", {})
_bi = _cfg.get("barge_in", {})

VAD_SPEECH_THRESHOLD  = _ep.get("speech_threshold",  0.5)
VAD_MIN_SPEECH_MS     = _ep.get("min_speech_ms",     250)
VAD_MAX_UTTERANCE_MS  = _ep.get("max_utterance_ms",  15_000)
VAD_SILENCE_MS        = _ep.get("silence_ms",        1_100)
BARGE_MIN_SPEECH_MS   = _bi.get("min_speech_ms",     200)
BARGE_SPEECH_THRESHOLD = _bi.get("speech_threshold", 0.7)

# --- source documents (VOX-029) ---------------------------------------------------------------
# The PDF corpus the POC answers from, and where the extracted chunks land. Both are gitignored:
# these are internal HR policies, so the documents and the text pulled out of them are the same
# disclosure either way. `make index` rebuilds the chunk file, so nothing here is precious.
SOURCES_DIR = Path(os.environ.get("VOX_SOURCES_DIR", REPO_ROOT / "sources"))
CHUNKS_FILE = Path(os.environ.get("VOX_CHUNKS_FILE", RUNS_DIR / "chunks.jsonl"))

# Chunk geometry, in *model* tokens rather than words — see TOKENIZER_REPO for why that is worth
# a tokenizer. 300/50 is the ticket's number, kept as config because VOX-030's retrieval quality
# is the thing that decides whether it was right, and that measurement has not happened yet.
CHUNK_TOKENS = int(os.environ.get("VOX_CHUNK_TOKENS", "300"))
CHUNK_OVERLAP_TOKENS = int(os.environ.get("VOX_CHUNK_OVERLAP_TOKENS", "50"))

# Counting tokens with the tokenizer of the model that will read the chunks, so "300 tokens" means
# what VOX-031's prompt budget means by it. Splitting on whitespace would have needed no download
# and would also have made the number a different unit from the one that matters.
#
# The default LLM arm is meta-llama/Llama-3.1-8B-Instruct, whose repo is gated: from_pretrained
# 401s without an HF token, so a clean clone could not build an index. This repo id is a mirror of
# those same Llama-3.1 tokenizer files (128k vocab, byte-level BPE, verified against
# unsloth/Meta-Llama-3.1-8B-Instruct on the same string). It is named as an HF repo id like every
# other model here; it is a tokenizer, so it is never called and costs nothing.
TOKENIZER_REPO = os.environ.get("VOX_TOKENIZER_REPO", "NousResearch/Meta-Llama-3.1-8B-Instruct")

# --- retrieval (VOX-030) -----------------------------------------------------------------------
# How many chunks a query gets back. The ticket's number. Five 300-token chunks is ~1500 tokens of
# context, which is what VOX-031's answer prompt has to fit around — so if this rises, that prompt
# budget is the thing that pays for it.
RETRIEVAL_TOP_K = int(os.environ.get("VOX_RETRIEVAL_TOP_K", "5"))

# The score below which nothing is returned, so "that is not in these documents" is a real answer
# state rather than an empty string. src/retrieval.py divides the raw BM25 sum by the query's own
# ceiling, so the number is a fraction of the query's information content and is comparable between
# a three-word question and a ten-word one. It is still corpus-specific: change the corpus, the chunk
# geometry or the stopword list and re-measure with `uv run python scripts/ask.py --calibrate`.
#
# MEASURED, 2026-08-20, 215 chunks over 15 documents, 13 dev queries (7 answerable, 6 deliberately
# absent) in evals/dev/retrieval_floor_queries.json:
#
#     answerable  min 0.320  max 0.615
#     absent      min 0.126  max 0.234
#
# Separable — every answerable query outscored every absent one, and top-1 landed in a document
# that answers the question on 7 of 7. 0.28 is the midpoint of the [0.234, 0.320] gap, a margin of
# ~0.04 on the tight side. It is a 13-query dev measurement, so it is a starting point and not a
# settled number; VOX-033's gate is what tests it at scale.
#
# The first attempt used raw BM25 and was NOT separable (an absent query scored 9.64 against a
# weakest answerable 7.38) — that failure is why the score is normalised at all.
RETRIEVAL_SCORE_FLOOR = float(os.environ.get("VOX_RETRIEVAL_SCORE_FLOOR", "0.28"))

# Okapi BM25's own two knobs, at rank_bm25's defaults. k1 is how fast term frequency saturates; b is
# how hard a long chunk is penalised for being long. Surfaced here for the same reason CHUNK_TOKENS
# is: retrieval quality is what decides whether the defaults were right for a corpus of short policy
# pages, and that measurement is VOX-033's gate, not something this ticket settled.
BM25_K1 = float(os.environ.get("VOX_BM25_K1", "1.5"))
BM25_B = float(os.environ.get("VOX_BM25_B", "0.75"))

# BM25Okapi floors the IDF of a term appearing in over half the corpus at epsilon * average_idf — a
# positive number, so an ultra-common term still adds score. That inflates every score including a
# miss's, which is exactly what the floor above has to see through. src/retrieval.py's stopword list
# is the first defence; this is the dial if it is not enough.
BM25_EPSILON = float(os.environ.get("VOX_BM25_EPSILON", "0.25"))

CONSENT_NOTICE = (
    "VOX records microphone audio for this turn only. Audio stays on this machine, is sent to "
    "the STT provider for transcription, and is not written to disk. Internal use only — do not "
    "speak customer PII. Ctrl-C to abort."
)
