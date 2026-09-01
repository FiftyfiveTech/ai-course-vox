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
import sys
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
    # 2026-09-01: `meta-llama/Llama-3.1-8B-Instruct` and `-70B-Instruct` were the two arms here and
    # both are gone — NIM retired them on 2026-08-26 and now answers 410 with an end-of-life date.
    # A retired model is not a rate limit and not a bug in the request, so it earns its own failure
    # class (errors.ModelGone) and `scripts/preflight.py` asks both catalogues before a session
    # starts rather than finding out mid-turn.
    #
    # The replacement is the arm that was already here for the provider-swap lesson. gpt-oss is a
    # reasoning model, and with the default effort it spent all 120 tokens thinking and returned an
    # empty reply (measured 2026-08-18), so `reasoning_effort` is not a tuning knob on this arm, it
    # is part of how it has to be called at all. 312-555 ms on two consecutive calls, 29 tokens.
    Arm(repo_id="openai/gpt-oss-120b", provider="groq", provider_model="openai/gpt-oss-120b",
        backend="openai-chat", alias="gpt-oss", api_base=GROQ, key_env="GROQ_API_KEY",
        request={"reasoning_effort": "low"}),
    # The same weights on the other free tier, which is why this row exists rather than a different
    # model on NIM: the arm table's job at this stage is to make one variable movable at a time, and
    # holding the model fixed makes the provider the only difference between this row and the one
    # above. 1766 ms and 5801 ms against Groq's 312-555 ms, so this is the slow leg, not a spare
    # default. It also keeps the stage on two providers — a single-provider stage is what made a
    # model retirement stop a session instead of costing it one arm.
    #
    # `resolve()` refuses a bare `openai/gpt-oss-120b` now that two providers serve it, which is the
    # behaviour it already documented; name an arm by alias or by `repo_id@provider`.
    Arm(repo_id="openai/gpt-oss-120b", provider="nvidia-nim",
        provider_model="openai/gpt-oss-120b", backend="openai-chat", alias="gpt-oss-nim",
        api_base=NIM, key_env="NVIDIA_API_KEY", request={"reasoning_effort": "low"}),
    # The fastest arm measured on either free tier — 259 ms and 272 ms, 16 tokens, and no
    # `reasoning_effort` needed because it does not narrate its thinking into `content` the way
    # `qwen/qwen3.6-27b` does. Kept as the cross-family contrast the 70B row used to provide.
    Arm(repo_id="Qwen/Qwen3.8-27B", provider="groq", provider_model="qwen/qwen3.8-27b",
        backend="openai-chat", alias="qwen3.8", api_base=GROQ, key_env="GROQ_API_KEY"),
    # The size contrast ARCHITECTURE.md open question 1 asks for, now within one family instead of
    # across two: same weights lineage as the default, a sixth of the parameters. 302-408 ms — the
    # 120B is not meaningfully slower on this tier, which is itself the answer to that question.
    Arm(repo_id="openai/gpt-oss-20b", provider="groq", provider_model="openai/gpt-oss-20b",
        backend="openai-chat", alias="gpt-oss-20b", api_base=GROQ, key_env="GROQ_API_KEY",
        request={"reasoning_effort": "low"}),
    # The stage's local fallback, and the only local LLM arm. Ollama serves the same
    # OpenAI-compatible /chat/completions the hosted arms speak, so it costs no new adapter —
    # `openai_chat` just has to stop sending an Authorization header it has no key for.
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


# Sentence encoders for the dense half of retrieval. Local, both of them: an embedding runs once per
# chunk at index time and once per *turn* at query time, so a network hop here would be a network hop
# inside the latency budget, and the corpus is internal HR policy — it does not leave the machine to
# be vectorised.
#
# `pooling` and `query_prefix` are part of *which model this is*, not tuning: BGE was trained with
# CLS pooling and an asymmetric query instruction, MiniLM with mean pooling and none. Pooling one
# the other's way does not degrade it a little, it produces vectors from a space the model was never
# trained to put anything in.
EMBED_ARMS = (
    Arm(repo_id="BAAI/bge-small-en-v1.5", provider="local",
        provider_model="BAAI/bge-small-en-v1.5", backend="transformers-embed", alias="bge-small",
        pooling="cls", dim=384,
        query_prefix="Represent this sentence for searching relevant passages: "),
    # No prefix and mean pooling — the symmetric alternative, and the smaller download. Kept as an
    # arm rather than a comment so the choice is measurable: `LLM=... make ...` has an equivalent
    # here, and two encoders' numbers can be put side by side the way VOX-013 puts arms side by side.
    Arm(repo_id="sentence-transformers/all-MiniLM-L6-v2", provider="local",
        provider_model="sentence-transformers/all-MiniLM-L6-v2", backend="transformers-embed",
        alias="minilm", pooling="mean", dim=384, query_prefix=""),
)

ARMS = {"stt": STT_ARMS, "llm": LLM_ARMS, "tts": TTS_ARMS, "embed": EMBED_ARMS}
STAGE_ENV = {"stt": "VOX_STT_MODEL", "llm": "VOX_LLM_MODEL", "tts": "VOX_TTS_MODEL",
             "embed": "VOX_EMBED_MODEL"}

DEFAULT_STT, DEFAULT_LLM, DEFAULT_TTS = STT_ARMS[0], LLM_ARMS[0], TTS_ARMS[0]
DEFAULT_EMBED = EMBED_ARMS[0]

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
#   embed local — once per chunk at index time and once per turn at query time; and the corpus is
#         internal policy, so vectorising it remotely would be the disclosure the gitignore avoids
PIPELINE = {"vad": "local", "stt": "remote", "llm": "remote", "tts": "local", "embed": "local"}

# Where a stage goes when its arm fails in a way another arm could survive. Every value must name
# a *local* arm on the same stage; tests/unit/test_fallback.py asserts exactly that, because a
# fallback that is itself remote would fail for the same reason the primary just did.
#
# `embed` deliberately has none. A fallback encoder would answer the query in a *different vector
# space* from the one the cached chunk vectors live in, so every cosine it produced would be
# arithmetic between two unrelated bases — not a degraded answer, a meaningless one. If the encoder
# cannot load, the dense half is skipped and BM25 answers alone; that is in src/retrieval.py, not
# here, because it is not a substitution.
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
    "quality": {"stt": "large-v3",    "llm": "gpt-oss",      "tts": "kokoro"},
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

# --- session length (`make demo`) ------------------------------------------------------------
# How long a conversational run lasts when it is bounded by the clock rather than by a turn count.
# Three minutes is a demo: long enough to ask a few things, talk over one of them, and hear the
# session end by itself. Env-configurable because a dev iterating on a stage wants one or two turns
# and not three minutes of talking:
#
#   VOX_SESSION_MINUTES=0.5 make demo
#
# The deadline is only ever checked between turns (src.loop.Budget), so a run overruns by at most
# one reply — a value smaller than a turn takes means one turn, not a truncated one.
SESSION_MINUTES = float(os.environ.get("VOX_SESSION_MINUTES", "3"))

# How many consecutive listens may hear nothing before a timed session gives up. A pause inside a
# conversation is not the end of it, so silence does not stop a timed run the way it stops a
# `--turns` one — but a muted mic, an unplugged headset or a device the OS handed to something else
# all look exactly like a thoughtful pause, and without a bound they would spin quietly for the
# whole three minutes and blame the user for saying nothing. Two, because vad.listen waits 30 s each
# time: a minute of silence is a broken mic or a person who has walked away, and both want the same
# answer. Set it to 1 to get the old "silence ends the run" behaviour inside a timed session.
SESSION_QUIET_LIMIT = int(os.environ.get("VOX_SESSION_QUIET_LIMIT", "2"))

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
#
# 2026-09-01, left deliberately unchanged when the default LLM arm moved off Llama-3.1: this is now
# the tokenizer of a model nothing calls, so "300 tokens" no longer means 300 gpt-oss tokens. Not
# repointed here, because CHUNK_TOKENS and both retrieval floors were measured on chunks cut with
# these 128k-vocab boundaries, and swapping the tokenizer re-cuts every chunk — which moves the
# floors without anyone re-measuring them. Re-cutting the corpus is its own ticket with its own
# `make floors` run, not a side effect of repairing the arm table.
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


# --- dense retrieval and fusion (hybrid) -------------------------------------------------------
# Why there is a dense half at all, measured before it was written. BM25 matches terms, not
# meanings, so on 2026-08-20 a live `make demo` turn asked "how many paternal leaves am I entitled
# to according to policy" and the one chunk that answers it (leave-policy:p12, "ADDITIONAL LEAVES
# 1. Paternity Leave") came back at **rank 110 of 137**: `paternal` is not `paternity` and `leaves`
# is not `leave`, so neither term matched at all. The same target ranks 1 for the query "paternity
# leave". Three lexical variants of one information need, `scripts/ask.py`:
#
#     paternity leave                                                 top-1 0.800   target rank 1
#     how many paternity leaves am I entitled to                      top-1 0.276   target rank 11
#     How many paternal leaves am I entitled to according to policy    top-1 0.140   target rank 110
#
# That is not a floor that needs moving. It is a scorer that cannot see the question.

# Where the chunk vectors are cached. Gitignored with the chunks they belong to, and keyed by the
# encoder that made them — vectors from two encoders are not interchangeable, so the file records
# which arm wrote it and is rebuilt rather than reused when that changes.
EMBEDDINGS_FILE = Path(os.environ.get("VOX_EMBEDDINGS_FILE", RUNS_DIR / "embeddings.npz"))

# How many chunks are encoded per forward pass at index time. Only affects index-build speed and
# peak memory; a query is one text and never batches.
EMBED_BATCH = int(os.environ.get("VOX_EMBED_BATCH", "16"))

# The cosine similarity below which a dense hit is not a hit. The dense analogue of
# RETRIEVAL_SCORE_FLOOR, and it needs one for the same reason: without it every query returns its
# five nearest chunks, and "not in these documents" stops being a state the system can be in.
#
# It is a *cosine*, so unlike the lexical score it has an absolute scale and does not move with the
# number of words in the question — which is the whole property the lexical half lacks.
#
# MEASURED, 2026-08-20, same 13 dev queries, `scripts/ask.py --calibrate`, and the measurement is
# NEGATIVE — this number is a compromise and the docs say so rather than implying a gap that is not
# there:
#
#     answerable  min 0.676  max 0.799
#     absent      min 0.620  max 0.738     <- "how many days of sabbatical leave can I take"
#
# NOT separable. An absent query outscores an answerable one, so no cosine floor routes all 13
# correctly. Three other dense signals were tried and none separates either: the top-1 z-score
# against the corpus (2.14 | 2.67), its margin over the mean (0.135 | 0.159), and the gap to the
# 6th best chunk (0.015 | 0.047). A 33M-parameter encoder does not know when it is lost.
#
# 0.65 is therefore chosen, not derived: below every answerable query's cosine with 0.026 of margin,
# and above two of the six absent ones (tuition 0.620, canteen 0.621), which are refused with no
# model call at all. The other four reach the model and the grounded prompt refuses them there —
# measured end to end below.
#
# The alternative was 0.674, which sits under the weakest answerable query (0.676) and over five of
# the six absent ones. It was rejected: 0.002 of margin is not a threshold, it is a coincidence that
# survives until someone rephrases a question. The cost of being wrong on this side is a false
# refusal to a real employee; the cost on the other side is one free-tier call that ends in the
# right refusal. Those are not symmetric, and the floor is set accordingly.
#
# That is also the design working as written rather than a hole in it: this floor is a cheap
# pre-filter, and prompts/answer_from_source_v1.md is what actually stops an ungrounded answer.
DENSE_SCORE_FLOOR = float(os.environ.get("VOX_DENSE_SCORE_FLOOR", "0.65"))

# Reciprocal-rank-fusion constant. A chunk's fused score is sum over the two rankings of
# 1/(RRF_K + rank), so the two halves are combined by *rank* and never by score — a BM25 fraction
# and a cosine are different units, and adding them with weights would be inventing an exchange
# rate between them. 60 is the value the RRF paper uses; it flattens the difference between rank 1
# and rank 2 relative to the difference between rank 1 and rank 20, which is the behaviour wanted
# when one half is confident and the other has no idea.
RRF_K = float(os.environ.get("VOX_RRF_K", "60"))

# How many candidates each half contributes to the fusion before the top-k cut. Wider than
# RETRIEVAL_TOP_K on purpose: a chunk the dense half ranks 8th and BM25 ranks 3rd should be able to
# win, and it cannot if each half only ever offers five.
FUSION_CANDIDATES = int(os.environ.get("VOX_FUSION_CANDIDATES", "20"))

# Whether the dense half runs at all. Off means BM25 alone — exactly VOX-030's behaviour, kept
# reachable because it is the baseline every hybrid number is measured against, and because it is
# the fallback when the encoder cannot load (no weights on this machine, no network on first run).
HYBRID_RETRIEVAL = os.environ.get("VOX_HYBRID_RETRIEVAL", "1") not in ("0", "false", "False", "")

# --- conversation history (VOX-034) ------------------------------------------------------------
# How many previous turns a session keeps. Three, because the thing history is for here is
# resolving a reference — "it", "those", "and privilege leaves" — and a reference reaches back one
# or two turns in speech, not ten. A longer window costs nothing in latency (the retry is arithmetic
# over a 331 KB vector file) but it does widen what a rewritten query can drag in, and a query
# rewritten from a topic three turns dead is worse than no rewrite at all.
HISTORY_TURNS = int(os.environ.get("VOX_HISTORY_TURNS", "3"))

# Whether history is used at all. Off is exactly pre-VOX-034 behaviour, kept reachable for the same
# reason HYBRID_RETRIEVAL is: it is the baseline column the follow-up gate measures against, and an
# A/B whose "before" arm has to be recovered from git history is not an A/B anyone re-runs.
HISTORY_ENABLED = os.environ.get("VOX_HISTORY", "1") not in ("0", "false", "False", "")

# --- the one inferred constant (VOX-034, decision reversed 2026-08-24) --------------------------
# The encashment formula in leave-policy:p7 divides by "number of days within a year" and the corpus
# never says what that number is. VOX-034 first decided to refuse rather than assume, so every
# encashment question stated the rule and computed nothing unless the person volunteered the figure.
# That decision was reversed: assume 365.
#
# What the reversal costs, recorded here because the code cannot warn about it at run time. 365 is
# not in the documents, so a figure computed with it carries an assumption the person is never told
# about — and the assumption is wrong one year in four. A leap year makes the same balance worth
# slightly more than this arithmetic says. The number is env-configurable so a leap-year run is a
# flag rather than an edit, but nothing detects the year for you.
#
# It is ONE named constant and not a general licence to infer, which is the whole boundary: an
# operand only ever traces to a constant when its name says it is counting days in a year (see
# figures.allowed_constant). Anything else the model supplies from general knowledge still fails to
# trace, so "which constants" cannot quietly become "any constant".
DAYS_IN_YEAR = float(os.environ.get("VOX_DAYS_IN_YEAR", "365"))

# How much of a quoted formula's distinctive vocabulary must appear in the excerpts before the
# arithmetic is trusted (figures.formula_grounded). Operand tracing checks where each NUMBER came
# from and says nothing about whether the SUM is the one the documents state — the gap that let a
# live turn invent "(eligible_balance - 24) * basic_salary", compute 80000 and say it aloud with
# every operand traced. High on purpose: a formula is a short, specific string, so an honest quote
# scores near 1.0, and the cost of being wrong is a confident wrong number about someone's pay.
FORMULA_OVERLAP = float(os.environ.get("VOX_FORMULA_OVERLAP", "0.7"))

# --- the date path (VOX-034 part D) --------------------------------------------------------------
# Two assumptions the corpus forces, both the same shape as DAYS_IN_YEAR above: needed to answer at
# all, absent from every document, and therefore said out loud in the reply rather than buried here.
#
# A person speaking says "my last working day is the eighteenth of August" and never says the year.
# The corpus is a 2026 corpus — every policy is "with effect from 01 January 2026" and the only
# holiday calendar in it is 2026 — so a bare date is read as that year. What this costs: a question
# asked in December about next March gets the wrong year, silently, unless VOX_ANCHOR_YEAR is set.
# `dates.spoken()` names the year it used for exactly that reason.
ANCHOR_YEAR = int(os.environ.get("VOX_ANCHOR_YEAR", "2026"))

# "Within 30-45 working days", "after 21 working days" — no document in this corpus defines a working
# day. Weekends are excluded under any reading. Whether the company's own published holidays also
# come out is a judgement, and it moves a real answer: 45 working days from 18 August 2026 is
# 20 October counting weekends only and 22 October once Gandhi Jayanti and Dusshera come out.
#
# On, because the holiday list is a document we have rather than a fact we are inventing — but the
# dates come from the EXCERPTS RETRIEVED FOR THE TURN and never from a table hardcoded here. So a
# turn that never retrieved the calendar counts weekends only and says so, and a turn that did
# retrieve it counts the holidays it actually read. The alternative — a holiday list in config —
# would make the figure depend on a number no listener can trace to a source.
WORKING_DAYS_SKIP_HOLIDAYS = os.environ.get(
    "VOX_WORKING_DAYS_SKIP_HOLIDAYS", "1") not in ("0", "false", "False", "")

CONSENT_NOTICE = (
    "VOX records microphone audio for this turn only. Audio stays on this machine, is sent to "
    "the STT provider for transcription, and is not written to disk. Internal use only — do not "
    "speak customer PII. Ctrl-C to abort."
)


def utf8_console():
    """Make this process's stdout and stderr encode anything. Call it from an entry point.

    A function and not an import-time side effect, because a library import has no business
    rewriting the caller's streams — pytest replaces both with objects of its own, and a `make demo`
    piped into `tee` is a different stream again.

    The reason it exists is a turn that died: a Windows console encodes to cp1252 unless something
    has changed it, `print` raises `UnicodeEncodeError` rather than dropping a character it cannot
    map, and a turn prints two things it does not control — the transcript and the reply. A single
    non-cp1252 character out of Whisper or the model therefore kills the turn *after* every stage
    has succeeded, and it does it at the console, where no fallback and no retry can see it. The
    VOX-026 dry run hit the same failure on a character the repo *does* control (a box-drawing dash
    in VOX-011's barge-in line, since replaced); the general case is not fixable by choosing nicer
    glyphs, only by saying what the stream encodes.

    `errors="replace"` rather than the default: a demo losing one character to a `?` is nothing, and
    losing the turn is the thing this whole ticket is about.
    """
    for stream in (sys.stdout, sys.stderr):
        # Absent under pytest's capture and on any stream that is not a TextIOWrapper.
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
