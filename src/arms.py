"""The one interface every model call goes through (VOX-006).

    stt(audio, model_id)  -> transcript
    llm(msgs,  model_id)  -> reply text
    tts(text,  model_id)  -> Speech(audio, sample_rate)

`model_id` is a Hugging Face repo id — `openai/whisper-base`, or `repo/id@provider` when two
providers serve the same weights, or the short alias for typing at a prompt. None means the stage
default, so a caller that does not care about arms reads exactly as it did before this ticket.

Three things happen here and nowhere else:

  resolve  the flag or env var becomes an Arm (config.py owns the table)
  log      one `log_call` around dispatch, so every arm is logged identically and the logged
           model_id is the arm's repo id by construction rather than by each author remembering
  dispatch arm.backend picks the adapter out of the stage module

Backends have the signature `fn(arm, payload, rec) -> result` and add their own measured facts to
the mutable `rec`. That is the whole contract — a new arm is a row in config.py, and a new *way of
running* one is one function plus one BACKENDS entry.
"""
from collections import namedtuple

from src import nlu, stt as stt_mod, tts as tts_mod
from src.config import ARMS, SAMPLE_RATE, STT_LANGUAGE, resolve
from src.telemetry import log_call

# TTS arms do not agree on a sample rate, so synthesis returns the rate it actually produced
# instead of leaving the speaker to assume one.
Speech = namedtuple("Speech", "audio sample_rate")

_MODULES = {"stt": stt_mod, "llm": nlu, "tts": tts_mod}


def available(stage=None):
    """-> every registered Arm, or just one stage's. The first per stage is that stage's default."""
    stages = (stage,) if stage else tuple(ARMS)
    return [arm for s in stages for arm in ARMS[s]]


def add_flags(parser):
    """Add --stt/--llm/--tts to an argparse parser.

    Lives here rather than in each entry point because `make demo` and the mic-free fixture harness
    have to offer the same flags — a difference between them would make the offline number describe
    a run the live loop cannot reproduce.
    """
    for stage, stage_arms in ARMS.items():
        parser.add_argument(
            f"--{stage}", metavar="MODEL_ID", default=None,
            help=f"{stage} arm; default {stage_arms[0].repo_id}. " +
                 "One of: " + ", ".join(f"{a.repo_id} ({a.alias})" for a in stage_arms),
        )
    return parser


def select(args, warm_up=True):
    """-> {stage: Arm} from parsed --stt/--llm/--tts, loading local weights before anything is timed."""
    chosen = {}
    for stage in ARMS:
        model_id = getattr(args, stage, None)
        chosen[stage] = warm(stage, model_id) if warm_up else resolve(stage, model_id)
    return chosen


def _backend(stage, arm):
    backends = _MODULES[stage].BACKENDS
    if arm.backend not in backends:
        raise RuntimeError(
            f"{arm.id} names backend {arm.backend!r}, which {_MODULES[stage].__name__} does not "
            f"implement. Known {stage} backends: {', '.join(sorted(backends))}."
        )
    return backends[arm.backend]


def warm(stage, model_id=None):
    """Load a local arm's weights before anything is timed. -> the resolved Arm.

    Kokoro takes ~10 s to load and SpeechT5 more on a cold cache. Leaving that inside the turn
    would bury it in t_tts and make the VOX-003 latency split a lie. A no-op for hosted arms.
    """
    arm = resolve(stage, model_id)
    loader = _MODULES[stage].LOADERS.get(arm.backend)
    if loader is not None:
        loader(arm)
    return arm


def _call(stage, arm, payload, turn_id, **extra):
    fn = _backend(stage, arm)
    with log_call(stage, arm, turn_id, **extra) as rec:
        return fn(arm, payload, rec)


def stt(audio, model_id=None, *, turn_id):
    """-> transcript text. Raises on a provider error rather than returning a plausible blank."""
    return _call("stt", resolve("stt", model_id), audio, turn_id,
                 audio_s=round(len(audio) / SAMPLE_RATE, 3), language=STT_LANGUAGE)


def llm(msgs, model_id=None, *, turn_id, **extra):
    """-> the assistant's reply. `msgs` is an OpenAI-shaped message list; see nlu.messages()."""
    return _call("llm", resolve("llm", model_id), msgs, turn_id, messages=len(msgs), **extra)


def tts(text, model_id=None, *, turn_id):
    """-> Speech(float32 mono samples, sample rate) — not played; audio.py does that."""
    arm = resolve("tts", model_id)
    sample_rate = arm.extra["sample_rate"]
    audio = _call("tts", arm, text, turn_id, chars=len(text), sample_rate=sample_rate)
    return Speech(audio, sample_rate)
