"""The one interface every model call goes through (VOX-006).

    stt(audio, model_id)  -> transcript
    llm(msgs,  model_id)  -> reply text
    tts(text,  model_id)  -> Speech(audio, sample_rate)
    embed(texts, model_id) -> (n, dim) unit vectors

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

Because dispatch happens here and only here, this is also where a stage falls back to its local arm
when the remote one fails in a way another arm could survive. See `_call`.
"""
import sys
from collections import namedtuple

from src import (cooldown, embeddings as embed_mod, errors, nlu, stt as stt_mod,
                 tts as tts_mod, vocab_bias)
from src.config import (ARMS, DEFAULT_COOLDOWN_S, FALLBACKS, SAMPLE_RATE, STT_LANGUAGE,
                        resolve)
from src.telemetry import log_call

# TTS arms do not agree on a sample rate, so synthesis returns the rate it actually produced
# instead of leaving the speaker to assume one.
Speech = namedtuple("Speech", "audio sample_rate")

_MODULES = {"stt": stt_mod, "llm": nlu, "tts": tts_mod, "embed": embed_mod}


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


# Stages whose fallback is warmed up front. The remote ones, because those are the stages that
# actually fall back — and a cold fallback would pay a model load inside a timed turn, which is the
# exact thing warm() exists to prevent.
#
# Not tts: it is already local, so its fallback only covers a broken Kokoro install, and SpeechT5 is
# ~650 MB. Loading that on every `make demo` to insure against a setup problem costs real startup
# time on every run for a failure that should be fixed rather than routed around.
WARM_FALLBACK_STAGES = ("stt", "llm")


def select(args, warm_up=True):
    """-> {stage: Arm} from parsed --stt/--llm/--tts, loading local weights before anything is timed."""
    chosen = {}
    for stage in ARMS:
        model_id = getattr(args, stage, None)
        chosen[stage] = warm(stage, model_id) if warm_up else resolve(stage, model_id)
    if warm_up:
        warm_fallbacks(chosen)
    return chosen


def describe(chosen):
    """The startup banner: what runs each stage, where, and what covers it. -> one string.

    Lives here rather than in each entry point because `make demo`, the fixture harness and the
    phase gate all have to say the same thing. They had a copy each, which is one copy per chance
    for the gate to describe a pipeline the loop does not actually run.
    """
    lines = [f"  {'vad':<5} snakers4/silero-vad  (local, silero)"]
    for stage, arm in chosen.items():
        where = "local" if arm.local else f"remote via {arm.provider}"
        lines.append(f"  {stage:<5} {arm.repo_id}  ({where}, {arm.backend})")
        fb = fallback_for(stage, arm)
        if fb is not None:
            lines.append(f"  {'':<5}   fallback -> {fb.repo_id} (local, {fb.backend})")
    return "\n".join(lines)


def warm_fallbacks(chosen):
    """Load the local arms the remote stages fall back to. -> {stage: Arm} that were warmed.

    A failure here is reported and swallowed on purpose. Not being able to warm the fallback is a
    reason to say so — it is the difference between a rate limit costing a turn and costing the
    session — but it is not a reason to refuse to start when the remote arms are fine.
    """
    warmed = {}
    for stage in WARM_FALLBACK_STAGES:
        fb = fallback_for(stage, chosen[stage])
        if fb is None:
            continue
        try:
            warmed[stage] = warm(stage, fb.id)
        except Exception as e:
            print(f"  warning: {stage} fallback {fb.repo_id} is not ready ({type(e).__name__}: "
                  f"{e}).\n  The remote arm still works; a fallback would fail.", file=sys.stderr)
    return warmed


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


def fallback_for(stage, arm, enabled=True):
    """-> the local Arm this stage falls back to, or None if `arm` has nowhere to go.

    None when the stage declares no fallback, when `arm` already *is* the fallback — an arm cannot
    rescue itself, and pretending otherwise would double every local failure — or when the caller
    asked for no fallback at all.

    `enabled=False` exists for measurement. `scripts/check_arms.py` times each arm individually, so
    a silent substitution there would print the local arm's latency on the remote arm's row and call
    it that model's number. Anything comparing arms has to see the failure, not be rescued from it.
    """
    alias = FALLBACKS.get(stage)
    if not enabled or alias is None:
        return None
    fb = resolve(stage, alias)
    return None if fb.id == arm.id else fb


def _dispatch(stage, arm, payload, turn_id, extra, sink=None, options=None):
    """Run one arm through the logger. `sink` receives the call record, failure included.

    The record is handed out rather than returned because on a failure there is no return — and the
    failed attempt's latency is exactly what the turn record needs. `log_call` stamps `latency_ms`
    in its `finally`, so by the time the exception reaches `_call` the dict in `sink` already
    carries the same number that reached calls.jsonl. One measurement, two readers.
    """
    fn = _backend(stage, arm)
    # `prompt` is an STT-only kwarg; strip it from the logging dict and pass directly to the backend.
    backend_kw = {}
    log_extra = dict(extra)
    if "prompt" in log_extra:
        backend_kw["prompt"] = log_extra.pop("prompt")
    with log_call(stage, arm, turn_id, **log_extra) as rec:
        if sink is not None:
            sink.append(rec)
        # `options` are call parameters the *backend* takes (a temperature, a timeout), as opposed
        # to `extra`, which are facts about the call for the log. They are kept apart because they
        # travel in opposite directions: one goes to the provider, the other to calls.jsonl.
        # `backend_kw` is the same kind of parameter arriving the other way round: callers pass
        # `prompt` in with `extra`, and it is split back out above.
        return fn(arm, payload, rec, **backend_kw, **(options or {}))


def _call(stage, arm, payload, turn_id, on_fallback=None, fallback=True, options=None, **extra):
    """Run `arm`; on a transient failure run the stage's local arm instead.

    -> (result, the arm that actually produced it). Callers need the second value because an arm is
    not only a name on a log line — a TTS arm carries the sample rate its samples are at, and
    reading that off the arm we *asked for* rather than the one that answered would pitch-shift
    every fallback reply.

    What counts as transient is `errors.is_transient` and nothing else; a bad key or a malformed
    request raises exactly as it did before. Both attempts write their own calls.jsonl line, so
    nothing is swallowed — the failed one with ok:false, the local one marked `fallback_for`.

    If the fallback also fails, the *original* exception is what propagates. The free tier refusing
    is the fact worth reporting; a local arm failing behind it is a second symptom of the same turn.
    """
    fb = fallback_for(stage, arm, fallback)

    # Skip the call entirely while the provider's own Retry-After window is still open. Without
    # this, a rate-limited session pays a doomed round-trip on every single turn.
    parked = cooldown.blocked(arm)
    if parked and fb is not None:
        reason = f"{arm.id} is rate-limited for another {parked:g}s"
        return _run_fallback(stage, arm, fb, payload, turn_id, extra, on_fallback,
                             reason, None, None, options)

    attempt = []
    try:
        return _dispatch(stage, arm, payload, turn_id, extra, attempt, options), arm
    except Exception as e:
        if fb is None or not errors.is_transient(e):
            raise
        if isinstance(e, errors.RateLimited):
            cooldown.block(arm, e.retry_after or DEFAULT_COOLDOWN_S)
        failed_ms = attempt[0].get("latency_ms") if attempt else None
        return _run_fallback(stage, arm, fb, payload, turn_id, extra, on_fallback,
                             f"{type(e).__name__}: {e}", failed_ms, e, options)


def _run_fallback(stage, arm, fb, payload, turn_id, extra, on_fallback, reason, failed_ms,
                  original, options=None):
    """Run `fb` in place of `arm`, loudly. -> (result, fb)."""
    print(f"{stage.upper()} FALLBACK — {reason}\n"
          f"  running {fb.repo_id} ({fb.provider}) locally instead of {arm.repo_id}.",
          file=sys.stderr, flush=True)
    if on_fallback is not None:
        on_fallback(stage, arm, fb, reason, failed_ms)
    try:
        result = _dispatch(stage, fb, payload, turn_id, {**extra, "fallback_for": arm.id},
                           options=options)
    except Exception as fb_error:
        if original is None:
            raise
        raise original from fb_error
    return result, fb


def stt(audio, model_id=None, *, turn_id, on_fallback=None, fallback=True):
    """-> transcript text. Raises on a provider error rather than returning a plausible blank."""
    extra = {"audio_s": round(len(audio) / SAMPLE_RATE, 3), "language": STT_LANGUAGE}
    bias = vocab_bias.prompt_if_enabled()
    if bias:
        extra["prompt"] = bias
    result, _arm = _call("stt", resolve("stt", model_id), audio, turn_id, on_fallback, fallback,
                         **extra)
    return result


def llm(msgs, model_id=None, *, turn_id, on_fallback=None, fallback=True, temperature=None,
        json_mode=False, max_tokens=None, **extra):
    """-> the assistant's reply. `msgs` is an OpenAI-shaped message list; see nlu.messages().

    `temperature` reaches the backend rather than the log: a grounded answer asks for 0 and a spoken
    reply keeps nlu.TEMPERATURE. It is also recorded, because two turns sampled differently are not
    comparable and a latency table has no way to know.

    `json_mode` and `max_tokens` are the same kind of thing (VOX-034): call options, not arm
    properties, because one arm serves both the spoken reply and the figure extractor and only the
    second one wants a parseable object and a bigger ceiling. They go through `options` so every arm
    reaches them through its own backend rather than through a special case here.
    """
    options = {} if temperature is None else {"temperature": temperature}
    if json_mode:
        options["json_mode"] = True
    if max_tokens is not None:
        options["max_tokens"] = max_tokens
    result, _arm = _call("llm", resolve("llm", model_id), msgs, turn_id, on_fallback, fallback,
                         options=options, messages=len(msgs),
                         temperature=nlu.TEMPERATURE if temperature is None else temperature,
                         **extra)
    return result


def tts(text, model_id=None, *, turn_id, on_fallback=None, fallback=True):
    """-> Speech(float32 mono samples, sample rate) — not played; audio.py does that.

    The rate comes off the arm that *answered*. Kokoro is 24 kHz and SpeechT5 16 kHz, so taking it
    from the arm we asked for would play every fallback reply 1.5x fast.
    """
    arm = resolve("tts", model_id)
    audio, ran = _call("tts", arm, text, turn_id, on_fallback, fallback,
                       chars=len(text), sample_rate=arm.extra["sample_rate"])
    return Speech(audio, ran.extra["sample_rate"])


def embed(texts, model_id=None, *, turn_id, is_query=False, fallback=False, **extra):
    """-> (n, dim) float32 unit vectors for `texts`, from the named encoder or the stage default.

    `fallback` defaults to **False**, which is the opposite of every other arm here, and it is not
    an oversight: an encoder's output only means anything against vectors from the same encoder. A
    substitution would answer the query in one vector space and compare it against a cached index
    built in another, and every cosine that came out would be arithmetic between unrelated bases —
    silently, since a meaningless cosine is still a number between -1 and 1. `config.FALLBACKS` has
    no `embed` entry for the same reason; this argument exists only so the signature does not lie
    about what `_call` supports.

    `is_query=True` prepends the arm's query instruction for an asymmetric encoder. Chunks are
    encoded without it. See src/embeddings.py.
    """
    arm = resolve("embed", model_id)
    payload = {"texts": texts, "is_query": is_query}
    result, _ran = _call("embed", arm, payload, turn_id, None, fallback, **extra)
    return result
