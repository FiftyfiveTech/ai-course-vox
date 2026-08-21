"""The shared cost/latency logger. Every model call goes through this — no exceptions.

Two logs, deliberately not merged:

  runs/calls.jsonl   one line per *model call* — what it cost and how long the provider took.
  runs/turns.jsonl   one line per *turn* — the five-field latency split (VOX-003).

They answer different questions. A call record cannot tell you where a turn's wall clock went,
because the gaps between calls (endpointing hangover, encoding, handing samples to the device)
belong to no call. A turn record cannot tell you which provider was slow. `turn_id` joins them.

Cost is logged as 0.0 with the tier that justifies it. A non-zero number here means the zero
spend constraint has been broken and the run should stop.
"""
import json
import time
import uuid
from contextlib import contextmanager

from src.config import RUNS_DIR

CALLS_LOG = RUNS_DIR / "calls.jsonl"
TURNS_LOG = RUNS_DIR / "turns.jsonl"

# Free-tier endpoints and local weights. Anything not on this list is a STOP-and-ask.
# ollama is "local-weights" like `local` is: the weights are on this machine and no one is billed.
# It has a name of its own only because it is reached over HTTP rather than loaded in-process.
FREE_TIERS = {"groq": "free-tier", "nvidia-nim": "free-tier", "local": "local-weights",
              "ollama": "local-weights"}


def new_turn_id():
    return uuid.uuid4().hex[:12]


@contextmanager
def log_call(stage, arm, turn_id, **extra):
    """Time one model call and append a record. Re-raises after logging the failure.

    Yields a dict the caller can add measured facts to (chars, tokens, audio seconds) —
    whatever the stage actually knows.
    """
    if arm.provider not in FREE_TIERS:
        raise RuntimeError(
            f"provider {arm.provider!r} for {arm.repo_id} is not a known free tier. Zero spend "
            f"is a hard constraint — stop and ask before calling it."
        )
    record = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "turn_id": turn_id,
        "stage": stage,
        "model_id": arm.repo_id,          # HF repo id, never the provider's string
        "provider": arm.provider,
        "tier": FREE_TIERS[arm.provider],
        "cost_usd": 0.0,
        **extra,
    }
    t0 = time.perf_counter()
    try:
        yield record
    except Exception as e:
        record["ok"] = False
        record["error"] = f"{type(e).__name__}: {e}"
        raise
    else:
        record["ok"] = True
    finally:
        record["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        _append(record)


def _append(record, path=None):
    # Looked up at call time, not bound as a default: a default argument would capture CALLS_LOG at
    # import and quietly ignore a test that redirects it, so an arm test would append to the real log.
    path = path or CALLS_LOG
    RUNS_DIR.mkdir(exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


# --- per-turn latency split (VOX-003) --------------------------------------------------------

STAGES = ("vad", "stt", "llm", "tts")

# The five fields the ticket asks for, in the order a turn produces them. A single total is
# useless — which of these moved is the whole question, so the record always carries all five
# and writes null rather than omitting one, so a missing measurement cannot read as a fast stage.
TURN_FIELDS = tuple(f"t_{s}_ms" for s in STAGES) + ("time_to_first_audio_ms",)


class TurnTimer:
    """Times one turn and writes a single line to runs/turns.jsonl.

    The clock that matters starts when the user stops talking, not when the endpointer notices:
    time_to_first_audio is measured from `Capture.speech_end_t` through to the moment the output
    device pulls its first block. So it contains t_vad's hangover, the three model calls, and the
    glue between them — if the five stage numbers do not roughly add up to it, the gap is real
    work nobody has attributed yet, which is the point of logging both.
    """

    def __init__(self, turn_id, source="mic"):
        self.turn_id = turn_id
        self.source = source              # "mic" for a live turn, else the file it was driven from
        self.ms = dict.fromkeys(STAGES)
        self.speech_end_t = None
        self.first_audio_t = None
        self.extra = {}
        self.error = None
        self.written = None               # the record that reached disk, so prints cannot drift

    def arms(self, **by_stage):
        """Record which arm ran each stage, as `<repo id>@<provider>` (VOX-006).

        Without this a turn line is a latency split with no model behind it, and once arms are
        selectable by flag two lines that look comparable may not be. Joining to calls.jsonl on
        turn_id would also answer it, but the comparison VOX-013 has to make is per turn.

        Named arms and timed stages are not the same set, and that is the point of the check below.
        `embed` is a registered arm that runs inside a turn — it ranks the chunks — but it is not one
        of the five VOX-003 fields, so it gets a `<stage>_model` on the record and no timing slot.
        Anything not in the registry at all is still a typo and still raises: a silently ignored
        stage name would leave a turn line claiming an arm it never named.
        """
        from src.config import ARMS                       # local: config imports nothing from here
        for stage, arm in by_stage.items():
            if stage not in self.ms and stage not in ARMS:
                raise ValueError(
                    f"unknown stage {stage!r} — expected a timed stage {STAGES} or a registered "
                    f"arm stage {tuple(ARMS)}"
                )
            self.extra[f"{stage}_model"] = arm.id if hasattr(arm, "id") else arm

    def fallback(self, stage, from_arm, to_arm, reason, failed_ms=None):
        """A stage did not run on the arm this turn was started with. Rewrite the record to say so.

        `arms()` stamps `<stage>_model` before the turn begins, from what was *selected*. After a
        fallback that field would name an arm that never ran, and VOX-013's per-turn comparison
        reads exactly that field — so the wrong model would get the credit for the latency.

        `<stage>_failed_ms` is the other half. `t_<stage>_ms` now spans the dead remote round-trip
        plus the local call, which is honest about what the user waited for but would otherwise
        attribute a provider timeout to local inference. The breakdown keeps both readable.
        """
        if stage not in self.ms:
            raise ValueError(f"unknown stage {stage!r} — expected one of {STAGES}")
        self.extra[f"{stage}_model"] = to_arm.id
        self.extra[f"{stage}_fallback_from"] = from_arm.id
        self.extra[f"{stage}_fallback_reason"] = reason
        self.extra[f"{stage}_failed_ms"] = failed_ms
        self.extra.setdefault("fell_back", []).append(stage)

    def barge(self, stop_ms, played_s, reply_s, out_latency_s=None):
        """The user talked over this reply and it was cut short (VOX-011).

        `stop_ms` is measured from the first speech frame, not from the moment the decision was
        made, so BARGE_MIN_SPEECH_MS is inside the number rather than hidden behind it. It is the
        interval between the user starting to speak and VOX stopping sending samples.

        `out_latency_s` is the output device's buffer, which `abort()` cannot recall. Recorded
        beside the stop latency because the two together bound what the user actually heard, and
        without it the printed number reads as silence-by-then, which it is not.

        The five-field split is untouched: this turn ran every stage and did reach first audio, so
        it stays in the latency percentiles the phase gates read. Being interrupted is a fact about
        the reply, not a failed turn.
        """
        self.extra["barged_in"] = True
        self.extra["barge_stop_ms"] = stop_ms
        self.extra["played_s"] = played_s
        self.extra["reply_s"] = reply_s
        self.extra["cut_s"] = round(reply_s - played_s, 3)
        # Written even when unknown, for the reason TURN_FIELDS gives: a field that vanishes when it
        # was not measured reads as a zero-length tail, which is the flattering answer.
        self.extra["out_latency_s"] = out_latency_s

    def vad(self, capture):
        """Adopt the endpointer's marks: t_vad, and the origin the whole turn is measured from."""
        self.speech_end_t = capture.speech_end_t
        self.ms["vad"] = capture.t_vad_ms
        self.extra["vad_infer_ms"] = capture.infer_ms
        self.extra["speech_s"] = round(capture.spoken_s, 3)

    @contextmanager
    def stage(self, name):
        """Time one stage as the loop experiences it — the call plus its glue, not just the call.

        This is deliberately wider than the matching calls.jsonl record. The difference between
        the two is encoding, parsing and waiting, which is latency the user feels and no provider
        will report.
        """
        if name not in self.ms:
            raise ValueError(f"unknown stage {name!r} — expected one of {STAGES}")
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.ms[name] = round((time.perf_counter() - t0) * 1000, 1)

    @contextmanager
    def retrieval(self):
        """Time the retrieval step and record it as `t_retrieval_ms` (VOX-032).

        Deliberately *not* a sixth entry in STAGES. TURN_FIELDS, stage_sum_ms and `ok` are all
        derived from STAGES and the phase gates read exactly those, so adding retrieval there
        would redefine the five-field split VOX-003 measured, and would make `ok` false for a turn
        that ran without an index — a turn that spoke perfectly well.

        Timed outside `stage("llm")` and not inside it: BM25 over a few hundred chunks is
        milliseconds against an LLM call of seconds, so folding the two together is how a cheap
        step disappears and a model is blamed for latency it never spent.
        """
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.extra["t_retrieval_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    def grounding(self, hits, grounded, sources=()):
        """What this turn's reply was grounded in (VOX-032).

        `grounded` is the field VOX-033's gate sums into a rate, which is why it is written on
        every turn that got as far as a reply rather than only on the grounded ones — a rate needs
        a denominator that was recorded while the turns were happening.

        `sources` is the provenance of the context that was passed to the model, in Hit.source
        form, so "leave-policy:p4" is spelled here exactly as retrieval and the answer print it.
        Empty on a refusal: see src/answer.py on why a refusal cites nothing.
        """
        self.extra["retrieved"] = len(hits)
        self.extra["grounded"] = bool(grounded)
        self.extra["sources"] = list(sources)
        self.extra["top_score"] = round(hits[0].score, 4) if hits else None

    def first_audio(self):
        """Stamp the first block reaching the speaker. Idempotent — the callback fires per block."""
        if self.first_audio_t is None:
            self.first_audio_t = time.perf_counter()

    def record(self):
        ttfa = None
        if self.speech_end_t is not None and self.first_audio_t is not None:
            ttfa = round((self.first_audio_t - self.speech_end_t) * 1000, 1)
        measured = [self.ms[s] for s in STAGES if self.ms[s] is not None]
        return {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "turn_id": self.turn_id,
            "source": self.source,
            **{f"t_{s}_ms": self.ms[s] for s in STAGES},
            "time_to_first_audio_ms": ttfa,
            "stage_sum_ms": round(sum(measured), 1) if measured else None,
            "ok": self.error is None and None not in self.ms.values() and ttfa is not None,
            **({"error": self.error} if self.error else {}),
            **self.extra,
        }

    def write(self):
        """Append the turn line. -> the record, or None if no turn ever started."""
        if self.speech_end_t is None:
            return None      # the mic stayed quiet; there is no turn to describe
        self.written = self.record()
        _append(self.written, TURNS_LOG)
        return self.written


@contextmanager
def turn_timer(turn_id, source="mic"):
    """One turn's timings, written even if the turn raises — a failed turn still has a latency."""
    t = TurnTimer(turn_id, source)
    try:
        yield t
    except Exception as e:
        t.error = f"{type(e).__name__}: {e}"
        raise
    finally:
        t.write()
