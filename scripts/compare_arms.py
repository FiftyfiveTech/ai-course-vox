"""Two complete architectures, end to end, with the five-field latency split for both (VOX-013).

    uv run python scripts/compare_arms.py                  both arms, 3 turns each
    uv run python scripts/compare_arms.py --repeat 1
    uv run python scripts/compare_arms.py --arm fast
    uv run python scripts/compare_arms.py --fixture evals/dev/utt_006_entity.wav

`make arms` already times arms one stage at a time — eight independent single-stage calls that
deliberately write no turn record. That cannot answer what a whole *architecture* costs, because the
gaps between the calls belong to no call. This runs both pipelines as real turns and reads the
VOX-003 split off runs/turns.jsonl.

The arm sets are `config.ARCHITECTURES`, named by alias exactly as the CLI names an arm, so re-pinning
a leg is a table edit in config.py and not a change here.

Four things this script does on purpose, each of which would otherwise put a lie in the table:

  no fallback     `fallback=False` throughout, for the reason `arms.fallback_for` gives: a rescued
                  remote arm prints the local arm's latency on the remote arm's row, and attributing
                  numbers to models is the entire point. A refused stage shows as FAILED.
  endpointed per turn, then checked
                  it is tempting to endpoint the fixture once and hand the same `Capture` to every
                  turn, so no arm sees a different segment. That silently breaks the fifth stage: a
                  `Capture` carries the `perf_counter` stamp of when its speech ended, and
                  `time_to_first_audio` is measured from it — so a reused capture measures every
                  later turn from a moment further and further in the past. It read 6.5 s, 39.4 s,
                  63.6 s across three identical turns before this was caught. So each turn
                  endpoints for itself, and the segment it produced is asserted byte-identical to
                  the first one. Silero on the same frames is deterministic; the assertion is what
                  turns that from an assumption into a checked fact.
  interleaved     A, B, A, B — not AAA then BBB. ARCHITECTURE.md's reading of `make arms` is that the
                  hosted spread is free-tier queueing rather than model speed, and running one arm's
                  repeats back to back hands that drift entirely to one column.
  warm first      every local arm in *both* arm sets is loaded before anything is timed, and the load
                  is printed as its own column. A cold piper or faster-whisper inside t_tts/t_stt is
                  the same lie `arms.warm` exists to prevent.

And one thing it changes: `--remote-timeout` (120 s) is far wider than the loop's REMOTE_TIMEOUT_S.
`meta-llama/Llama-3.1-70B-Instruct` measured 8.1 s and 39.3 s on the NIM free tier, so under the
shipped 10 s budget the quality arm's llm row would be blank. Both numbers are printed together,
because the gap between what an arm needs and what the loop allows is a finding and not a footnote.
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import arms, config, harness, vad                                 # noqa: E402
from src.config import ARCHITECTURES, SAMPLE_RATE, resolve                 # noqa: E402
from src.telemetry import CALLS_LOG, TURN_FIELDS, TURNS_LOG                # noqa: E402

DEFAULT_FIXTURE = Path("tests/fixtures/hello_testing_voice.mp3")

# Wide enough that a 120 s answer lands in the table instead of raising a timeout. The point is not
# that the loop should wait this long — it must not — but that an arm cannot be compared on a number
# it was never given the chance to produce.
COMPARE_REMOTE_TIMEOUT_S = 120.0

# The five fields the ticket means by "all five stages": the four stage timings plus the one that
# spans them. Taken from telemetry so this table cannot describe a different split than the log.
ROWS = TURN_FIELDS


class Result:
    """One arm set's turns across the whole run — the successes and the refusals."""

    def __init__(self, name, chosen):
        self.name = name
        self.chosen = chosen
        self.records = []       # turn records that completed
        self.failures = []      # (turn_id, stage-ish label, message)
        self.transcripts = []
        self.replies = []
        self.speech = []        # (seconds, sample_rate)
        self.load_ms = {}
        self.drifted = 0        # turns whose endpointed segment did not match the first one

    def values(self, field):
        """-> the non-null values of one field across the completed turns."""
        return [r[field] for r in self.records if r.get(field) is not None]

    def complete(self):
        """-> True when at least one turn measured all five fields. The criterion, per arm."""
        return any(all(r.get(f) is not None for f in ROWS) for r in self.records)

    @property
    def turn_ids(self):
        return [r["turn_id"] for r in self.records] + [t for t, _, _ in self.failures]


def stats(values):
    """-> (min, median, max) or None when nothing was measured. None prints n/a, never 0."""
    if not values:
        return None
    return min(values), statistics.median(values), max(values)


def cell(values):
    """Three columns of 8 for one arm, or three n/a. A missing measurement is not a fast stage."""
    s = stats(values)
    if s is None:
        return f"{'n/a':>8}{'n/a':>8}{'n/a':>8}"
    return "".join(f"{v:>8.0f}" for v in s)


def resolve_architecture(name):
    """-> {stage: Arm} for one named architecture. Raises with the known names on a bad one."""
    if name not in ARCHITECTURES:
        raise SystemExit(f"no architecture called {name!r}. Known: {', '.join(ARCHITECTURES)}")
    return {stage: resolve(stage, alias) for stage, alias in ARCHITECTURES[name].items()}


def warm(result):
    """Load every local arm in this arm set and record what the load cost. Hosted arms are no-ops."""
    for stage, arm in result.chosen.items():
        t0 = time.perf_counter()
        arms.warm(stage, arm.id)
        result.load_ms[stage] = round((time.perf_counter() - t0) * 1000, 1)


def show_arms(result):
    """What this arm set actually is, one line per stage. Repo id first — it is the only name."""
    where = {s: ("local" if a.local else f"remote via {a.provider}")
             for s, a in result.chosen.items()}
    hosted = [s for s, a in result.chosen.items() if not a.local]
    summary = "all local, no credential" if not hosted else f"hosted: {', '.join(hosted)}"
    print(f'  ARM "{result.name}" — {summary}')
    for stage, arm in result.chosen.items():
        # 22 for `where`: "remote via nvidia-nim" is 21 characters and ran into the backend column.
        print(f"    {stage:<4}{arm.repo_id:<45}{where[stage]:<22}{arm.backend:<16}"
              f"load {result.load_ms.get(stage, 0):>7.0f} ms")


def show_table(results, repeat, fixtures):
    """The table the criterion names: five rows, one column group per arm, min/median/max."""
    names = [r.name for r in results]
    gap = "   "
    width = 2 + 24 + 24 * len(names) + len(gap) * (len(names) - 1)

    print("=" * width)
    print("  VOX-013 — ARCHITECTURE COMPARISON        all times in ms")
    print(f"  n={repeat} per arm per fixture, interleaved · {len(fixtures)} fixture(s): "
          + ", ".join(str(f) for f in fixtures))
    print("=" * width)
    print("  " + f"{'':<24}" + gap.join(f"{n:^24}" for n in names))
    print("  " + f"{'stage':<24}" + gap.join(f"{'min':>8}{'median':>8}{'max':>8}" for _ in names))
    print("  " + "-" * (width - 2))
    for field in ROWS:
        label = field[:-3] if field.endswith("_ms") else field
        print("  " + f"{label:<24}" + gap.join(cell(r.values(field)) for r in results))
    print("  " + "-" * (width - 2))
    print("  " + f"{'stage_sum':<24}" + gap.join(cell(r.values("stage_sum_ms")) for r in results))
    print("=" * width)


def show_outputs(results):
    """What came out, not just how long it took. The transcript difference is the non-latency finding."""
    print("\n  --- what each arm produced ------------------------------------------------")
    for r in results:
        distinct = sorted(set(r.transcripts))
        chars = [len(x) for x in r.replies]
        secs = [s for s, _ in r.speech]
        rates = {sr for _, sr in r.speech}
        print(f'  ARM "{r.name}"')
        print(f"    turns        {len(r.records)} completed, {len(r.failures)} failed")
        if distinct:
            print(f"    transcript   {len(distinct)} distinct: "
                  + "; ".join(repr(t) for t in distinct))
        else:
            print("    transcript   none — STT never answered")
        if chars:
            print(f"    reply        {min(chars)}-{max(chars)} chars over {len(chars)} turns")
        if secs:
            rate = ", ".join(f"{sr} Hz" for sr in sorted(rates))
            print(f"    speech       {min(secs):.2f}-{max(secs):.2f} s at {rate}")
            # t_tts is not held equal across arms: the two LLMs write different-length replies, so
            # part of the TTS difference is the text. Per-char is the comparable number.
            per_char = [rec["t_tts_ms"] / n for rec, n in zip(r.records, chars)
                        if rec.get("t_tts_ms") and n]
            if per_char:
                print(f"    tts          {min(per_char):.2f}-{max(per_char):.2f} ms per reply char")
        for turn_id, where, msg in r.failures:
            print(f"    FAILED       [{where}] {turn_id}: {msg}")


def show_log(path, wanted):
    """The lines these turns appended. Show the log, not a summary of it — the board's rule."""
    print(f"\n  --- {path} — the lines these turns appended ---")
    if not path.exists():
        print("  (no such file — nothing was logged)")
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if rec.get("turn_id") in wanted:
            print("  " + json.dumps(rec, ensure_ascii=False))


def run_turn(result, clip, reference, source, play):
    """One turn on one arm set. Records the outcome on `result` either way. -> None.

    `reference` is the segment the first endpointing of this fixture produced. Every turn endpoints
    for itself — see the module docstring on why reusing a `Capture` breaks time_to_first_audio —
    and the result is checked against `reference`, because "every arm saw the same audio" has to be
    a verified fact and not a property of silero someone remembered.
    """
    try:
        run = harness.fixture_turn(result.chosen, clip, source, play=play, fallback=False)
    except Exception as e:
        # fallback=False means a refusing free tier raises here. The record was still written, with
        # whatever stages did complete, and the next repeat has to run — so this is caught, counted,
        # and printed, not allowed to end the comparison.
        record = getattr(e, "turn_record", None)
        turn_id = getattr(e, "turn_id", "?")
        stage = next((s for s in ("stt", "llm", "tts")
                      if record and record.get(f"t_{s}_ms") is None), "?")
        result.failures.append((turn_id, stage, f"{type(e).__name__}: {e}"))
        print(f"    {result.name:<9} FAILED at {stage}: {type(e).__name__}: {e}", flush=True)
        return

    if not np.array_equal(run.capture.segment, reference):
        # Not fatal, but it invalidates the comparison, so it is said loudly and on the record
        # rather than left for a reader to wonder about.
        result.drifted += 1
        print(f"    {result.name:<9} WARNING: endpointed segment differs from the first one "
              f"({len(run.capture.segment)} samples vs {len(reference)}) — the arms are no longer "
              f"being compared on identical audio.", flush=True)

    result.records.append(run.record)
    result.transcripts.append(run.transcript)
    result.replies.append(run.reply)
    result.speech.append((len(run.speech.audio) / run.speech.sample_rate, run.speech.sample_rate))
    ttfa = run.record.get("time_to_first_audio_ms")
    print(f"    {result.name:<9} ok  stt {run.record['t_stt_ms']:>7.0f}  "
          f"llm {run.record['t_llm_ms']:>7.0f}  tts {run.record['t_tts_ms']:>7.0f}  "
          f"-> ttfa {f'{ttfa:.0f}' if ttfa is not None else 'n/a':>8}", flush=True)


def main():
    ap = argparse.ArgumentParser(
        description="Two architectures end to end, five-stage split for both (VOX-013)")
    ap.add_argument("--arm", action="append", choices=tuple(ARCHITECTURES), metavar="NAME",
                    help=f"only this architecture; repeatable. Default: all of "
                         f"{', '.join(ARCHITECTURES)}")
    ap.add_argument("--repeat", type=int, default=3, metavar="N",
                    help="turns per arm per fixture (default 3). Every stage in this repo varies "
                         "by 2-4x between runs, so n=1 is an anecdote")
    ap.add_argument("--fixture", type=Path, nargs="+", default=[DEFAULT_FIXTURE],
                    help=f"recording(s) to drive the turns from (default {DEFAULT_FIXTURE}). "
                         f"evals/dev/*.wav once scripts/gen_utterances.py has been run")
    ap.add_argument("--silent", action="store_true",
                    help="skip playback. Leaves time_to_first_audio_ms unmeasured, which means "
                         "the table cannot show all five stages — so this fails the criterion")
    ap.add_argument("--remote-timeout", type=float, default=COMPARE_REMOTE_TIMEOUT_S, metavar="S",
                    help=f"seconds a hosted arm gets to answer (default {COMPARE_REMOTE_TIMEOUT_S:g}). "
                         f"The loop allows {config.REMOTE_TIMEOUT_S:g} s; the large LLM needs more")
    args = ap.parse_args()

    if args.repeat < 1:
        sys.exit("--repeat must be at least 1")
    missing = [f for f in args.fixture if not f.is_file()]
    if missing:
        sys.exit("no such recording: " + ", ".join(str(f) for f in missing))

    shipped_timeout = config.REMOTE_TIMEOUT_S
    # Read at call time by `Arm.timeout_s`, so reassigning the module global is enough and no arm
    # needs a second definition. Printed below beside the shipped value, never instead of it.
    config.REMOTE_TIMEOUT_S = args.remote_timeout

    names = args.arm or list(ARCHITECTURES)
    results = [Result(n, resolve_architecture(n)) for n in names]

    print("resolving arms and loading local models…", flush=True)
    vad._vad_model()
    for r in results:
        warm(r)

    print()
    for r in results:
        show_arms(r)
        print()
    print(f"  remote timeout for this run: {args.remote_timeout:g} s "
          f"(the loop allows {shipped_timeout:g} s — see the finding, not just the table)")
    print("  fallback: OFF. A refused stage is printed as FAILED, never rescued by a local arm.")

    for fixture in args.fixture:
        clip = harness.load_16k_mono(fixture)
        # Endpointed here only to establish what the segment *should* be. Each turn below
        # endpoints again for itself, so its speech_end_t is its own — see the module docstring.
        reference, state = vad.endpoint_frames(vad.frames_from(clip))
        if reference is None:
            sys.exit(f"endpointer found no turn in {fixture} (state={state})")
        print(f"\n  {fixture} — {len(clip) / SAMPLE_RATE:.2f}s clip, endpoints to "
              f"{len(reference) / SAMPLE_RATE:.2f}s ({reference.spoken_s:.2f}s speech), "
              f"state={state}. Every turn re-endpoints and is checked against this segment.")
        for i in range(args.repeat):
            print(f"  repeat {i + 1}/{args.repeat}")
            for r in results:                       # interleaved: A, B, A, B, …
                run_turn(r, clip, reference.segment, str(fixture), play=not args.silent)

    print()
    show_table(results, args.repeat, args.fixture)
    show_outputs(results)

    wanted = {t for r in results for t in r.turn_ids}
    show_log(TURNS_LOG, wanted)
    show_log(CALLS_LOG, wanted)

    drifted = [r.name for r in results if r.drifted]
    if drifted:
        print()
        print("DRIFT — the endpointed segment changed mid-run for: " + ", ".join(drifted))
        print("  The arms were not compared on identical audio, so the table above is not a "
              "comparison. Do not quote it.")
        return 1

    incomplete = [r.name for r in results if not r.complete()]
    print()
    if incomplete:
        print("INCOMPLETE — no turn measured all five stages for: " + ", ".join(incomplete))
        for r in results:
            if not r.complete():
                print(f"  {r.name}: {len(r.records)} completed turn(s), "
                      f"{len(r.failures)} failed. Missing fields cannot be filled in by hand.")
        return 1
    print(f"COMPLETE — all five stages measured for {len(results)} arm(s): "
          + ", ".join(r.name for r in results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
