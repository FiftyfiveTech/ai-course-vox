"""The scripted demo session, run end to end with no microphone (VOX-026).

    uv run --no-sync python scripts/dry_run.py           the ten turns, paced at real time
    uv run --no-sync python scripts/dry_run.py --fast    no pacing — a wiring check, not a latency read
    uv run --no-sync python scripts/dry_run.py --only t04 t05
    uv run --no-sync python scripts/dry_run.py --script evals/demo/session_v1.json

`--no-sync` is not decoration: `uv run` re-resolves the environment, `en-core-web-sm` is pinned to a
GitHub release URL, and a 504 from GitHub is a failed run before a line of VOX executes. It happened
once while this script was being written. Two minutes before a demo is the wrong time to find out.

What this is
------------
The running order of the demo, in `evals/demo/session_v1.json`, executed against the real stages:
silero endpointing, whisper-large-v3 at Groq, hybrid retrieval over the internal corpus,
Llama-3.1-8B at NVIDIA NIM, and a local voice. Ten turns, one barge-in whose interrupting utterance
becomes the next turn's input, and two confirmations — one confirmed, one cancelled.

Each turn's `expect` block is asserted against the *turn record*, not against the reply text: a
reply is a model output at a temperature and pinning its words would fail the rehearsal for the
wrong reason. `grounded`, `sources`, `retrieved`, `barged_in`, `confirmation` and the extractor's
`intent` are decisions the code made, and those are what a PASS means here.

What it is not
--------------
Not a gate: it prints latencies and a per-turn PASS/FAIL, and no threshold is asserted on the
timings — VOX-003's budget is missed by a factor of 4 and is recorded as missed in ARCHITECTURE.md,
so a latency floor here would either be a lie or a permanent failure.

Not an acoustic test. The user's lines are synthesised locally (Kokoro by default, a different voice
from the reply's, so a recording of the session has two speakers in it) and handed to STT as
samples. They never cross a room or a microphone, so per-turn transcript accuracy here is an upper
bound on what a person in an open-plan office gets. Synthesis happens in a preparation phase, is
cached under runs/rehearsal/, and is deliberately **not** routed through `arms.tts` — the user's
voice is a prop, not a stage of the system under test, and a calls.jsonl line for it would join to
no turn. Put a path in a turn's `recording` field and that file is used verbatim instead, which is
how a real recorded voice replaces the synthetic one on the demo machine.

`--fast` skips real-time pacing. Then the VAD_SILENCE_MS hangover collapses to silero compute, every
`t_vad` and `time_to_first_audio` reads about a second better than a person will experience it, and
the run says so on every table it prints. Use it to check wiring, never to quote a number.
"""
import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np                                                        # noqa: E402
import soundfile as sf                                                    # noqa: E402

from src import (answer as answer_mod, arms, config, confirm, harness,    # noqa: E402
                 loop, state, tts as tts_mod, vad, vocab_bias)
from src.config import RUNS_DIR, VAD_SILENCE_MS, utf8_console             # noqa: E402
from src.telemetry import CALLS_LOG, TURNS_LOG                            # noqa: E402

SCRIPT_FILE = REPO_ROOT / "evals" / "demo" / "session_v1.json"
AUDIO_DIR = RUNS_DIR / "rehearsal" / "audio"
REPORT_DIR = RUNS_DIR / "rehearsal"

# Silence appended after every synthesised line, so the endpointer reaches DONE by hearing the user
# stop rather than by running out of file. `flush()` would end the utterance either way, but only one
# of those two is what happens live, and t_vad is the difference between them.
TAIL_SILENCE_MS = VAD_SILENCE_MS + 400


# --- preflight ----------------------------------------------------------------------------------

def env_names_no_module_reads():
    """-> the VOX_* names set in .env that no module in src/ reads. Empty is the healthy answer.

    A knob nobody reads is worse than a missing one: it reads as a setting in a file someone will
    trust under pressure. `.env` on this machine sets VOX_STT_FIXUPS=1 with nine lines of comment
    explaining what it protects the demo from, and the mechanism it names lives on an unmerged
    branch. That is the class of surprise this ticket exists to remove.
    """
    import re

    dotenv = REPO_ROOT / ".env"
    if not dotenv.is_file():
        return []
    set_here = set(re.findall(r"^\s*(?:export\s+)?(VOX_[A-Z0-9_]+)\s*=", dotenv.read_text(
        encoding="utf-8"), flags=re.M))
    read_by_code = set()
    for folder in ("src", "scripts", "tests", "tools"):
        for py in (REPO_ROOT / folder).rglob("*.py"):
            read_by_code |= set(re.findall(r"VOX_[A-Z0-9_]+", py.read_text(encoding="utf-8")))
    return sorted(set_here - read_by_code)


def preflight(chosen, idx, silent):
    """Everything that only fails in front of an audience, checked before turn 1. -> [blockers].

    Blockers stop the run. A warning that scrolls past above the first turn is the same thing as no
    warning, so anything that would make a scripted turn fail its own `expect` is a blocker here.
    """
    blockers, warnings = [], []

    print(arms.describe(chosen))
    print(f"  paths  chunks={config.CHUNKS_FILE.name} "
          f"turns={TURNS_LOG.name} calls={CALLS_LOG.name}")
    print(f"  tunables  RETRIEVAL_SCORE_FLOOR={config.RETRIEVAL_SCORE_FLOOR} "
          f"DENSE_SCORE_FLOOR={config.DENSE_SCORE_FLOOR} TOP_K={config.RETRIEVAL_TOP_K} "
          f"HYBRID={config.HYBRID_RETRIEVAL}")
    print(f"            REMOTE_TIMEOUT_S={config.REMOTE_TIMEOUT_S} "
          f"LOCAL_TIMEOUT_S={config.LOCAL_TIMEOUT_S} "
          f"SESSION_QUIET_LIMIT={config.SESSION_QUIET_LIMIT} "
          f"STT_BIAS={'on' if vocab_bias.enabled() else 'off'}")

    if idx is None:
        blockers.append("no knowledge base: `make index` first, or six of the ten turns cannot "
                        "be grounded and will fail their own expectations")

    if silent:
        blockers.append("--silent leaves time_to_first_audio unmeasured, so no turn can be clean. "
                        "It is for reading stdout on a machine with no speaker, not for a run "
                        "anyone quotes")
    else:
        try:
            import sounddevice as sd
            out = sd.query_devices(kind="output")
            print(f"  speaker  {out['name']} ({out['default_samplerate']:.0f} Hz)")
            print("           headphones, not open speakers: there is no echo cancellation, so on "
                  "a speaker silero hears the reply and it interrupts itself")
        except Exception as e:
            blockers.append(f"no usable output device ({type(e).__name__}: {e}) — "
                            "time_to_first_audio cannot be measured without one")

    for stage in ("stt", "llm"):
        fb = arms.fallback_for(stage, chosen[stage])
        if fb is None:
            warnings.append(f"{stage} has no local fallback — a rate limit costs the turn")

    dead = env_names_no_module_reads()
    if dead:
        warnings.append("set in .env and read by no module in src/: " + ", ".join(dead)
                        + " — a knob that does nothing, not a setting")

    for w in warnings:
        print(f"  WARN  {w}")
    for b in blockers:
        print(f"  STOP  {b}")
    return blockers


# --- the user's voice ---------------------------------------------------------------------------

def synth(text, arm, tail_ms=TAIL_SILENCE_MS, lead_s=0.0):
    """Synthesise one of the user's lines. -> float32 mono at SAMPLE_RATE.

    Called straight into the backend rather than through `arms.tts`: this audio is the demo's
    *input*, not one of its stages, and a calls.jsonl line for it would carry a turn_id that joins
    to nothing — the one thing the two-log design exists to prevent.
    """
    samples = tts_mod.BACKENDS[arm.backend](arm, text, {})
    rate = arm.extra["sample_rate"]
    pad = np.zeros(int(rate * tail_ms / 1000), dtype="float32")
    lead = np.zeros(int(rate * lead_s), dtype="float32")
    return np.concatenate([lead, np.asarray(samples, dtype="float32"), pad]), rate


def clip_for(text, arm, lead_s=0.0, regen=False, say=print):
    """The wav for one line, synthesised once and cached. -> float32 mono at 16 kHz.

    The filename carries a hash of what determines the audio — the words, the arm, the lead-in — so
    editing a line in the session script produces a new file instead of silently reusing the old
    one. That is the whole cache-invalidation policy and it needs no sidecar.
    """
    key = hashlib.sha256(f"{arm.id}|{lead_s}|{text}".encode()).hexdigest()[:10]
    slug = "".join(c if c.isalnum() else "-" for c in text.lower())[:40].strip("-")
    path = AUDIO_DIR / f"{slug}-{key}.wav"

    if regen or not path.is_file():
        AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        samples, rate = synth(text, arm, lead_s=lead_s)
        sf.write(path, samples, rate)
        say(f"    synthesised {path.name} ({len(samples) / rate:.2f}s @ {rate} Hz)")
    return harness.load_16k_mono(path), path


# --- the turns ----------------------------------------------------------------------------------

def listener_over(clip, paced, say=print):
    """A stand-in for `vad.listen` that reads a recording. -> callable(**kw) -> Capture or None.

    Passed to `loop.speak_and_watch` and to `loop.confirmation_leg`, which is the whole point: both
    of those keep their own logic — the barge threshold, the confirm window, the mark the stop
    latency is measured from, the yes/no classifier — and only where the audio comes from changes.
    """
    def listen(on_speech=None, threshold=None, announce=True, **_):
        frames = vad.frames_from(clip)
        cap, _state = vad.endpoint_frames(vad.paced(frames, echo=say) if paced else frames,
                                          on_speech=on_speech, threshold=threshold)
        return cap

    return listen


def run_turn(turn, chosen, idx, *, clips, carried, paced, play, say=print):
    """One scripted turn. -> (TurnRun, TurnState or None).

    Raises whatever the turn raised, with `exc.turn_record` attached by the harness.
    """
    held = {}

    def extract_state(text, tid, model_id=None, **_):
        """The plain path, as the live loop wires it: the structured extractor writes the reply, and
        `next_action` is what the confirmation gate below reads."""
        held["state"] = state.build(text, tid, model_id=model_id)
        return held["state"].reply

    def after_play(t, turn_id, ch):
        st = held.get("state")
        if st is None:
            return                      # a grounded answer has no state and nothing to confirm
        t.extra.update(intent=st.intent, next_action=st.next_action,
                       state_confidence=round(st.confidence, 3))
        if confirm.needs_confirmation(st):
            wants = turn.get("confirm_with")
            if wants is None:
                # Scripted as an ordinary turn but the extractor asked for a yes/no. Not an error
                # here — the turn's `expect` block is what decides — but the run must not hang
                # waiting on audio the script never wrote.
                say("    (confirmation asked for, and the script supplies no answer to it)")
                t.extra["confirmation"] = "unanswered"
                t.extra["confirmation_required"] = True
                return
            loop.confirmation_leg(t, turn_id, ch, st, play=play,
                                  listen=listener_over(clips["confirm"], paced, say=say),
                                  say=lambda line: say("  " + line))

    watch = None
    if turn["kind"] == "barge":
        watch = listener_over(clips["barge"], paced, say=say)

    run = harness.fixture_turn(
        chosen, clips.get("say"), f"rehearsal:{turn['id']}",
        play=play, idx=idx, echo=lambda line: say("    " + line), paced=paced,
        plain=extract_state, watch=watch, after_play=after_play,
        segment=carried if turn["kind"] == "carried" else None,
        extra={"rehearsal_turn": turn["id"], "paced": paced,
               **({"input": "carried-in"} if turn["kind"] == "carried" else {})})
    return run, held.get("state")


# --- what a clean turn is -----------------------------------------------------------------------

# Every key this function knows how to assert. An `expect` block naming anything else is a typo
# that would otherwise assert nothing and print PASS, which is the one way a rehearsal can lie.
CHECKED_KEYS = frozenset({"path", "sources_any", "retrieved", "retrieved_min", "intent_in",
                          "barged_in", "confirmation", "input"})


def check(turn, rec, run):
    """Compare one finished turn against its `expect` block. -> [failures], empty when clean.

    Structural facts only. The reply's wording is a sampled model output and asserting on it would
    fail a rehearsal for the wrong reason; every key below is a decision the code made and wrote
    down. `ok` is the harness's own: all five VOX-003 fields measured and no error.
    """
    want = turn.get("expect", {})
    bad = []

    unknown = sorted(set(want) - CHECKED_KEYS)
    if unknown:
        bad.append(f"the script expects {unknown}, which this checker does not implement — an "
                   f"expectation nobody asserts is worse than none")

    if not rec.get("ok"):
        bad.append(f"ok={rec.get('ok')} — a stage was not measured"
                   + (f" ({rec['error']})" if rec.get("error") else ""))

    path, grounded, retrieved = want.get("path"), rec.get("grounded"), rec.get("retrieved")
    if path == "grounded" and not grounded:
        bad.append(f"expected a grounded answer, got grounded={grounded} "
                   f"retrieved={retrieved} sources={rec.get('sources')}")
    if path == "refused" and grounded:
        bad.append(f"expected a refusal, got grounded from {rec.get('sources')}")
    if path == "plain" and grounded:
        bad.append(f"expected the plain path, got a grounded answer from {rec.get('sources')}")

    if "sources_any" in want:
        got = rec.get("sources") or []
        if not set(want["sources_any"]) & set(got):
            bad.append(f"none of {want['sources_any']} in the context: {got}")
    if "retrieved" in want and retrieved != want["retrieved"]:
        bad.append(f"retrieved={retrieved}, expected {want['retrieved']}")
    if "retrieved_min" in want and (retrieved or 0) < want["retrieved_min"]:
        bad.append(f"retrieved={retrieved}, expected at least {want['retrieved_min']}")
    if "intent_in" in want and rec.get("intent") not in want["intent_in"]:
        bad.append(f"intent={rec.get('intent')!r}, expected one of {want['intent_in']}")
    if want.get("barged_in") and not rec.get("barged_in"):
        bad.append("the reply was not interrupted — abort() found nothing left to cut, so "
                   "barge_after_s is too late or the reply was too short")
    if "confirmation" in want and rec.get("confirmation") != want["confirmation"]:
        bad.append(f"confirmation={rec.get('confirmation')!r}, expected "
                   f"{want['confirmation']!r}")
    if "input" in want and rec.get("input") != want["input"]:
        bad.append(f"input={rec.get('input')!r}, expected {want['input']!r}")
    if turn["kind"] == "barge" and run is not None and run.carried is None:
        bad.append("nothing was captured during the reply, so the next turn has no input to run on")
    return bad


# --- the report ---------------------------------------------------------------------------------

def ms(rec, key):
    v = rec.get(key)
    return f"{v:>7.0f}" if isinstance(v, (int, float)) else "      -"


def table(rows, paced):
    print("\n" + "=" * 118)
    print("  per-turn timings, all ms. " + ("paced at real time — the VAD hangover is paid, so "
          "these are live numbers with no mic" if paced else
          "NOT paced: t_vad and ttfa read ~1s better than a person will hear"))
    print("=" * 118)
    print(f"  {'turn':<5}{'path':<10}{'vad':>8}{'stt':>8}{'llm':>8}{'tts':>8}{'retr':>8}"
          f"{'ttfa':>8}  {'notes'}")
    for turn, rec, bad in rows:
        if rec is None:
            print(f"  {turn['id']:<5}{'FAILED':<10}{'no record — the turn raised before it started'}")
            continue
        notes = []
        if rec.get("sources"):
            notes.append(",".join(rec["sources"][:2]))
        elif rec.get("intent"):
            notes.append(f"intent={rec['intent']}")
        if rec.get("barged_in"):
            notes.append(f"barge {rec['barge_stop_ms']:.0f}ms cut {rec['cut_s']:.1f}s")
        if rec.get("confirmation"):
            notes.append(f"confirm={rec['confirmation']}")
        if rec.get("fell_back"):
            notes.append("FELL BACK: " + ",".join(rec["fell_back"]))
        path = ("grounded" if rec.get("grounded") else
                "refused" if rec.get("retrieved") else "plain")
        print(f"  {turn['id']:<5}{path:<10}{ms(rec, 't_vad_ms')}{ms(rec, 't_stt_ms')}"
              f"{ms(rec, 't_llm_ms')}{ms(rec, 't_tts_ms')}{ms(rec, 't_retrieval_ms')}"
              f"{ms(rec, 'time_to_first_audio_ms')}  {'; '.join(notes)}")
        if bad:
            for line in bad:
                print(f"         FAIL  {line}")
    print("=" * 118)


def summarise(rows, asked_for_grounding):
    """The three numbers worth quoting, with their denominators (VOX-033's lesson)."""
    done = [rec for _t, rec, _b in rows if rec]
    ttfa = sorted(r["time_to_first_audio_ms"] for r in done
                  if r.get("time_to_first_audio_ms") is not None)
    if ttfa:
        print(f"  time_to_first_audio over {len(ttfa)}/{len(rows)} turns: "
              f"median {statistics.median(ttfa):.0f}ms, min {ttfa[0]:.0f}ms, max {ttfa[-1]:.0f}ms")
    for field, label in (("t_llm_ms", "llm"), ("t_tts_ms", "tts"), ("t_retrieval_ms", "retrieval")):
        got = sorted(r[field] for r in done if r.get(field) is not None)
        if got:
            print(f"  {label:<10} median {statistics.median(got):>7.0f}ms  "
                  f"band {got[0]:.0f}-{got[-1]:.0f}ms  n={len(got)}")
    fell = [r["rehearsal_turn"] for r in done if r.get("fell_back")]
    print(f"  fell back on {len(fell)}/{len(rows)} turns" + (f": {', '.join(fell)}" if fell else ""))
    grounded = [r for r in done if r.get("grounded")]
    print(f"  grounded {len(grounded)}/{len(rows)} turns; the script asks for grounding on "
          f"{asked_for_grounding} of them")


def main():
    # First line of the run: a rehearsal that is being piped into a file, or watched on a console
    # that encodes to cp1252, must not lose a turn to a character in a transcript. See
    # config.utf8_console() — this is the failure that cost the first t04 of this ticket.
    utf8_console()
    ap = argparse.ArgumentParser(description="The scripted demo session, end to end (VOX-026)")
    ap.add_argument("--script", type=Path, default=SCRIPT_FILE)
    ap.add_argument("--only", nargs="*", metavar="ID",
                    help="run just these turn ids. A subset that skips t04 cannot run t05, which "
                         "has no audio of its own")
    ap.add_argument("--fast", action="store_true",
                    help="do not pace the frames at real time. Checks wiring; the timings it "
                         "prints are not the ones a person hears")
    ap.add_argument("--silent", action="store_true",
                    help="do not play the replies. Leaves time_to_first_audio unmeasured, so no "
                         "turn can be clean — refused by the preflight for exactly that reason")
    ap.add_argument("--regen", action="store_true", help="re-synthesise the user's lines")
    ap.add_argument("--user-voice", default="kokoro",
                    help="the arm that speaks the user's lines. A different one from the reply's, "
                         "so a recording of the session has two distinguishable voices in it")
    arms.add_flags(ap)
    args = ap.parse_args()

    script = json.loads(args.script.read_text(encoding="utf-8"))
    turns = script["turns"]
    if args.only:
        turns = [t for t in turns if t["id"] in set(args.only)]
    paced, play = not args.fast, not args.silent

    print(f"VOX — dry run of {args.script.name} ({script['version']}), "
          f"{len(turns)} of {len(script['turns'])} turns")
    print(f"\n{config.CONSENT_NOTICE}\n")

    # Everything loadable is loaded before the first turn is timed, exactly as src/loop.py does it:
    # a cold Kokoro import inside turn 1 would land in t_tts and make the whole table a lie.
    print("resolving arms and loading local models…", flush=True)
    vad._vad_model()
    chosen = arms.select(args)
    idx = answer_mod.knowledge_base()
    user_arm = arms.warm("tts", args.user_voice)

    if preflight(chosen, idx, args.silent):
        print("\npreflight failed — nothing was run.", file=sys.stderr)
        return 2

    print(f"\npreparing the user's lines ({user_arm.id}, cached in {AUDIO_DIR})…", flush=True)
    clips = {}
    for turn in turns:
        c = {}
        if turn.get("recording"):
            c["say"] = harness.load_16k_mono(REPO_ROOT / turn["recording"])
            print(f"    {turn['id']}: {turn['recording']} (recorded, not synthesised)")
        elif turn.get("say"):
            c["say"], _ = clip_for(turn["say"], user_arm, regen=args.regen)
        if turn.get("barge_with"):
            b = turn["barge_with"]
            # The lead-in silence *is* the barge offset: the frames are paced, so N seconds of
            # zeros in front of the utterance is N seconds of the reply playing before the user
            # starts talking over it. No sleep anywhere, and nothing measures its own delay.
            c["barge"], _ = clip_for(b["say"], user_arm, lead_s=b["after_s"], regen=args.regen)
        if turn.get("confirm_with"):
            c["confirm"], _ = clip_for(turn["confirm_with"], user_arm, regen=args.regen)
        clips[turn["id"]] = c

    rows, carried, t0 = [], None, time.perf_counter()
    for turn in turns:
        said = repr(turn["say"]) if turn.get("say") else "(the audio that interrupted the last reply)"
        print(f"\n--- {turn['id']} · {turn['kind']} · {said}")
        if turn["kind"] == "carried" and carried is None:
            print("    SKIPPED: nothing was carried out of the previous turn")
            rows.append((turn, None, ["no carried-in audio: the barge turn did not produce any"]))
            continue
        try:
            run, _st = run_turn(turn, chosen, idx, clips=clips[turn["id"]], carried=carried,
                                paced=paced, play=play)
            rec = run.record
            carried = run.carried
        except Exception as e:
            rec = getattr(e, "turn_record", None)
            carried = None
            print(f"    RAISED {type(e).__name__}: {e}", file=sys.stderr)
            rows.append((turn, rec, [f"{type(e).__name__}: {e}"]))
            continue
        bad = check(turn, rec, run)
        print(f"    {'PASS' if not bad else 'FAIL'}  " + loop.report(rec))
        rows.append((turn, rec, bad))

    elapsed = time.perf_counter() - t0
    table(rows, paced)
    summarise(rows, sum(1 for t in turns if t.get("expect", {}).get("path") == "grounded"))

    failed = [t["id"] for t, _rec, bad in rows if bad]
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / f"run-{time.strftime('%Y%m%dT%H%M%S')}.json"
    out.write_text(json.dumps({
        "script": script["version"], "paced": paced, "played": play,
        "wall_clock_s": round(elapsed, 1), "user_voice": user_arm.id,
        "arms": {stage: arm.id for stage, arm in chosen.items()},
        "clean": not failed,
        "turns": [{"id": t["id"], "kind": t["kind"], "expect": t.get("expect", {}),
                   "failures": bad, "record": rec} for t, rec, bad in rows],
    }, indent=2), encoding="utf-8")

    print(f"\n  wall clock {elapsed:.0f}s for {len(rows)} turns · report {out}")
    print(f"  turns: {TURNS_LOG}\n  calls: {CALLS_LOG}")
    if failed:
        print(f"\nNOT CLEAN — {len(failed)}/{len(rows)} turns failed: {', '.join(failed)}",
              file=sys.stderr)
        return 1
    print(f"\nCLEAN — {len(rows)}/{len(rows)} turns met their expectations.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
