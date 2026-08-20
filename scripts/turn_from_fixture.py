"""Run one chained turn from a recording instead of the microphone.

    uv run python scripts/turn_from_fixture.py tests/fixtures/hello_testing_voice.mp3

Same stages, same telemetry, one line appended to runs/turns.jsonl — the only difference from
`make demo` is where the frames come from. The turn itself is `src/harness.fixture_turn`, shared
with `scripts/compare_arms.py` so the comparison cannot drift from the turn this script measures.

Read `t_vad` and `time_to_first_audio` from a run of this script with care, and never as a stand-in
for the live number. Frames arrive here as fast as the CPU can push them, so the VAD_SILENCE_MS
hangover that a live turn actually waits out collapses to silero compute — roughly a second that a
person at the mic pays and this script does not. `source` on the turn record says which run it was,
so the two cannot be quietly averaged together. What this script *is* good for is the three model
calls and the synthesis-to-speaker gap, which behave the same either way.

`--kb` runs the VOX-032 grounded path: retrieval after STT, and the answer written from the chunks
that cleared the floor. It is off by default because most reads of this script are latency reads and
the grounded path puts ~1500 tokens of policy text into the llm call — a real cost, and one that
belongs to the KB and not to the arm.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import answer as answer_mod, arms, harness, vad                  # noqa: E402
from src.config import SAMPLE_RATE                                        # noqa: E402
from src.loop import grounding, report                                    # noqa: E402
from src.telemetry import TURNS_LOG                                       # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="One chained turn driven from a recording (VOX-003)")
    ap.add_argument("recording", type=Path, nargs="?",
                    default=Path("tests/fixtures/hello_testing_voice.mp3"))
    ap.add_argument("--kb", action="store_true",
                    help="answer from the policy documents when they cover what was said "
                         "(VOX-032). Off by default: a fixture turn is usually being read as a "
                         "latency measurement, and the grounded path carries ~1500 more tokens")
    ap.add_argument("--silent", action="store_true",
                    help="skip playback — leaves time_to_first_audio unmeasured, so the turn "
                         "line is written with ok=false")
    arms.add_flags(ap)
    args = ap.parse_args()

    if not args.recording.is_file():
        sys.exit(f"no such recording: {args.recording}")

    # Same as the live loop: local weights load before the turn, so a ~10 s Kokoro import does
    # not land inside t_tts and make the split a lie.
    print("resolving arms and loading local models…", flush=True)
    vad._vad_model()
    chosen = arms.select(args)
    print(arms.describe(chosen))

    # Built before the turn, exactly as src/loop.py does it — indexing is per-process work and
    # would otherwise land inside the turn it is being measured with.
    idx = answer_mod.knowledge_base() if args.kb else None

    clip = harness.load_16k_mono(args.recording)
    print(f"\n--- turn · {args.recording} ({len(clip) / SAMPLE_RATE:.2f}s) ---")

    try:
        run = harness.fixture_turn(chosen, clip, str(args.recording), play=not args.silent,
                                   echo=print, idx=idx)
    except (harness.NoSpeech, harness.EmptyTranscript) as e:
        sys.exit(str(e))

    if args.kb:
        # Reassembled from the TurnRun rather than printed inside the harness: `grounding()` is one
        # wording shared with `make demo`, and the harness stays free of console output.
        hits = run.answer.hits if run.answer else []
        print("  " + grounding(answer_mod.Reply(run.reply, run.answer, hits),
                               kb=idx is not None))

    print(f"\nturn {run.turn_id}: " + report(run.record))
    print(f"turn line appended to {TURNS_LOG}")
    return 0 if run.record["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
