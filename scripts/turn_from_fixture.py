"""Run one chained turn from a recording instead of the microphone.

    uv run python scripts/turn_from_fixture.py tests/fixtures/hello_testing_voice.mp3

Same stages, same telemetry, one line appended to runs/turns.jsonl — the only difference from
`make demo` is where the frames come from. That seam is `Endpointer`, which takes one 32 ms frame
and knows nothing about audio devices, so `listen()` stays the only code that opens a mic and
`make demo` stays live-mic only.

Read `t_vad` and `time_to_first_audio` from a run of this script with care, and never as a stand-in
for the live number. Frames arrive here as fast as the CPU can push them, so the VAD_SILENCE_MS
hangover that a live turn actually waits out collapses to silero compute — roughly a second that a
person at the mic pays and this script does not. `source` on the turn record says which run it was,
so the two cannot be quietly averaged together. What this script *is* good for is the three model
calls and the synthesis-to-speaker gap, which behave the same either way.
"""
import argparse
import sys
from pathlib import Path

import soundfile as sf
import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import audio as audio_out                                          # noqa: E402
from src import arms, nlu, vad                                              # noqa: E402
from src.config import SAMPLE_RATE                                          # noqa: E402
from src.loop import report                                                 # noqa: E402
from src.telemetry import TURNS_LOG, new_turn_id, turn_timer                # noqa: E402


def load_16k_mono(path):
    """-> float32 mono at SAMPLE_RATE, which is what silero and whisper both want."""
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    mono = torch.from_numpy(data.mean(axis=1))
    if sr != SAMPLE_RATE:
        mono = torchaudio.functional.resample(mono, sr, SAMPLE_RATE)
    return mono.numpy()


def main():
    ap = argparse.ArgumentParser(description="One chained turn driven from a recording (VOX-003)")
    ap.add_argument("recording", type=Path, nargs="?",
                    default=Path("tests/fixtures/hello_testing_voice.mp3"))
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
    for stage, arm in chosen.items():
        print(f"  {stage:<4} {arm.repo_id}  ({arm.provider}, {arm.backend})")

    clip = load_16k_mono(args.recording)
    turn_id = new_turn_id()
    print(f"\n--- turn {turn_id} · {args.recording} ({len(clip) / SAMPLE_RATE:.2f}s) ---")

    with turn_timer(turn_id, source=str(args.recording)) as turn:
        turn.arms(**chosen)
        cap, state = vad.endpoint_frames(vad.frames_from(clip))
        if cap is None:
            sys.exit(f"endpointer found no turn in {args.recording} (state={state})")
        turn.vad(cap)
        print(f"endpointed: {len(cap) / SAMPLE_RATE:.2f}s ({cap.spoken_s:.2f}s speech), "
              f"state={state}")

        with turn.stage("stt"):
            transcript = arms.stt(cap.segment, chosen["stt"].id, turn_id=turn_id)
        print(f"you said : {transcript!r}")
        if not transcript:
            sys.exit("empty transcript from STT — not calling the LLM.")

        with turn.stage("llm"):
            answer = nlu.reply(transcript, turn_id, model_id=chosen["llm"].id)
        print(f"vox says : {answer!r}")

        with turn.stage("tts"):
            speech = arms.tts(answer, chosen["tts"].id, turn_id=turn_id)

        if args.silent:
            print("(--silent: not playing, time_to_first_audio will be null)")
        else:
            print("speaking…", flush=True)
            audio_out.play(speech.audio, sample_rate=speech.sample_rate,
                           on_first_audio=turn.first_audio)

    print("\n" + report(turn.written))
    print(f"turn line appended to {TURNS_LOG}")
    return 0 if turn.written["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
