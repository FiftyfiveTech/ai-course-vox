"""One chained turn, end to end: mic -> VAD -> STT -> LLM -> TTS -> speaker.

    uv run python -m src.loop          one turn, then exit
    uv run python -m src.loop --turns 3

Deliberately one turn by default: VOX-002 is "you speak, you hear a reply". Barge-in is VOX-011
and the five-field timing breakdown is VOX-003, so neither is here.
"""
import argparse
import sys

from src import audio, nlu, stt, tts, vad
from src.config import CONSENT_NOTICE, LLM, STT, TTS
from src.telemetry import CALLS_LOG, new_turn_id


def one_turn():
    """-> True if a reply was spoken, False if the mic stayed quiet."""
    turn_id = new_turn_id()
    print(f"\n--- turn {turn_id} ---")

    segment = vad.listen()
    if segment is None:
        print("nothing heard — stopping.")
        return False

    transcript = stt.transcribe(segment, turn_id)
    print(f"you said : {transcript!r}")
    if not transcript:
        # Whisper returning empty on real audio is a provider problem, not a quiet user.
        print("empty transcript from STT — not calling the LLM.", file=sys.stderr)
        return False

    answer = nlu.reply(transcript, turn_id)
    print(f"vox says : {answer!r}")

    speech = tts.synthesize(answer, turn_id)
    print("speaking…", flush=True)
    audio.play(speech)
    return True


def main():
    ap = argparse.ArgumentParser(description="VOX — one chained turn (VOX-002)")
    ap.add_argument("--turns", type=int, default=1, help="how many turns before exiting")
    args = ap.parse_args()

    print("VOX — chained turn loop")
    print(f"  stt  {STT.repo_id}  ({STT.provider})")
    print(f"  llm  {LLM.repo_id}  ({LLM.provider})")
    print(f"  tts  {TTS.repo_id}  ({TTS.provider})")

    # Load the local weights before the turn starts. Kokoro takes ~10 s to load and silero a
    # moment; leaving that inside the turn would bury it in t_tts and t_vad and make the
    # latency split a lie. VOX-003 measures the warm path, which is the one users feel.
    print("\nloading local models…", flush=True)
    vad._vad_model()
    tts._kokoro()

    print(f"\n{CONSENT_NOTICE}\n")

    spoken = 0
    for _ in range(args.turns):
        if not one_turn():
            break
        spoken += 1

    print(f"\n{spoken} turn(s) completed. Call log: {CALLS_LOG}")
    return 0 if spoken else 1


if __name__ == "__main__":
    sys.exit(main())
