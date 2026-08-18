"""One chained turn, end to end: mic -> VAD -> STT -> LLM -> TTS -> speaker.

    uv run python -m src.loop          one turn, then exit
    uv run python -m src.loop --turns 3

    uv run python -m src.loop --stt openai/whisper-base --tts microsoft/speecht5_tts

Deliberately one turn by default: VOX-002 is "you speak, you hear a reply". Barge-in is VOX-011,
so that is not here. The five-field latency split is VOX-003 and it is here from the first
commit that has a turn to measure — retrofitting timings onto a loop that already runs means
tuning against numbers nobody watched being taken.

Which model runs each stage is a flag (VOX-006), and the arms are named on the turn record, so two
runs with different arms cannot be quietly averaged together.
"""
import argparse
import sys

from src import arms, audio, nlu, vad
from src.config import CONSENT_NOTICE
from src.errors import RateLimited
from src.telemetry import CALLS_LOG, TURNS_LOG, new_turn_id, turn_timer


def one_turn(chosen):
    """-> (spoken, keep_going). `chosen` maps stage -> Arm.

    Two separate facts, because one bool used to carry both and they came apart at the TTS stage:
    `spoken` is whether audio actually reached the speaker, which is what the run counts, and
    `keep_going` is whether the session can continue. A failed voice with a good reply is
    (False, True) — nothing was heard, but there is no reason to end the conversation.
    """
    turn_id = new_turn_id()
    print(f"\n--- turn {turn_id} ---")

    with turn_timer(turn_id, source="mic") as turn:
        turn.arms(**chosen)
        cap = vad.listen()
        if cap is None:
            print("nothing heard — stopping.")
            return False, False
        turn.vad(cap)

        with turn.stage("stt"):
            transcript = arms.stt(cap.segment, chosen["stt"].id, turn_id=turn_id)
        print(f"you said : {transcript!r}")
        if not transcript:
            # Whisper returning empty on real audio is a provider problem, not a quiet user.
            print("empty transcript from STT — not calling the LLM.", file=sys.stderr)
            return False, False

        with turn.stage("llm"):
            answer = nlu.reply(transcript, turn_id, model_id=chosen["llm"].id)
        print(f"vox says : {answer!r}")

        try:
            with turn.stage("tts"):
                speech = arms.tts(answer, chosen["tts"].id, turn_id=turn_id)
        except Exception as e:
            # The reply is fine; only the voice failed. Losing the whole turn over that throws away
            # work the user waited for, so degrade to text — but loudly, on stderr and on the turn
            # record. A quiet degrade would read downstream as a turn that simply never spoke, and
            # the gate percentiles would improve because a turn dropped out of them.
            turn.extra["degraded"] = "tts"
            turn.extra["degrade_reason"] = f"{type(e).__name__}: {e}"
            print(f"TTS FAILED ({type(e).__name__}: {e}) — text only, nothing was spoken:\n"
                  f"  {answer}", file=sys.stderr)
            return False, True

        print("speaking…", flush=True)
        audio.play(speech.audio, sample_rate=speech.sample_rate, on_first_audio=turn.first_audio)

    print("  " + report(turn.written))
    return True, True


def report(rec):
    """The line a human reads. The JSONL line is the record; this is so you see it happen."""
    def ms(key):
        v = rec.get(key)
        return f"{v:.0f}ms" if v is not None else "n/a"

    return (f"vad {ms('t_vad_ms')} + stt {ms('t_stt_ms')} + llm {ms('t_llm_ms')} + "
            f"tts {ms('t_tts_ms')}  ->  time_to_first_audio {ms('time_to_first_audio_ms')}")


def main():
    ap = argparse.ArgumentParser(description="VOX — one chained turn (VOX-002)")
    ap.add_argument("--turns", type=int, default=1, help="how many turns before exiting")
    arms.add_flags(ap)
    args = ap.parse_args()

    print("VOX — chained turn loop")

    # Resolving and loading happen before the turn starts. Kokoro takes ~10 s to load and silero a
    # moment; leaving that inside the turn would bury it in t_tts and t_vad and make the
    # latency split a lie. VOX-003 measures the warm path, which is the one users feel.
    print("resolving arms and loading local models…", flush=True)
    vad._vad_model()
    chosen = arms.select(args)
    for stage, arm in chosen.items():
        print(f"  {stage:<4} {arm.repo_id}  ({arm.provider}, {arm.backend})")

    print(f"\n{CONSENT_NOTICE}\n")

    spoken = 0
    for _ in range(args.turns):
        try:
            heard, keep_going = one_turn(chosen)
        except RateLimited as e:
            # Caught here and not inside the turn: a turn cannot decide the session is over, and
            # the wait is longer than a turn anyway. The turn record already carries the error,
            # written on the way out — this is so the user reads a sentence, not a traceback.
            print(f"\nRATE LIMITED — {e}", file=sys.stderr)
            break
        spoken += 1 if heard else 0
        if not keep_going:
            break

    print(f"\n{spoken} turn(s) completed.")
    print(f"  calls: {CALLS_LOG}")
    print(f"  turns: {TURNS_LOG}")
    return 0 if spoken else 1


if __name__ == "__main__":
    sys.exit(main())
