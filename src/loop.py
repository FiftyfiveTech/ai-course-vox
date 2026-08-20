"""One chained turn, end to end: mic -> VAD -> STT -> LLM -> TTS -> speaker.

    uv run python -m src.loop          one turn, then exit
    uv run python -m src.loop --turns 3

    uv run python -m src.loop --stt openai/whisper-base --tts microsoft/speecht5_tts

Still one turn by default: VOX-002 is "you speak, you hear a reply". The five-field latency split is
VOX-003 and it is here from the first commit that has a turn to measure — retrofitting timings onto a
loop that already runs means tuning against numbers nobody watched being taken.

Barge-in (VOX-011) is what `--turns 2` or more buys. Every turn but the last plays its reply with the
mic still open, so speaking over VOX stops it mid-sentence and the words that stopped it are carried
into the next turn as its input. There is no second listener: the ordinary endpointer runs across the
whole reply and past it, so an interruption and a normal next utterance are the same code path and
differ only in whether anything was still playing when they arrived.

Which model runs each stage is a flag (VOX-006), and the arms are named on the turn record, so two
runs with different arms cannot be quietly averaged together.
"""
import argparse
import sys
from collections import namedtuple

from src import arms, audio, nlu, state, vad
from src.config import BARGE_SPEECH_THRESHOLD, CONSENT_NOTICE, SAMPLE_RATE
from src.errors import RateLimited
from src.telemetry import CALLS_LOG, TURNS_LOG, new_turn_id, turn_timer

# `pending` is the third fact one_turn has to hand back. A turn that was interrupted already holds
# the next turn's audio, and returning it is what stops the loop from opening the mic to ask for
# something it has been given. None means the next turn listens for itself, as every turn used to.
TurnResult = namedtuple("TurnResult", "spoken keep_going pending")


def speak_and_watch(turn, speech):
    """Play the reply with the mic open, and stop it if the user talks over it (VOX-011).

    -> the Capture that arrived during or after the reply, or None if nothing was said.

    Two streams, not one duplex stream: the mic runs at 16 kHz for silero and whisper, and Kokoro
    emits 24 kHz. A single duplex stream takes one sample rate, so it would mean resampling the reply
    to match the microphone — which is the one thing config.py and tts.py both refuse to do.

    No thread is started here either. Playback already runs on the output device's own callback
    thread, so the mic loop below can simply have the main thread while it happens.
    """
    playback = audio.play(speech.audio, sample_rate=speech.sample_rate,
                          on_first_audio=turn.first_audio, block=False)
    cut = {}

    def on_speech(first_speech_t):
        stopped_t = playback.abort()
        if stopped_t is None:
            # The reply had already played out, so nothing was interrupted. This is just the user
            # taking their next turn, and calling it a barge-in would inflate the numbers.
            return
        # Read while the stream is still open — `close()` below takes the latency with it.
        cut.update(stop_ms=round((stopped_t - first_speech_t) * 1000, 1),
                   played_s=playback.played_s, out_latency_s=playback.out_latency_s)
        print(f"  ── barge-in: stopped {cut['stop_ms']:.0f}ms after you started speaking "
              f"({cut['played_s']:.1f}s of {playback.reply_s:.1f}s played)", flush=True)

    try:
        # The stricter threshold applies to this whole capture, not only to the barge decision, so
        # the utterance carried into the next turn is endpointed slightly more conservatively than a
        # turn that began in silence. That is the price of one listener instead of two, and it is
        # visible: the next turn's record says its input was carried in.
        cap = vad.listen(on_speech=on_speech, threshold=BARGE_SPEECH_THRESHOLD, announce=False)
    finally:
        playback.close()

    if cut:
        turn.barge(cut["stop_ms"], cut["played_s"], playback.reply_s, cut["out_latency_s"])
    return cap


def one_turn(chosen, pending=None, watch=False):
    """-> TurnResult(spoken, keep_going, pending). `chosen` maps stage -> Arm.

    Three separate facts, because one bool used to carry the first two and they came apart at the TTS
    stage: `spoken` is whether audio actually reached the speaker, which is what the run counts,
    `keep_going` is whether the session can continue, and `pending` is audio the next turn already
    has. A failed voice with a good reply is (False, True, None) — nothing was heard, but there is no
    reason to end the conversation.

    `pending` in is an utterance captured while the previous reply was playing. This turn then does
    not open the mic at all; it already has its input.

    `watch` plays the reply interruptibly. False on the last turn of a run, because holding the mic
    open with no turn left to carry an interruption into would only make the process sit through
    max_wait_s before exiting — so `make demo` with its default single turn behaves exactly as
    VOX-002 and VOX-003 measured it.
    """
    turn_id = new_turn_id()
    print(f"\n--- turn {turn_id} ---")

    with turn_timer(turn_id, source="mic") as turn:
        turn.arms(**chosen)
        if pending is not None:
            cap = pending
            # A fact about this turn worth recording: these frames arrived while the previous reply
            # was still playing, so t_vad and time_to_first_audio here are measured from a mark that
            # falls inside the previous turn. Whether that previous reply was actually cut short is
            # on *its* record as `barged_in`, not on this one.
            turn.extra["input"] = "carried-in"
            print(f"  carried in: {len(cap) / SAMPLE_RATE:.2f}s captured during the last reply")
        else:
            cap = vad.listen()
        if cap is None:
            print("nothing heard — stopping.")
            return TurnResult(False, False, None)
        turn.vad(cap)

        with turn.stage("stt"):
            transcript = arms.stt(cap.segment, chosen["stt"].id, turn_id=turn_id,
                                  on_fallback=turn.fallback)
        print(f"you said : {transcript!r}")
        if not transcript:
            # Whisper returning empty on real audio is a provider problem, not a quiet user.
            print("empty transcript from STT — not calling the LLM.", file=sys.stderr)
            return TurnResult(False, False, None)

        with turn.stage("llm"):
            turn_state = state.build(transcript, turn_id, model_id=chosen["llm"].id)
            answer = turn_state.reply
        print(f"vox says : {answer!r}  "
              f"[intent={turn_state.intent} conf={turn_state.confidence:.2f} "
              f"next={turn_state.next_action}]")

        try:
            with turn.stage("tts"):
                speech = arms.tts(answer, chosen["tts"].id, turn_id=turn_id,
                                  on_fallback=turn.fallback)
        except Exception as e:
            # The reply is fine; only the voice failed. Losing the whole turn over that throws away
            # work the user waited for, so degrade to text — but loudly, on stderr and on the turn
            # record. A quiet degrade would read downstream as a turn that simply never spoke, and
            # the gate percentiles would improve because a turn dropped out of them.
            turn.extra["degraded"] = "tts"
            turn.extra["degrade_reason"] = f"{type(e).__name__}: {e}"
            print(f"TTS FAILED ({type(e).__name__}: {e}) — text only, nothing was spoken:\n"
                  f"  {answer}", file=sys.stderr)
            # Nothing is playing, so there is nothing to interrupt and nothing to carry forward. The
            # next turn opens the mic for itself, exactly as it did before barge-in existed.
            return TurnResult(False, True, None)

        print("speaking…", flush=True)
        if watch:
            next_cap = speak_and_watch(turn, speech)
        else:
            audio.play(speech.audio, sample_rate=speech.sample_rate,
                       on_first_audio=turn.first_audio)
            next_cap = None

    # A watched turn's record closes only once the *next* utterance has been endpointed, because one
    # listener spans both. So this line, and the turn's `ts`, land after the user has spoken again.
    print("  " + report(turn.written))

    if watch and next_cap is None:
        print("nothing heard after the reply — stopping.")
        return TurnResult(True, False, None)
    return TurnResult(True, True, next_cap)


def report(rec):
    """The line a human reads. The JSONL line is the record; this is so you see it happen."""
    def ms(key):
        v = rec.get(key)
        return f"{v:.0f}ms" if v is not None else "n/a"

    line = (f"vad {ms('t_vad_ms')} + stt {ms('t_stt_ms')} + llm {ms('t_llm_ms')} + "
            f"tts {ms('t_tts_ms')}  ->  time_to_first_audio {ms('time_to_first_audio_ms')}")
    # A turn that fell back is not comparable to one that did not, so the human-readable line says
    # so too rather than leaving it only in the JSONL.
    fell_back = rec.get("fell_back")
    line += f"   [fell back: {', '.join(fell_back)}]" if fell_back else ""

    # The stop latency VOX-011 is measured on. Printed on the same line as the split it belongs to,
    # not only at the moment of the interruption, so it survives in a scrollback and in a recording.
    if rec.get("barged_in"):
        buffered = rec.get("out_latency_s")
        line += (f"   [barge-in: stopped {ms('barge_stop_ms')} after speech started, "
                 f"{rec['played_s']:.1f}s of {rec['reply_s']:.1f}s played"
                 + (f", {buffered * 1000:.0f}ms of output buffer behind it]" if buffered else "]"))
    return line


def main():
    ap = argparse.ArgumentParser(description="VOX — chained turns with barge-in (VOX-002, VOX-011)")
    ap.add_argument("--turns", type=int, default=1,
                    help="how many turns before exiting. Every turn but the last plays its reply "
                         "with the mic open, so 2 or more is what makes barge-in demonstrable")
    arms.add_flags(ap)
    args = ap.parse_args()

    print("VOX — chained turn loop")

    # Resolving and loading happen before the turn starts. Kokoro takes ~10 s to load and silero a
    # moment; leaving that inside the turn would bury it in t_tts and t_vad and make the
    # latency split a lie. VOX-003 measures the warm path, which is the one users feel.
    print("resolving arms and loading local models…", flush=True)
    vad._vad_model()
    chosen = arms.select(args)
    print(arms.describe(chosen))

    print(f"\n{CONSENT_NOTICE}\n")

    spoken = 0
    pending = None
    for i in range(args.turns):
        try:
            result = one_turn(chosen, pending=pending, watch=i < args.turns - 1)
        except RateLimited as e:
            # Caught here and not inside the turn: a turn cannot decide the session is over, and
            # the wait is longer than a turn anyway. The turn record already carries the error,
            # written on the way out — this is so the user reads a sentence, not a traceback.
            print(f"\nRATE LIMITED — {e}", file=sys.stderr)
            break
        spoken += 1 if result.spoken else 0
        pending = result.pending
        if not result.keep_going:
            break

    print(f"\n{spoken} turn(s) completed.")
    print(f"  calls: {CALLS_LOG}")
    print(f"  turns: {TURNS_LOG}")
    return 0 if spoken else 1


if __name__ == "__main__":
    sys.exit(main())
