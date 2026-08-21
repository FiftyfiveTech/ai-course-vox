"""One chained turn, end to end: mic -> VAD -> STT -> LLM -> TTS -> speaker.

    uv run python -m src.loop          one turn, then exit
    uv run python -m src.loop --turns 3
    uv run python -m src.loop --minutes      talk to it until the session clock runs out
    VOX_SESSION_MINUTES=10 uv run python -m src.loop --minutes

    uv run python -m src.loop --stt openai/whisper-base --tts microsoft/speecht5_tts

Still one turn by default: VOX-002 is "you speak, you hear a reply". The five-field latency split is
VOX-003 and it is here from the first commit that has a turn to measure — retrofitting timings onto a
loop that already runs means tuning against numbers nobody watched being taken.

A run is bounded by a turn count or by a wall clock, and the two differ in what ends them, not only
in how long they last. `--turns N` is N replies and stops at the first silence: with a fixed count
there is nothing to wait for. `--minutes M` is a conversation — turns keep coming until the deadline,
every reply is interruptible, and a pause inside it is a person thinking rather than a session that
finished. That is what `make demo` runs, because "one turn, then exit" demonstrates a pipeline and
not an agent you can talk to. How long it lasts is config.SESSION_MINUTES (VOX_SESSION_MINUTES), so
a dev iterating on one stage can have half a minute without editing anything.

Barge-in (VOX-011) is what `--turns 2` or more buys. Every turn but the last plays its reply with the
mic still open, so speaking over VOX stops it mid-sentence and the words that stopped it are carried
into the next turn as its input. There is no second listener: the ordinary endpointer runs across the
whole reply and past it, so an interruption and a normal next utterance are the same code path and
differ only in whether anything was still playing when they arrived.

Which model runs each stage is a flag (VOX-006), and the arms are named on the turn record, so two
runs with different arms cannot be quietly averaged together.

A turn about the policy documents is answered *from* them (VOX-032). After STT the transcript goes
to retrieval, and the chunks that clear the score floor are what the reply is written from — with
the doc:page it came from printed under the answer and logged on the turn line. `--no-kb` forces
that path off for a whole run, which is how the two are compared without editing code.

What retrieval does *not* claim is a request rather than a question, and it goes to VOX-019's
structured extractor: `state.build` returns the reply to speak together with the intent, the
entities and the `next_action` that VOX-020's confirmation gate reads — so an action that changes
data is read back and waits for a spoken yes or no before it proceeds. One LLM call per turn on
either path; which one ran is visible on the printed line and on the turn record.
"""
import argparse
import sys
import time
from collections import namedtuple

from src import answer as answer_mod, arms, audio, confirm, state, vad
from src.config import (BARGE_SPEECH_THRESHOLD, CONSENT_NOTICE, SAMPLE_RATE, SESSION_MINUTES,
                        SESSION_QUIET_LIMIT)
from src.errors import RateLimited
from src.telemetry import CALLS_LOG, TURNS_LOG, new_turn_id, turn_timer

# `pending` is the third fact one_turn has to hand back. A turn that was interrupted already holds
# the next turn's audio, and returning it is what stops the loop from opening the mic to ask for
# something it has been given. None means the next turn listens for itself, as every turn used to.
TurnResult = namedtuple("TurnResult", "spoken keep_going pending")


class Budget:
    """What ends a run: a turn count (VOX-002, VOX-011) or a wall clock (`--minutes`).

    One object so the loop in `main()` reads the same either way, and so the two policies are stated
    once, here, instead of being spread across three `if timed:` branches down there.

    `clock` is injectable because the alternative is a unit test that takes three real minutes. It
    is looked up here rather than defaulted in the signature, so that a default argument bound at
    import time cannot quietly outrank what a test patched.
    """

    def __init__(self, turns=None, minutes=None, clock=None):
        if turns is not None and minutes is not None:
            raise ValueError("a run is bounded by turns or by minutes, not both")
        self.clock = clock or time.monotonic
        self.turns = 1 if (turns is None and minutes is None) else turns
        self.started = self.clock()
        self.deadline = self.started + minutes * 60 if minutes is not None else None
        self.taken = 0
        self.quiet_streak = 0

    @property
    def timed(self):
        return self.deadline is not None

    def left_s(self):
        """Seconds still on the clock, or None for a run counted in turns."""
        return max(0.0, self.deadline - self.clock()) if self.timed else None

    def elapsed_s(self):
        return self.clock() - self.started

    def open(self):
        """Is there room for another turn? Asked before each one.

        The deadline is checked at the *start* of a turn and never during it: a turn that began in
        time gets to finish, so the run overruns by at most one reply rather than cutting the user
        off mid-sentence to hit a number nobody is measuring.
        """
        if self.quiet_streak >= SESSION_QUIET_LIMIT:
            return False
        return self.left_s() > 0 if self.timed else self.taken < self.turns

    def watch(self):
        """Should this turn's reply play with the mic open? -> True if a turn can follow it.

        Timed runs watch every reply, which is the point of them — barge-in is only reachable when
        there is a next turn to carry the interruption into, and inside a conversation there always
        is until the clock says otherwise.
        """
        if not self.open():
            return False               # the wind-down turn: nothing after it to carry audio into
        return True if self.timed else self.taken < self.turns - 1

    def took(self, quiet):
        """Record a finished turn. `quiet` is a turn whose listen heard nothing."""
        self.taken += 1
        self.quiet_streak = self.quiet_streak + 1 if quiet else 0

    def clock_str(self, seconds=None):
        """m:ss, for the line a person reads between turns."""
        total = int(round(self.elapsed_s() if seconds is None else seconds))
        return f"{total // 60}:{total % 60:02d}"


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


def one_turn(chosen, pending=None, watch=False, idx=None):
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
    max_wait_s before exiting — so a single-turn run behaves exactly as VOX-002 and VOX-003
    measured it.

    A turn that hears nothing comes back as keep_going=False, and that is all it says: whether an
    empty listen ends the session is `main()`'s call, because it depends on what bounds the run. In
    a fixed count it does; inside a timed conversation a pause is not an ending.

    `idx` is the retrieval index, built once by `main()` before any turn starts. None means this run
    has no knowledge base — either nothing indexed or `--no-kb` — and every reply comes from the
    plain prompt.
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
            print("nothing heard.")
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

        # Two tickets rewrote this stage and retrieval is what decides which one runs. VOX-032
        # asks the documents first: a question they cover is answered *from* them, and a policy
        # answer changes no data, so there is no state to extract and nothing to confirm. What
        # retrieval does not claim is a request rather than a question, and that is VOX-019's —
        # `state.build` writes the reply *and* the intent/next_action that VOX-020's gate reads
        # further down. `idx=None` (no corpus on this machine) skips retrieval entirely and every
        # turn takes the second path; that is announced once at startup, not once per turn.
        #
        # One LLM call per turn either way. Routing on the measured floor rather than running both
        # prompts is what keeps t_llm comparable with every turn logged before this merge — two
        # calls a turn would double the stage that most of the VOX-003 budget is spent in.
        turn_state = None

        def extract_state(text, tid, model_id=None, **_):
            """The plain-reply path, as VOX-019 now writes it: structured state, and its `reply`
            field is what gets spoken. Anything retrieval vouched for goes to the grounded prompt
            instead and never arrives here."""
            nonlocal turn_state
            turn_state = state.build(text, tid, model_id=model_id)
            return turn_state.reply

        reply = answer_mod.turn_reply(transcript, turn_id, idx=idx, turn=turn,
                                      model_id=chosen["llm"].id, on_fallback=turn.fallback,
                                      plain=extract_state)
        print(f"vox says : {reply.text!r}" + (
            f"  [intent={turn_state.intent} conf={turn_state.confidence:.2f} "
            f"next={turn_state.next_action}]" if turn_state is not None else ""))
        print("  " + grounding(reply, kb=idx is not None))

        try:
            with turn.stage("tts"):
                speech = arms.tts(reply.text, chosen["tts"].id, turn_id=turn_id,
                                  on_fallback=turn.fallback)
        except Exception as e:
            # The reply is fine; only the voice failed. Losing the whole turn over that throws away
            # work the user waited for, so degrade to text — but loudly, on stderr and on the turn
            # record. A quiet degrade would read downstream as a turn that simply never spoke, and
            # the gate percentiles would improve because a turn dropped out of them.
            turn.extra["degraded"] = "tts"
            turn.extra["degrade_reason"] = f"{type(e).__name__}: {e}"
            print(f"TTS FAILED ({type(e).__name__}: {e}) — text only, nothing was spoken:\n"
                  f"  {reply.text}", file=sys.stderr)
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

        # VOX-020: if the LLM asked for confirmation, listen for yes/no. A grounded answer has no
        # TurnState and cannot reach this — it read a document out loud, which is not an action to
        # confirm.
        if turn_state is not None and confirm.needs_confirmation(turn_state):
            print("  [confirmation required — listening for yes/no]", flush=True)
            yn_cap = vad.listen(announce=False)
            if yn_cap is None:
                print("  nothing heard — treating as cancel")
                yn_response = "no"
            else:
                with turn.stage("stt"):
                    yn_transcript = arms.stt(yn_cap.segment, chosen["stt"].id,
                                             turn_id=turn_id, on_fallback=turn.fallback)
                yn_response = confirm.classify_response(yn_transcript)
                print(f"  confirmation response: {yn_transcript!r} -> {yn_response}")

            if yn_response == "yes":
                print("  confirmed — action would proceed here")
            elif yn_response == "no":
                cancel_text = confirm.cancelled_reply()
                with turn.stage("tts"):
                    cancel_speech = arms.tts(cancel_text, chosen["tts"].id,
                                             turn_id=turn_id, on_fallback=turn.fallback)
                audio.play(cancel_speech.audio, sample_rate=cancel_speech.sample_rate)
                print(f"  cancelled: {cancel_text!r}")
            else:
                unclear_text = confirm.unclear_reply()
                with turn.stage("tts"):
                    unclear_speech = arms.tts(unclear_text, chosen["tts"].id,
                                              turn_id=turn_id, on_fallback=turn.fallback)
                audio.play(unclear_speech.audio, sample_rate=unclear_speech.sample_rate)
                print(f"  unclear: {unclear_text!r}")

    # A watched turn's record closes only once the *next* utterance has been endpointed, because one
    # listener spans both. So this line, and the turn's `ts`, land after the user has spoken again.
    print("  " + report(turn.written))

    if watch and next_cap is None:
        print("nothing heard after the reply.")
        return TurnResult(True, False, None)
    return TurnResult(True, True, next_cap)


def grounding(reply, kb=True):
    """The line under the answer that says what it was grounded in, or why it was not.

    Printed on every turn rather than only on the grounded ones. A run where retrieval silently
    stopped contributing — a re-index that produced nothing, a floor edited upwards — looks from the
    outside like a run of questions the documents happen not to cover, and those two need to be
    distinguishable while the demo is happening, not afterwards in the JSONL.

    `kb` is False when this run has no index at all, which is why "nothing was retrieved" and
    "nothing was retrievable" do not read the same here.
    """
    if reply.answer is None:
        return ("plain reply — nothing was retrieved: no chunk cleared the floor" if kb
                else "plain reply — no knowledge base on this run")
    if not reply.answer.grounded:
        return (f"refused by the model — {len(reply.hits)} chunk(s) cleared the floor "
                f"({', '.join(h.source for h in reply.hits)}) but do not answer it")
    return (f"grounded in {', '.join(reply.answer.labels)} "
            f"(top score {reply.hits[0].score:.3f}, {len(reply.hits)} chunks in context)")


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
    length = ap.add_mutually_exclusive_group()
    length.add_argument("--turns", type=int, default=None,
                        help="how many turns before exiting (default 1). Every turn but the last "
                             "plays its reply with the mic open, so 2 or more is what makes "
                             "barge-in demonstrable")
    length.add_argument("--minutes", type=float, nargs="?", const=SESSION_MINUTES, default=None,
                        help=f"talk to it for this long instead of for a fixed number of turns: "
                             f"turns keep coming until the clock runs out, every reply is "
                             f"interruptible, and a pause is a pause rather than the end of the "
                             f"session. Bare `--minutes` is config.SESSION_MINUTES "
                             f"(VOX_SESSION_MINUTES, currently {SESSION_MINUTES:g}), which is what "
                             f"`make demo` runs")
    ap.add_argument("--no-kb", action="store_true",
                    help="skip retrieval and answer every turn from the plain reply prompt "
                         "(VOX-032). The pre-RAG loop, for comparing against it without an edit")
    arms.add_flags(ap)
    args = ap.parse_args()

    budget = Budget(turns=args.turns, minutes=args.minutes)
    print("VOX — chained turn loop")

    # Resolving and loading happen before the turn starts. Kokoro takes ~10 s to load and silero a
    # moment; leaving that inside the turn would bury it in t_tts and t_vad and make the
    # latency split a lie. VOX-003 measures the warm path, which is the one users feel.
    print("resolving arms and loading local models…", flush=True)
    vad._vad_model()
    chosen = arms.select(args)
    print(arms.describe(chosen))

    # Built here for the same reason the weights are loaded here: it is per-process work, and a
    # build inside the first turn would land in that turn's numbers. None is a supported state —
    # see answer_mod.knowledge_base() — and the reason is printed there rather than per turn.
    idx = None if args.no_kb else answer_mod.knowledge_base()
    if args.no_kb:
        print("knowledge base: off (--no-kb) — every reply comes from the plain reply prompt")

    if budget.timed:
        print(f"talking for {args.minutes:g} minute(s) — keep going as long as you like, "
              f"talk over a reply to interrupt it, Ctrl-C to stop early")

    print(f"\n{CONSENT_NOTICE}\n")

    spoken = 0
    pending = None
    # The second clause is the wind-down, and it belongs to timed runs only: the deadline passed
    # while the last reply was playing and the user answered it anyway. Their words are already
    # captured, so the session spends one more unwatched turn replying to them rather than exiting
    # on a sentence it heard and dropped. A run counted in turns has no use for it — there the
    # count is the whole bound, and `--turns 3` means three turns however the third one ends.
    while budget.open() or (budget.timed and pending is not None):
        if budget.timed:
            print(f"\n  {budget.clock_str(budget.left_s())} left in this session"
                  if budget.open() else "\n  time is up — one last reply to what you just said.")
        try:
            result = one_turn(chosen, pending=pending, watch=budget.watch(), idx=idx)
        except RateLimited as e:
            # Caught here and not inside the turn: a turn cannot decide the session is over, and
            # the wait is longer than a turn anyway. The turn record already carries the error,
            # written on the way out — this is so the user reads a sentence, not a traceback.
            print(f"\nRATE LIMITED — {e}", file=sys.stderr)
            break
        except KeyboardInterrupt:
            # The documented way out of a timed session — CONSENT_NOTICE says so — so it ends the
            # run the way the clock does, with the counts and the log paths below, rather than with
            # a traceback out of whichever stage happened to be running.
            print("\nstopped.", file=sys.stderr)
            break
        spoken += 1 if result.spoken else 0
        pending = result.pending
        budget.took(quiet=not result.keep_going)
        if not result.keep_going:
            if not budget.timed:
                break
            if budget.open():
                # Only inside a timed run, and only while the clock agrees: silence here is a
                # pause, and SESSION_QUIET_LIMIT is what keeps that from meaning "forever".
                print("  still listening — say something, or Ctrl-C to stop.")

    print(f"\n{spoken} turn(s) completed in {budget.clock_str()}.")
    print(f"  calls: {CALLS_LOG}")
    print(f"  turns: {TURNS_LOG}")
    return 0 if spoken else 1


if __name__ == "__main__":
    sys.exit(main())
