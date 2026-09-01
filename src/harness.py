"""One chained turn driven from a recording, for everything that is not a live mic.

`scripts/turn_from_fixture.py` and `scripts/compare_arms.py` both need the same thing: the exact
stage sequence `src/loop.py` runs, with the frames coming from a file instead of a device. That
sequence was already copied once; a third copy is how a comparison ends up measuring a pipeline the
loop does not actually run, which is the one failure a per-stage table cannot survive. So it lives
here, once, and the scripts are argument parsing and printing.

What is *not* here is `vad.listen`. The mic stays in `src/loop.py`, because the seam between the two
is `Endpointer` — it takes one 32 ms frame and knows nothing about audio devices — and keeping the
seam there is what makes `make demo` live-mic only.

Read `t_vad` and `time_to_first_audio` from a fixture turn with care. Frames arrive as fast as the
CPU can push them, so the `VAD_SILENCE_MS` hangover a person at a mic actually waits out collapses
to silero compute — roughly a second the live loop pays and this does not. `source` on the turn
record says which kind of run produced it, so the two cannot be quietly averaged.

`paced=True` is the exception, and it exists for the rehearsal (VOX-026): the frames are fed in at
one every 32 ms of wall clock, so the hangover is paid by the same code a person waits for and the
two numbers above become the live ones with the microphone removed. It is off by default because
every other caller here is measuring model stages and would only be paying for the clip's duration.
"""
from collections import namedtuple

import soundfile as sf
import torch
import torchaudio

from src import answer as answer_mod, arms, audio as audio_out, vad
from src.config import SAMPLE_RATE
from src.telemetry import new_turn_id, turn_timer

# What one fixture turn produced. `record` is the line that reached runs/turns.jsonl — the same
# object, not a copy of the numbers, so nothing printed can drift from what was logged.
#
# `reply` is the text that was spoken either way; `answer` is the VOX-031 Answer when the turn took
# the grounded path and None when it took the plain one, so a caller can see which ran without
# re-deriving it. Appended last with a default, so every existing positional caller is unaffected.
#
# `carried` is the utterance captured while this reply was playing, when the caller asked for a
# watched reply (VOX-026). None on every ordinary turn, and on a watched one where nobody spoke.
TurnRun = namedtuple("TurnRun", "record turn_id capture transcript reply speech answer carried")
TurnRun.__new__.__defaults__ = (None, None)


class NoSpeech(RuntimeError):
    """The endpointer found no turn in the clip. Nothing was measured and no turn line was written."""


class EmptyTranscript(RuntimeError):
    """STT returned an empty string on real audio. A provider problem, not a quiet speaker."""


def load_16k_mono(path):
    """-> float32 mono at SAMPLE_RATE, which is what silero and whisper both want."""
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    mono = torch.from_numpy(data.mean(axis=1))
    if sr != SAMPLE_RATE:
        mono = torchaudio.functional.resample(mono, sr, SAMPLE_RATE)
    return mono.numpy()


def fixture_turn(chosen, clip, source, *, play=True, fallback=True, on_fallback=None, echo=None,
                 segment=None, idx=None, plain=None, paced=False, watch=None, after_play=None,
                 extra=None, history=None):
    """Run one turn on `clip` with the arms in `chosen`. -> TurnRun.

    `chosen` maps stage -> Arm, exactly as `arms.select()` returns it.

    The last four arguments are what a scripted session needs (VOX-026) and no other caller passes:

    `plain` is `answer_mod.turn_reply`'s hook of the same name, passed straight through. Without it
    the un-retrieved path is `nlu.reply` and there is no `TurnState`, so nothing can ask for
    confirmation — which is why `make turn` has never shown VOX-020.

    `paced` feeds the endpointer one frame every 32 ms of wall clock instead of as fast as the CPU
    allows, so the `VAD_SILENCE_MS` hangover is paid in real time and `t_vad` / `time_to_first_audio`
    become the live numbers rather than the optimistic ones this docstring warns about below.

    `watch` is a listener, and passing one plays the reply interruptibly through
    `loop.speak_and_watch` — the live barge-in path, not a copy of it. Whatever it captures comes
    back as `TurnRun.carried`, for the next turn to run on exactly as the loop does.

    `after_play(turn, turn_id, chosen)` runs inside the turn timer once the reply has been spoken.
    It is where a scripted run puts `loop.confirmation_leg`, so the yes/no exchange is timed on this
    turn's record and the *policy* about which turn confirms stays out of the harness.

    `history` is a src.history.History for a scripted multi-turn session (VOX-034), passed
    straight through to turn_reply. Defaults to None — *not* to a fresh History — for the same
    reason `idx` defaults to None: scripts/compare_arms.py measures arms, and a rewritten query
    would change what was retrieved between two rows of the same table.

    `extra` is stamped onto the turn record as-is — facts the caller knows and this function cannot,
    such as `input="carried-in"` for a turn running on audio captured during the previous reply.
    `src/loop.py` writes that same field for the same reason.

    `fallback` is passed straight through to all three `arms.*` calls, and it is the one argument a
    caller has to think about. The loop wants True: a rate limit should cost the quality, not the
    turn. Anything *comparing* arms wants False, for the reason `arms.fallback_for` gives — a
    rescued remote arm prints the local arm's latency on the remote arm's row, and attributing
    numbers to models is the whole point of a comparison.

    `play=False` leaves `time_to_first_audio_ms` null. That is not a faster turn, it is an
    unmeasured one, and the turn record's `ok` goes false to say so.

    `on_fallback` defaults to the turn's own `TurnTimer.fallback`, not to nothing. A fallback that
    does not reach the record leaves `<stage>_model` naming an arm that never ran, and every
    per-turn comparison reads exactly that field — so the default has to be the loud one.

    `idx` is a retrieval index, and passing one turns this turn's reply into the grounded path when
    the documents cover what was said (VOX-032). It defaults to None — *not* to the process-wide
    index — because the callers here measure things: `scripts/compare_arms.py` is comparing arms and
    a grounded turn carries ~1500 extra tokens of context, which would land in the llm column as if
    the arm were slower. A caller that wants the grounded path says so.

    `segment` skips endpointing and runs on an already-endpointed `Capture`.

    **It invalidates `time_to_first_audio_ms`, so pass it only when you are not reading that field.**
    A `Capture` carries `speech_end_t`, the `perf_counter` stamp of when its speech ended, and
    time_to_first_audio is measured from there — the user has been waiting since they stopped
    talking. Reuse the capture on a second turn and that origin is a moment in the past, so the
    number silently grows by everything that happened in between. Three identical turns read 6.5 s,
    39.4 s and 63.6 s that way. `t_vad_ms` is repeated rather than re-measured for the same reason.

    It exists for tests, which supply a fake capture and no speaker. `scripts/compare_arms.py`
    deliberately does *not* use it: it endpoints every turn afresh and asserts the segment came out
    identical, which buys the same fairness without freezing the clock.

    On failure the exception propagates with the turn record attached as `exc.turn_record`, because
    the failed turn's latency split is exactly what a caller needs and there is no return value to
    put it in. The exception type is untouched, so a `RateLimited` is still a `RateLimited`.
    """
    say = echo if echo is not None else (lambda *a, **k: None)
    turn_id = new_turn_id()

    try:
        with turn_timer(turn_id, source=source) as turn:
            turn.arms(**chosen)
            turn.extra.update(extra or {})
            notify = turn.fallback if on_fallback is None else on_fallback

            cap = segment
            if cap is None:
                frames = vad.frames_from(clip)
                cap, state = vad.endpoint_frames(
                    vad.paced(frames, echo=say) if paced else frames)
                if cap is None:
                    raise NoSpeech(f"endpointer found no turn in {source} (state={state})")
                say(f"endpointed: {len(cap) / SAMPLE_RATE:.2f}s ({cap.spoken_s:.2f}s speech), "
                    f"state={state}")
            turn.vad(cap)

            with turn.stage("stt"):
                transcript = arms.stt(cap.segment, chosen["stt"].id, turn_id=turn_id,
                                      on_fallback=notify, fallback=fallback)
            say(f"you said : {transcript!r}")
            if not transcript:
                raise EmptyTranscript(
                    f"{chosen['stt'].id} returned an empty transcript for {source} — not calling "
                    f"the LLM.")

            # The same routing the live loop runs, from the same function (VOX-032) — a second copy
            # here is exactly how a fixture-driven comparison ends up timing a pipeline `make demo`
            # does not. `idx=None` is the default, so nothing that was not handed a knowledge base
            # changes behaviour: `scripts/compare_arms.py` still times the plain reply path.
            answered = answer_mod.turn_reply(transcript, turn_id, idx=idx, turn=turn,
                                             model_id=chosen["llm"].id, on_fallback=notify,
                                             fallback=fallback, plain=plain, history=history)
            reply = answered.text
            say(f"vox says : {reply!r}")

            with turn.stage("tts"):
                speech = arms.tts(reply, chosen["tts"].id, turn_id=turn_id,
                                  on_fallback=notify, fallback=fallback)

            carried = None
            if play and watch is not None:
                # Borrowed from the loop rather than reimplemented: the stop latency, the mark it is
                # measured from and the carry-forward are VOX-011's and belong in one place. Imported
                # here and not at module scope to keep the direction of this module's dependencies as
                # its own docstring states them — the mic lives over there.
                from src import loop
                say("speaking… (interruptible)")
                carried = loop.speak_and_watch(turn, speech, listen=watch)
            elif play:
                say("speaking…")
                audio_out.play(speech.audio, sample_rate=speech.sample_rate,
                               on_first_audio=turn.first_audio)
            else:
                say("(not playing: time_to_first_audio will be null)")

            if after_play is not None:
                after_play(turn, turn_id, chosen)
    except Exception as e:
        # `turn` is bound by the `with` before its body runs, so the record is reachable here even
        # though the body did not finish. turn_timer already wrote it, error included.
        e.turn_record = turn.written
        e.turn_id = turn_id
        raise

    return TurnRun(turn.written, turn_id, cap, transcript, reply, speech, answered.answer, carried)
