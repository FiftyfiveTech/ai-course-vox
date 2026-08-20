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
"""
from collections import namedtuple

import soundfile as sf
import torch
import torchaudio

from src import arms, audio as audio_out, nlu, vad
from src.config import SAMPLE_RATE
from src.telemetry import new_turn_id, turn_timer

# What one fixture turn produced. `record` is the line that reached runs/turns.jsonl — the same
# object, not a copy of the numbers, so nothing printed can drift from what was logged.
TurnRun = namedtuple("TurnRun", "record turn_id capture transcript reply speech")


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
                 segment=None):
    """Run one turn on `clip` with the arms in `chosen`. -> TurnRun.

    `chosen` maps stage -> Arm, exactly as `arms.select()` returns it.

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
            notify = turn.fallback if on_fallback is None else on_fallback

            cap = segment
            if cap is None:
                cap, state = vad.endpoint_frames(vad.frames_from(clip))
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

            with turn.stage("llm"):
                reply = nlu.reply(transcript, turn_id, model_id=chosen["llm"].id,
                                  on_fallback=notify, fallback=fallback)
            say(f"vox says : {reply!r}")

            with turn.stage("tts"):
                speech = arms.tts(reply, chosen["tts"].id, turn_id=turn_id,
                                  on_fallback=notify, fallback=fallback)

            if play:
                say("speaking…")
                audio_out.play(speech.audio, sample_rate=speech.sample_rate,
                               on_first_audio=turn.first_audio)
            else:
                say("(not playing: time_to_first_audio will be null)")
    except Exception as e:
        # `turn` is bound by the `with` before its body runs, so the record is reachable here even
        # though the body did not finish. turn_timer already wrote it, error included.
        e.turn_record = turn.written
        e.turn_id = turn_id
        raise

    return TurnRun(turn.written, turn_id, cap, transcript, reply, speech)
