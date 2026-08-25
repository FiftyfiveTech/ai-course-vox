"""Self-echo rejection: telling VOX's own voice apart from a person, without cancelling anything.

There is no acoustic echo cancellation in this repo and none is in scope (ARCHITECTURE.md). On open
speakers that costs more than a cosmetic self-interrupt: `speak_and_watch` holds the mic open across
the reply, silero hears Kokoro, the capture is carried into the next turn as `pending`, and a turn
with a `pending` never opens the mic at all — it transcribes the machine's own reply and answers it.
The loop then feeds itself until the session clock runs out.

Cancelling the echo would mean an adaptive filter, a far-end reference aligned to the mic clock, and
a drift tracker between two devices with separate crystals. **Detecting** it needs none of that. The
question here is not "what did the person say underneath the echo" but "is this the person at all",
and that is answerable from the amplitude envelope:

  - Echo tracks the reply's envelope. Whatever the room, the speaker and the mic do to the waveform,
    loud stays loud and a pause stays a pause — delayed by the output buffer and the flight time.
  - A person talking over the reply does not. Their envelope is their own.

Envelopes and not waveforms is also what lets the 16 kHz mic and the 24 kHz reply be compared at all
without a resampler inside a live turn: both collapse to the same *time* base here, never to a common
sample rate.

## How much audio this needs, which is the finding that shaped the design

Measured on synthetic echo against synthetic speech (`scripts/measure_echo_guard.py`), at the
threshold that keeps 95% of echo, the fraction of real speech wrongly called echo is:

    200 ms of audio    the guard refuses to decide at all (see MIN_DECISION_MS)
    400 ms             18.3% of real speech wrongly called echo
    700 ms              0.0%
   1200 ms              0.0%   — and the margin is wide: echo 0.82, speech at most 0.53

Short windows have almost no envelope structure to match — 200 ms of speech is barely one syllable,
and one syllable's rise-and-fall correlates with any other's. So the decision is taken **where the
audio is**, not where it would be most convenient:

  the envelope      asked once per utterance, at the barge-in confirmation point, once
                    `MIN_DECISION_MS` of speech has accumulated. The guard, when it is on, holds
                    that decision open that long: cutting the reply at 200 ms stops the echo that
                    would have gone on to make the decision possible. It costs stop latency on a
                    guarded run, and `echo_guard` on the turn record says so.

  the transcript    `echoes_reply`, after STT, on carried-in audio only. Catches what the envelope
                    could not — an utterance that never reached `MIN_DECISION_MS` because the reply
                    ended, a machine whose delay is past `max_delay_ms`.

There is deliberately no third check when the turn ends. A guard that compares mic audio against
what the speaker has played needs both to mean the same instant, and at DONE they do not:
`VAD_SILENCE_MS` of hangover has passed and the reply has played on, so the echo sits further back
in the reference than the delay search reaches. That check does not fail loudly — it quietly answers
about the wrong stretch of reply, which is worse than not asking. The transcript layer covers the
same ground and needs no alignment at all.

Discarding a real utterance is worse than the bug, so the only place the guard discards is the place
where both the numbers and the alignment hold.

**The threshold is not measured on real acoustics.** A synthetic room proves the mechanism, not the
value: a real speaker/mic pair adds nonlinearity and a real room adds reflections this does not have.
Tuning `corr_threshold` needs a recording from a machine that actually has the problem —
docs/learning/vox-035-concepts.md says how to take one.
"""
import re

import numpy as np

# 8 ms — 125 envelope points a second. Short enough that the minimum below is still 87 points to
# correlate over, long enough that what is correlated is an envelope and not the waveform.
HOP_MS = 8.0

# Below this the answer is "not echo" regardless of what the correlation says: this is where the
# measurement above goes to zero, and 400 ms — the value that looked sufficient before the numbers
# were taken — throws away nearly one real utterance in five.
#
# It is also why `speak_and_watch` widens the barge-in confirmation window when the guard is on. A
# reply cut at 200 ms stops the echo that would have gone on to make the decision possible, so a
# guard that let the abort happen first would be handed 400 ms of echo and told to call it speech.
MIN_DECISION_MS = 700.0

# Silence would otherwise be the loudest feature in a log envelope: log10(1e-12) against log10(0.1)
# is a swing no speech detail can compete with, so two unrelated pauses lining up would score higher
# than a real echo. Clamped 40 dB below the window's own peak, which is quiet by any measure.
FLOOR_DB = 40.0


def envelope(samples, sample_rate, hop_ms=HOP_MS):
    """Per-hop RMS in dB, floored. -> float32 array at 1000/hop_ms Hz, whatever `sample_rate` was.

    dB rather than linear because the echo is 20-30 dB down: in linear RMS the correlation is carried
    by the loudest syllable in the window and the rest contributes nothing, which is precisely the
    structure the decision needs.
    """
    hop = max(1, int(round(sample_rate * hop_ms / 1000)))
    n = len(samples) // hop
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    frames = np.asarray(samples[:n * hop], dtype=np.float32).reshape(n, hop)
    db = 20.0 * np.log10(np.sqrt((frames ** 2).mean(axis=1)) + 1e-12)
    return np.maximum(db, db.max() - FLOOR_DB).astype(np.float32)


def _z(x):
    """Zero mean, unit variance. None for a flat window — silence correlates with nothing."""
    sd = float(x.std())
    return None if sd < 1e-6 else (x - x.mean()) / sd


def best_correlation(mic, ref, mic_rate, ref_rate, max_delay_ms=500.0, hop_ms=HOP_MS):
    """How well the mic audio's envelope matches the reply's, at the best lag. -> (r, delay_ms).

    `ref` must *end at what the speaker has actually played* — `Playback.reference` is exactly that —
    because the lag is searched backwards from that end. The echo lags the reply by the output
    buffer (0.182 s on the demo machine's MME device) plus the flight time across the room, and
    those are the delays `max_delay_ms` has to cover.

    Searching backwards from the end and not forwards from the start is not a detail: forwards, the
    window that gets searched is the *oldest* `max_delay_ms` of the reference, which is the one
    stretch of it the echo cannot be in. That version found delays of 350 ms for a 180 ms echo.

    r is Pearson, so it is bounded by 1 and means the same at any volume — which is what makes the
    echo being 26 dB down irrelevant. (0.0, 0.0) whenever the question cannot be asked: too little
    mic audio, too little reference, silence either side. A caller reads that as "not echo", which is
    the safe direction — a missed echo costs one turn, a false positive eats a real utterance.
    """
    a = envelope(mic, mic_rate, hop_ms)
    b = envelope(ref, ref_rate, hop_ms)
    span = len(b) - len(a)
    # Floor-divided, not compared in milliseconds: 700 ms of 16 kHz audio is 87 whole 8 ms hops and
    # 700/8 is 87.5, so a window of exactly the minimum length was refusing itself.
    if len(a) < int(MIN_DECISION_MS // hop_ms) or span < 0:
        return 0.0, 0.0

    za = _z(a)
    if za is None:
        return 0.0, 0.0

    lo = max(0, span - int(round(max_delay_ms / hop_ms)))
    best_r, best_lag = 0.0, span
    for lag in range(lo, span + 1):
        zb = _z(b[lag:lag + len(a)])
        if zb is None:
            continue
        r = float(np.dot(za, zb) / len(za))
        if r > best_r:
            best_r, best_lag = r, lag
    # Reported so a real recording can say whether max_delay_ms is generous enough, or whether the
    # true delay is being clipped and the guard is correlating against the wrong stretch of reply.
    return best_r, (span - best_lag) * hop_ms


def is_self_echo(mic, ref, mic_rate, ref_rate, threshold, max_delay_ms=500.0):
    """-> (bool, r, delay_ms). True when this mic audio is the reply coming back through the room."""
    r, delay_ms = best_correlation(mic, ref, mic_rate, ref_rate, max_delay_ms)
    return r >= threshold, round(r, 3), round(delay_ms, 1)


# --- the second layer: the transcript -------------------------------------------------------------

_WORD = re.compile(r"[a-z0-9']+")


def _words(text):
    return set(_WORD.findall((text or "").lower()))


def echoes_reply(transcript, reply, overlap=0.70, min_words=4):
    """Is this transcript the reply we just spoke, heard back? -> bool.

    The layer that catches what the envelope missed — a machine whose delay is past `max_delay_ms`,
    a capture too short to judge. It runs after STT, on carried-in audio only, and it is cheap
    exactly because by then the hard part has been done by whisper.

    Overlap is measured against the *transcript*, not against the reply: an echo is a second or two
    captured out of a reply that runs ten, so it is a fragment — nearly all of its words are in the
    reply, while the reply is mostly not in it. A symmetric similarity scores that pair low and
    misses the one case this exists for.

    `min_words` keeps "yes" and "no" out of it. A confirmation answer is short by nature and its
    words appear in the read-back it is answering, so VOX-020's gate would lose every "yes, go ahead"
    to this function otherwise.
    """
    t, r = _words(transcript), _words(reply)
    if len(t) < min_words or not r:
        return False
    return len(t & r) / len(t) >= overlap
