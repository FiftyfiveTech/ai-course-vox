# VOX-035 — concept primer: detecting an echo you cannot cancel

The ticket: *on open speakers the reply is captured by the mic, carried into the next turn as its
input, and answered — the session talks to itself until the clock runs out. Stop the loop without
adding acoustic echo cancellation, without changing what a run on headphones measures, and without
ever discarding a real utterance.*

Written alongside the implementation rather than before it — this started as a support question
about a colleague's machine, not as a planned ticket. Where this file and the code disagree, the
code and `ARCHITECTURE.md` win and this file is wrong.

---

## 1 · The bug is the carry-forward, not the self-interrupt

Everyone describes this bug as "it interrupts itself". That is the visible half and the harmless
half. The mechanism that actually costs you the session is four hops in `src/loop.py`:

1. `speak_and_watch` holds the mic open across the reply — that is what makes barge-in cheap.
2. Silero hears Kokoro. It is not wrong to: speaker bleed *is* speech.
3. The capture is returned as `TurnResult.pending`.
4. `one_turn` with a `pending` **never opens the mic**. It transcribes what it was given.

So the next turn answers a mangled transcription of the previous reply, speaks the answer, and hop 1
starts again. Cutting the reply short at hop 2 does not break the cycle; only refusing at hop 3 or 4
does.

**Why here:** the same shape appears any time a component is fed its own output through a channel
that does not know where the audio came from. The fix is never "make the detector stricter" — it is
to put an identity check on the edge where output can re-enter as input.

**Pitfall:** debugging the VAD threshold. `BARGE_SPEECH_THRESHOLD` is already 0.7 for exactly this
reason and no value fixes it, because there is no threshold at which a recording of speech stops
being speech.

---

## 2 · Detection is a different problem from cancellation, and much cheaper

Acoustic echo cancellation *subtracts* the echo: an adaptive filter, a far-end reference aligned to
the mic clock, and a drift tracker between two devices with separate crystals. That is a real
dependency and a real ticket.

The question here is not "what did the person say underneath the echo". It is "is this the person at
all" — a **classification**, and it can be answered from the amplitude envelope, because echo tracks
the reply's envelope while a person does not. Pearson correlation over a log-RMS envelope, searched
over a delay window. Two array operations and a short loop.

That reframing also dissolves the 16 kHz / 24 kHz problem `ARCHITECTURE.md` warns about. Envelopes
are computed to a common **time** base (8 ms hops), never to a common sample rate, so no resampler
runs inside a live turn.

**Why here:** "can I answer a cheaper question" is worth asking before importing a library. The
expensive version of this problem is famous enough that reaching for `webrtc-audio-processing` feels
like the professional move — and it would have been the wrong one for what the loop actually needs.

**Pitfall:** believing detection is *free*. See §3: it needs more evidence than the barge-in decision
does, and that costs latency.

---

## 3 · The measurement changed the design twice

The first implementation checked at 200 ms — the barge-in confirmation window — because that is when
the reply gets cut. `scripts/measure_echo_guard.py` says what that costs, at the threshold that
keeps 95% of echo:

| audio available | real speech wrongly called echo |
|---|---|
| 200 ms | unusable |
| 400 ms | 18.3% |
| 700 ms | 0.0% |
| 1200 ms | 0.0%, margin 0.82 vs 0.53 |

200 ms of speech is barely one syllable, and one syllable's rise and fall correlates with any
other's. So `MIN_DECISION_MS` is 700, and `speak_and_watch` **holds the barge-in decision open that
long when the guard is on** — because cutting the reply at 200 ms stops the echo that would have
gone on to make the decision possible. A guarded run has worse stop latency and its turn records say
`echo_guard: true` so those numbers are never averaged in with VOX-011's.

The second change: the check was also being made when the turn ended, on the whole utterance, which
looks like strictly more evidence. It is not — it is *misaligned* evidence. By DONE, `VAD_SILENCE_MS`
of hangover has passed and the reply has played on, so the echo sits further back in the reference
than the delay search reaches. That check does not fail loudly; it quietly answers about the wrong
stretch of reply. It was deleted, and the transcript layer covers the same ground with no alignment
to get wrong.

**Why here:** both changes came from a number, and neither was visible from reading the code. This is
what the repo's "measured gates, not vibes" rule buys.

**Pitfall:** a fixture that flatters you. The first version of the synthetic voice was a 4 Hz
amplitude modulation, so any window shorter than a syllable was nearly a straight line — unrelated
speech scored 0.99 and the guard looked useless. It was the fixture. Phone-scale segments with their
own level, spectrum and pauses are the minimum that makes a short window contain anything to match.

---

## 3b · A guard that rearms leaves a residue, and the residue is the bug again

Reported from a real run once the first version was in use: *the guard ran, but sometimes after
finishing it detected the last words.* Two holes, and neither is a threshold problem.

**The output buffer.** `finished` is when the device stopped *pulling*, not when the room went
quiet — 0.182 s of buffer is still on its way out on the demo machine, and every millisecond of it
is audible echo. The guard returned `False` the moment `finished` was set, which is precisely when
the reply's last words are still in the air. It now keeps looking for the buffer plus one delay
window, and widens the delay search by however long ago the device stopped: the reference cannot
advance past the end of the reply, so as the mic runs on, the echo sits further and further back
inside it.

**The rearm residue.** The envelope guard rearms every time it rejects. Near the end of a reply the
last rearm leaves a capture of a second or less — below `MIN_DECISION_MS`, so the envelope test
declines to judge it, and below `min_words`, so the overlap test declines too. It becomes a turn.

The second one cannot be fixed by lowering `MIN_DECISION_MS`: §3's table is why that floor exists.
It is fixed one layer up, where whisper's words make a *narrower* question askable — not "do these
words appear in the reply" but "are these words how the reply **ended**". A tail echo is a suffix by
construction. `tail_slack` is 1, which is the whole trade: at 2, a reply ending "...before two years
of allotment" makes "two years" a tail and a user asking about those two years loses their turn.

**Why here:** a detector that resets on each firing does not fail cleanly at the boundary — it fails
by leaving a fragment just under every threshold it has. Worth expecting in anything that rearms.

**Pitfall:** the fix that reaches for `min_words = 2`. That is the same question asked more loosely,
and it takes VOX-020's "yes, go ahead" with it. A different, stricter question is what was needed —
and the carried-in path still asks `confirm.classify_response` first, because a real confirmation
said over a read-back arrives as carried-in audio and no echo rule should be able to eat one.

---

## 4 · Asymmetric costs decide where a classifier is allowed to act

Two ways to be wrong, and they are not equally bad:

- **Missed echo** — one wasted turn. The transcript layer catches most of it. Recoverable.
- **False positive** — a real utterance is discarded. The user repeats themselves and it may be
  discarded again. Worse than the bug being fixed.

Everything ambiguous therefore resolves to "not echo": too little audio, too little reference,
silence on either side, a reply that has finished playing. And the guard is **off by default**, so a
machine on headphones runs the code path VOX-011 was measured on, untouched.

**Why here:** it is the same reasoning as VOX-031's refusal to answer from a weak retrieval, and the
same as `abort()` returning `None` rather than counting a patient user as an interruption. When two
errors cost different amounts, the threshold does not go in the middle.

**Pitfall:** reporting a guard's quality as accuracy. 95% accurate is meaningless here; "at the
threshold that keeps 95% of echo, how much real speech is lost" is the question, and only the second
number decides whether the thing can ship.

---

## 5 · What is still not true

The threshold is measured on a **synthetic** room: a delay, a one-pole lowpass, two reflections, an
attenuation and noise. A real speaker/mic pair is nonlinear and a real room has a tail this does not
model. `corr_threshold: 0.75` is the mechanism working, not a value for anyone's machine.

To tune it where the problem actually lives, make the machine report its own correlations by
running the guard with a threshold nothing can reach:

```
VOX_ECHO_CORR=1.01 VOX_ECHO_GUARD=1 make demo    # on speakers; let a turn or two run away
```

Nothing is rejected at 1.01, but every asking is recorded, so `runs/turns.jsonl` carries
`self_echo_r` per turn. Put `corr_threshold` between the runaway turns and the ones you spoke in.

**And headphones remain the real fix.** This guard stops a session talking to itself. It does not
give you working barge-in on open speakers, it costs stop latency, and it is one more thing that can
be wrong. `docs/CONTRIBUTING.md` and the dry-run checklist still say headphones, and they are right.
