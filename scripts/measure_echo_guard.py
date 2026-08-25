"""Measure how much audio the self-echo guard needs before it can be trusted (VOX-035).

The guard in src/echo.py discards a capture whose amplitude envelope matches the reply that was
playing. Discarding a *real* utterance is worse than the bug it fixes, so the number that decides
the design is not "how well does echo score" — it is "at the threshold that keeps almost all echo,
how much real speech is thrown away", against how much audio the decision was taken on.

Usage:
    .venv/Scripts/python scripts/measure_echo_guard.py          # Windows
    .venv/bin/python scripts/measure_echo_guard.py

    --trials N        utterances per window length (default 60)

## What is synthetic here, and what that costs

With no recording, both the reply and the interrupting speech are generated: 40-120 ms phone-like
segments with their own spectral tilt and level, 18% of them silent, and the "room" is a delay, a
one-pole lowpass, two reflections, an attenuation and noise. That is enough to show how the decision
degrades as the window shortens, and it is **not** enough to set `corr_threshold` for a real machine:
a real speaker/mic pair is nonlinear and a real room has a tail this does not model.

This script cannot do that tuning for you, because the number it would need is the correlation
*your* room produces. The machine measures that itself, and the way to make it say so is to run the
guard with a threshold nothing can reach:

    VOX_ECHO_CORR=1.01 VOX_ECHO_GUARD=1 make demo     # on speakers; let a turn or two run away

Nothing is ever rejected at 1.01, but every asking is recorded, so `runs/turns.jsonl` then carries
`self_echo_r` — the best correlation that turn's capture scored against the reply — and
`self_echo_delay_ms` beside it. Compare the turns that ran away with the turns where you spoke, and
put `corr_threshold` between the two populations. If `self_echo_delay_ms` sits at the top of the
search window, raise `max_delay_ms` first: the guard is looking in the wrong place, not failing.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import echo                                                    # noqa: E402
from src.config import ECHO_CORR_THRESHOLD, SAMPLE_RATE                 # noqa: E402

REPLY_RATE = 24_000            # Kokoro's, so the rate change the real path crosses is crossed here
WINDOWS_MS = (200, 400, 700, 1200)   # 400 is kept in the table: it is why 700 is the floor


def speechlike(seconds, rate, seed=0):
    """Audio with phone-scale structure: the thing a 4 Hz syllable envelope is not.

    An earlier version of this measurement generated speech as a slow amplitude modulation, and
    every window shorter than a syllable was very nearly a straight line — so unrelated speech
    correlated at 0.99 and the guard looked useless. It was the fixture. Segments with their own
    length, level and spectrum, and real pauses between them, are the minimum that makes a 200 ms
    window contain anything to match.
    """
    rng = np.random.default_rng(seed)
    out, total = [], 0
    while total < int(seconds * rate):
        n = int(rng.uniform(0.04, 0.12) * rate)
        total += n
        if rng.random() < 0.18:
            out.append(np.zeros(n, dtype=np.float32))
            continue
        t = np.arange(n) / rate
        f0, harmonics, tilt = rng.uniform(90, 220), int(rng.integers(3, 12)), rng.uniform(0.3, 2.5)
        sig = sum((h ** -tilt) * np.sin(2 * np.pi * f0 * h * t + rng.uniform(0, 6.3))
                  for h in range(1, harmonics + 1))
        if rng.random() < 0.3:
            sig = sig * 0.3 + rng.normal(0, 0.5, n)          # a fricative
        shape = np.hanning(n) if rng.random() < 0.5 else np.ones(n)
        out.append((sig * shape * rng.uniform(0.2, 1.0)).astype(np.float32))
    return np.concatenate(out)[:int(seconds * rate)]


def through_a_room(x, rate, delay_ms=180.0, gain=0.05, seed=1):
    """What the mic hears when the speaker plays `x`.

    `delay_ms` defaults to the demo machine's measured MME output-buffer latency (0.182 s), which is
    the floor — flight time across a desk is on top of it. `gain` 0.05 is about 26 dB down.
    """
    rng = np.random.default_rng(seed)
    delayed = np.concatenate([np.zeros(int(delay_ms / 1000 * rate), dtype=np.float32), x])
    out, acc = np.empty_like(delayed), 0.0
    for i, v in enumerate(delayed):                          # one-pole lowpass: the cone and the air
        acc = 0.7 * acc + 0.3 * float(v)
        out[i] = acc
    reflections = np.zeros_like(out)
    for lag, g in ((int(0.013 * rate), 0.4), (int(0.029 * rate), 0.25)):
        reflections[lag:] += out[:-lag] * g
    return ((out + reflections) * gain + rng.normal(0, 2e-4, len(out))).astype(np.float32)


def resample_to(x, src_rate, dst_rate):
    n = int(len(x) * dst_rate / src_rate)
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)


def one_trial(seed, window_ms, ref_s=2.0):
    """-> (r for the echo, r for an unrelated voice, the delay the echo was found at)."""
    reply = speechlike(3.0, REPLY_RATE, seed=seed)
    ref = reply[:int(ref_s * REPLY_RATE)]                    # what the speaker has played by "now"
    mic = resample_to(through_a_room(reply, REPLY_RATE), REPLY_RATE, SAMPLE_RATE)
    end = int(ref_s * SAMPLE_RATE)                           # the mic's "now" is the same instant
    heard = mic[end - int(window_ms / 1000 * SAMPLE_RATE):end]
    person = speechlike(window_ms / 1000, SAMPLE_RATE, seed=10_000 + seed)

    r_echo, delay = echo.best_correlation(heard, ref, SAMPLE_RATE, REPLY_RATE)
    r_person, _ = echo.best_correlation(person, ref, SAMPLE_RATE, REPLY_RATE)
    return r_echo, r_person, delay


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--trials", type=int, default=60)
    args = ap.parse_args()

    print(f"self-echo guard — SYNTHETIC room, {REPLY_RATE // 1000}k reply -> "
          f"{SAMPLE_RATE // 1000}k mic, {args.trials} trials per row")
    print(f"  MIN_DECISION_MS={echo.MIN_DECISION_MS:.0f}  config corr_threshold="
          f"{ECHO_CORR_THRESHOLD:.2f}\n")
    print(f"  {'window':>7} | {'echo p05':>8} {'echo mean':>9} {'delay':>7} | "
          f"{'speech max':>10} {'speech mean':>11} | {'speech lost':>11}")
    print("  " + "-" * 76)

    worst = 0.0
    for window_ms in WINDOWS_MS:
        e, p, d = zip(*(one_trial(s, window_ms) for s in range(args.trials)))
        e, p = np.array(e), np.array(p)
        # The operating point: a threshold that keeps 95% of echo, and what it costs in real speech.
        if window_ms < echo.MIN_DECISION_MS:
            # Not a loss rate: the guard declines to answer below MIN_DECISION_MS, and declining is
            # the safe answer. Printed so the row that set that constant stays visible.
            print(f"  {window_ms:>5}ms | {'-':>8} {'-':>9} {'-':>7} | {'-':>10} {'-':>11} | "
                  f"{'refused':>11}   <- below MIN_DECISION_MS")
            continue
        keeps_95 = float(np.quantile(e, 0.05))
        lost = float((p >= keeps_95).mean())
        worst = max(worst, lost)
        print(f"  {window_ms:>5}ms | {keeps_95:>8.3f} {e.mean():>9.3f} {np.median(d):>6.0f}ms | "
              f"{p.max():>10.3f} {p.mean():>11.3f} | {lost * 100:>10.1f}%")

    print(f"\n  worst speech-lost rate at or above MIN_DECISION_MS: {worst * 100:.1f}%")
    print("  Synthetic. This is the mechanism working, not a threshold for your machine — see the\n"
          "  module docstring for how to take a real recording.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
