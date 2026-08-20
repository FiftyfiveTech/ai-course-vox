#!/usr/bin/env python3
"""VOX-012 — measure the effect of endpointing.silence_ms on the dev utterances.

Sweeps silence_ms over a range, runs the real VAD endpointer on every dev fixture, and
prints a table showing segment length, spoken length, and trailing silence cost.

Usage:
    .venv/bin/python scripts/tune_vad.py
    .venv/bin/python scripts/tune_vad.py --values 700 900 1100 1300 1500
    .venv/bin/python scripts/tune_vad.py --fixture tests/fixtures/hello_testing_voice.mp3
"""
import argparse
import sys
from pathlib import Path

import soundfile as sf
import torch
import torchaudio

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import src.config as _config_mod              # noqa: E402  (patched per sweep value)
from src import vad                            # noqa: E402
from src.config import SAMPLE_RATE            # noqa: E402

DEV_DIR = REPO_ROOT / "evals" / "dev"

DEFAULT_SWEEP = [500, 700, 900, 1100, 1300, 1500]


def load_16k_mono(path):
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    mono = torch.from_numpy(data.mean(axis=1))
    if sr != SAMPLE_RATE:
        mono = torchaudio.functional.resample(mono, sr, SAMPLE_RATE)
    return mono.numpy()


def endpoint_with(clip, silence_ms):
    """Run the real endpointer with silence_ms patched into src.config."""
    _config_mod.VAD_SILENCE_MS = silence_ms
    cap, _state = vad.endpoint_frames(vad.frames_from(clip))
    if cap is None:
        return None, None, False
    seg_s = len(cap) / SAMPLE_RATE
    return seg_s, cap.spoken_s, True


def main():
    ap = argparse.ArgumentParser(description="Sweep silence_ms over dev fixtures")
    ap.add_argument("--values", nargs="+", type=int, default=DEFAULT_SWEEP,
                    metavar="MS", help="silence_ms values to sweep")
    ap.add_argument("--fixture", type=Path, default=None,
                    help="single fixture (default: all dev wavs)")
    args = ap.parse_args()

    fixtures = ([args.fixture] if args.fixture
                else sorted(DEV_DIR.glob("*.wav")) + sorted(DEV_DIR.glob("*.mp3")))
    fixtures = [f for f in fixtures if f.exists()]

    if not fixtures:
        sys.exit(f"No fixtures found in {DEV_DIR}")

    print(f"\nSweeping silence_ms over {len(fixtures)} fixture(s)")
    print(f"Values: {args.values}\n")

    vad._vad_model()   # load once before timing

    print(f"  {'silence_ms':>10}  {'detected':>8}  {'avg_seg_s':>10}  "
          f"{'avg_spoken_s':>13}  {'avg_trail_s':>12}  {'detect_rate':>11}")
    print("  " + "-" * 72)

    original_ms = _config_mod.VAD_SILENCE_MS

    for ms in args.values:
        segs, spk, trails, detected = [], [], [], 0
        for f in fixtures:
            clip = load_16k_mono(f)
            seg_s, spoken_s, ok = endpoint_with(clip, ms)
            if ok:
                detected += 1
                segs.append(seg_s)
                spk.append(spoken_s)
                trails.append(seg_s - spoken_s)

        n = len(fixtures)
        avg_seg   = sum(segs)   / len(segs)   if segs   else 0.0
        avg_spk   = sum(spk)    / len(spk)    if spk    else 0.0
        avg_trail = sum(trails) / len(trails) if trails else 0.0
        rate      = detected / n

        marker = "  <-- current (config.yaml)" if ms == original_ms else ""
        print(f"  {ms:>10}  {detected:>5}/{n:<2}  {avg_seg:>10.2f}  "
              f"{avg_spk:>13.2f}  {avg_trail:>12.2f}  {rate:>10.0%}{marker}")

    # Restore original value
    _config_mod.VAD_SILENCE_MS = original_ms

    print()
    print("avg_trail_s = avg_seg_s - avg_spoken_s  "
          "(silence cost per turn; lands in time_to_first_audio)")
    print("detect_rate = fraction of fixtures where speech was found")


if __name__ == "__main__":
    main()
