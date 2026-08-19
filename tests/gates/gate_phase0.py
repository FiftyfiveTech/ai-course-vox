#!/usr/bin/env python3
"""Phase 0 gate: one fixture turn through the full pipeline.

PASS when:
  - t_stt_ms, t_llm_ms, t_tts_ms are all non-null and positive
  - every calls.jsonl entry for this turn carries a HF repo id and cost_usd == 0.0
  - the 5-way PASS table is printed to stdout

time_to_first_audio_ms requires a speaker and is not measured here.
The gate prints "n/a" rather than zero — a zero would be a lie.

Usage:
    uv run python tests/gates/gate_phase0.py
    uv run python tests/gates/gate_phase0.py --fixture evals/dev/utt_006_entity.wav
"""
import argparse
import json
import sys
from pathlib import Path

import soundfile as sf
import torch
import torchaudio

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src import arms, nlu, vad                                         # noqa: E402
from src.config import SAMPLE_RATE                                     # noqa: E402
from src.telemetry import CALLS_LOG, TURNS_LOG, new_turn_id, turn_timer  # noqa: E402

DEFAULT_FIXTURE = REPO_ROOT / "evals" / "dev" / "utt_001_greet.wav"


def load_16k_mono(path):
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    mono = torch.from_numpy(data.mean(axis=1))
    if sr != SAMPLE_RATE:
        mono = torchaudio.functional.resample(mono, sr, SAMPLE_RATE)
    return mono.numpy()


def main():
    ap = argparse.ArgumentParser(description="Phase 0 gate — 5-way latency split")
    ap.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE,
                    help="wav file from evals/dev/ to drive the turn (default: utt_001_greet.wav)")
    # VOX-006 made every stage an arm. The gate certifies the defaults unless told otherwise, so
    # an unflagged run still measures what Phase 0 ships; the flags let the same gate be re-run
    # per arm without a second script.
    arms.add_flags(ap)
    args = ap.parse_args()

    if not args.fixture.is_file():
        sys.exit(f"fixture not found: {args.fixture}")

    print("resolving arms and loading local models…", flush=True)
    vad._vad_model()
    chosen = arms.select(args)
    print(arms.describe(chosen))

    clip = load_16k_mono(args.fixture)
    turn_id = new_turn_id()
    print(f"\n--- gate_phase0  turn {turn_id} · {args.fixture.name} ---")

    # Snapshot log sizes before the run so we can read only this turn's entries afterwards.
    calls_pos = CALLS_LOG.stat().st_size if CALLS_LOG.exists() else 0

    with turn_timer(turn_id, source=str(args.fixture)) as turn:
        cap, state = vad.endpoint_frames(vad.frames_from(clip))
        if cap is None:
            sys.exit(f"endpointer found no speech in {args.fixture} (state={state})")
        turn.vad(cap)
        turn.arms(**chosen)

        with turn.stage("stt"):
            transcript = arms.stt(cap.segment, chosen["stt"].id, turn_id=turn_id,
                                  on_fallback=turn.fallback)
        print(f"you said : {transcript!r}")
        if not transcript:
            sys.exit("empty transcript from STT — cannot continue")

        with turn.stage("llm"):
            answer = nlu.reply(transcript, turn_id, model_id=chosen["llm"].id,
                               on_fallback=turn.fallback)
        print(f"vox says : {answer!r}")

        with turn.stage("tts"):
            _speech = arms.tts(answer, chosen["tts"].id, turn_id=turn_id,
                               on_fallback=turn.fallback)

        print("(gate: speaker skipped, time_to_first_audio not measured)")

    rec = turn.written

    # Read only the calls appended during this run.
    calls = []
    if CALLS_LOG.exists():
        with CALLS_LOG.open(encoding="utf-8") as fh:
            fh.seek(calls_pos)
            for line in fh:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    if entry.get("turn_id") == turn_id:
                        calls.append(entry)

    # --- print PASS table ---
    def ms(key):
        v = rec.get(key)
        return f"{v:>8.0f} ms" if v is not None else "       n/a"

    print()
    print("=" * 55)
    print("  PHASE 0 GATE — 5-way latency split")
    print("=" * 55)
    print(f"  t_vad_ms               {ms('t_vad_ms')}")
    print(f"  t_stt_ms               {ms('t_stt_ms')}")
    print(f"  t_llm_ms               {ms('t_llm_ms')}")
    print(f"  t_tts_ms               {ms('t_tts_ms')}")
    print(f"  time_to_first_audio_ms {ms('time_to_first_audio_ms')}  (speaker skipped)")
    print("-" * 55)
    stage_sum = rec.get("stage_sum_ms")
    print(f"  stage_sum_ms           "
          f"{f'{stage_sum:>8.0f} ms' if stage_sum is not None else '       n/a'}")
    print()
    print("  model calls this turn:")
    for c in calls:
        mark = "     " if c.get("ok") else "  !  "
        note = f"  -> fell back from {c['fallback_for']}" if c.get("fallback_for") else ""
        print(f"  {mark}[{c['stage']:>3}] {c['model_id']}  "
              f"provider={c['provider']}  cost_usd={c['cost_usd']}{note}")
        if not c.get("ok"):
            print(f"          FAILED: {c.get('error', '')}")

    fell_back = rec.get("fell_back")
    if fell_back:
        print()
        print(f"  FELL BACK: {', '.join(fell_back)} — this turn did not run the arms it selected.")
        for stage in fell_back:
            print(f"    {stage}: {rec.get(f'{stage}_fallback_from')} -> {rec.get(f'{stage}_model')}"
                  f"  ({rec.get(f'{stage}_failed_ms')} ms lost to the failed attempt)")
        print("  The split above is therefore not comparable to a clean run.")
    print("=" * 55)

    # --- assertions ---
    failures = []

    for field in ("t_stt_ms", "t_llm_ms", "t_tts_ms"):
        v = rec.get(field)
        if v is None:
            failures.append(f"{field} is null — stage did not complete")
        elif v <= 0:
            failures.append(f"{field} = {v} (must be positive)")

    # A stage passes when *something* answered it, not when nothing failed. A remote arm that 429s
    # and is covered by its local fallback produces an ok:false line and a working turn; failing the
    # gate on that would mean a pipeline built to survive a rate limit cannot certify during one.
    # The failure is still printed above and still on the turn record — it is reported, not ignored.
    expected_stages = {"stt", "llm", "tts"}
    served = {c["stage"] for c in calls if c.get("ok")}
    for missing in sorted(expected_stages - served):
        failures.append(f"no successful calls.jsonl record for stage '{missing}'")

    for c in calls:
        if not c.get("model_id"):
            failures.append(f"[{c['stage']}] missing model_id (must be a HF repo id)")
        if c.get("cost_usd", -1) != 0.0:
            failures.append(f"[{c['stage']}] cost_usd={c['cost_usd']} — zero spend violated")

    if failures:
        print("\nFAIL")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)

    print("\nPASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
