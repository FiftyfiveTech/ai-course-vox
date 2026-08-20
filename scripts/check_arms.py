"""Call every registered arm once and print what it did (VOX-006's verification).

    uv run python scripts/check_arms.py --list          the table, no calls
    uv run python scripts/check_arms.py                 every arm, one real call each
    uv run python scripts/check_arms.py --stage stt

This is the evidence for the ticket, not a smoke test: the criterion is "call each arm by flag and
show the model id in the log", so the script makes the calls, then re-reads the lines it just
appended to runs/calls.jsonl and prints them. What you see is what reached disk.

Each arm gets its own turn_id — these are eight independent single-stage calls, not a turn, so no
line is written to runs/turns.jsonl and the VOX-003 latency table stays a table of real turns.
Weights load before the call is timed, the same as in the loop, so a cold cache shows up as a slow
`load` column rather than as a slow arm.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import soundfile as sf
import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import arms, nlu, vad                                              # noqa: E402
from src.config import ARMS, FALLBACKS, PIPELINE, SAMPLE_RATE               # noqa: E402
from src.telemetry import CALLS_LOG, new_turn_id                            # noqa: E402

FIXTURE = Path("tests/fixtures/hello_testing_voice.mp3")

# Held identical across the arms of a stage. Comparing arms on different inputs would measure the
# inputs.
LLM_TRANSCRIPT = "What is on my board for today?"
TTS_TEXT = "You have three tasks due today."


def clip_16k(path):
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    mono = torch.from_numpy(data.mean(axis=1))
    if sr != SAMPLE_RATE:
        mono = torchaudio.functional.resample(mono, sr, SAMPLE_RATE)
    return mono.numpy()


def show_table():
    print(f"{'stage':<6}{'HF repo id':<45}{'provider':<12}{'backend':<22}{'where':<7}alias")
    for stage, stage_arms in ARMS.items():
        for i, a in enumerate(stage_arms):
            marks = " (default)" if i == 0 else ""
            marks += " (fallback)" if a.alias == FALLBACKS.get(stage) else ""
            where = "local" if a.local else "remote"
            print(f"{stage:<6}{a.repo_id:<45}{a.provider:<12}{a.backend:<22}{where:<7}"
                  f"{a.alias}{marks}")
    # Read off the registry rather than restated: the counts moved once already (VOX-013 added the
    # piper arm) and a hand-written "2 tts" then described a table that had three rows.
    counts = ", ".join(f"{len(v)} {k}" for k, v in ARMS.items())
    minimums = {"stt": 3, "llm": 2, "tts": 2}
    print(f"\n{counts} — VOX-006's criterion is at least "
          + ", ".join(f"{n} {s}" for s, n in minimums.items()))
    print("pipeline: " + " -> ".join(f"{s} {p}" for s, p in PIPELINE.items()))


def run_one(stage, arm, segment):
    """-> (turn_id, load_ms, call_ms, output_summary) or raises."""
    turn_id = new_turn_id()

    t0 = time.perf_counter()
    arms.warm(stage, arm.id)
    load_ms = round((time.perf_counter() - t0) * 1000, 1)

    # fallback=False throughout: this script measures arms against each other, so a rate-limited
    # Groq arm has to show up as FAILED on its own row. Letting it be rescued would print the local
    # arm's latency and transcript beside the remote arm's name — the one error this table cannot
    # afford, because its whole purpose is attributing numbers to models.
    t0 = time.perf_counter()
    if stage == "stt":
        out = repr(arms.stt(segment, arm.id, turn_id=turn_id, fallback=False))
    elif stage == "llm":
        out = repr(nlu.reply(LLM_TRANSCRIPT, turn_id, model_id=arm.id, fallback=False))
    else:
        speech = arms.tts(TTS_TEXT, arm.id, turn_id=turn_id, fallback=False)
        out = f"{len(speech.audio) / speech.sample_rate:.2f}s at {speech.sample_rate} Hz"
    call_ms = round((time.perf_counter() - t0) * 1000, 1)

    return turn_id, load_ms, call_ms, out


def main():
    ap = argparse.ArgumentParser(description="Call every arm once and show the log (VOX-006)")
    ap.add_argument("--stage", choices=tuple(ARMS), help="only this stage")
    ap.add_argument("--list", action="store_true", help="print the arm table and exit")
    args = ap.parse_args()

    if args.list:
        show_table()
        return 0

    if not FIXTURE.is_file():
        sys.exit(f"no such fixture: {FIXTURE}")

    stages = (args.stage,) if args.stage else tuple(ARMS)
    show_table()

    print("\nendpointing the fixture once, so every stt arm sees the same segment…", flush=True)
    vad._vad_model()
    cap, state = vad.endpoint_frames(vad.frames_from(clip_16k(FIXTURE)))
    if cap is None:
        sys.exit(f"endpointer found no turn in {FIXTURE} (state={state})")
    print(f"  {len(cap) / SAMPLE_RATE:.2f}s segment, {cap.spoken_s:.2f}s speech")
    print(f"  llm input : {LLM_TRANSCRIPT!r}")
    print(f"  tts input : {TTS_TEXT!r}")

    turn_ids, failed = [], []
    for stage in stages:
        print(f"\n=== {stage} ===")
        for arm in ARMS[stage]:
            print(f"{arm.id} …", end=" ", flush=True)
            try:
                turn_id, load_ms, call_ms, out = run_one(stage, arm, cap.segment)
            except Exception as e:
                failed.append((arm.id, f"{type(e).__name__}: {e}"))
                print(f"FAILED  {type(e).__name__}: {e}")
                continue
            turn_ids.append(turn_id)
            print(f"ok  load {load_ms:.0f}ms  call {call_ms:.0f}ms  -> {out}")

    # The criterion says "show the model id in the log", so show the log — not a summary of it.
    print(f"\n=== {CALLS_LOG} — the lines these calls appended ===")
    wanted = set(turn_ids)
    for line in CALLS_LOG.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if rec.get("turn_id") in wanted:
            print(json.dumps(rec, ensure_ascii=False))

    called = len(turn_ids)
    print(f"\n{called} arm(s) called, {len(failed)} failed.")
    for arm_id, err in failed:
        print(f"  {arm_id}: {err}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
