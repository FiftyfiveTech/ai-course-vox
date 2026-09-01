#!/usr/bin/env python3
"""Phase 1B gate: entity capture rate, confirmation coverage, and state validity on heldout-v1.

Runs every utterance in evals/heldout/ through:
  STT -> state.build() -> entity scoring + confirmation check

Reports:
  - Overall entity capture rate
  - Confirmation rate on write-intent turns
  - State validity rate
  - Per-category breakdown (names / dates / ids / other)
  - Per-utterance detail

Done when numbers are printed. VOX-024 sets the pass threshold.

Usage:
    source .env
    .venv/bin/python tests/gates/gate_phase1b.py
"""
import json
import sys
import time
from pathlib import Path
from collections import defaultdict

import soundfile as sf
import torch
import torchaudio

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src import arms, vad, state as state_mod                         # noqa: E402
from src.config import SAMPLE_RATE                                    # noqa: E402
from src.confirm import WRITE_INTENTS, needs_confirmation             # noqa: E402
from src.scorer import score_utterance                                # noqa: E402
from src.telemetry import new_turn_id                                 # noqa: E402

HELDOUT_DIR = REPO_ROOT / "evals" / "heldout"
LABELS_FILE = HELDOUT_DIR / "labels.json"

# Entity field → display category for the breakdown table
FIELD_CATEGORY = {
    "person":     "names",
    "team":       "names",
    "date":       "dates",
    "project":    "ids",
    "duration":   "other",
    "time":       "other",
    "recurrence": "other",
    "intent":     "other",
}

# Map gold intent labels → TurnState intent literals
GOLD_TO_STATE = {
    "greet":          "greet",
    "log_hours":      "capture",
    "set_reminder":   "capture",
    "query_calendar": "clarify",
    "escalate":       "escalate",
    "refuse":         "refuse",
    "ambiguous":      "clarify",
}


def load_16k_mono(path):
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    mono = torch.from_numpy(data.mean(axis=1))
    if sr != SAMPLE_RATE:
        mono = torchaudio.functional.resample(mono, sr, SAMPLE_RATE)
    return mono.numpy()


def fixture_path(uid):
    """Find the WAV for an utterance id regardless of category suffix."""
    matches = list(HELDOUT_DIR.glob(f"{uid}_*.wav"))
    return matches[0] if matches else None


def main():
    labels = json.loads(LABELS_FILE.read_text())

    print("loading local models…", flush=True)
    vad._vad_model()
    arms.warm("tts", arms.resolve("tts").id)

    print(f"\n{'='*70}")
    print("  PHASE 1B GATE — heldout-v1 evaluation (VOX-023)")
    print(f"{'='*70}")
    print(f"  utterances: {len(labels)}")
    print()

    rows = []
    skipped = []

    for label in labels:
        uid = label["id"]
        gold = label.get("gold", {})

        wav = fixture_path(uid)
        if wav is None:
            print(f"  SKIP {uid} — WAV not found", flush=True)
            skipped.append((uid, "wav not found"))
            continue

        print(f"  {uid} …", end=" ", flush=True)

        turn_id = new_turn_id()

        # --- STT ---
        try:
            clip = load_16k_mono(wav)
            cap, state_vad = vad.endpoint_frames(vad.frames_from(clip))
            if cap is None:
                raise RuntimeError(f"no speech (state={state_vad})")
            transcript = arms.stt(cap.segment, turn_id=turn_id)
        except Exception as e:
            print(f"STT FAIL: {e}", flush=True)
            skipped.append((uid, f"stt: {e}"))
            continue

        # --- state.build() ---
        ts = None
        state_valid = False
        state_error = None
        try:
            ts = state_mod.build(transcript, turn_id)
            state_valid = True
        except Exception as e:
            state_error = f"{type(e).__name__}: {e}"

        # --- entity scoring ---
        gold_entities = {k: v for k, v in gold.items() if k != "intent"}
        extracted_entities = dict(ts.entities) if ts else {}
        score = score_utterance(gold_entities, extracted_entities)

        # --- confirmation check ---
        gold_intent = gold.get("intent", "")
        needs_conf = gold_intent in WRITE_INTENTS
        conf_fired = ts is not None and needs_confirmation(ts)

        # --- intent match ---
        expected_intent = GOLD_TO_STATE.get(gold_intent, "unknown")
        intent_ok = ts is not None and ts.intent == expected_intent

        rows.append({
            "id": uid,
            "gold_intent": gold_intent,
            "transcript": transcript,
            "state_valid": state_valid,
            "state_error": state_error,
            "intent_ok": intent_ok,
            "predicted_intent": ts.intent if ts else None,
            "correct": score["correct"],
            "total_gold": score["total_gold"],
            "slot_results": score["slot_results"],
            "false_positives": score["false_positives"],
            "needs_conf": needs_conf,
            "conf_fired": conf_fired,
        })

        mark = "ok" if state_valid else "STATE_FAIL"
        conf_mark = "" if not needs_conf else (" conf✓" if conf_fired else " conf✗")
        print(f"{mark}{conf_mark}  {score['correct']}/{score['total_gold']} slots"
              f"  intent={'✓' if intent_ok else '✗'}"
              f"  {transcript[:55]!r}", flush=True)

        time.sleep(0.3)   # stay within Groq rate limits

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print()
    print(f"{'='*70}")
    print("  RESULTS")
    print(f"{'='*70}")

    n = len(rows)
    valid = sum(1 for r in rows if r["state_valid"])
    validity_rate = valid / n if n else 0

    total_correct = sum(r["correct"] for r in rows)
    total_gold    = sum(r["total_gold"] for r in rows)
    capture_rate  = total_correct / total_gold if total_gold else 0

    write_rows = [r for r in rows if r["needs_conf"]]
    conf_ok_n  = sum(1 for r in write_rows if r["conf_fired"])
    conf_rate  = conf_ok_n / len(write_rows) if write_rows else 1.0

    intent_ok_n = sum(1 for r in rows if r["intent_ok"])
    intent_rate = intent_ok_n / n if n else 0

    print(f"\n  utterances processed : {n}  (skipped: {len(skipped)})")
    print(f"  state validity       : {valid}/{n}  ({validity_rate:.0%})")
    print(f"  entity capture rate  : {total_correct}/{total_gold}  ({capture_rate:.0%})")
    print(f"  confirmation rate    : {conf_ok_n}/{len(write_rows)} write-intent turns  ({conf_rate:.0%})")
    print(f"  intent accuracy      : {intent_ok_n}/{n}  ({intent_rate:.0%})")

    # --- per entity-field-category breakdown ---
    cat_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in rows:
        for field, result in r["slot_results"].items():
            cat = FIELD_CATEGORY.get(field, "other")
            cat_stats[cat]["total"] += 1
            if result == "correct":
                cat_stats[cat]["correct"] += 1

    print()
    print("  entity capture by field category:")
    print(f"  {'category':<10}  {'correct':>8}  {'total':>6}  {'rate':>6}")
    print("  " + "-" * 36)
    for cat in ("names", "dates", "ids", "other"):
        s = cat_stats.get(cat, {"correct": 0, "total": 0})
        rate = s["correct"] / s["total"] if s["total"] else 0
        print(f"  {cat:<10}  {s['correct']:>8}  {s['total']:>6}  {rate:>5.0%}")

    # --- per gold-intent breakdown ---
    intent_stats = defaultdict(lambda: {"correct": 0, "total_gold": 0, "count": 0})
    for r in rows:
        gi = r["gold_intent"] or "unknown"
        intent_stats[gi]["correct"]    += r["correct"]
        intent_stats[gi]["total_gold"] += r["total_gold"]
        intent_stats[gi]["count"]      += 1

    print()
    print("  entity capture by gold intent:")
    print(f"  {'intent':<18}  {'turns':>6}  {'slots':>8}  {'correct':>8}  {'rate':>6}")
    print("  " + "-" * 54)
    for gi, s in sorted(intent_stats.items()):
        rate = s["correct"] / s["total_gold"] if s["total_gold"] else 0
        print(f"  {gi:<18}  {s['count']:>6}  {s['total_gold']:>8}  "
              f"{s['correct']:>8}  {rate:>5.0%}")

    # --- per-utterance detail ---
    print()
    print("  per-utterance detail:")
    print(f"  {'id':<12}  {'intent':<16}  {'slots':>6}  {'state':>6}  {'conf':>5}  transcript")
    print("  " + "-" * 90)
    for r in rows:
        slot_str = f"{r['correct']}/{r['total_gold']}"
        state_str = "ok" if r["state_valid"] else "FAIL"
        conf_str = ("n/a" if not r["needs_conf"]
                    else ("✓" if r["conf_fired"] else "✗"))
        print(f"  {r['id']:<12}  {r['gold_intent']:<16}  {slot_str:>6}  "
              f"{state_str:>6}  {conf_str:>5}  {r['transcript'][:55]!r}")

    if skipped:
        print("\n  skipped:")
        for uid, reason in skipped:
            print(f"    {uid}: {reason}")

    state_fails = [r for r in rows if not r["state_valid"]]
    if state_fails:
        print("\n  state failures:")
        for r in state_fails:
            print(f"    {r['id']}: {r['state_error']}")

    conf_fails = [r for r in write_rows if not r["conf_fired"]]
    if conf_fails:
        print("\n  confirmation missed:")
        for r in conf_fails:
            print(f"    {r['id']} ({r['gold_intent']}): predicted_intent={r['predicted_intent']}")

    print()
    print(f"{'='*70}")
    print(f"  state_validity={validity_rate:.0%}  "
          f"entity_capture={capture_rate:.0%}  "
          f"confirmation={conf_rate:.0%}  "
          f"intent_accuracy={intent_rate:.0%}")
    print(f"{'='*70}")
    print("\nPASS — numbers printed, breakdown complete (VOX-023 done; VOX-024 sets the threshold)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
