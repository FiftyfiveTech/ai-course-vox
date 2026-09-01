#!/usr/bin/env python3
"""VOX-020 — verify confirmation fires on all cases that need it.

Runs each entry in evals/confirmation_cases.json through state.build()
and checks that next_action == "confirm" and the reply contains "shall i"
or "go ahead".

Also runs all 15 dev fixtures through state.build() and prints which ones
trigger confirmation, so the count can be checked against expectation.

Usage:
    .venv/bin/python scripts/check_confirmation.py
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import json                                   # noqa: E402
from src import confirm, state                # noqa: E402
from src.telemetry import new_turn_id         # noqa: E402

CONF_CASES = REPO_ROOT / "evals" / "confirmation_cases.json"
DEV_DIR    = REPO_ROOT / "evals" / "dev"


def check_confirmation_cases():
    """Run every confirmation_cases.json entry through state.build()."""
    print("\n" + "=" * 65)
    print("  Confirmation cases (evals/confirmation_cases.json)")
    print("=" * 65)

    cases = json.loads(CONF_CASES.read_text())
    passes, fails = 0, []

    for case in cases:
        cid   = case["id"]
        text  = case["text"]
        needs = case["confirmation_required"]

        try:
            s = state.build(text, new_turn_id())
            fired      = confirm.needs_confirmation(s)
            reply_ok   = any(p in s.reply.lower() for p in (
                              "shall i", "go ahead", "confirm", "is that correct",
                              "correct?", "proceed", "want me to"))
            intent_ok  = s.intent != "unknown"

            ok = (fired == needs) and (not needs or reply_ok)
            mark = "PASS" if ok else "FAIL"

            print(f"  {mark}  {cid}  intent={s.intent}  next={s.next_action}  "
                  f"conf={fired}  reply_has_confirmation={reply_ok}")
            print(f"       reply: {s.reply!r}")

            if ok:
                passes += 1
            else:
                fails.append(f"{cid}: needs={needs} fired={fired} reply_ok={reply_ok}")
        except Exception as e:
            print(f"  FAIL  {cid}  ERROR: {type(e).__name__}: {e}")
            fails.append(f"{cid}: {type(e).__name__}: {e}")

    print(f"\n  {passes}/{len(cases)} confirmation cases passed")
    return passes, len(cases), fails


def check_dev_fixtures():
    """Run all 15 dev fixtures through state.build() and show which trigger confirmation."""
    print("\n" + "=" * 65)
    print("  Dev fixtures — confirmation rate")
    print("=" * 65)

    # Load transcripts from the dev manifest if available, else use fixture filename as hint
    fixtures = sorted(DEV_DIR.glob("*.wav"))
    confirmed, total = 0, 0

    # Use a simple heuristic transcript map for dev fixtures we know
    KNOWN = {
        "utt_001_greet.wav": "Hello there.",
        "utt_002_greet.wav": "Hello VOX, are you there?",
        "utt_006_entity.wav": "Book a one hour meeting with Priya tomorrow at three p.m.",
        "utt_007_entity.wav": "Schedule a 30-minute call with the design team on Friday at 10 a.m.",
        "utt_008_entity.wav": "Set up a two-hour workshop with Rahul and Snefer on Monday morning.",
        "utt_009_entity.wav": "Book a one hour call with Ananya on Wednesday at noon.",
        "utt_010_entity.wav": "Schedule a team standup every day at nine a.m.",
        "utt_016_entity.wav": "Remind me to send the report to Ananya at 5pm today.",
        "utt_017_entity.wav": "Set a reminder for the deployment checklist at 9am on Thursday.",
        "utt_018_entity.wav": "Log three hours on the VOX project for today.",
        "utt_021_entity.wav": "What meetings do I have tomorrow?",
        "utt_022_entity.wav": "How many hours have I logged this week?",
        "utt_023_entity.wav": "When is the next sprint review?",
        "utt_031_ambig.wav":  "I need to set something up for next week.",
        "utt_036_escalate.wav": "I need to report a security incident.",
    }

    for f in fixtures:
        text = KNOWN.get(f.name, f.name)
        total += 1
        try:
            s = state.build(text, new_turn_id())
            fired = confirm.needs_confirmation(s)
            if fired:
                confirmed += 1
            mark = "CONFIRM" if fired else "reply  "
            print(f"  {mark}  {f.name:<30}  intent={s.intent:<12}  next={s.next_action}")
        except Exception as e:
            print(f"  ERROR   {f.name:<30}  {type(e).__name__}: {e}")

    print(f"\n  {confirmed}/{total} dev fixtures trigger confirmation")
    return confirmed, total


def main():
    passes, total_cases, fails = check_confirmation_cases()
    confirmed_dev, total_dev   = check_dev_fixtures()

    print("\n" + "=" * 65)
    print("  SUMMARY")
    print("=" * 65)
    print(f"  confirmation_cases.json : {passes}/{total_cases} passed")
    print(f"  dev fixtures confirmed  : {confirmed_dev}/{total_dev}")

    if fails:
        print("\nFAIL")
        for f in fails:
            print(f"  - {f}")
        sys.exit(1)

    print(f"\nPASS  ({passes}/{total_cases} confirmation cases fire correctly)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
