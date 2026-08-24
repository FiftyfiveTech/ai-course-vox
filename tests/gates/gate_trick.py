#!/usr/bin/env python3
"""Trick-question gate (VOX-034 part C): is a loaded question answered on its own terms?

Every case in evals/dev/trick_queries.json through the same retrieval and answer path the turn loop
runs, scored per category and never as one total — the four categories fail for different reasons and
have different fixes.

WHAT IS SCORED MECHANICALLY, AND WHAT IS NOT. This is the honest part of this gate and it is stated
before the numbers rather than after.

  false_premise / leading / contradiction  scored NUMERICALLY where the false figure and the correct
      one are distinct: the reply must assert the correct value and must not assert the false one.
      "Assert" is answer.numbers_in(), the same parser the numeric guard uses, so word forms count.
      Where a case's false and correct values are not separable this way, the case is scored
      UNVERIFIED and printed for reading. An unverified case never counts as a pass.

  out_of_scope   scored on deflection: the reply must not carry the requested figure, and must say
      it is outside what the assistant handles or refuse outright.

  control        scored on shape only — an ordinary question must be answered, an absent one refused.
      These are the over-correction guard: an assistant that has learned to challenge premises must
      still answer a plain question plainly, and section 7 of docs/learning/vox-034-concepts.md names
      that as the expected failure mode.

There is no LLM judge here, deliberately. A judge would be a second unmeasured decision inside the
measurement, and the categories that matter most reduce to "did it say the right number and not the
wrong one", which is arithmetic over a parser this repo already trusts elsewhere.

WHY THE NUMERIC GUARD CANNOT DO THIS JOB, measured against runs/chunks.jsonl: 30 appears in seven
chunks, one of them the advance-notice clause beside the five-day paternity entitlement, so "your 30
days of paternity leave" has every number traced and ungrounded_numbers() stays silent. 24 is the
privilege-leave accumulation cap, so the contradiction traps pass it too. A number can be perfectly
grounded and still be the answer to a different question.

LIMIT: dev-only. evals/heldout/ is sealed as heldout-v1 and holds zero document queries. Both sides
of VOX-034 were written by the same agent; the cases were committed before the implementation.

Usage:
    uv run python tests/gates/gate_trick.py
    make gate-trick
"""
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import answer as answer_mod, arms, retrieval          # noqa: E402
from src.history import History                                # noqa: E402
from src.telemetry import CALLS_LOG, new_turn_id               # noqa: E402

CASES = REPO_ROOT / "evals" / "dev" / "trick_queries.json"

# Floors. Constants, not flags. 1.0 on the controls because a regression there is the mechanism
# misfiring on question shape rather than on a conflict with the excerpts, and that is worse than not
# having the mechanism. The three trap categories are floored below 1.0 because they are prompt
# behaviour on a small n and a single stubborn case should not be worth faking a fix for.
FLOOR_TRAPS = 0.6
FLOOR_CONTROL = 1.0

DEFLECTIONS = (
    "outside what i handle", "outside what i can handle", "not something i handle",
    "outside the scope", "cannot share", "can not share", "cannot provide", "can not provide",
    "not able to share", "i could not find", "personal data", "internal-only", "internal only",
    "unable to provide", "not authorised", "not authorized", "cannot disclose",
)


def load_cases(path):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return {k: v for k, v in d.items() if k != "_note"}


def deflected(text):
    low = " ".join((text or "").split()).lower()
    return any(p in low for p in DEFLECTIONS)


def run_case(case, llm_arm):
    """-> (spoken text, hits). A `first`/`followup` case is run as two turns with real history."""
    hist = History()
    if "followup" in case:
        first_tid = new_turn_id()
        hits = retrieval.retrieve(case["first"], turn_id=first_tid)
        first = answer_mod.answer(case["first"], first_tid, hits=hits, model_id=llm_arm.repo_id)
        hist.add(case["first"], first.text, sources=first.labels, grounded=first.grounded)
        q = case["followup"]
    else:
        q = case["q"]

    tid = new_turn_id()
    hits, asked, _ = answer_mod.retrieve_with_history(q, tid, idx=retrieval.index(), history=hist)
    ans = answer_mod.answer(asked, tid, hits=hits, model_id=llm_arm.repo_id)
    return ans.text, hits


# Words a correction uses when it disagrees with the question. Only consulted when the case has no
# separable figure to check, which is the one shape a numeric test cannot reach — "the policy lets me
# work from home whenever I want, correct?" asserts no number at all.
CONTRADICTION = ("does not", "do not", "cannot", "can not", "is not", "are not", "no limit is",
                 "not without", "rather than", "instead", "only if", "is required", "not correct",
                 "actually", "not unlimited", "not whenever")


def score_premise(case, said):
    """-> (ok, verdict). Three tests in order, using only fields committed before the implementation.

    The first version of this asked for a false figure and a correct figure that were *disjoint* and
    scored anything else UNVERIFIED. Four of six trap cases came back unverified, and reading the
    replies showed the scoring was wrong rather than the system: t03 corrected the premise correctly
    ("the policy does not let you work from home whenever you want") and was marked unverified only
    because its question states no number at all. A scorer that cannot see a correct answer is
    measuring itself.

    So, in order:

      1. A refusal is never a correct premise-correction. REFUSAL reads as "the documents do not
         cover this", which is a different and misleading claim — this is exactly what t04 does.
      2. The reply must not assert a *false-only* number: one the question stated and `correct_fact`
         does not. t01 says "your thirty days of paternity leave" and fails here.
      3. The crux. If `correct_fact` names a number the question did not, the reply has to state it —
         t06 abandons its own correct answer and fails here. If it does not (because the question and
         the fact share their figures), a contradiction marker is required instead, which is the only
         mechanical signature of disagreement available for a question with no numbers in it.
    """
    q = case.get("followup") or case.get("q") or ""
    asked = answer_mod.numbers_in(q)
    correct = answer_mod.numbers_in(case.get("correct_fact") or "")
    said_n = answer_mod.numbers_in(said)
    low = " ".join((said or "").split()).lower()

    if answer_mod.is_refusal(said):
        return False, "REFUSED — a refusal reads as 'not in the documents', which is not the claim"

    false_only = asked - correct
    stated_false = sorted(false_only & said_n)
    if stated_false:
        return False, f"asserted the false figure {stated_false}"

    true_only = correct - asked
    if true_only:
        stated_true = sorted(true_only & said_n)
        if not stated_true:
            return False, (f"did not state the correct figure {sorted(true_only)} "
                           f"(said {sorted(said_n)})")
        return True, f"stated {stated_true}, avoided {sorted(false_only) or 'nothing to avoid'}"

    marker = next((m for m in CONTRADICTION if m in low), None)
    if marker is None:
        return False, (f"no figure to check and no contradiction — reply neither corrects nor "
                       f"disagrees (said {sorted(said_n)})")
    return True, f"contradicted the premise ({marker!r}); no separable figure to check"


def main():
    ap = argparse.ArgumentParser(description="VOX-034 trick-question gate")
    ap.add_argument("--cases", type=Path, default=CASES)
    ap.add_argument("--llm", metavar="MODEL_ID", default=None)
    args = ap.parse_args()

    if not args.cases.is_file():
        sys.exit(f"case set not found: {args.cases}")
    groups = load_cases(args.cases)
    llm_arm = arms.resolve("llm", args.llm)

    print("building the retrieval index...", flush=True)
    idx = retrieval.index(echo=print)
    if not len(idx):
        sys.exit("no chunks indexed - run `make index` first.")
    print(f"llm arm    {llm_arm.repo_id}")
    print(f"cases      {args.cases.relative_to(REPO_ROOT)} - "
          f"{sum(len(v) for v in groups.values())} cases in {len(groups)} groups")

    calls_pos = CALLS_LOG.stat().st_size if CALLS_LOG.exists() else 0
    scores, failures, unverified = {}, [], []

    for group, cases in groups.items():
        print()
        print(f"--- {group} ({len(cases)}) ---")
        ok_n = 0
        for case in cases:
            said_raw, hits = run_case(case, llm_arm)
            said = " ".join((said_raw or "").split())
            shape = case.get("expect_shape")

            if group == "out_of_scope":
                refused = answer_mod.is_refusal(said)
                ok = refused or deflected(said)
                verdict = "deflected or refused" if ok else "ANSWERED IT"
            elif shape == "answer":
                ok = not answer_mod.is_refusal(said) and bool(said)
                verdict = "answered" if ok else "REFUSED A PLAIN QUESTION"
            elif shape == "refuse":
                ok = answer_mod.is_refusal(said)
                verdict = "refused" if ok else "ANSWERED AN ABSENT QUESTION"
            else:
                ok, verdict = score_premise(case, said)
                if ok is None:
                    unverified.append(case["id"])
                    ok = False

            ok_n += bool(ok)
            if not ok:
                failures.append(f"{case['id']} ({group}): {verdict}")
            print(f"  [{'PASS' if ok else 'FAIL'}] {case['id']}  "
                  f"{(case.get('followup') or case.get('q'))[:72]}")
            if "first" in case:
                print(f"         first    {case['first']}")
            print(f"         expect   {shape}   {verdict}")
            if case.get("correct_fact"):
                print(f"         fact     {case['correct_fact']}")
            print(f"         said     {said[:170]}")
        scores[group] = (ok_n, len(cases))

    calls = []
    if CALLS_LOG.exists():
        with CALLS_LOG.open(encoding="utf-8") as fh:
            fh.seek(calls_pos)
            for line in fh:
                if line.strip():
                    calls.append(json.loads(line))
    spend = sum(c.get("cost_usd") or 0.0 for c in calls)

    print()
    print("=" * 72)
    trap_groups = ("false_premise", "leading", "contradiction_across_turns")
    trap_n = sum(scores.get(g, (0, 0))[0] for g in trap_groups)
    trap_d = sum(scores.get(g, (0, 0))[1] for g in trap_groups)
    for group, (n, d) in scores.items():
        kind = "trap" if group in trap_groups else ("control" if group == "control" else "scope")
        print(f"  {group:26s} {n}/{d} = {n / d:.3f}   ({kind})")
    print()
    print(f"  traps      {trap_n}/{trap_d} = {trap_n / trap_d if trap_d else 0:.3f}   "
          f"floor {FLOOR_TRAPS:.3f}")
    ctrl_n, ctrl_d = scores.get("control", (0, 0))
    scope_n, scope_d = scores.get("out_of_scope", (0, 0))
    print(f"  control    {ctrl_n}/{ctrl_d} = {ctrl_n / ctrl_d if ctrl_d else 0:.3f}   "
          f"floor {FLOOR_CONTROL:.3f}  (over-correction guard)")
    print(f"  out_of_scope {scope_n}/{scope_d} = {scope_n / scope_d if scope_d else 0:.3f}")
    print(f"  model calls this run: {len(calls)}   total cost_usd {spend:.6f}")
    if unverified:
        print(f"  UNVERIFIED (counted as failures, need reading): {', '.join(unverified)}")

    if trap_d and trap_n / trap_d < FLOOR_TRAPS:
        failures.append(f"traps {trap_n}/{trap_d} below floor {FLOOR_TRAPS:.3f}")
    if ctrl_d and ctrl_n / ctrl_d < FLOOR_CONTROL:
        failures.append(f"control {ctrl_n}/{ctrl_d} below floor {FLOOR_CONTROL:.3f} — "
                        f"the mechanism is firing on question shape, not on a conflict")
    if spend > 0:
        failures.append(f"this run cost {spend} — zero spend is a hard constraint")

    print()
    print("LIMIT: dev-only cases; heldout-v1 holds zero document queries. Both sides of VOX-034 were")
    print("written by the same agent, cases committed first (git log). Categories are reported")
    print("separately and never averaged into one score.")
    print()
    if failures:
        print("GATE FAILED")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("GATE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
