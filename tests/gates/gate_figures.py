#!/usr/bin/env python3
"""Figure gate (VOX-034 part B): can it compute, and does it still refuse?

Every case in evals/dev/figure_queries.json through the same retrieval and answer path the turn loop
runs. Scored on TWO numbers, never averaged:

    accuracy   over `computable`       the right value, and the derivation traced
    refusal    over `missing_operand`  no figure produced at all
    plus       `no_arithmetic`         a cap is stated, not turned into a subtraction
    plus       `trap`                  a false premise is not agreed with

The pair is the point. A system that computes eagerly scores well on accuracy and badly on refusal;
one that refuses everything scores the reverse. prompts/answer_from_source_v2.md alone scores 0/4 and
4/4 — it forbids arithmetic outright — so the "before" column is known in advance and is what this
ticket is measured against.

PASS when accuracy == 1.0 AND refusal == 1.0. Both floors are 1.0 and neither is negotiable at this
n: a wrong computed figure is a person told the wrong number about their own pay, and a 90% there
means one person in ten. VOX-033 set the precedent for refusing to call a small denominator a rate —
these are fractions, printed with their denominators.

A NOTE ON WHAT "accuracy" MEANS HERE. The expected value is compared against the number Python
computed, read off the Figure, not parsed out of the spoken sentence. src/figures.py composes the
sentence around the computed value, so the two cannot disagree — and a gate that regex-scraped a
number out of prose would be measuring the regex.

LIMIT, printed again in the output: dev-only. evals/heldout/ is sealed as heldout-v1 and holds zero
document queries. Both sides of VOX-034 were written by the same agent; the cases were committed
before the implementation (git log) and that ordering is the only thing standing in for blind
labelling.

Usage:
    uv run python tests/gates/gate_figures.py
    make gate-figures
"""
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import answer as answer_mod, arms, figures, retrieval    # noqa: E402
from src.telemetry import CALLS_LOG, new_turn_id                  # noqa: E402

CASES = REPO_ROOT / "evals" / "dev" / "figure_queries.json"

FLOOR_ACCURACY = 1.0
FLOOR_REFUSAL = 1.0

# How close a computed value has to be to the expected one. Not equality: the encashment case divides
# before multiplying, so a float is involved even when the answer is an integer.
TOLERANCE = 0.01


def load_cases(path):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return {k: v for k, v in d.items() if k != "_note"}


def run_case(case, idx, llm_arm, computable):
    """-> (Figure or None, the spoken text, the hits). ONE model call per case.

    The first version of this called `figures.compute()` for the value and `answer_mod.answer()` for
    the sentence, which is two extractions per case — so the value printed and the sentence printed
    came from different runs of a non-deterministic call and could disagree. They did: g01 reported a
    correct computed 3 next to a sentence from the other extraction. Fixed by running one path per
    case: `compute()` for the computable group, where the assertion is about the arithmetic, and
    `answer()` for every other group, where the assertion is about what gets said.
    """
    tid = new_turn_id()
    hits = retrieval.retrieve(case["q"], turn_id=tid)
    if computable and hits and figures.states_a_number(case["q"]):
        fig = figures.compute(case["q"], hits, tid, model_id=llm_arm.repo_id)
        return fig, (fig.spoken() if fig else ""), hits
    ans = answer_mod.answer(case["q"], tid, hits=hits, model_id=llm_arm.repo_id)
    # For the non-computable groups the claim is "no figure was produced", and the Answer is the
    # thing a listener gets. Re-deriving a Figure here would be a second call measuring nothing.
    fig = None
    if hits and figures.states_a_number(case["q"]):
        fig = figures.compute(case["q"], hits, tid, model_id=llm_arm.repo_id)
    return fig, ans.text, hits


def main():
    ap = argparse.ArgumentParser(description="VOX-034 figure gate")
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
    print(f"llm arm    {llm_arm.repo_id}  "
          f"({'local' if llm_arm.local else 'remote via ' + llm_arm.provider})")
    print(f"cases      {args.cases.relative_to(REPO_ROOT)} - "
          f"{sum(len(v) for v in groups.values())} cases in {len(groups)} groups")

    calls_pos = CALLS_LOG.stat().st_size if CALLS_LOG.exists() else 0
    scores, failures = {}, []

    for group, cases in groups.items():
        print()
        print(f"--- {group} ({len(cases)}) ---")
        ok_n = 0
        for case in cases:
            fig, text, hits = run_case(case, idx, llm_arm, group == "computable")
            said = " ".join((text or "").split())

            if group == "computable":
                want = case["expect_value"]
                got = fig.value if fig else None
                ok = got is not None and abs(got - want) <= TOLERANCE
                verdict = f"want {want}  got {got}"
                if not ok:
                    failures.append(f"{case['id']} computed {got}, expected {want} "
                                    f"({case['working']})")
            else:
                # Every other group must produce NO figure. That is one assertion, not three: a cap
                # turned into a subtraction, an agreed-with false premise and a computation over a
                # missing operand are the same failure wearing different clothes.
                got = fig.value if fig else None
                ok = got is None
                verdict = ("no figure" if ok else f"PRODUCED A FIGURE: {got}")
                if not ok:
                    failures.append(f"{case['id']} produced {got} when it should have stated the "
                                    f"rule ({case.get('expect_shape')})")
            ok_n += ok

            print(f"  [{'PASS' if ok else 'FAIL'}] {case['id']}  {case['q'][:78]}")
            print(f"         expect   {case.get('expect_shape') or 'computed'}   {verdict}")
            if case.get("working"):
                print(f"         working  {case['working']}")
            if fig is not None:
                print(f"         formula  {fig.expression or '(none extracted)'}")
                print(f"         operands {[(o.get('name'), o.get('value'), o.get('source')) for o in fig.operands]}")
                if fig.missing:
                    print(f"         missing  {', '.join(fig.missing)}")
            print(f"         said     {said[:150]}")
        scores[group] = (ok_n, len(cases))

    # --- this run's calls: zero spend, HF repo ids ---
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
    acc_n, acc_d = scores.get("computable", (0, 0))
    ref_n, ref_d = scores.get("missing_operand", (0, 0))
    acc = acc_n / acc_d if acc_d else 0.0
    ref = ref_n / ref_d if ref_d else 0.0
    print(f"  accuracy   {acc_n}/{acc_d} = {acc:.3f}   floor {FLOOR_ACCURACY:.3f}  (computable)")
    print(f"  refusal    {ref_n}/{ref_d} = {ref:.3f}   floor {FLOOR_REFUSAL:.3f}  (missing_operand)")
    for group in ("no_arithmetic", "trap"):
        if group in scores:
            n, d = scores[group]
            print(f"  {group:10s} {n}/{d} = {n / d:.3f}   asserted, no figure may be produced")
    print(f"  model calls this run: {len(calls)}   total cost_usd {spend:.6f}")

    if acc < FLOOR_ACCURACY:
        failures.append(f"accuracy {acc:.3f} below floor {FLOOR_ACCURACY:.3f}")
    if ref < FLOOR_REFUSAL:
        failures.append(f"refusal {ref:.3f} below floor {FLOOR_REFUSAL:.3f}")
    if spend > 0:
        failures.append(f"this run cost {spend} - zero spend is a hard constraint")

    print()
    print("LIMIT: dev-only cases. evals/heldout/ is sealed as heldout-v1 and holds zero document")
    print("queries. Both sides of VOX-034 were written by the same agent; the cases were committed")
    print("before the implementation (git log), which is the only stand-in for blind labelling.")
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
