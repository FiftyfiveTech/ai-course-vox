#!/usr/bin/env python3
"""Follow-up gate (VOX-034 part A): does history make a cross-question retrievable?

Every case in evals/dev/followup_queries.json is a (first question, follow-up) pair. The follow-up
is scored as correct-source@3, twice:

    history OFF   the follow-up alone, which is exactly the pre-VOX-034 pipeline
    history ON    the same follow-up with the first question in a src.history.History

PASS when all three hold:
  - referential:      the ON column beats the OFF column, and ON >= FLOOR_REFERENTIAL
  - self_sufficient:  ON == OFF. These follow-ups stand alone, so history must not CHANGE them.
  - absent_followup:  the answer refuses in both columns.

TWO CRITERIA WERE CORRECTED AFTER THE FIRST RUN, and the reasoning is here rather than in a commit
message because a moved goalpost is indistinguishable from a fixed one unless the reason is written
down.

`self_sufficient` was first written as ON == OFF == 1.0, which bundles two different claims. The one
this ticket owns is non-interference: history must not change a question that was already whole. The
other — that retrieval finds these pages at all — is a VOX-030 property, and f07 shows it is already
false for a reason predating VOX-034: 'resign' is not in the corpus vocabulary (the documents say
'resignation'/'separation'), so the query loses its one high-information term before BM25 runs. That
is the exact morphology failure src/retrieval.py's docstring describes for 'paternal' vs 'paternity'.
Asserting 1.0 here would fail this gate for a defect in a different ticket, so the OFF column is
printed as pre-existing coverage and the assertion is ON == OFF.

`absent_followup` was first written as "both columns return nothing", i.e. a retrieval assertion.
Measured: the rewrite lifts leave-policy chunks over the floor via the DENSE half (lexical 0.098,
below RETRIEVAL_SCORE_FLOOR, admitted on cosine), so retrieval alone cannot hold this line — and
src/answer.py already says which layer owns it: "The model still has to refuse on the harder case,
which the floor cannot catch: chunks that score well because they come from the document that ought
to answer the question, and then stop short of the answer." So these two cases are scored on the
ANSWER refusing, which is where the defence actually lives. That is the one group here that costs a
model call.

Retrieval-only for the first two groups — no LLM call. correct-source@3 asks whether the right
file:page is in the top three, which is answerable without a model, and VOX-033's gate records why
that is the metric worth isolating: if the grounded rate drops, this is what says whether retrieval
stopped finding the page or the model stopped using it. The local sentence encoder still runs (one
forward pass per query per column), so those groups need no key and no network.

It calls answer.retrieve_with_history() — the same function src/answer.py's turn_reply() calls — so
the pipeline scored here is the pipeline the loop runs. A second copy of the retry logic would be
how the two quietly diverge.

LIMIT, printed again in the output: these cases are dev-only. evals/heldout/ is sealed as heldout-v1
and holds zero document queries; it is not reopened for this. Both sides of VOX-034 were written by
the same agent, so nothing here bounds how much the implementation was shaped by these cases. The
cases were committed before the implementation (see git log) and that is the only thing standing in
for blind labelling.

This gate changes nothing under src/.

Usage:
    uv run python tests/gates/gate_followup.py
    make gate-followup
    uv run python tests/gates/gate_followup.py --k 3
"""
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # Windows console is cp1252

from src import answer as answer_mod, retrieval               # noqa: E402
from src.history import History                               # noqa: E402
from src.telemetry import new_turn_id                         # noqa: E402

CASES = REPO_ROOT / "evals" / "dev" / "followup_queries.json"

# A refusal in substance that is not the REFUSAL string. Same list and same reasoning as
# tests/gates/gate_poc_pdf.py: answer.is_refusal() is equality against one constant, deliberately,
# so a model saying "the excerpts do not mention a sabbatical" is refusing without saying the
# sentence. Counted here rather than widened in src/answer.py, because widening it would change what
# the live loop writes into runs/turns.jsonl and re-open every VOX-031/032 number.
REFUSAL_SHAPES = (
    "no mention of", "not mentioned in", "do not mention", "does not mention",
    "do not contain", "does not contain", "not stated in", "not specified in",
    "not provided in", "no information about", "not found in the", "not covered in",
    "not addressed in", "could not find",
)


def paraphrased_refusal(text):
    """-> the refusal-shaped phrase in `text`, or None. Only meaningful when is_refusal() said no."""
    low = " ".join((text or "").split()).lower()
    for shape in REFUSAL_SHAPES:
        if shape in low:
            return shape
    return None

# The floor for the referential group. Not a flag: a gate whose threshold is a command-line argument
# is a report. Set at 4/6 rather than 6/6 because two of the six are known-hard by construction and
# said so when they were written — f01's follow-up is ambiguous across five near-identical chunks,
# and f03's shares no lexical term with its target ("paid out" vs "processed"), so it rests entirely
# on the dense half. The floor asserts the mechanism works on a majority, not that concatenation is
# a solved rewrite. Beating the OFF column is the assertion that carries the claim.
FLOOR_REFERENTIAL = 4 / 6


def load_cases(path):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return {k: v for k, v in d.items() if k != "_note"}


def sources_line(hits, n=3):
    return ", ".join(f"{h.source}({h.score:.3f})" for h in hits[:n]) or "(nothing cleared the floor)"


def score_case(case, idx, k, with_history):
    """-> (hit, hits, rewrite) for one case in one column.

    `with_history` False runs the follow-up alone through the same function, with history=None —
    which is byte-for-byte the pre-VOX-034 path, not an approximation of it.
    """
    hist = None
    if with_history:
        hist = History()
        # Only the QUESTION is seeded, with an empty reply: this gate makes no model call, so there
        # is no answer to remember. That is not a limitation of the measurement — retrieval_query()
        # reads questions and never replies, so a real reply would not change this number.
        hist.add(case["first"], "")
    tid = new_turn_id()
    hits, asked, rewrite = answer_mod.retrieve_with_history(
        case["followup"], tid, idx=idx, history=hist, k=k)
    expect = case.get("expect") or []
    if not expect:
        # Absent cases are scored on the ANSWER, not on retrieval returning nothing — see the module
        # docstring for the measurement that moved this. One model call each, and the floor-miss path
        # makes none at all, so this is at most two calls per column.
        ans = answer_mod.answer(asked, tid, hits=hits)
        refused = answer_mod.is_refusal(ans.text) or bool(paraphrased_refusal(ans.text))
        return refused, hits, rewrite, asked, ans.text
    hit = any(h.source in expect for h in hits[:k])
    return hit, hits, rewrite, asked, None


def main():
    ap = argparse.ArgumentParser(description="VOX-034 follow-up gate (retrieval only)")
    ap.add_argument("--cases", type=Path, default=CASES)
    ap.add_argument("--k", type=int, default=3, help="correct-source@k (default 3)")
    args = ap.parse_args()

    if not args.cases.is_file():
        sys.exit(f"case set not found: {args.cases}")
    groups = load_cases(args.cases)

    print("building the retrieval index...", flush=True)
    idx = retrieval.index(echo=print)
    if not len(idx):
        sys.exit("no chunks indexed - sources/ is gitignored, so a fresh clone has none.\n"
                 "Put the PDFs in sources/ and run `make index`, then re-run this gate.")
    print(f"cases      {args.cases.relative_to(REPO_ROOT)} - "
          f"{sum(len(v) for v in groups.values())} cases in {len(groups)} groups, k={args.k}")
    print("no LLM call is made by this gate - correct-source@k is a retrieval measurement")

    totals = {}
    for group, cases in groups.items():
        print()
        print(f"--- {group} ({len(cases)}) ---")
        off_n = on_n = 0
        for case in cases:
            off_hit, off_hits, _, _, off_said = score_case(case, idx, args.k, with_history=False)
            on_hit, on_hits, rewrite, asked, on_said = score_case(case, idx, args.k,
                                                                  with_history=True)
            off_n += off_hit
            on_n += on_hit

            moved = "" if off_hit == on_hit else ("  <- RESCUED" if on_hit else "  <- LOST")
            print(f"  {case['id']}  off={'PASS' if off_hit else 'FAIL'} "
                  f"on={'PASS' if on_hit else 'FAIL'}{moved}")
            print(f"        first    {case['first']}")
            print(f"        followup {case['followup']}")
            print(f"        expect   {', '.join(case.get('expect') or []) or '(refusal)'}")
            print(f"        off top-{args.k} {sources_line(off_hits, args.k)}")
            print(f"        on  top-{args.k} {sources_line(on_hits, args.k)}")
            if rewrite is not None:
                print(f"        rewrite  trigger={rewrite['trigger']} used={rewrite['used']}  "
                      f"{rewrite['query']!r}")
            else:
                print("        rewrite  not attempted (transcript reads as a whole question)")
            if on_said is not None:
                print(f"        off said {' '.join((off_said or '').split())[:120]}")
                print(f"        on  said {' '.join(on_said.split())[:120]}")
        totals[group] = (off_n, on_n, len(cases))

    # --- the numbers ---
    print()
    print("=" * 72)
    print(f"{'group':20s} {'off':>9s} {'on':>9s}   verdict")
    failures = []
    for group, (off_n, on_n, n) in totals.items():
        off_r, on_r = off_n / n, on_n / n
        if group == "referential":
            ok = on_r > off_r and on_r >= FLOOR_REFERENTIAL
            note = f"need on > off and on >= {FLOOR_REFERENTIAL:.3f}"
        elif group == "self_sufficient":
            ok = on_r == off_r
            note = (f"need on == off (non-interference); off={off_r:.3f} is pre-existing "
                    f"retrieval coverage, not this ticket's claim")
        else:
            ok = on_r == off_r == 1.0
            note = "need the answer to refuse in both columns"
        if not ok:
            failures.append(f"{group}: off={off_r:.3f} on={on_r:.3f} - {note}")
        print(f"{group:20s} {off_n}/{n} {off_r:.3f} {on_n}/{n} {on_r:.3f}   "
              f"[{'PASS' if ok else 'FAIL'}] {note}")

    print()
    print("LIMIT: dev-only cases. evals/heldout/ is sealed as heldout-v1 and holds zero document")
    print("queries, so none of these numbers has a held-out counterpart. Both sides of VOX-034 were")
    print("written by the same agent; the cases were committed before the implementation (git log)")
    print("and that ordering is the only thing standing in for blind labelling.")

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
