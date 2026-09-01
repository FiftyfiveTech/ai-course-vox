#!/usr/bin/env python3
"""POC gate: the grounded-answer rate, printed.

Ten written queries from evals/dev/pdf_queries.json through the same retrieval and answer path the
turn loop runs — 8 answerable with an expected source file:page, 2 deliberately absent from the
corpus.

PASS when:
  - correct-source@3 over the 8 answerable is >= 7/8 (0.875)
  - the refusal rate over the 2 absent is 2/2 (1.000)
  - every calls.jsonl entry for this run carries a HF repo id and cost_usd == 0.0

The grounded-answer rate is printed with its denominator and NOT asserted. It is bounded above by
correct-source@3 (an answer cannot be grounded in a page retrieval did not return) and below by the
numeric guard's correct refusals, so asserting it too would fail this gate twice for one cause.

LIMIT, stated here and again in the output: these queries are dev-only. evals/heldout/ is sealed as
heldout-v1 and holds zero document queries, and it is not reopened for this, so none of these
numbers has a held-out counterpart.

This gate changes nothing under src/. It reads the index and calls the same functions the loop
calls, so a failing run cannot degrade the demo.

Usage:
    uv run python tests/gates/gate_poc_pdf.py
    make gate-poc
    uv run python tests/gates/gate_poc_pdf.py --llm gpt-oss --k 5
"""
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # Windows console is cp1252

from src import answer as answer_mod, arms, retrieval            # noqa: E402
from src.telemetry import CALLS_LOG, TURNS_LOG, new_turn_id     # noqa: E402

QUERIES = REPO_ROOT / "evals" / "dev" / "pdf_queries.json"

# The floors. Constants, not flags — a gate whose threshold is a command-line argument is a report.
#
# correct-source@3 >= 7/8: `make floors` already measures top-1 in an expected document 7/7 for both
# retrieval halves, so 8/8 is the expectation over labels read off the actual chunks. The one query
# of slack is spent, deliberately and in advance, on q03 — the STT-damaged form of q02, which the
# pdf_queries.json _note explains. A regression on any clean query takes this to 6/8 and fails.
#
# refusal 2/2: end to end this stack already refuses all six of the floor set's absent queries. A
# confident answer to a question the documents do not address is the failure that costs trust, and
# n=2 is too small to negotiate a fraction over.
FLOOR_CORRECT_SOURCE = 7 / 8
FLOOR_REFUSAL = 1.0

# A paraphrased refusal, counted here and not in src/answer.py.
#
# answer.is_refusal() is equality against one constant string, deliberately — "not a parser", says
# its docstring. So "There is no mention of sabbatical leave in the provided excerpts" is a refusal
# in substance, is not the REFUSAL string, and the turn logs grounded: true. ARCHITECTURE.md records
# that as a shape rather than a fixed bug and leaves the rate's problem with it to this gate.
#
# It is fixed here rather than there because widening is_refusal() changes what the live turn loop
# writes into runs/turns.jsonl, which re-opens every VOX-031/032 number already published. That is
# its own before/after measurement, not a side effect of writing a gate.
#
# Every phrase names the *source* — mentioned in, contain, stated in — rather than being a bare
# negation. "There is no cap on carry forward" is an answer, and a pattern for "there is no" would
# refuse it. Each catch is printed with the reply that produced it, because being generous about
# what counts as a refusal is generous in the direction that makes this gate easier to pass, and
# that has to be auditable rather than trusted.
REFUSAL_SHAPES = (
    "no mention of",
    "not mentioned in",
    "do not mention",
    "does not mention",
    "do not contain",
    "does not contain",
    "not stated in",
    "not specified in",
    "not provided in",
    "no information about",
    "not found in the",
    "not covered in",
    "not addressed in",
)


def paraphrased_refusal(text):
    """-> the refusal-shaped phrase in `text`, or None. Only meaningful when is_refusal() said no."""
    low = " ".join((text or "").split()).lower()
    for shape in REFUSAL_SHAPES:
        if shape in low:
            return shape
    return None


def load_queries(path):
    spec = json.loads(path.read_text(encoding="utf-8"))
    return spec["answerable"], spec["absent"]


def sources_line(hits):
    return ", ".join(f"{h.source}({h.score:.3f})" for h in hits) or "-"


def rate(n, d):
    return (n / d) if d else 0.0


def turns_log_rate():
    """-> (grounded, total) summed off runs/turns.jsonl, or (None, None) if the field is absent.

    ARCHITECTURE.md: `grounded` is written on every turn that reached a reply, not only the grounded
    ones, precisely so this rate has a denominator that was recorded while the turns were happening.
    Historic and cumulative across every run on this machine — informational, not this gate's number.
    """
    if not TURNS_LOG.exists():
        return None, None
    grounded = total = 0
    for line in TURNS_LOG.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if "grounded" in rec:
            total += 1
            grounded += bool(rec["grounded"])
    return (grounded, total) if total else (None, None)


def main():
    ap = argparse.ArgumentParser(description="POC gate — correct-source@3 and the refusal rate")
    ap.add_argument("--queries", type=Path, default=QUERIES,
                    help=f"query set to score (default: {QUERIES.relative_to(REPO_ROOT)})")
    ap.add_argument("--k", type=int, default=3,
                    help="top-k for correct-source@k (default 3; the floors are set for k=3)")
    # Only the llm arm is flagged. This gate never runs STT or TTS, and offering --stt/--tts would
    # advertise a knob that cannot change any number it prints.
    ap.add_argument("--llm", metavar="MODEL_ID", default=None,
                    help="llm arm; default is config's. Same aliases as make demo.")
    args = ap.parse_args()

    if not args.queries.is_file():
        sys.exit(f"query set not found: {args.queries}")
    answerable, absent = load_queries(args.queries)
    llm_arm = arms.resolve("llm", args.llm)

    print("building the retrieval index…", flush=True)
    idx = retrieval.index(echo=print)
    if not len(idx):
        sys.exit("no chunks indexed — sources/ is gitignored, so a fresh clone has none.\n"
                 "Put the PDFs in sources/ and run `make index`, then re-run this gate.")
    print(f"llm arm    {llm_arm.repo_id}  "
          f"({'local' if llm_arm.local else 'remote via ' + llm_arm.provider})")
    print(f"queries    {args.queries.relative_to(REPO_ROOT)} — "
          f"{len(answerable)} answerable, {len(absent)} absent, k={args.k}")

    calls_pos = CALLS_LOG.stat().st_size if CALLS_LOG.exists() else 0
    turn_ids = set()
    failures = []
    paraphrases = []

    # --- the 8 answerable: is the expected file:page in the top k, and did a model answer from it --
    print()
    print(f"--- answerable ({len(answerable)}) ---")
    found = grounded_n = grounded_right = 0
    for case in answerable:
        tid = new_turn_id()
        turn_ids.add(tid)
        hits = retrieval.retrieve(case["q"], k=args.k, turn_id=tid)
        got = [h.source for h in hits]
        hit = any(src in case["expect"] for src in got)
        found += hit

        ans = answer_mod.answer(case["q"], tid, hits=hits, model_id=llm_arm.repo_id)
        shape = paraphrased_refusal(ans.text) if ans.grounded else None
        if shape:
            paraphrases.append((case["id"], shape, ans.text))
        really_grounded = ans.grounded and not shape
        grounded_n += really_grounded
        grounded_right += really_grounded and hit

        print(f"  [{'PASS' if hit else 'FAIL'}] {case['id']}  {case['q']}")
        print(f"         expect  {', '.join(case['expect'])}")
        print(f"         top-{args.k}   {sources_line(hits)}")
        if not hit:
            # The floored result is what a caller gets, so it is what is scored. The unfloored rank
            # separates "the floor rejected the right page" from "the right page was never ranked" —
            # two different fixes, and otherwise two identical-looking failures.
            unfloored = ", ".join(f"{h.source}({h.score:.3f})"
                                  for h in idx.rank(case["q"])[:args.k])
            print(f"         unfloored lexical {unfloored}")
        print(f"         grounded={really_grounded}"
              f"{' (paraphrased refusal: ' + repr(shape) + ')' if shape else ''}")
        print(f"         said    {' '.join(ans.text.split())[:160]}")

    # --- the 2 absent: both refusal paths, and neither may answer ---
    print()
    print(f"--- absent ({len(absent)}) ---")
    refused = 0
    for case in absent:
        tid = new_turn_id()
        turn_ids.add(tid)
        hits = retrieval.retrieve(case["q"], k=args.k, turn_id=tid)
        ans = answer_mod.answer(case["q"], tid, hits=hits, model_id=llm_arm.repo_id)

        exact = answer_mod.is_refusal(ans.text)
        shape = None if exact else paraphrased_refusal(ans.text)
        if shape:
            paraphrases.append((case["id"], shape, ans.text))
        ok = exact or bool(shape)
        refused += ok
        # Which refusal path ran, because src/answer.py distinguishes them and only one involves a
        # model: no hits means the floor refused it with no call at all.
        path = "floor (no model call)" if not hits else "model"
        print(f"  [{'PASS' if ok else 'FAIL'}] {case['id']}  {case['q']}")
        print(f"         top-{args.k}   {sources_line(hits)}")
        print(f"         refused via {path}"
              f"{'' if ok else ' — NOTHING REFUSED, this is an answer to an unanswerable question'}")
        if shape:
            print(f"         paraphrased refusal: {shape!r}")
        print(f"         said    {' '.join(ans.text.split())[:160]}")
        if not ok:
            failures.append(f"{case['id']} answered a question the corpus does not cover: "
                            f"{' '.join(ans.text.split())[:120]}")

    # --- this run's model calls ---
    calls = []
    if CALLS_LOG.exists():
        with CALLS_LOG.open(encoding="utf-8") as fh:
            fh.seek(calls_pos)
            for line in fh:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    if entry.get("turn_id") in turn_ids:
                        calls.append(entry)

    cs_rate = rate(found, len(answerable))
    ref_rate = rate(refused, len(absent))
    g_rate = rate(grounded_n, len(answerable))

    print()
    print("=" * 70)
    print("  POC GATE — grounded answers over a written query set")
    print("=" * 70)
    print(f"  correct-source@{args.k}   {found}/{len(answerable)} = {cs_rate:.3f}"
          f"   floor {FLOOR_CORRECT_SOURCE:.3f}")
    print(f"  refusal rate        {refused}/{len(absent)} = {ref_rate:.3f}"
          f"   floor {FLOOR_REFUSAL:.3f}")
    print(f"  grounded-answer     {grounded_n}/{len(answerable)} = {g_rate:.3f}"
          f"   printed, not asserted")
    # The intersection, and the more honest reading of the line above it. `grounded` only says a
    # model answered from the excerpts it was handed — it is true even when those excerpts came off
    # the wrong pages, which is exactly what a rate reported alone would hide. Not asserted either:
    # it cannot exceed correct-source@k, which is already the floor being enforced.
    print(f"  ...on an expected source  {grounded_right}/{len(answerable)} = "
          f"{rate(grounded_right, len(answerable)):.3f}   grounded AND correct-source@{args.k}")
    print("-" * 70)
    print(f"  paraphrased refusals caught: {len(paraphrases)}"
          f"   (answer.is_refusal() would have read these as answers)")
    for qid, shape, text in paraphrases:
        print(f"    {qid}  matched {shape!r}: {' '.join(text.split())[:110]}")
    hist_g, hist_n = turns_log_rate()
    if hist_n:
        print(f"  runs/turns.jsonl    {hist_g}/{hist_n} = {rate(hist_g, hist_n):.3f} grounded — "
              f"every live turn on this machine, cumulative. Not this gate's number.")
    else:
        print("  runs/turns.jsonl    no turns carry `grounded` yet.")
    print()
    print(f"  model calls this run: {len(calls)}")
    for c in calls:
        mark = "     " if c.get("ok") else "  !  "
        note = f"  -> fell back from {c['fallback_for']}" if c.get("fallback_for") else ""
        print(f"  {mark}[{c['stage']:>5}] {c['model_id']}  provider={c['provider']}  "
              f"cost_usd={c['cost_usd']}{note}")
        if not c.get("ok"):
            print(f"          FAILED: {c.get('error', '')}")
    print()
    print("  LIMIT: dev-only. evals/heldout/ is sealed as heldout-v1 and holds zero document")
    print("         queries; it was not reopened for this, so none of these numbers has a")
    print("         held-out counterpart. Every query above is one the Builder can read.")

    # --- assertions ---
    if cs_rate < FLOOR_CORRECT_SOURCE:
        failures.append(f"correct-source@{args.k} = {cs_rate:.3f} is below the floor "
                        f"{FLOOR_CORRECT_SOURCE:.3f} ({found}/{len(answerable)})")
    if ref_rate < FLOOR_REFUSAL:
        failures.append(f"refusal rate = {ref_rate:.3f} is below the floor {FLOOR_REFUSAL:.3f} "
                        f"({refused}/{len(absent)})")

    for c in calls:
        if not c.get("model_id"):
            failures.append(f"[{c['stage']}] missing model_id (must be a HF repo id)")
        if c.get("cost_usd", -1) != 0.0:
            failures.append(f"[{c['stage']}] cost_usd={c['cost_usd']} — zero spend violated")
        if not c.get("ok"):
            failures.append(f"[{c['stage']}] call failed: {c.get('error', '')}")

    # Infrastructure failures are warnings, not gate failures — the same rule gate_phase1.py uses.
    # This gate makes up to ten live calls on a free tier; a rate limit covered by the local
    # fallback must not read as a retrieval regression.
    infra, real = [], []
    for f in failures:
        (infra if any(s in f for s in ("ollama daemon", "ReadTimeout", "ConnectError")) else
         real).append(f)

    print("=" * 70)
    if infra:
        print("\n  infrastructure skips (not gate failures):")
        for f in infra:
            print(f"  - {f}")

    if real:
        print("\nFAIL")
        for f in real:
            print(f"  - {f}")
        sys.exit(1)

    print("\nPASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
