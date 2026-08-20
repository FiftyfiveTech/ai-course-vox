"""Retrieve over the chunk file and print what came back (VOX-030), optionally answer it (VOX-031).

    make ask Q="how many casual leaves am I entitled to in a year"
    make answer Q="how many casual leaves am I entitled to in a year"
    uv run python scripts/ask.py "what notice period do I serve" --k 3
    uv run python scripts/ask.py "how much casual leave" --answer --llm gpt-oss
    uv run python scripts/ask.py --calibrate          pick the score floor from real queries

Retrieval is what this prints by default, and it stays that way on purpose. The criterion for
VOX-030 is "a retrieval function returns ranked results with provenance fields", so the script calls
`retrieve()` and prints the fields it actually returned — doc_id, page, chunk_idx, score — rather
than a summary of them. Two runs are that verification: one query the corpus answers, one it does
not. A miss prints the best score it *did* see next to the floor that rejected it, because "nothing
found" and "the floor is set wrong" look identical otherwise, and only one of them is an answer.

Retrieval that way needs no model, no network and no key, and `--calibrate` sweeps thirteen queries
in one run — so the LLM call is behind `--answer` rather than in front of it. BM25 is arithmetic, so
the default path still writes nothing to runs/calls.jsonl (see the note in src/retrieval.py);
`--answer` writes exactly one line there, through `arms.llm` like every other model call.

VOX-031 is what `--answer` prints: the spoken answer plus the doc:page it was grounded in, or the
refusal. The refusal has two sources and the output says which — a query that clears no chunk is
refused here with no model call at all, and one whose chunks do not contain the answer is refused by
the model. See src/answer.py on why both say the same sentence.

The elapsed_ms figures are measured here rather than inside the modules because they are
measurements of this run. Putting retrieval and answer latency on the *turn* record is VOX-032.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # Windows console is cp1252

from src import answer as answer_mod, retrieval, telemetry                      # noqa: E402
from src.config import (BM25_B, BM25_EPSILON, BM25_K1, CHUNKS_FILE, DEFAULT_LLM,  # noqa: E402
                        RETRIEVAL_SCORE_FLOOR, RETRIEVAL_TOP_K, REPO_ROOT)

FLOOR_QUERIES = REPO_ROOT / "evals" / "dev" / "retrieval_floor_queries.json"
SNIPPET = 96


def snippet(text, width=SNIPPET):
    text = " ".join(text.split())
    return text if len(text) <= width else text[:width - 1] + "…"


def show_index(idx):
    lengths = [len(t) for t in idx.corpus]
    avg = sum(lengths) / max(len(lengths), 1)
    print(f"chunks     {CHUNKS_FILE}")
    print(f"index      {len(idx)} chunks over {len(idx.doc_ids)} documents, "
          f"{avg:.0f} scoreable terms per chunk (min {min(lengths)}, max {max(lengths)})")
    print(f"bm25       Okapi k1={BM25_K1} b={BM25_B} epsilon={BM25_EPSILON}")


def ask(idx, query, k, floor):
    """Print the ranked hits for one query. -> the hits above the floor, best first.

    The hits are returned rather than counted so `--answer` can pass these exact chunks to
    src/answer.py instead of retrieving a second time. Retrieving twice would print one set of
    scores and ground the answer in another, and nothing in the output would say so.
    """
    terms = retrieval.tokenize(query)
    print(f"\nquery      {query!r}")
    print(f"terms      {terms}   (stopwords and single letters dropped)")
    if not terms:
        print("\nnothing found — the query has no scoreable terms left after stopword removal.")
        return []

    t0 = time.perf_counter()
    hits = retrieval.retrieve(query, k=k, floor=floor, idx=idx)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    ranked = idx.rank(query)                    # unfloored, so a miss can show what it did see
    best = ranked[0].score if ranked else 0.0
    print(f"retrieval  {elapsed_ms:.1f} ms   top-1 score {best:.3f}   floor {floor:.3f}")

    if not hits:
        print(f"\nnothing found — best score {best:.3f} does not clear the floor {floor:.3f}. "
              f"Not in the documents.")
        return []

    width = max(len(h.source) for h in hits)
    print(f"\n  {'#':<3}{'score':>7}{'raw':>8}  {'source':<{width}}  chunk  text")
    for rank, h in enumerate(hits, start=1):
        print(f"  {rank:<3}{h.score:>7.3f}{h.raw:>8.2f}  {h.source:<{width}}  {h.chunk_idx:>5}  "
              f"{snippet(h.text)}")
    print(f"\n{len(hits)} of {len(ranked)} scoring chunks returned "
          f"(k={k}), fields: {', '.join(hits[0].as_dict())}")
    return hits


def answer(query, hits, model_id):
    """Print the grounded answer for one query. -> exit code.

    The arm banner and the turn_id are printed only when there is going to be a call. Printing them
    on a floor miss would name a model that was never asked anything and a turn_id that joins to no
    calls.jsonl line — which is precisely the kind of output this repo's gates are supposed to be
    able to trust.
    """
    if not hits:
        # No arm, no turn_id, no call. See src/answer.py: nothing cleared the floor, so there is no
        # context to be grounded in and nothing for a model to do but invent.
        result = answer_mod.answer(query, turn_id=None, hits=[])
        print("\nanswer     REFUSED before any model call — no chunk cleared the floor, so there "
              "was nothing to ground an answer in.")
        print(f"\n  {result.text}")
        print(f"\nsources    {result.sources}   (empty: a refusal cites nothing)")
        return 0

    from src import arms                      # local: importing arms pulls in the whole stack

    arm = arms.warm("llm", model_id)          # a no-op for a hosted arm; a real load for ollama
    turn_id = telemetry.new_turn_id()
    print(f"\narm        {arm.repo_id}  ({'local' if arm.local else 'remote'} via {arm.provider}, "
          f"{arm.backend})")
    fb = arms.fallback_for("llm", arm)
    if fb is not None:
        print(f"fallback   {fb.repo_id} (local, {fb.backend}) if the free tier refuses")
    print(f"prompt     {answer_mod.PROMPT_FILE.name}   context {len(hits)} chunks, "
          f"{len(answer_mod.context_block(hits))} chars")
    print(f"turn_id    {turn_id}   (joins this run to its runs/calls.jsonl line)")

    t0 = time.perf_counter()
    result = answer_mod.answer(query, turn_id, hits=hits, model_id=model_id)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    # A refusal *here* is the model's own, on chunks that did clear the floor — the case the floor
    # cannot catch. The listener hears the same sentence either way (see src/answer.py); a reader of
    # this output needs to know which happened, so the two are worded differently here and only here.
    if not result.grounded:
        print(f"\nanswer     {elapsed_ms:.0f} ms   REFUSED by {arm.repo_id} — the chunks cleared "
              f"the floor but do not contain the answer.")
        print(f"\n  {result.text}")
        print(f"\nsources    {result.sources}   (empty: a refusal cites nothing)")
        return 0

    print(f"\nanswer     {elapsed_ms:.0f} ms   grounded in {len(result.sources)} of "
          f"{len(hits)} chunks")
    print(f"\n  {result.text}")
    print(f"\nsources    {', '.join(result.labels)}")
    for source in result.sources:
        print(f"           doc_id={source['doc_id']!r}  page={source['page']}")
    return 0


def calibrate(idx):
    """Print the top-1 score for known-answerable and known-absent queries. -> exit code.

    The floor is whatever number sits in the gap between the two columns. If they overlap, there is
    no such number and the honest output is to say so rather than to split the difference — the
    stopword list, the chunk geometry or k1/b is what needs changing, and then this is re-run.
    """
    if not FLOOR_QUERIES.is_file():
        sys.exit(f"no calibration queries at {FLOOR_QUERIES}")
    spec = json.loads(FLOOR_QUERIES.read_text(encoding="utf-8"))

    def scores(entries):
        out = []
        for e in entries:
            ranked = idx.rank(e["q"])
            top = ranked[0] if ranked else None
            out.append((e["q"], e.get("expect_doc"), top))
        return out

    hits = scores(spec["hit"])
    misses = scores(spec["miss"])

    for label, rows in (("answerable", hits), ("absent from the corpus", misses)):
        print(f"\n=== {label} ({len(rows)} queries) ===")
        print(f"  {'top-1':>7}{'raw':>8}  {'source':<44}  query")
        for query, expect, top in rows:
            src = top.source if top else "-"
            if expect:
                src += " ok" if top and top.doc_id in expect else f" (want {'|'.join(expect)})"
            print(f"  {(top.score if top else 0.0):>7.3f}{(top.raw if top else 0.0):>8.2f}  "
                  f"{src:<44}  {query}")

    hit_scores = sorted(t.score if t else 0.0 for _, _, t in hits)
    miss_scores = sorted(t.score if t else 0.0 for _, _, t in misses)
    lo_hit, hi_miss = hit_scores[0], miss_scores[-1]
    print(f"\nanswerable  min {lo_hit:.3f}  max {hit_scores[-1]:.3f}")
    print(f"absent      min {miss_scores[0]:.3f}  max {hi_miss:.3f}")

    # Informational, not this ticket's number: correct-source@k over a written query set is what
    # VOX-033's gate prints, over its own queries. Here it is a sanity check that the floor is being
    # calibrated on queries that retrieve the right thing in the first place.
    right_doc = sum(1 for _, expect, top in hits if expect and top and top.doc_id in expect)
    print(f"top-1 in an expected document: {right_doc}/{len(hits)}")

    if lo_hit > hi_miss:
        floor = round((lo_hit + hi_miss) / 2, 2)
        print(f"\nseparable: every answerable query outscores every absent one.")
        print(f"midpoint of the gap [{hi_miss:.3f}, {lo_hit:.3f}] -> "
              f"RETRIEVAL_SCORE_FLOOR = {floor}")
        return 0

    print(f"\nNOT separable: an absent query scores {hi_miss:.3f}, above the weakest answerable one "
          f"at {lo_hit:.3f}.")
    print("No single floor separates them. Tune the stopword list, the chunk geometry or k1/b and "
          "re-run — do not split the difference.")
    return 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("query", nargs="*", help="the question to retrieve for")
    ap.add_argument("--k", type=int, default=RETRIEVAL_TOP_K,
                    help=f"how many chunks to return (default {RETRIEVAL_TOP_K})")
    ap.add_argument("--floor", type=float, default=RETRIEVAL_SCORE_FLOOR,
                    help=f"score floor (default {RETRIEVAL_SCORE_FLOOR})")
    ap.add_argument("--calibrate", action="store_true",
                    help=f"score {FLOOR_QUERIES.name} and print the floor the gap implies")
    ap.add_argument("--answer", action="store_true",
                    help="also send the retrieved chunks to the LLM arm for a grounded answer "
                         "(VOX-031). This is the only path here that makes a model call.")
    ap.add_argument("--llm", metavar="MODEL_ID", default=None,
                    help="which LLM arm answers, as an HF repo id or alias; default "
                         f"{DEFAULT_LLM.repo_id}. Only used with --answer.")
    args = ap.parse_args()

    query = " ".join(args.query).strip()
    if not query and not args.calibrate:
        ap.error('nothing to ask. Try: make ask Q="how many casual leaves do I get"')
    if args.calibrate and args.answer:
        # Thirteen queries would be thirteen calls to measure a threshold that is pure arithmetic.
        ap.error("--calibrate measures the retrieval floor and needs no model. Drop --answer.")

    t0 = time.perf_counter()
    idx = retrieval.index()
    build_ms = (time.perf_counter() - t0) * 1000
    show_index(idx)
    print(f"build      {build_ms:.0f} ms (once per process, not per turn)")

    if args.calibrate:
        return calibrate(idx)
    hits = ask(idx, query, args.k, args.floor)
    if args.answer:
        return answer(query, hits, args.llm)
    return 0


if __name__ == "__main__":
    sys.exit(main())
