"""Retrieve over the chunk file and print what came back (VOX-030's verification).

    make ask Q="how many casual leaves am I entitled to in a year"
    uv run python scripts/ask.py "what notice period do I serve" --k 3
    uv run python scripts/ask.py --calibrate          pick the score floor from real queries

This is the ticket's evidence, not a smoke test bolted on afterwards: the criterion is "a retrieval
function returns ranked results with provenance fields", so the script calls `retrieve()` and prints
the fields it actually returned — doc_id, page, chunk_idx, score — rather than a summary of them.

Two runs are the verification: one query the corpus answers, one it does not. A miss prints the best
score it *did* see next to the floor that rejected it, because "nothing found" and "the floor is set
wrong" look identical otherwise, and only one of them is an answer.

No model, no network, no key. BM25 is arithmetic, so nothing here writes to runs/calls.jsonl — see
the note on that in src/retrieval.py. The elapsed_ms below is measured here rather than inside the
module because it is a measurement of this run; putting retrieval latency on the *turn* record is
VOX-032.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # Windows console is cp1252

from src import retrieval                                                       # noqa: E402
from src.config import (BM25_B, BM25_EPSILON, BM25_K1, CHUNKS_FILE,             # noqa: E402
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
    """Print the ranked hits for one query. -> the number of hits above the floor."""
    terms = retrieval.tokenize(query)
    print(f"\nquery      {query!r}")
    print(f"terms      {terms}   (stopwords and single letters dropped)")
    if not terms:
        print("\nnothing found — the query has no scoreable terms left after stopword removal.")
        return 0

    t0 = time.perf_counter()
    hits = retrieval.retrieve(query, k=k, floor=floor, idx=idx)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    ranked = idx.rank(query)                    # unfloored, so a miss can show what it did see
    best = ranked[0].score if ranked else 0.0
    print(f"retrieval  {elapsed_ms:.1f} ms   top-1 score {best:.3f}   floor {floor:.3f}")

    if not hits:
        print(f"\nnothing found — best score {best:.3f} does not clear the floor {floor:.3f}. "
              f"Not in the documents.")
        return 0

    width = max(len(h.source) for h in hits)
    print(f"\n  {'#':<3}{'score':>7}{'raw':>8}  {'source':<{width}}  chunk  text")
    for rank, h in enumerate(hits, start=1):
        print(f"  {rank:<3}{h.score:>7.3f}{h.raw:>8.2f}  {h.source:<{width}}  {h.chunk_idx:>5}  "
              f"{snippet(h.text)}")
    print(f"\n{len(hits)} of {len(ranked)} scoring chunks returned "
          f"(k={k}), fields: {', '.join(hits[0].as_dict())}")
    return len(hits)


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
    args = ap.parse_args()

    query = " ".join(args.query).strip()
    if not query and not args.calibrate:
        ap.error('nothing to ask. Try: make ask Q="how many casual leaves do I get"')

    t0 = time.perf_counter()
    idx = retrieval.index()
    build_ms = (time.perf_counter() - t0) * 1000
    show_index(idx)
    print(f"build      {build_ms:.0f} ms (once per process, not per turn)")

    if args.calibrate:
        return calibrate(idx)
    ask(idx, query, args.k, args.floor)
    return 0


if __name__ == "__main__":
    sys.exit(main())
