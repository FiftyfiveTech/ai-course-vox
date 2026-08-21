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
                        DENSE_SCORE_FLOOR, HYBRID_RETRIEVAL, RETRIEVAL_SCORE_FLOOR,
                        RETRIEVAL_TOP_K, REPO_ROOT)

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
    if idx.has_vectors:
        print(f"dense      {idx.embed_arm}, {idx.vectors.shape[1]}-dim, "
              f"floor {DENSE_SCORE_FLOOR:.3f}   fused by rank (RRF)")
    else:
        print("dense      off — BM25 alone. `make index` builds the vectors; "
              "VOX_HYBRID_RETRIEVAL=0 asks for this deliberately.")


def ask(idx, query, k, floor, dense_floor):
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
    hits = retrieval.retrieve(query, k=k, floor=floor, idx=idx, dense_floor=dense_floor)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    ranked = idx.rank(query)                    # unfloored, so a miss can show what it did see
    best = ranked[0].score if ranked else 0.0
    dense = idx.dense_rank(query) if idx.has_vectors else []
    best_dense = dense[0][1] if dense else None

    line = f"retrieval  {elapsed_ms:.1f} ms   lexical top-1 {best:.3f} (floor {floor:.3f})"
    if best_dense is not None:
        line += f"   dense top-1 {best_dense:.3f} (floor {dense_floor:.3f})"
    print(line)

    if not hits:
        # Both halves have to miss now, so the sentence has to name both — "the floor rejected it"
        # was a complete explanation when there was one floor and is a half-truth with two.
        why = f"best lexical {best:.3f} < {floor:.3f}"
        if best_dense is not None:
            why += f", best cosine {best_dense:.3f} < {dense_floor:.3f}"
        print(f"\nnothing found — {why}. Not in the documents.")
        return []

    width = max(len(h.source) for h in hits)
    print(f"\n  {'#':<3}{'lex':>6}{'lr':>4}{'cos':>7}{'dr':>4}{'fused':>8}  "
          f"{'source':<{width}}  chunk  text")
    for rank, h in enumerate(hits, start=1):
        cos = "     -" if h.dense is None else f"{h.dense:>7.3f}"
        lr = "  -" if h.lex_rank is None else f"{h.lex_rank:>4}"
        dr = "  -" if h.dense_rank is None else f"{h.dense_rank:>4}"
        print(f"  {rank:<3}{h.score:>6.3f}{lr}{cos}{dr}{h.fused:>8.4f}  {h.source:<{width}}  "
              f"{h.chunk_idx:>5}  {snippet(h.text, 60)}")
    # lr/dr are where each half put the chunk, and they are the interesting column: a hit with
    # lr 110 and dr 1 is one only the encoder found, which is the whole reason it exists.
    print(f"\n{len(hits)} returned (k={k}) out of {len(ranked)} lexically scoring chunks"
          + (f" and {len(dense)} ranked by cosine" if dense else "")
          + f", fields: {', '.join(hits[0].as_dict())}")
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


def separation(label, hit_scores, miss_scores, floor_name):
    """Print the gap between answerable and absent for one scorer. -> the implied floor, or None.

    Factored out because there are two scorers now and the arithmetic is the same for both: the
    floor is whatever number sits in the gap between the two columns, and if they overlap there is
    no such number. Saying so is the honest output; splitting the difference would be picking a
    threshold that is known to be wrong for some query in the set.
    """
    lo_hit, hi_miss = min(hit_scores), max(miss_scores)
    print(f"\n{label}")
    print(f"  answerable  min {lo_hit:.3f}  max {max(hit_scores):.3f}")
    print(f"  absent      min {min(miss_scores):.3f}  max {hi_miss:.3f}")
    if lo_hit > hi_miss:
        floor = round((lo_hit + hi_miss) / 2, 3)
        print(f"  separable — gap [{hi_miss:.3f}, {lo_hit:.3f}], midpoint -> {floor_name} = {floor}")
        return floor
    print(f"  NOT separable — an absent query scores {hi_miss:.3f}, above the weakest answerable "
          f"one at {lo_hit:.3f}.")
    return None


def calibrate(idx):
    """Print what each half scores on known-answerable and known-absent queries. -> exit code.

    Two floors to set now, and they are not interchangeable. The lexical one is a fraction of the
    query's own information content; the dense one is a cosine, which has an absolute scale and does
    not move with question length. Both are corpus-specific and both are measured here rather than
    guessed — and the union rule at the bottom is the one that actually decides a refusal, so it is
    the one that has to separate.
    """
    if not FLOOR_QUERIES.is_file():
        sys.exit(f"no calibration queries at {FLOOR_QUERIES}")
    spec = json.loads(FLOOR_QUERIES.read_text(encoding="utf-8"))

    def scores(entries):
        out = []
        for e in entries:
            ranked = idx.rank(e["q"])
            top = ranked[0] if ranked else None
            dense = idx.dense_rank(e["q"]) if idx.has_vectors else []
            top_dense = dense[0] if dense else None
            out.append((e["q"], e.get("expect_doc"), top, top_dense))
        return out

    hits = scores(spec["hit"])
    misses = scores(spec["miss"])

    for label, rows in (("answerable", hits), ("absent from the corpus", misses)):
        print(f"\n=== {label} ({len(rows)} queries) ===")
        print(f"  {'lex':>6}{'cos':>7}  {'lexical top-1':<40}  {'cosine top-1':<24}  query")
        for query, expect, top, top_dense in rows:
            src = top.source if top else "-"
            if expect:
                src += " ok" if top and top.doc_id in expect else f" (want {'|'.join(expect)})"
            dsrc, dscore = "-", 0.0
            if top_dense is not None:
                row, dscore = top_dense
                chunk = idx.chunks[row]
                dsrc = f"{chunk['doc_id']}:p{chunk['page']}"
                if expect:
                    dsrc += " ok" if chunk["doc_id"] in expect else " (miss)"
            print(f"  {(top.score if top else 0.0):>6.3f}{dscore:>7.3f}  {src:<40}  "
                  f"{dsrc:<24}  {query}")

    lex_hit = [t.score if t else 0.0 for _, _, t, _ in hits]
    lex_miss = [t.score if t else 0.0 for _, _, t, _ in misses]
    lex_floor = separation("lexical (BM25 / query ceiling)", lex_hit, lex_miss,
                           "RETRIEVAL_SCORE_FLOOR")

    dense_floor = None
    if idx.has_vectors:
        cos_hit = [d[1] if d else 0.0 for _, _, _, d in hits]
        cos_miss = [d[1] if d else 0.0 for _, _, _, d in misses]
        dense_floor = separation(f"dense (cosine, {idx.embed_arm})", cos_hit, cos_miss,
                                 "DENSE_SCORE_FLOOR")

    # Informational, not this script's number: correct-source@k over a written query set is what
    # VOX-033's gate prints, over its own queries. Here it is a sanity check that the floors are
    # being calibrated on queries that retrieve the right thing in the first place — and it is now
    # per half, because "the encoder found it and BM25 did not" is exactly what this bought.
    def right(rows, dense_side):
        n = 0
        for _q, expect, top, top_dense in rows:
            if not expect:
                continue
            if dense_side:
                n += bool(top_dense and idx.chunks[top_dense[0]]["doc_id"] in expect)
            else:
                n += bool(top and top.doc_id in expect)
        return n

    print(f"\ntop-1 in an expected document:  lexical {right(hits, False)}/{len(hits)}"
          + (f"   dense {right(hits, True)}/{len(hits)}" if idx.has_vectors else ""))

    if not idx.has_vectors:
        return 0 if lex_floor is not None else 1

    # What actually decides a refusal is the union: a chunk is kept if either half vouches for it,
    # so the state "not in the documents" survives only if *both* halves reject every absent query
    # at the chosen floors. Separability of each column on its own does not imply that, which is why
    # it is checked here rather than inferred.
    lex_at = RETRIEVAL_SCORE_FLOOR if lex_floor is None else lex_floor
    dense_at = DENSE_SCORE_FLOOR if dense_floor is None else dense_floor
    print(f"\nunion rule at lexical {lex_at:.3f} / cosine {dense_at:.3f} "
          f"(a chunk is kept if either half vouches for it):")
    kept = 0
    for label, rows, want in (("answerable", hits, True), ("absent", misses, False)):
        for query, _expect, top, top_dense in rows:
            got = bool(top and top.score > lex_at) or bool(top_dense and top_dense[1] >= dense_at)
            kept += got == want
            if got != want:
                print(f"  {'FALSE REFUSAL' if want else 'FALSE HIT    '}  "
                      f"lex {(top.score if top else 0.0):.3f} cos "
                      f"{(top_dense[1] if top_dense else 0.0):.3f}  {query}")
    total = len(hits) + len(misses)
    print(f"  {kept}/{total} queries routed correctly")
    return 0 if kept == total else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("query", nargs="*", help="the question to retrieve for")
    ap.add_argument("--k", type=int, default=RETRIEVAL_TOP_K,
                    help=f"how many chunks to return (default {RETRIEVAL_TOP_K})")
    ap.add_argument("--floor", type=float, default=RETRIEVAL_SCORE_FLOOR,
                    help=f"score floor (default {RETRIEVAL_SCORE_FLOOR})")
    ap.add_argument("--dense-floor", type=float, default=DENSE_SCORE_FLOOR,
                    help=f"cosine floor for the dense half (default {DENSE_SCORE_FLOOR})")
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

    if idx.has_vectors:
        # Load the encoder before anything is timed, for the reason every other stage does it: the
        # first forward pass otherwise carries a ~9 s model load, and the `retrieval` line below
        # would report it as query latency. src/loop.py gets this from arms.select() at startup.
        from src import arms                                              # noqa: E402
        t0 = time.perf_counter()
        arms.warm("embed")
        print(f"encoder    loaded in {(time.perf_counter() - t0) * 1000:.0f} ms "
              f"(once per process, not per query)")

    if args.calibrate:
        return calibrate(idx)
    hits = ask(idx, query, args.k, args.floor, args.dense_floor)
    if args.answer:
        return answer(query, hits, args.llm)
    return 0


if __name__ == "__main__":
    sys.exit(main())
