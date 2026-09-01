"""Retrieval over the VOX-029 chunks: lexical BM25 (VOX-030) fused with a dense encoder.

Query string in, top-k chunks out, each carrying the provenance VOX-029 wrote: `doc_id`, `page`,
`chunk_idx`, plus the BM25 `score`, the dense cosine, and the `text` itself.

**Why there are two halves.** BM25 matches terms, not meanings. A live turn on 2026-08-20 asked
"how many paternal leaves am I entitled to according to policy" and the one chunk that answers it —
`leave-policy:p12`, "ADDITIONAL LEAVES 1. Paternity Leave" — came back at rank **110 of 137**:
`paternal` is not `paternity`, `leaves` is not `leave`, so neither term matched anything. The same
chunk ranks 1 for the query "paternity leave". No floor fixes that, because the ranking itself was
wrong; only a scorer that can see past the surface form does. See `config.DENSE_SCORE_FLOOR` for the
three-row measurement that bought this.

BM25 stays. It is exact where the dense half is fuzzy — a policy that names "Form 16" or "Keka
Portal" is found by the term, and an encoder trained on general English will happily rank a
paragraph about something adjacent above it. The two are fused by **rank**, never by score
(`RRF_K`), because a normalised BM25 fraction and a cosine are different units and adding them with
weights would be inventing an exchange rate between them.

The lexical half is arithmetic — no model, no network, no key. The dense half is a model, so it goes
through `arms.embed()` and therefore through the cost logger like every other model call in this
repo; the sentence that used to be here about retrieval being the one stage with no cost line is no
longer true, and `src/embeddings.py` says so too. Retrieval also costs *time*, which is VOX-032's
`t_retrieval_ms` on the turn record — and that number now contains one encoder forward pass per
turn, which is the price of the ranking above.

Four decisions worth knowing before reading the code:

**The stopword list is load-bearing, not hygiene.** `rank_bm25`'s BM25Okapi floors the IDF of a term
appearing in more than half the corpus at `epsilon * average_idf` — a *positive* number, not zero.
So without stopwords, "what is the policy on ..." scores every chunk in the corpus on `what`, `is`,
`the` and `on`, every score lands well above zero, and there is no floor that can separate a real hit
from a question about something the documents have never heard of. Dropping function words is what
makes the "not in the documents" state reachable at all.

**A floor, so "not in the documents" is an answer and not an empty string.** `retrieve` returns `[]`
when the best chunk does not clear `config.RETRIEVAL_SCORE_FLOOR`. That number is measured, not
guessed — `scripts/ask.py --calibrate` prints the top-1 score for known-answerable and known-absent
queries and the floor is set in the gap between them. VOX-032 routes the turn on this: hits go to the
grounded answer, an empty list leaves the ordinary reply path alone.

**The score is BM25 divided by the query's own ceiling, and that is what makes the floor possible.**
Raw Okapi BM25 is a *sum* over query terms, so it grows with query length, and no single
threshold can serve both a three-word question and a ten-word one. Measured, first attempt, on raw
scores: the absent question "is there a canteen subsidy for lunch on working days" scored 9.64 on
`attendance-policy:p4` — six terms, several of them common in a document about working days — while
the answerable "am I responsible for the laptop assigned to me" scored 7.38 on three terms, all of
them right. Not separable, so no floor existed.

So `score` is the raw sum over `(k1 + 1) * Σ idf(term)`, the score a chunk containing every query
term to saturation would get: an IDF-weighted fraction of the query's information content, in [0, 1).
The division is by a constant per query, so the *ranking* is untouched — only the threshold becomes
comparable between a three-word question and a ten-word one. The raw sum stays on the hit as `raw`
so the arithmetic can be checked.

A term the corpus has never seen is scored as if it appeared in exactly one chunk — the most
informative a real term here can be — so it counts in full in the denominator and not at all in the
numerator. That is deliberate and it is what makes a miss fall: "sabbatical" and "canteen" are the
whole reason those questions are unanswerable, and treating them as zero-information (which is what
BM25 does with an out-of-vocabulary term) is what let the miss look like a hit.

**Ties break deterministically.** Chunks overlap by 50 tokens, so two chunks from the same page
routinely score identically on a short query. Sorting by `(-score, doc_id, chunk_idx)` rather than by
score alone means the same query returns the same five chunks on every run — which is the difference
between a gate number that can be re-run and one that drifts.
"""
import hashlib
import math
import pathlib
import re

import numpy as np
from rank_bm25 import BM25Okapi

from src.config import (BM25_B, BM25_EPSILON, BM25_K1, DENSE_SCORE_FLOOR, EMBEDDINGS_FILE,
                        FUSION_CANDIDATES, HYBRID_RETRIEVAL, RETRIEVAL_SCORE_FLOOR,
                        RETRIEVAL_TOP_K, RRF_K, resolve)
from src.sources import load_chunks

_WORD = re.compile(r"[a-z0-9]+")

# English function words. Deliberately written out rather than pulled from a corpus package: it is
# the one tuning knob in this module that changes retrieval behaviour, so it belongs where it can be
# read and argued with. Nothing domain-bearing is in here — "leave", "days", "notice", "policy" and
# every other word the HR corpus is actually about score normally, and BM25's IDF is what discounts
# the ones that turn out to be common in these particular documents.
STOPWORDS = frozenset("""
a about above after again against all also am an and any are as at
be because been before being below between both but by
can cannot could did do does doing done down during
each either else few for from further
had has have having he her here hers herself him himself his how however
i if in into is it its itself
just
me more most much must my myself
no nor not now
of off on once only or other others our ours ourselves out over own
same she should so some such
than that the their theirs them themselves then there these they this those through to too
under until up upon us
very
was we were what when where whether which while who whom whose why will with would
you your yours yourself yourselves
""".split())


class Hit:
    """One retrieved chunk: the four provenance fields VOX-029 wrote, plus its score.

    A plain class rather than a dict so `source` exists in one place — VOX-031 reads it out loud and
    VOX-032 puts it on the turn record, and "leave-policy:p4" being spelled the same way in both is
    the whole point of carrying provenance this far.

    `score` is the normalised **lexical** one, in [0, 1) — what BM25 ranked on and what
    `RETRIEVAL_SCORE_FLOOR` compares against. It stays the lexical number after the dense half
    arrived, rather than becoming a blend: it is the field VOX-030 calibrated, VOX-031 logs and
    VOX-032 puts on the turn record, and quietly redefining it would invalidate every one of those
    measurements while every name still looked right.

    `raw` is the BM25 sum it came from, carried so the division can be checked rather than trusted.
    `dense` is the cosine against the query, or None when the dense half did not run — None and 0.0
    are different facts (no encoder, versus an encoder that saw no similarity).
    `fused` is the reciprocal-rank-fusion score the hit was ordered by, and `lex_rank`/`dense_rank`
    are where each half put it — kept because "BM25 had it 110th and the encoder had it 1st" is the
    only way to read why a chunk is in the list.
    """

    __slots__ = ("doc_id", "page", "chunk_idx", "score", "text", "raw",
                 "dense", "fused", "lex_rank", "dense_rank")

    def __init__(self, doc_id, page, chunk_idx, score, text, raw=None,
                 dense=None, fused=None, lex_rank=None, dense_rank=None):
        self.doc_id = doc_id
        self.page = page
        self.chunk_idx = chunk_idx
        self.score = score
        self.text = text
        self.raw = score if raw is None else raw
        self.dense = dense
        self.fused = fused
        self.lex_rank = lex_rank
        self.dense_rank = dense_rank

    @property
    def source(self):
        """-> "leave-policy:p4" — the citation, in the form a person can check."""
        return f"{self.doc_id}:p{self.page}"

    def as_dict(self):
        """-> exactly the five fields the acceptance criterion names, JSON-serialisable."""
        return {"doc_id": self.doc_id, "page": self.page, "chunk_idx": self.chunk_idx,
                "score": self.score, "text": self.text}

    def __repr__(self):
        dense = "-" if self.dense is None else f"{self.dense:.3f}"
        return (f"Hit({self.source} #{self.chunk_idx} lex={self.score:.3f} dense={dense} "
                f"raw={self.raw:.3f})")

    def __eq__(self, other):
        return isinstance(other, Hit) and self.as_dict() == other.as_dict()


def tokenize(text):
    """-> the scoreable terms in `text`: lowercase, alphanumeric, no stopwords, no single letters.

    Single characters go with the stopwords: after the regex has split "employee's" into "employee"
    and "s", a bare "s" is punctuation debris that appears in most chunks, not a term.
    """
    return [w for w in _WORD.findall((text or "").lower())
            if len(w) > 1 and w not in STOPWORDS]


def _rrf(rank):
    """-> this ranking's contribution to a fused score: 1/(RRF_K + rank), or 0 if it never ranked.

    Rank and not score, deliberately. A normalised BM25 fraction and a cosine are different units,
    and combining them with weights would mean inventing an exchange rate between them and then
    tuning it — a knob with no measurement behind it. RRF only asks each half where it put the
    chunk. `RRF_K` is what makes rank 1 vs rank 2 a small difference and rank 1 vs rank 20 a large
    one, which is the behaviour wanted when one half is confident and the other has no idea.
    """
    return 0.0 if rank is None else 1.0 / (RRF_K + rank)


class Index:
    """A BM25 index over chunk records. Built once; queried per turn."""

    def __init__(self, chunks, k1=None, b=None, epsilon=None, vectors=None,
                 embed_arm=None):
        if not chunks:
            raise RuntimeError(
                "no chunks to index — run `make index` first (and check that sources/ has PDFs "
                "with extractable text; `make index` names every page that produced none)."
            )
        self.chunks = list(chunks)
        self.corpus = [tokenize(c.get("text")) for c in self.chunks]
        self.bm25 = BM25Okapi(
            self.corpus,
            k1=BM25_K1 if k1 is None else k1,
            b=BM25_B if b is None else b,
            epsilon=BM25_EPSILON if epsilon is None else epsilon,
        )
        # What a term appearing in exactly one chunk would score — the most informative a real term
        # in this corpus can be. Charged to any query term the corpus has never seen: see the
        # module docstring for why an unknown word has to cost the query rather than be free.
        n = self.bm25.corpus_size
        self.oov_idf = math.log(n - 1 + 0.5) - math.log(1 + 0.5) if n > 1 else 0.0

        # The dense half, or None. Rows line up with `self.chunks` by position and by nothing else,
        # so a vector file built against a different chunk file is not a stale cache to be
        # tolerated — it is an index that cites the wrong page. `vectors_for()` fingerprints the
        # chunk text to make that unrepresentable; this only checks the shape it was handed.
        if vectors is not None and len(vectors) != len(self.chunks):
            raise RuntimeError(
                f"{len(vectors)} vectors for {len(self.chunks)} chunks — the vector cache was "
                f"built against a different chunk file. Run `make index` to rebuild both."
            )
        self.vectors = vectors
        self.embed_arm = embed_arm        # which encoder wrote them, for the banner and the log

    @property
    def has_vectors(self):
        """-> is there a dense half? False means BM25 alone, which is VOX-030 exactly."""
        return self.vectors is not None and len(self.vectors) > 0

    def __len__(self):
        return len(self.chunks)

    @property
    def doc_ids(self):
        """-> the distinct documents in the index, in the order build_index wrote them."""
        return list(dict.fromkeys(c["doc_id"] for c in self.chunks))

    def ceiling(self, terms):
        """-> the raw BM25 score a chunk holding every one of `terms` to saturation would get.

        BM25's per-term contribution is `idf * tf(k1+1) / (tf + k1 * ...)`, and that fraction tends
        to `k1 + 1` as tf grows. So this is the sum the score is a fraction *of* — the query's total
        information content, out-of-vocabulary terms charged at `oov_idf`.
        """
        total = sum(self.bm25.idf.get(t, self.oov_idf) for t in terms)
        return (self.bm25.k1 + 1) * total

    def _lexical(self, query):
        """-> ([Hit] best first with `lex_rank` set, {row index: the same Hit}).

        The row map is what fusion needs: `rank()` drops every chunk that shares no term with the
        query, so a hit's position in that list says nothing about which chunk it is, and the dense
        half indexes by row.
        """
        terms = tokenize(query)
        if not terms:
            return [], {}
        ceiling = self.ceiling(terms) or 1.0
        scores = self.bm25.get_scores(terms)
        by_row = {}
        for row, (c, sc) in enumerate(zip(self.chunks, scores)):
            if sc > 0:
                by_row[row] = Hit(c["doc_id"], c["page"], c["chunk_idx"], float(sc) / ceiling,
                                  c["text"], float(sc))
        hits = sorted(by_row.values(), key=lambda h: (-h.score, h.doc_id, h.chunk_idx))
        for position, h in enumerate(hits, start=1):
            h.lex_rank = position
        return hits, by_row

    def rank(self, query):
        """-> every chunk sharing a term with `query`, best first. No floor, no k, no dense half.

        Separate from `search` so the floor can be *inspected* rather than only applied: a miss has
        to be able to print the best score it did see, or "nothing found" is indistinguishable from
        an empty index. `scripts/ask.py --calibrate` scores against this, and it stays purely
        lexical so the number it calibrates is the number `RETRIEVAL_SCORE_FLOOR` means.
        """
        return self._lexical(query)[0]

    def dense_rank(self, query, turn_id=None, model_id=None):
        """-> [(row index, cosine)] best first over every chunk. `[]` if there is no dense half.

        One encoder forward pass, through `arms.embed()` so it is logged like every other model
        call in this repo. The cosine is a dot product because both sides are unit vectors — see
        src/embeddings.py on why normalising happens once, there.
        """
        if not self.has_vectors or not (query or "").strip():
            return []
        from src import arms                 # local: arms imports every stage module, this is one
        q = arms.embed(query, model_id or self.embed_arm, turn_id=turn_id, is_query=True,
                       corpus_chunks=len(self.chunks))
        sims = self.vectors @ np.asarray(q, dtype="float32").reshape(-1)
        order = np.argsort(-sims, kind="stable")
        return [(int(row), float(sims[row])) for row in order]

    def search(self, query, k=None, floor=None, dense_floor=None, turn_id=None, hybrid=None):
        """-> the top `k` chunks, best first. `[]` means not in the documents.

        A chunk is a candidate if **either** half vouches for it: lexical score above `floor`, or
        cosine at least `dense_floor`. Union and not intersection — the two halves fail on different
        questions, and requiring both would keep only the questions BM25 could already answer, which
        is the behaviour this was built to replace.

        Candidates are ordered by reciprocal rank fusion, so what decides the top five is where each
        half *placed* a chunk rather than a blend of two incomparable numbers. Each half offers at
        most `FUSION_CANDIDATES` rows — wider than `k` on purpose, so a chunk the encoder ranks 8th
        and BM25 ranks 3rd can still win.

        **A half with no evidence abstains from the ordering.** If BM25's own best chunk does not
        clear `floor`, it has not found this question and its ranking is noise; fusing noise with a
        confident ranking is how the answer gets buried under chunks both halves are lukewarm
        about. See the comment on `lex_confident` for the measurement that forced this.

        `[]` needs both halves to miss, which makes the refusal a stronger claim than it was: a
        question the documents do not cover now has to fail twice.

        `hybrid=False` (or `VOX_HYBRID_RETRIEVAL=0`) is BM25 alone: VOX-030 behaviour, kept reachable
        because it is the baseline every hybrid number here is measured against.
        """
        k = RETRIEVAL_TOP_K if k is None else k
        floor = RETRIEVAL_SCORE_FLOOR if floor is None else floor
        dense_floor = DENSE_SCORE_FLOOR if dense_floor is None else dense_floor
        hybrid = HYBRID_RETRIEVAL if hybrid is None else hybrid

        lex, by_row = self._lexical(query)
        dense = self.dense_rank(query, turn_id=turn_id) if hybrid else []

        # Attach the cosine to every lexical hit that has one, and mint a Hit for the rows only the
        # dense half found. `dense` covers every chunk, so this is where the two views meet.
        candidates = {row: hit for row, hit in by_row.items() if hit.lex_rank <= FUSION_CANDIDATES}
        for position, (row, cos) in enumerate(dense, start=1):
            hit = by_row.get(row)
            if hit is None:
                c = self.chunks[row]
                hit = Hit(c["doc_id"], c["page"], c["chunk_idx"], 0.0, c["text"], 0.0)
            hit.dense = cos
            hit.dense_rank = position
            if position <= FUSION_CANDIDATES:
                candidates[row] = hit

        keep = [h for h in candidates.values()
                if h.score > floor or (h.dense is not None and h.dense >= dense_floor)]

        # When BM25 found nothing at all for this query, its *ordering* is noise too, and fusing
        # noise with a confident ranking loses the answer. Measured, 2026-08-20: for "how many
        # paternal leaves am I entitled to according to policy" the chunk that answers it —
        # leave-policy:p12, "5 calendar days leave in one go" — is dense rank **1** and lexical rank
        # **110**, and equal-weight RRF still kept it out of the top five, because two mediocre ranks
        # (lex 5 + dense 3, on chunks that answer nothing) outscore one excellent rank plus one
        # terrible one. So the half that has no evidence abstains rather than voting.
        #
        # "No evidence" is `RETRIEVAL_SCORE_FLOOR` applied to the query instead of to the chunk, and
        # it is the same measured number for the same reason: below it, BM25 cannot tell an
        # answerable question from an absent one. That is what makes this a threshold and not a
        # patch.
        #
        # The dense half does not get the same courtesy, and that asymmetry is measured rather than
        # assumed: on the 13 dev queries the lexical score separates answerable from absent
        # (0.234 | 0.320) and no dense signal does — not the raw cosine, not its z-score against the
        # corpus, not its margin over the mean, not the gap to the 6th best. An encoder that cannot
        # tell when it is lost cannot be asked to abstain.
        lex_confident = bool(lex and lex[0].score > floor)
        for h in keep:
            h.fused = (_rrf(h.lex_rank) if lex_confident else 0.0) + _rrf(h.dense_rank)
        # Ties broken by doc_id then chunk_idx, for the reason the module docstring gives: the same
        # query has to return the same chunks on every run or no gate number can be re-run.
        keep.sort(key=lambda h: (-h.fused, -h.score, h.doc_id, h.chunk_idx))
        return keep[:k]


# --- the chunk vectors ------------------------------------------------------------------------
# Encoding 215 chunks is seconds, not milliseconds, so unlike the BM25 index it is cached to disk.
# What makes that cache safe is the fingerprint: rows are matched to chunks by *position*, so a
# vector file that is one re-index out of date does not degrade retrieval, it cites the wrong page.


def fingerprint(chunks):
    """-> a hash of exactly what was encoded: the chunk texts, in order, and nothing else.

    Not the chunk file's bytes. `make index` rewrites that file every run, and a rebuild that
    produced identical text should not force a re-encode; a reordering or a re-chunk must. Hashing
    the thing the vectors are *of* is the only version of this that cannot be wrong.
    """
    h = hashlib.sha1()
    for c in chunks:
        h.update((c.get("text") or "").encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def load_vectors(chunks, arm, path=None):
    """-> the cached (n, dim) array for `chunks` under `arm`, or None if there is no usable cache.

    None covers every way a cache can fail to apply — missing, written by another encoder, built
    over different text, unreadable. All four are the same decision (re-encode) and none of them is
    an error, so this returns rather than raises, and the caller says out loud what it is doing.
    """
    path = pathlib.Path(path or EMBEDDINGS_FILE)
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as z:
            if str(z["arm"]) != arm.id or str(z["fingerprint"]) != fingerprint(chunks):
                return None
            return z["vectors"].astype("float32")
    except Exception:
        return None                     # a corrupt cache is a cache to rebuild, not a crash


def save_vectors(vectors, chunks, arm, path=None):
    """Write the cache with the two facts that decide whether it may be reused. -> the path."""
    path = pathlib.Path(path or EMBEDDINGS_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, vectors=np.asarray(vectors, dtype="float32"), arm=arm.id,
             fingerprint=fingerprint(chunks), dim=int(np.shape(vectors)[1]) if len(vectors) else 0)
    return path


def vectors_for(chunks, arm=None, path=None, rebuild=False, turn_id=None, echo=None):
    """-> ((n, dim) float32 unit vectors, the Arm that made them), encoding only if it has to.

    `echo` is called with a sentence when an encode actually happens, because it is the one part of
    starting up that takes real time — and a silent multi-second pause reads as a hang. Nothing is
    printed when the cache hits.
    """
    say = echo if echo is not None else (lambda *a, **k: None)
    arm = arm or resolve("embed")
    if not rebuild:
        cached = load_vectors(chunks, arm, path)
        if cached is not None:
            return cached, arm

    from src import arms                     # local, for the reason dense_rank() gives
    say(f"encoding {len(chunks)} chunks with {arm.repo_id} (once — cached to "
        f"{pathlib.Path(path or EMBEDDINGS_FILE).name})…")
    vectors = arms.embed([c.get("text") or "" for c in chunks], arm.id,
                         turn_id=turn_id or "index", corpus_chunks=len(chunks))
    save_vectors(vectors, chunks, arm, path)
    return vectors, arm


_INDEX = None


def build(chunks=None, path=None, dense=None, echo=None, rebuild_vectors=False):
    """-> a fresh Index over `chunks`, or over the chunk file `path` (default config.CHUNKS_FILE).

    `dense` defaults to config.HYBRID_RETRIEVAL. When it is on and the encoder cannot be reached —
    no weights on this machine, no network on a first run — the failure is reported and the index is
    built lexical-only rather than refused. That is the one substitution this module makes, and it is
    safe in a way an encoder swap is not: BM25 alone is a *worse* ranking, not a meaningless one.
    """
    say = echo if echo is not None else (lambda *a, **k: None)
    records = load_chunks(path) if chunks is None else chunks
    if dense is None:
        dense = HYBRID_RETRIEVAL
    if not dense:
        return Index(records)
    try:
        vectors, arm = vectors_for(records, echo=echo, rebuild=rebuild_vectors)
    except Exception as e:
        say(f"  warning: the dense half is off — {type(e).__name__}: {e}\n"
            f"  BM25 alone will answer, so a question phrased unlike the documents may miss. "
            f"`make setup` fetches the encoder.")
        return Index(records)
    return Index(records, vectors=vectors, embed_arm=arm.id)


def index(rebuild=False, echo=None, dense=None):
    """-> the process-wide Index, built on first use.

    Building BM25 over 215 chunks is milliseconds; encoding them is seconds on a cold cache. Both
    are per-process work and not per-turn work, which is why VOX-032 calls this at startup — so the
    first spoken question is not slower than the second.
    """
    global _INDEX
    if _INDEX is None or rebuild:
        _INDEX = build(echo=echo, dense=dense)
    return _INDEX


def fuse(hit_lists, k=None):
    """-> one ranked list of Hits fused from several rankings by reciprocal rank. Best first.

    The same mechanism `Index.search` uses to combine its lexical and dense halves, lifted so a
    caller can combine whole *queries* the same way (VOX-034 fuses a follow-up with its rewritten
    form). Rank and not score, for the reason `_rrf` gives: two rankings of the same corpus produced
    by different queries have incomparable scores, and weighting them would mean inventing an
    exchange rate and then tuning it.

    A chunk is identified by `(doc_id, chunk_idx)`, which is what makes a chunk the same chunk across
    two lists. The Hit kept is the one from the list that ranked it best, so its `score`, `dense` and
    `lex_rank` still describe a real ranking rather than an average of two.

    Ties break on `(doc_id, chunk_idx)` for the reason the module docstring gives: the same input has
    to produce the same output or no gate number can be re-run.
    """
    best, fused = {}, {}
    for hits in hit_lists:
        for rank, h in enumerate(hits or (), start=1):
            key = (h.doc_id, h.chunk_idx)
            fused[key] = fused.get(key, 0.0) + _rrf(rank)
            if key not in best or rank < best[key][0]:
                best[key] = (rank, h)
    out = [best[key][1] for key in fused]
    out.sort(key=lambda h: (-fused[(h.doc_id, h.chunk_idx)], h.doc_id, h.chunk_idx))
    return out if k is None else out[:k]


def retrieve(query, k=None, floor=None, idx=None, dense_floor=None, turn_id=None, hybrid=None):
    """-> the top `k` chunks for `query` with provenance and scores, best first; `[]` for a miss.

    The function the acceptance criterion names. `idx` is for callers that hold their own index —
    tests, and the calibration script; left alone it uses the process-wide one. `turn_id` joins the
    query's encoder call to the turn it was made for in runs/calls.jsonl.
    """
    return (idx or index()).search(query, k=k, floor=floor, dense_floor=dense_floor,
                                   turn_id=turn_id, hybrid=hybrid)
