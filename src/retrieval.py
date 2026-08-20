"""Lexical retrieval over the VOX-029 chunks (VOX-030).

Query string in, top-k chunks out, each carrying the provenance VOX-029 wrote: `doc_id`, `page`,
`chunk_idx`, plus the BM25 `score` and the `text` itself. No model, no network, no key — Okapi BM25
is arithmetic over term frequencies, so this is the one stage of the pipeline that does not touch
`src/telemetry.py`'s cost logger. It must not: `log_call` checks the provider against `FREE_TIERS`
and stamps a `cost_usd`, and there is no provider here to name. Retrieval still costs *time*, and
that is VOX-032's business — it adds `t_retrieval_ms` to the turn record.

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
import math
import re

from rank_bm25 import BM25Okapi

from src.config import BM25_B, BM25_EPSILON, BM25_K1, RETRIEVAL_SCORE_FLOOR, RETRIEVAL_TOP_K
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

    `score` is the normalised one, in [0, 1) — it is what ranks and what the floor compares against.
    `raw` is the BM25 sum it came from, carried so the division can be checked rather than trusted.
    """

    __slots__ = ("doc_id", "page", "chunk_idx", "score", "text", "raw")

    def __init__(self, doc_id, page, chunk_idx, score, text, raw=None):
        self.doc_id = doc_id
        self.page = page
        self.chunk_idx = chunk_idx
        self.score = score
        self.text = text
        self.raw = score if raw is None else raw

    @property
    def source(self):
        """-> "leave-policy:p4" — the citation, in the form a person can check."""
        return f"{self.doc_id}:p{self.page}"

    def as_dict(self):
        """-> exactly the five fields the acceptance criterion names, JSON-serialisable."""
        return {"doc_id": self.doc_id, "page": self.page, "chunk_idx": self.chunk_idx,
                "score": self.score, "text": self.text}

    def __repr__(self):
        return f"Hit({self.source} #{self.chunk_idx} score={self.score:.3f} raw={self.raw:.3f})"

    def __eq__(self, other):
        return isinstance(other, Hit) and self.as_dict() == other.as_dict()


def tokenize(text):
    """-> the scoreable terms in `text`: lowercase, alphanumeric, no stopwords, no single letters.

    Single characters go with the stopwords: after the regex has split "employee's" into "employee"
    and "s", a bare "s" is punctuation debris that appears in most chunks, not a term.
    """
    return [w for w in _WORD.findall((text or "").lower())
            if len(w) > 1 and w not in STOPWORDS]


class Index:
    """A BM25 index over chunk records. Built once; queried per turn."""

    def __init__(self, chunks, k1=None, b=None, epsilon=None):
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

    def rank(self, query):
        """-> every chunk that shares a term with `query`, best first. No floor, no k.

        Separate from `search` so the floor can be *inspected* rather than only applied: a miss has
        to be able to print the best score it did see, or "nothing found" is indistinguishable from
        an empty index. `scripts/ask.py --calibrate` scores against this.
        """
        terms = tokenize(query)
        if not terms:
            return []
        ceiling = self.ceiling(terms) or 1.0
        scores = self.bm25.get_scores(terms)
        hits = [Hit(c["doc_id"], c["page"], c["chunk_idx"], float(s) / ceiling, c["text"], float(s))
                for c, s in zip(self.chunks, scores) if s > 0]
        hits.sort(key=lambda h: (-h.score, h.doc_id, h.chunk_idx))
        return hits

    def search(self, query, k=None, floor=None):
        """-> the top `k` chunks scoring above `floor`, best first. `[]` means not in the documents.

        `k` defaults to config.RETRIEVAL_TOP_K (5, the acceptance criterion's number) and `floor` to
        config.RETRIEVAL_SCORE_FLOOR. Above, not at: a floor of 0 drops the chunks that share no term
        with the query, which is what a score of exactly 0 means.
        """
        k = RETRIEVAL_TOP_K if k is None else k
        floor = RETRIEVAL_SCORE_FLOOR if floor is None else floor
        return [h for h in self.rank(query) if h.score > floor][:k]


_INDEX = None


def build(chunks=None, path=None):
    """-> a fresh Index over `chunks`, or over the chunk file `path` (default config.CHUNKS_FILE)."""
    return Index(load_chunks(path) if chunks is None else chunks)


def index(rebuild=False):
    """-> the process-wide Index, built on first use.

    Indexing 215 chunks is milliseconds, but it is still per-process work and not per-turn work:
    VOX-032 calls this at startup so the first spoken question is not slower than the second.
    """
    global _INDEX
    if _INDEX is None or rebuild:
        _INDEX = build()
    return _INDEX


def retrieve(query, k=None, floor=None, idx=None):
    """-> the top `k` chunks for `query` with provenance and score, best first; `[]` for a miss.

    The function the acceptance criterion names. `idx` is for callers that hold their own index —
    tests, and the calibration script; left alone it uses the process-wide one.
    """
    return (idx or index()).search(query, k=k, floor=floor)
