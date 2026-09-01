"""VOX-030. What the ranker promises, on a corpus small enough to reason about by hand.

`sources/` and `runs/chunks.jsonl` are both gitignored (internal HR policies), so anything that read
the real corpus would pass here and fail on a clean clone. Every chunk below is written out, and the
absolute scores are therefore meaningless — so nothing here asserts a score *value*. What is asserted
is the order, the fields, and the two behaviours a caller depends on: k caps the result and the floor
turns a weak match into no match at all.

The floor's actual number is not testable here either: it is a measured property of the real corpus,
recorded in config.RETRIEVAL_SCORE_FLOOR with the run that produced it, and re-measured by
`scripts/ask.py --calibrate`. What these tests pin is that the *mechanism* works and that the
normalisation which made a floor possible at all is really in place.
"""
import pytest

from src import config, retrieval, sources
from src.retrieval import Index, Hit

from test_chunking import WordTokenizer   # same directory; pytest puts tests/unit on sys.path


# Six chunks over two documents. "sabbatical" appears nowhere, "policy" appears everywhere, and
# "casual" only in leave-policy — the three cases the ranker has to tell apart.
CHUNKS = [
    {"doc_id": "leave-policy", "page": 1, "chunk_idx": 0,
     "text": "Leave policy. Employees are entitled to twelve days of casual leave every year."},
    {"doc_id": "leave-policy", "page": 2, "chunk_idx": 1,
     "text": "Leave policy. Privilege leave accrues monthly and may be encashed on separation."},
    {"doc_id": "leave-policy", "page": 2, "chunk_idx": 2,
     "text": "Leave policy. Casual leave cannot be availed while serving the notice period."},
    {"doc_id": "travel-policy", "page": 1, "chunk_idx": 0,
     "text": "Travel policy. Domestic travel is reimbursed at actuals against submitted bills."},
    {"doc_id": "travel-policy", "page": 3, "chunk_idx": 1,
     "text": "Travel policy. International travel needs approval from the reporting manager."},
    {"doc_id": "travel-policy", "page": 4, "chunk_idx": 2,
     "text": "Travel policy. Hotel booking is done centrally by the administration team."},
]


@pytest.fixture
def idx():
    return Index(CHUNKS)


# --- tokenizing ---------------------------------------------------------------------------------

def test_tokenize_lowercases_splits_on_punctuation_and_keeps_digits():
    assert retrieval.tokenize("Twelve (12) days' Casual Leave.") == \
        ["twelve", "12", "days", "casual", "leave"]


def test_tokenize_drops_stopwords_and_single_letters():
    """The 's' left behind by splitting "employee's" is debris, not a term — see tokenize()."""
    assert retrieval.tokenize("what is the notice period") == ["notice", "period"]
    assert "s" not in retrieval.tokenize("an employee's laptop")


def test_tokenize_survives_nothing():
    assert retrieval.tokenize("") == []
    assert retrieval.tokenize(None) == []
    assert retrieval.tokenize("!!! ??? ...") == []


# --- the fields the criterion names -------------------------------------------------------------

def test_every_hit_carries_the_five_named_fields_and_they_match_the_source_chunk(idx):
    hits = idx.search("casual leave", floor=0)
    assert hits
    for h in hits:
        assert sorted(h.as_dict()) == ["chunk_idx", "doc_id", "page", "score", "text"]
        origin = [c for c in CHUNKS
                  if c["doc_id"] == h.doc_id and c["chunk_idx"] == h.chunk_idx]
        assert len(origin) == 1                      # (doc_id, chunk_idx) is the unique key
        assert (h.page, h.text) == (origin[0]["page"], origin[0]["text"])


def test_source_is_the_citation_a_person_can_check(idx):
    hit = idx.search("twelve days casual leave", floor=0)[0]
    assert hit.source == "leave-policy:p1"


# --- ranking ------------------------------------------------------------------------------------

def test_the_chunk_holding_the_rare_term_wins(idx):
    """"casual" is in two of six chunks, "policy" in all six. The rare term has to decide."""
    assert idx.search("casual", floor=0)[0].doc_id == "leave-policy"
    assert {h.chunk_idx for h in idx.search("casual", floor=0)} == {0, 2}


def test_a_term_in_every_chunk_ranks_by_length_and_not_by_document(idx):
    """A query of nothing but the corpus-wide term must not prefer one document over the other.

    It still returns everything — "policy" is a real term, and BM25 floors its IDF rather than
    zeroing it. What decides the order is then only `b`, the length penalty: shortest chunk first.
    That is the honest answer to "which chunk is most about `policy`" when every chunk says it once,
    and it is why a floor is needed rather than a top-1.
    """
    hits = idx.search("policy", k=99, floor=0)
    assert len(hits) == len(CHUNKS)
    by_length = sorted(hits, key=lambda h: len(retrieval.tokenize(h.text)))
    assert [h.score for h in hits] == [h.score for h in by_length]


def test_results_are_ordered_by_descending_score(idx):
    scores = [h.score for h in idx.search("casual leave notice period", k=99, floor=0)]
    assert scores == sorted(scores, reverse=True)


def test_ties_break_deterministically_on_doc_id_then_chunk_idx(idx):
    """Overlapping chunks tie constantly, so the same query has to return the same five chunks.

    A gate number that changes between runs of the same command is not a number.
    """
    hits = idx.search("policy", k=99, floor=0)
    # Three of the six chunks are the same length and say "policy" once, so they score identically.
    tied = [(h.doc_id, h.chunk_idx) for h in hits if round(h.score, 9) == round(hits[1].score, 9)]
    assert len(tied) == 3, "the fixture has a tie group to order"
    assert tied == sorted(tied), "and it is ordered by doc_id then chunk_idx"

    first = [(h.doc_id, h.chunk_idx) for h in hits]
    for _ in range(3):
        assert [(h.doc_id, h.chunk_idx) for h in idx.search("policy", k=99, floor=0)] == first


def test_k_caps_the_result_count_and_defaults_to_the_configured_five(idx):
    assert len(idx.search("policy", k=2, floor=0)) == 2
    assert config.RETRIEVAL_TOP_K == 5, "the acceptance criterion's default"
    assert len(idx.search("policy", floor=0)) == config.RETRIEVAL_TOP_K


# --- the floor: "not in the documents" as a real answer state -----------------------------------

def test_a_query_sharing_no_term_with_the_corpus_returns_nothing(idx):
    assert idx.search("sabbatical gymnasium", floor=0) == []


def test_the_floor_is_what_suppresses_a_weak_match_not_the_scorer(idx):
    """The same query, twice: the chunks exist and score, and the floor is what withholds them."""
    weak = "hotel booking sabbatical gymnasium tuition"
    assert idx.search(weak, floor=0), "there is a match to suppress"
    assert idx.search(weak, floor=0.9) == [], "and the floor suppresses it"


def test_an_unknown_term_costs_the_query_rather_than_being_free(idx):
    """The normalisation that made a floor possible — see the module docstring in src/retrieval.py.

    Raw BM25 gives an out-of-vocabulary term idf 0, so padding a query with words the corpus has
    never seen would leave the score untouched. Here it must fall, because the query asked for
    information the chunk does not have.
    """
    focused = idx.search("casual leave", floor=0)[0].score
    padded = idx.search("casual leave sabbatical gymnasium tuition", floor=0)[0].score
    assert padded < focused


def test_the_score_is_the_raw_sum_over_the_query_ceiling(idx):
    """The arithmetic, checkable: score * ceiling == raw. `raw` exists so this can be asserted."""
    hit = idx.search("casual leave", floor=0)[0]
    ceiling = idx.ceiling(retrieval.tokenize("casual leave"))
    assert hit.raw == pytest.approx(hit.score * ceiling)
    assert 0 < hit.score < 1


def test_a_query_that_is_all_stopwords_returns_nothing_rather_than_raising(idx):
    assert idx.search("what is the", floor=0) == []
    assert idx.search("", floor=0) == []
    assert idx.rank("!!!") == []


# --- building -----------------------------------------------------------------------------------

def test_an_empty_index_says_to_run_make_index():
    with pytest.raises(RuntimeError, match="make index"):
        Index([])


def test_doc_ids_are_the_distinct_documents_in_write_order(idx):
    assert idx.doc_ids == ["leave-policy", "travel-policy"]
    assert len(idx) == len(CHUNKS)


def test_retrieve_with_no_index_reads_the_configured_chunk_file(tmp_path, monkeypatch):
    """The default path, end to end: VOX-029 writes the file, VOX-030 retrieves over it.

    Uses the chunker's fake tokenizer so no HF cache is needed, and a real (stub) PDF path so
    build_index's own walk runs. The two halves of the POC are only correct if they agree on the
    field names, and this is the test that would catch a rename in either.
    """
    # Four pages, not two: with a two-chunk corpus a term in exactly one of them has an IDF of
    # exactly zero (log(N - df + 0.5) - log(df + 0.5) with N=2, df=1), so every score would be 0 and
    # the test would be asserting against a degenerate index rather than against retrieval.
    monkeypatch.setattr(sources, "extract_pages",
                        lambda path: [(1, "casual leave twelve days"), (2, "travel reimbursed"),
                                      (3, "hotel booking administration"), (4, "notice period")])
    (tmp_path / "leave-policy.pdf").write_bytes(b"%PDF-1.4 stub")
    out = tmp_path / "chunks.jsonl"
    sources.build_index(root=tmp_path, out=out, tokenizer=WordTokenizer())

    monkeypatch.setattr(sources, "CHUNKS_FILE", out)
    monkeypatch.setattr(retrieval, "_INDEX", None)

    hits = retrieval.retrieve("casual leave", floor=0)
    assert [h.source for h in hits[:1]] == ["leave-policy:p1"]
    assert retrieval.index() is retrieval.index(), "built once per process, not per query"


def test_retrieve_without_a_chunk_file_says_to_run_make_index():
    """config.CHUNKS_FILE points at a path that does not exist — see conftest.chunks_file."""
    with pytest.raises(RuntimeError, match="no chunk file"):
        retrieval.retrieve("casual leave")


def test_hit_equality_is_by_the_named_fields():
    a = Hit("leave-policy", 1, 0, 0.5, "text", raw=9.0)
    assert a == Hit("leave-policy", 1, 0, 0.5, "text", raw=9.0)
    assert a != Hit("leave-policy", 1, 0, 0.5, "other", raw=9.0)
