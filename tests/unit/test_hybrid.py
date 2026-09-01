"""Hybrid retrieval: BM25 and a sentence encoder, fused by rank.

The encoder is faked at the same `BACKENDS` seam every other stage is faked at, so `arms.embed`
runs for real — resolve, the free-tier check, the cost logger — and only the forward pass is
replaced. Real weights are unreachable from here by construction: `no_real_encoder` in conftest.py
turns the dense half off for every test in the suite, and the tests below switch it back on for
themselves with vectors they wrote.

The fake is a bag-of-words vector over a fixed vocabulary, normalised. That is not an embedding
model and does not pretend to be — it gives the *shape* the fusion needs (a similarity per chunk,
comparable between chunks, high for a related query) without a download. Anything that depends on
what a real encoder actually knows is not testable here and is measured on the corpus instead:
`scripts/ask.py --calibrate`, and the numbers in config.DENSE_SCORE_FLOOR.

What is asserted here is the machinery, and it is exactly the set of things that were wrong at some
point during the build:

  the union      either half can put a chunk in the list, and both have to miss for a refusal
  the abstention BM25 stops voting on the *order* when its own floor says it found nothing — the
                 bug that kept a dense-rank-1 chunk out of the top five behind two lukewarm ones
  the cache      vectors are matched to chunks by position, so the fingerprint has to reject a
                 vector file built over different text rather than silently cite the wrong page
  the logging    a query embedding is a model call and lands in runs/calls.jsonl like the rest
"""
import json
import types

import numpy as np
import pytest
import torch

from src import arms, config, embeddings, retrieval
from src.config import EMBED_ARMS, resolve
from src.retrieval import Index

from test_retrieval import CHUNKS       # same directory; pytest puts tests/unit on sys.path

VOCAB = ("casual", "privilege", "leave", "travel", "reimbursement", "notice", "days", "policy")


def vector(text):
    """-> a unit vector over VOCAB. A stand-in for meaning, not a model."""
    v = np.array([float((text or "").lower().count(w)) for w in VOCAB], dtype="float32")
    n = float(np.linalg.norm(v))
    return v / n if n else v


def fake_encoder(monkeypatch, capture=None):
    """Answer every embed arm with `vector()`, recording what it was asked to encode."""
    seen = capture if capture is not None else []

    def backend(arm, payload, rec):
        texts = payload["texts"] if isinstance(payload, dict) else payload
        texts = [texts] if isinstance(texts, str) else texts
        seen.append({"arm": arm, "texts": list(texts),
                     "is_query": bool(isinstance(payload, dict) and payload.get("is_query"))})
        rec["fake"] = True
        return np.stack([vector(t) for t in texts])

    for arm in EMBED_ARMS:
        monkeypatch.setitem(embeddings.BACKENDS, arm.backend, backend)
    return seen


@pytest.fixture
def hybrid(monkeypatch):
    """An Index over the hand-written chunks with a dense half built by the fake encoder."""
    fake_encoder(monkeypatch)
    monkeypatch.setattr(config, "HYBRID_RETRIEVAL", True)
    monkeypatch.setattr(retrieval, "HYBRID_RETRIEVAL", True)
    vectors = np.stack([vector(c["text"]) for c in CHUNKS])
    return Index(CHUNKS, vectors=vectors, embed_arm=resolve("embed").id)


# --- the encoder is an arm like any other -------------------------------------------------------

def test_the_query_embedding_is_logged_like_every_other_model_call(monkeypatch, hybrid, calls_log):
    """The claim src/retrieval.py had to withdraw when it grew a dense half: it used to be the one
    stage with no cost line, because BM25 has no provider to name. An encoder does."""
    fake_encoder(monkeypatch)

    hybrid.dense_rank("casual leave", turn_id="t-embed")

    rec = json.loads(calls_log.read_text(encoding="utf-8").splitlines()[0])
    assert rec["stage"] == "embed" and rec["turn_id"] == "t-embed" and rec["ok"] is True
    assert rec["model_id"] == EMBED_ARMS[0].repo_id      # the HF repo id, never a provider string
    assert rec["cost_usd"] == 0.0 and rec["tier"] == "local-weights"


def test_a_query_is_encoded_as_a_query_and_a_chunk_is_not(monkeypatch):
    """BGE is asymmetric: the query side carries an instruction prefix and the document side does
    not. Getting that backwards is silent — the cosines stay in [-1, 1] and just mean less."""
    seen = fake_encoder(monkeypatch)
    idx = Index(CHUNKS, vectors=np.stack([vector(c["text"]) for c in CHUNKS]))

    idx.dense_rank("casual leave", turn_id="t")

    assert seen[-1]["is_query"] is True


@pytest.mark.parametrize("arm", EMBED_ARMS, ids=lambda a: a.alias)
def test_the_backend_prepends_the_prefix_to_a_query_and_never_to_a_chunk(monkeypatch, arm):
    """One level below the arm interface: the real `encode_transformers` is run against a stub
    tokenizer, so what is asserted is the string that actually reaches the model.

    Parametrized over both arms because the interesting half is the pair — BGE declares an
    instruction and MiniLM declares none, and a backend that ignored the registry would pass a
    single-arm test either way.
    """
    seen = []

    def tokenizer(texts, **kw):
        seen.append(list(texts))
        return {"attention_mask": torch.ones(len(texts), 1, dtype=torch.long)}

    class Model:
        def __call__(self, **batch):
            n = batch["attention_mask"].shape[0]
            return types.SimpleNamespace(last_hidden_state=torch.ones(n, 1, 4))

    monkeypatch.setitem(embeddings._LOADED, arm.repo_id, (tokenizer, Model()))
    prefix = arm.extra.get("query_prefix", "")

    embeddings.encode_transformers(arm, {"texts": ["casual leave"], "is_query": True}, {})
    embeddings.encode_transformers(arm, {"texts": ["casual leave"], "is_query": False}, {})

    assert seen[0] == [prefix + "casual leave"], "the query side carries the arm's instruction"
    assert seen[1] == ["casual leave"], "a chunk is never prefixed"


def test_the_two_registered_encoders_do_not_agree_on_how_to_be_called():
    """The reason `pooling` and `query_prefix` are registry fields and not constants in the
    backend. If a future arm row copied one from the other, the vectors would come out of a space
    the model was never trained to put anything in — and nothing downstream could tell."""
    bge, minilm = EMBED_ARMS[0], EMBED_ARMS[1]

    assert (bge.extra["pooling"], bool(bge.extra["query_prefix"])) == ("cls", True)
    assert (minilm.extra["pooling"], bool(minilm.extra["query_prefix"])) == ("mean", False)


def test_mean_pooling_ignores_padding(monkeypatch):
    """Otherwise a sentence embeds differently depending on the longest *other* text in its batch,
    which at index time is a silent, batch-order-dependent index."""
    arm = EMBED_ARMS[1]                       # the mean-pooling one

    def tokenizer(texts, **kw):
        # Two tokens each; the second is padding for the first text only.
        return {"attention_mask": torch.tensor([[1, 0], [1, 1]])}

    class Model:
        def __call__(self, **batch):
            # Token 0 is the same for both rows; the padded token is wildly different.
            return types.SimpleNamespace(last_hidden_state=torch.tensor(
                [[[1.0, 0.0], [99.0, 99.0]], [[1.0, 0.0], [1.0, 0.0]]]))

    monkeypatch.setitem(embeddings._LOADED, arm.repo_id, (tokenizer, Model()))

    out = embeddings.encode_transformers(arm, ["short", "short too"], {})

    assert np.allclose(out[0], out[1]), "the padded token leaked into the mean"


def test_the_encoder_stage_has_no_fallback_arm():
    """A second encoder answers in a different vector space from the cached chunk vectors, so its
    cosines would be arithmetic between unrelated bases. See src/arms.py: embed() defaults
    fallback=False, and config.FALLBACKS has no entry."""
    assert arms.fallback_for("embed", resolve("embed")) is None


# --- the union --------------------------------------------------------------------------------

def test_either_half_can_put_a_chunk_in_the_list(hybrid):
    """The point of the union. A chunk BM25 never scored is still retrievable."""
    hits = hybrid.search("privilege", k=5, floor=0.99, dense_floor=0.1)

    assert hits, "the lexical floor rejected everything; the dense half still vouched"
    assert all(h.dense is not None for h in hits)


def test_a_refusal_needs_both_halves_to_miss(hybrid):
    """`[]` is the "not in the documents" state, and it is a stronger claim than it was: a question
    the corpus does not cover now has to fail twice."""
    assert hybrid.search("sabbatical", k=5, floor=0.99, dense_floor=0.99) == []


def test_the_dense_half_can_be_switched_off_for_a_baseline(hybrid, calls_log):
    """VOX-030's exact behaviour, still reachable — it is what every hybrid number is measured
    against. And with it off, no encoder call is made at all."""
    hits = hybrid.search("casual leave", k=5, floor=0.0, hybrid=False)

    assert hits and all(h.dense is None for h in hits)
    assert not calls_log.exists() or calls_log.read_text(encoding="utf-8") == ""


def test_scores_from_the_two_halves_are_never_added(hybrid):
    """`score` stays the lexical number after the dense half arrived. It is what VOX-030
    calibrated, VOX-031 logs and VOX-032 puts on the turn record — a blend would invalidate all
    three while every name still looked right."""
    hits = hybrid.search("casual leave", k=5, floor=0.0, dense_floor=0.1)
    lexical_only = hybrid.rank("casual leave")
    by_source = {(h.doc_id, h.chunk_idx): h.score for h in lexical_only}

    for h in hits:
        if (h.doc_id, h.chunk_idx) in by_source:
            assert h.score == by_source[(h.doc_id, h.chunk_idx)]


def test_the_five_provenance_fields_are_unchanged(hybrid):
    """VOX-030's acceptance criterion names exactly these. The dense fields ride alongside on the
    Hit; they do not join the record a caller serialises."""
    hits = hybrid.search("casual leave", k=1, floor=0.0, dense_floor=0.1)

    assert list(hits[0].as_dict()) == ["doc_id", "page", "chunk_idx", "score", "text"]


# --- the abstention ----------------------------------------------------------------------------

def test_bm25_stops_voting_on_the_order_when_it_found_nothing(monkeypatch, hybrid):
    """The bug this exists for, in miniature.

    On the real corpus: "how many paternal leaves am I entitled to according to policy" put the
    answering chunk at dense rank 1 and lexical rank 110, and equal-weight RRF still kept it out of
    the top five — two lukewarm ranks beat one excellent one plus one terrible one. Here the same
    shape is built by hand: the dense half's favourite must lead when BM25's best is under its own
    floor, because under that floor BM25 has not found the question and its ordering is noise.
    """
    dense_favourite = 4                       # the row the fake encoder will rank first
    vectors = np.stack([vector(c["text"]) for c in CHUNKS])
    vectors[dense_favourite] = vector("privilege privilege privilege")
    idx = Index(CHUNKS, vectors=vectors, embed_arm=resolve("embed").id)
    fake_encoder(monkeypatch)

    hits = idx.search("privilege", k=5, floor=0.99, dense_floor=0.1)

    assert hits[0].chunk_idx == CHUNKS[dense_favourite]["chunk_idx"]
    assert hits[0].dense_rank == 1


def test_both_halves_vote_when_bm25_does_have_evidence(monkeypatch, hybrid):
    """The abstention is conditional, not a demotion of BM25. With its floor cleared, a chunk both
    halves like outranks one only the encoder likes."""
    fake_encoder(monkeypatch)

    hits = hybrid.search("casual leave", k=5, floor=0.0, dense_floor=0.1)

    assert hits[0].lex_rank is not None and hits[0].fused > 0
    top = hits[0]
    assert top.lex_rank <= 3 and top.dense_rank <= 3, "agreement is what won it the top slot"


def test_a_fused_score_is_recorded_so_an_ordering_can_be_read_back(hybrid):
    """`lr 110 / dr 1` is the only way to see why a chunk is in the list; the columns exist for
    the same reason `raw` does on the lexical side."""
    hits = hybrid.search("casual leave", k=3, floor=0.0, dense_floor=0.1)

    assert all(h.fused is not None for h in hits)
    assert [h.fused for h in hits] == sorted((h.fused for h in hits), reverse=True)


# --- the cache ---------------------------------------------------------------------------------

def test_vectors_are_cached_and_reused_without_a_second_encode(monkeypatch, tmp_path):
    """Encoding 215 chunks is seconds. The second startup must not pay it again."""
    seen = fake_encoder(monkeypatch)
    path = tmp_path / "vectors.npz"

    first, arm = retrieval.vectors_for(CHUNKS, path=path)
    second, _ = retrieval.vectors_for(CHUNKS, path=path)

    assert len(seen) == 1, "the second call read the cache"
    assert np.array_equal(first, second)
    assert path.is_file()


def test_a_cache_built_over_different_text_is_rejected_not_reused(monkeypatch, tmp_path):
    """The failure this prevents is not a stale ranking, it is a citation to the wrong page: rows
    are matched to chunks by position and by nothing else."""
    fake_encoder(monkeypatch)
    path = tmp_path / "vectors.npz"
    retrieval.vectors_for(CHUNKS, path=path)

    reworded = [dict(c, text=c["text"] + " (re-chunked)") for c in CHUNKS]

    assert retrieval.load_vectors(reworded, resolve("embed"), path=path) is None


def test_a_cache_written_by_another_encoder_is_rejected(monkeypatch, tmp_path):
    """Two encoders' vectors are not interchangeable, and nothing downstream could tell — a cosine
    between unrelated bases is still a number between -1 and 1."""
    fake_encoder(monkeypatch)
    path = tmp_path / "vectors.npz"
    retrieval.vectors_for(CHUNKS, arm=EMBED_ARMS[0], path=path)

    assert retrieval.load_vectors(CHUNKS, EMBED_ARMS[1], path=path) is None
    assert retrieval.load_vectors(CHUNKS, EMBED_ARMS[0], path=path) is not None


def test_a_corrupt_cache_is_rebuilt_rather_than_raised(monkeypatch, tmp_path):
    """Every way a cache can fail to apply is the same decision — re-encode — so none of them is
    an error, and a half-written file must not stop `make demo` from starting."""
    fake_encoder(monkeypatch)
    path = tmp_path / "vectors.npz"
    path.write_bytes(b"not an npz")

    assert retrieval.load_vectors(CHUNKS, resolve("embed"), path=path) is None
    vectors, _ = retrieval.vectors_for(CHUNKS, path=path)
    assert len(vectors) == len(CHUNKS)


def test_a_vector_count_that_does_not_match_the_chunks_is_refused_loudly(monkeypatch):
    """The one case that must raise rather than degrade: silently truncating would mean every hit
    after the mismatch cites a different chunk's provenance."""
    vectors = np.stack([vector(c["text"]) for c in CHUNKS])[:2]

    with pytest.raises(RuntimeError, match="vector cache"):
        Index(CHUNKS, vectors=vectors)


def test_an_index_falls_back_to_lexical_when_the_encoder_cannot_load(monkeypatch, tmp_path):
    """A clean clone with no weights and no network still has to answer. This is the one
    substitution retrieval makes, and it is safe in the way an encoder swap is not: BM25 alone is a
    worse ranking, not a meaningless one."""
    monkeypatch.setattr(config, "HYBRID_RETRIEVAL", True)
    monkeypatch.setattr(retrieval, "HYBRID_RETRIEVAL", True)

    def no_weights(arm, payload, rec):
        raise RuntimeError("no local weights and no network")

    for arm in EMBED_ARMS:
        monkeypatch.setitem(embeddings.BACKENDS, arm.backend, no_weights)
    said = []

    idx = retrieval.build(chunks=CHUNKS, echo=said.append)

    assert idx.has_vectors is False
    assert any("dense half is off" in line for line in said), "it says why, once"
    assert idx.search("casual leave", k=3, floor=0.0), "and it still answers"
