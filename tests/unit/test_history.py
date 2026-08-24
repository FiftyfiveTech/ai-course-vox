"""Conversation history and the follow-up retry (VOX-034).

Nothing here calls a model. The retry is a retrieval decision, and `answer.retrieve_with_history()`
is the one copy of it — the same function `turn_reply()` and `tests/gates/gate_followup.py` call, so
these tests pin the behaviour the gate scores rather than a parallel implementation of it.

The tests that matter most are the negative ones. `history=None` has to reproduce the pre-VOX-034
pipeline byte for byte, and a whole question must not be rewritten — a mechanism that helps
follow-ups by quietly changing every other query would improve the follow-up gate and regress
`make gate-poc`, and only one of those is being watched during development.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src import answer as answer_mod, retrieval          # noqa: E402
from src.history import History, Turn, elliptical        # noqa: E402


# --- the window -------------------------------------------------------------------------------

def test_history_is_bounded_and_keeps_the_most_recent_turns():
    h = History(maxlen=2)
    h.add("first", "a")
    h.add("second", "b")
    h.add("third", "c")
    assert [t.transcript for t in h] == ["second", "third"]
    assert h.last.transcript == "third"


def test_an_empty_history_is_falsey_so_callers_can_ask_whether_it_can_help():
    h = History()
    assert not h                      # nothing to refer back to
    h.add("something", "a reply")
    assert h


def test_a_disabled_history_is_falsey_even_when_it_holds_turns():
    """--no-history has to be indistinguishable from a fresh session at every call site."""
    h = History(enabled=False)
    h.add("something", "a reply")
    assert not h
    assert h.retrieval_query("and what about this") == "and what about this"


def test_the_rewritten_query_uses_questions_and_never_replies():
    """A reply is policy prose; its terms would drag retrieval back to what was already cited."""
    h = History()
    h.add("how many casual leaves do I get", "The leave policy says twelve days a year.")
    rq = h.retrieval_query("and privilege leaves")
    assert "how many casual leaves do I get" in rq
    assert "twelve" not in rq
    assert "policy says" not in rq


def test_the_current_question_comes_last_in_the_rewritten_query():
    h = History()
    h.add("first question", "")
    assert h.retrieval_query("second question").endswith("second question")


# --- ellipsis detection -----------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "and privilege leaves",
    "what about during a performance improvement plan",
    "how far in advance do I have to plan it",
    "when is it paid out",
    "what happens to the extra ones",
    "and if I am still on probation",
    "also can I carry them forward",
])
def test_a_fragment_is_recognised_as_elliptical(text):
    assert elliptical(text)


@pytest.mark.parametrize("text", [
    "what is the notice period when I resign",
    "how many floating holidays do I get in a year",
    "how many days of paternity leave am I entitled to",
    "is there a canteen subsidy for lunch on working days",
])
def test_a_whole_question_is_not_elliptical(text):
    """The failure that would matter: rewriting a question that needed nothing."""
    assert not elliptical(text)


def test_term_count_is_not_the_signal():
    """'what is the notice period when I resign' tokenizes to three terms and is complete.

    The obvious rule — few content words means elliptical — was tried and misfires on exactly this
    string, which is why the signal is openers and anaphors instead.
    """
    assert len(retrieval.tokenize("what is the notice period when I resign")) == 3
    assert not elliptical("what is the notice period when I resign")


def test_ellipsis_needs_something_to_refer_back_to():
    """A fragment with no antecedent is not resolvable, so the History says it is not elliptical."""
    h = History()
    assert not h.elliptical("and privilege leaves")
    h.add("how many casual leaves", "")
    assert h.elliptical("and privilege leaves")


# --- the retrieval decision -------------------------------------------------------------------

class _FakeHit:
    """Enough of retrieval.Hit for fuse() and the assertions here."""

    def __init__(self, doc_id, chunk_idx, page=1, score=0.5):
        self.doc_id, self.chunk_idx, self.page, self.score = doc_id, chunk_idx, page, score

    @property
    def source(self):
        return f"{self.doc_id}:p{self.page}"


def test_no_history_reproduces_the_pre_vox_034_path(monkeypatch):
    """One retrieval call on the raw transcript, and no rewrite recorded."""
    calls = []

    def fake_retrieve(query, **kw):
        calls.append(query)
        return [_FakeHit("leave-policy", 1)]

    monkeypatch.setattr(retrieval, "retrieve", fake_retrieve)
    hits, asked, rewrite = answer_mod.retrieve_with_history("a whole question", "t", idx=object())
    assert calls == ["a whole question"]
    assert asked == "a whole question"
    assert rewrite is None
    assert hits


def test_a_whole_question_is_never_rewritten_even_with_history(monkeypatch):
    """The property that keeps `make gate-poc` a no-regression check rather than a re-measurement."""
    calls = []
    monkeypatch.setattr(retrieval, "retrieve",
                        lambda query, **kw: calls.append(query) or [_FakeHit("leave-policy", 1)])
    h = History()
    h.add("how many casual leaves do I get in a year", "twelve")
    _, asked, rewrite = answer_mod.retrieve_with_history(
        "what is the notice period when I resign", "t", idx=object(), history=h)
    assert calls == ["what is the notice period when I resign"]
    assert rewrite is None
    assert asked == "what is the notice period when I resign"


def test_an_elliptical_follow_up_retrieves_on_both_forms_and_fuses(monkeypatch):
    """Fused, not replaced: attempt 2 replaced and lost as many cases as it rescued."""
    seen = []

    def fake_retrieve(query, **kw):
        seen.append(query)
        if "accumulate" in query:
            return [_FakeHit("leave-policy", 7, page=7)]      # what the rewrite finds
        return [_FakeHit("performance-management", 9, page=9)]  # what the fragment finds

    monkeypatch.setattr(retrieval, "retrieve", fake_retrieve)
    h = History()
    h.add("how many privilege leaves can I accumulate", "")
    hits, _, rewrite = answer_mod.retrieve_with_history(
        "what happens to the extra ones", "t", idx=object(), history=h)

    assert len(seen) == 2, "both the fragment and the resolved question must be retrieved"
    assert rewrite["trigger"] == "elliptical"
    assert rewrite["used"] is True
    got = {h_.source for h_ in hits}
    assert got == {"leave-policy:p7", "performance-management:p9"}, \
        "fusion keeps what each query found; replacing would drop one of them"


def test_a_miss_with_an_antecedent_gets_one_retry(monkeypatch):
    """The second trigger: a referential turn that ellipsis detection did not catch."""
    seen = []

    def fake_retrieve(query, **kw):
        seen.append(query)
        return [_FakeHit("leave-policy", 7)] if "accumulate" in query else []

    monkeypatch.setattr(retrieval, "retrieve", fake_retrieve)
    h = History()
    h.add("how many privilege leaves can I accumulate", "")
    hits, asked, rewrite = answer_mod.retrieve_with_history(
        "tell me more", "t", idx=object(), history=h)
    assert rewrite["trigger"] == "miss"
    assert rewrite["used"] is True
    assert hits and "accumulate" in asked


def test_a_rewrite_that_also_misses_is_recorded_as_attempted_but_unused(monkeypatch):
    """'fired and failed' and 'never fired' are different diagnoses and must stay distinguishable."""
    monkeypatch.setattr(retrieval, "retrieve", lambda query, **kw: [])
    h = History()
    h.add("how many privilege leaves can I accumulate", "")
    hits, asked, rewrite = answer_mod.retrieve_with_history(
        "tell me more", "t", idx=object(), history=h)
    assert hits == []
    assert rewrite["used"] is False
    assert asked == "tell me more", "a failed rewrite must not become the question asked"


# --- fusion -----------------------------------------------------------------------------------

def test_fuse_ranks_a_chunk_both_queries_found_above_one_only_seen_once():
    a = [_FakeHit("doc-a", 1), _FakeHit("doc-b", 2)]
    b = [_FakeHit("doc-c", 3), _FakeHit("doc-a", 1)]
    out = retrieval.fuse([a, b])
    assert out[0].source.startswith("doc-a"), "found by both rankings, so it wins"
    assert len(out) == 3, "the union, deduped by (doc_id, chunk_idx)"


def test_fuse_is_deterministic():
    a = [_FakeHit("doc-a", 1), _FakeHit("doc-b", 1)]
    b = [_FakeHit("doc-b", 1), _FakeHit("doc-a", 1)]
    first = [h.source for h in retrieval.fuse([a, b])]
    for _ in range(5):
        assert [h.source for h in retrieval.fuse([a, b])] == first


def test_fuse_tolerates_an_empty_ranking():
    out = retrieval.fuse([[], [_FakeHit("doc-a", 1)]])
    assert len(out) == 1


# --- what history records ---------------------------------------------------------------------

def test_a_turn_remembers_its_sources_and_whether_it_was_grounded():
    h = History()
    t = h.add("q", "a", sources=["leave-policy:p7"], grounded=True)
    assert isinstance(t, Turn)
    assert h.last.sources == ["leave-policy:p7"]
    assert h.last.grounded is True


def test_the_plain_path_prefix_alternates_user_and_assistant():
    h = History()
    h.add("first", "reply one")
    h.add("second", "reply two")
    assert h.messages_prefix() == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply one"},
        {"role": "user", "content": "second"},
        {"role": "assistant", "content": "reply two"},
    ]


def test_an_empty_history_contributes_no_messages():
    assert History().messages_prefix() == []
