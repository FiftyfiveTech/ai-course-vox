"""VOX-032: the turn loop answers from the documents when they cover the question.

VOX-031 proved the grounded answer works when a script hands it chunks. What this file tests is the
*routing* — the decision a live turn makes between two paths — and the three ways that decision can
be got wrong without anything looking broken:

  the route      chunks above the floor go through the answer prompt, an empty list goes through the
                 plain reply prompt. Asserted on the system message the model actually received,
                 because both paths return a fluent sentence and the sentence cannot tell you which
                 prompt produced it.
  the accounting retrieval is timed and it is *not* one of the five VOX-003 fields. A retrieval
                 folded into t_llm_ms would blame the model for milliseconds it never spent, and a
                 sixth entry in STAGES would redefine the split the phase gates read.
  the record     `grounded` on every turn that got as far as a reply, not only on the grounded ones.
                 VOX-033 sums it into a rate, and a rate needs the denominator recorded while the
                 turns were happening.

No key, no network, no mic, no speaker. The LLM is faked at the BACKENDS seam that test_answer.py
uses, so everything from `turn_reply` down to `arms.llm` — resolve, the cost logger, the free-tier
check — still runs for real.
"""
import time
import types

import pytest

from src import answer as answer_mod, harness, loop, nlu, retrieval
from src.config import LLM_ARMS, STT_ARMS, TTS_ARMS
from src.retrieval import Index
from src.telemetry import STAGES, TURN_FIELDS, TurnTimer

from conftest import fake_state             # same directory; pytest puts tests/unit on sys.path
from test_answer import fake_llm
from test_retrieval import CHUNKS

DEFAULTS = {"stt": STT_ARMS[0], "llm": LLM_ARMS[0], "tts": TTS_ARMS[0]}

# Hand-written chunks, so a query is answerable or absent by construction rather than by a score
# that would move the next time the real corpus is re-indexed. The floor is passed explicitly for
# the same reason test_answer.py does it — see test_retrieval.py on why these scores mean nothing.
ANSWERABLE = "casual privilege leave"
ABSENT = "sabbatical policy for research"


@pytest.fixture
def idx():
    return Index(CHUNKS)


class FakeCapture:
    """What vad.listen hands back, reduced to the marks a TurnTimer reads off it."""

    speech_end_t = 100.0
    endpointed_t = 101.1
    t_vad_ms = 1100.0
    spoken_s = 1.2
    infer_ms = 9.0

    def __init__(self):
        self.segment = [0.0] * 16_000

    def __len__(self):
        return len(self.segment)


def timed_turn():
    """A TurnTimer with the marks a turn record needs, so `write()` produces a real line."""
    t = TurnTimer("t-032")
    t.vad(FakeCapture())
    return t


def system_message(seen):
    return seen[0]["msgs"][0]["content"]


# --- the route --------------------------------------------------------------------------------

def test_a_covered_question_is_answered_from_the_documents(monkeypatch, idx):
    seen = fake_llm(monkeypatch, reply="The leave policy allows twelve casual leaves.")

    reply = answer_mod.turn_reply(ANSWERABLE, "t", idx=idx, floor=0.0)

    assert reply.answer is not None and reply.answer.grounded
    assert reply.text == "The leave policy allows twelve casual leaves."
    assert reply.hits, "the chunks it was grounded in come back with it"
    assert system_message(seen) == answer_mod.system_prompt()


def test_a_question_the_documents_do_not_cover_falls_back_to_the_plain_reply(monkeypatch, idx):
    """The pre-RAG path, unchanged. Not a refusal: VOX is still a voice agent for everything else."""
    seen = fake_llm(monkeypatch, reply="I can help with that.")

    reply = answer_mod.turn_reply(ABSENT, "t", idx=idx)

    assert reply.answer is None and reply.hits == []
    assert reply.text == "I can help with that."
    assert system_message(seen) == nlu.system_prompt()
    assert system_message(seen) != answer_mod.system_prompt()


def test_the_route_is_the_measured_floor_and_nothing_else(monkeypatch, idx):
    """The same query goes both ways on the floor alone — that is what makes it the router.

    If some second signal (question words, length, an intent guess) had crept in, this would fail:
    only `floor` differs between the two calls.
    """
    fake_llm(monkeypatch, reply="Twelve days.")

    over = answer_mod.turn_reply(ANSWERABLE, "t", idx=idx, floor=0.0)
    under = answer_mod.turn_reply(ANSWERABLE, "t", idx=idx, floor=0.99)

    assert over.answer is not None and under.answer is None


def test_a_model_refusal_is_spoken_and_is_not_counted_as_grounded(monkeypatch, idx):
    """The case the floor cannot catch: the right document, without the answer in it.

    The listener hears the refusal — that *is* the turn's reply, and TTS speaks it — but the turn
    record must not claim the answer was grounded, or VOX-033's rate counts a non-answer.
    """
    fake_llm(monkeypatch, reply=answer_mod.REFUSAL)
    turn = timed_turn()

    reply = answer_mod.turn_reply(ANSWERABLE, "t", idx=idx, turn=turn, floor=0.0)

    assert reply.text == answer_mod.REFUSAL
    assert reply.answer is not None and reply.answer.grounded is False
    assert turn.record()["grounded"] is False
    assert turn.record()["sources"] == [], "a refusal cites nothing"
    assert turn.record()["retrieved"] == len(reply.hits) > 0, "it did retrieve; it did not answer"


def test_no_knowledge_base_at_all_never_retrieves(monkeypatch):
    """A clean clone: sources/ is gitignored, so there is no corpus and no chunk file.

    `retrieve` is pointed at a raise, so a lookup fails the test instead of quietly succeeding
    against whatever index happened to be lying around in the process.
    """
    fake_llm(monkeypatch, reply="I can help with that.")
    monkeypatch.setattr(retrieval, "retrieve",
                        lambda *a, **kw: pytest.fail("no index means no retrieval"))

    reply = answer_mod.turn_reply(ANSWERABLE, "t", idx=None)

    assert reply.answer is None and reply.hits == []
    assert reply.text == "I can help with that."


def test_a_missing_chunk_file_is_reported_once_and_is_not_fatal():
    """`make demo` has to run for someone who has not put the corpus on their machine — and has to
    say why nothing is grounded, because silence here looks exactly like a run of uncovered
    questions. conftest points CHUNKS_FILE at a path that does not exist."""
    said = []

    assert answer_mod.knowledge_base(echo=said.append) is None
    assert any("no knowledge base" in line for line in said)
    assert any("make index" in line for line in said), "it names the fix"


# --- the accounting ---------------------------------------------------------------------------

def test_retrieval_is_not_a_sixth_stage():
    """The guard on the VOX-003 contract. TURN_FIELDS, stage_sum_ms and `ok` all derive from
    STAGES, and the phase gates read exactly those — so retrieval is timed beside them, not in
    them. Adding it here would also make `ok` false for a turn that ran without an index."""
    assert "retrieval" not in STAGES
    assert len(TURN_FIELDS) == 5


def test_retrieval_time_is_on_the_record_and_is_not_charged_to_the_model(monkeypatch, idx):
    """The failure this catches: retrieval timed inside `stage("llm")`.

    BM25 is milliseconds against an LLM call of seconds, so on the real corpus the mistake is
    invisible. Here retrieval is made deliberately slow and the model deliberately fast, which is
    the same arithmetic with the sizes swapped.
    """
    fake_llm(monkeypatch, reply="Twelve days.")
    slow = types.SimpleNamespace(search=lambda q, **kw:
                                 (time.sleep(0.12), idx.search(q, k=kw.get("k"), floor=0.0))[1])
    turn = timed_turn()

    answer_mod.turn_reply(ANSWERABLE, "t", idx=slow, turn=turn)
    rec = turn.record()

    assert rec["t_retrieval_ms"] >= 100
    assert rec["t_llm_ms"] < 100, "the retrieval wait landed in the llm stage"
    assert rec["stage_sum_ms"] == pytest.approx(
        sum(rec[f"t_{s}_ms"] for s in STAGES if rec[f"t_{s}_ms"] is not None), abs=0.2)


def test_a_run_with_no_index_leaves_the_retrieval_field_off_the_record(monkeypatch):
    """"Nothing was retrieved" and "nothing was retrievable" are different facts about a turn, and
    a null would read as the first. Absent means retrieval never ran."""
    fake_llm(monkeypatch, reply="I can help with that.")
    turn = timed_turn()

    answer_mod.turn_reply(ANSWERABLE, "t", idx=None, turn=turn)

    assert "t_retrieval_ms" not in turn.record()
    assert turn.record()["grounded"] is False, "the denominator is still recorded"


def test_the_turn_record_says_what_the_answer_was_grounded_in(monkeypatch, idx, turns_log):
    """Written to disk, not just returned: VOX-033 reads runs/turns.jsonl, not this process."""
    import json

    fake_llm(monkeypatch, reply="The leave policy allows twelve casual leaves.")
    turn = timed_turn()

    reply = answer_mod.turn_reply(ANSWERABLE, "t", idx=idx, turn=turn, floor=0.0)
    turn.write()

    rec = json.loads(turns_log.read_text(encoding="utf-8").strip())
    assert rec["grounded"] is True
    assert rec["sources"] == reply.answer.labels
    assert rec["retrieved"] == len(reply.hits)
    assert rec["top_score"] == pytest.approx(reply.hits[0].score, abs=1e-4)


def test_sources_are_spelled_the_way_retrieval_spells_them(monkeypatch, idx):
    """`leave-policy:p4` in one place only — Hit.source. A second spelling is a second format, and
    the turn line and the printed line would drift apart."""
    fake_llm(monkeypatch, reply="Twelve days.")
    turn = timed_turn()

    reply = answer_mod.turn_reply(ANSWERABLE, "t", idx=idx, turn=turn, floor=0.0)

    assert turn.record()["sources"][0] == reply.hits[0].source


# --- inside the live turn ---------------------------------------------------------------------

def spoken_turn(monkeypatch, transcript=ANSWERABLE):
    """One live turn with a fake mic, a fake speaker and real everything else. -> what TTS was given.

    Only the devices and STT are faked. `loop.one_turn` runs its own sequence, `turn_reply` routes
    for real and `arms.llm` is reached through the BACKENDS seam — because "the loop is wired to the
    grounded path" is the claim, and a monkeypatched turn_reply would assert nothing about it.
    """
    said = []
    monkeypatch.setattr(loop.vad, "listen", lambda *a, **kw: FakeCapture())
    monkeypatch.setattr(loop.arms, "stt", lambda *a, **kw: transcript)
    monkeypatch.setattr(loop.arms, "tts", lambda text, *a, **kw: said.append(text) or
                        types.SimpleNamespace(audio=[0.0] * 240, sample_rate=24_000))
    monkeypatch.setattr(loop.audio, "play", lambda samples, **kw: kw["on_first_audio"]())
    return said


def test_one_turn_speaks_the_grounded_answer_and_logs_its_sources(monkeypatch, idx, turns_log,
                                                                  capsys):
    """The acceptance criterion, at the level the demo runs it: a covered question spoken into the
    loop comes back out of the speaker as an answer from the documents, with provenance."""
    import json

    fake_llm(monkeypatch, reply="The leave policy allows twelve casual leaves.")
    said = spoken_turn(monkeypatch)
    monkeypatch.setattr(answer_mod, "turn_reply", _floorless(answer_mod.turn_reply))

    result = loop.one_turn(DEFAULTS, idx=idx)

    assert result.spoken is True
    assert said == ["The leave policy allows twelve casual leaves."], "TTS spoke the grounded text"
    rec = json.loads(turns_log.read_text(encoding="utf-8").strip())
    assert rec["grounded"] is True and rec["sources"]
    assert rec["t_retrieval_ms"] is not None
    assert "grounded in" in capsys.readouterr().out, "the demo says so on the terminal"


def test_one_turn_without_a_knowledge_base_behaves_as_it_did_before_this_ticket(monkeypatch,
                                                                                turns_log):
    """`--no-kb`, and every clean clone. The turn still runs; nothing claims to be grounded.

    What that path calls changed in the VOX-019/020 merge — it is the structured extractor now, not
    `nlu.reply` — but what this test asserts did not: the turn speaks, and the record says plainly
    that nothing was retrieved and nothing was grounded.
    """
    import json

    monkeypatch.setattr(loop.state, "build", lambda *a, **kw: fake_state("I can help with that."))
    said = spoken_turn(monkeypatch)

    result = loop.one_turn(DEFAULTS, idx=None)

    assert result.spoken is True and said == ["I can help with that."]
    rec = json.loads(turns_log.read_text(encoding="utf-8").strip())
    assert rec["grounded"] is False and rec["sources"] == []
    assert "t_retrieval_ms" not in rec


def test_the_five_field_split_survives_a_grounded_turn(monkeypatch, idx, turns_log):
    """A grounded turn is still a turn the phase gates can read: all five fields, ok true."""
    import json

    fake_llm(monkeypatch, reply="Twelve casual leaves.")
    spoken_turn(monkeypatch)
    monkeypatch.setattr(answer_mod, "turn_reply", _floorless(answer_mod.turn_reply))

    loop.one_turn(DEFAULTS, idx=idx)

    rec = json.loads(turns_log.read_text(encoding="utf-8").strip())
    assert all(rec[f] is not None for f in TURN_FIELDS)
    assert rec["ok"] is True


def test_the_fixture_harness_runs_the_same_routing(monkeypatch, idx):
    """`scripts/turn_from_fixture.py --kb` and the live loop share one implementation — the copy
    this repo has already been bitten by. Same fake LLM, same grounded answer, no mic."""
    fake_llm(monkeypatch, reply="The leave policy allows twelve casual leaves.")
    monkeypatch.setattr(harness.arms, "stt", lambda *a, **kw: ANSWERABLE)
    monkeypatch.setattr(harness.arms, "tts", lambda *a, **kw: types.SimpleNamespace(
        audio=[0.0] * 240, sample_rate=24_000))
    monkeypatch.setattr(answer_mod, "turn_reply", _floorless(answer_mod.turn_reply))
    run = harness.fixture_turn(DEFAULTS, [], "fake.mp3", play=False, segment=FakeCapture(),
                               idx=idx)

    assert run.answer is not None and run.answer.grounded
    assert run.reply == "The leave policy allows twelve casual leaves."
    assert run.record["sources"] == run.answer.labels


def test_the_fixture_harness_stays_on_the_plain_path_by_default(monkeypatch):
    """`scripts/compare_arms.py` times arms, and a grounded turn carries ~1500 more tokens into the
    llm call. Defaulting `idx` to the process-wide index would put the KB in the arm's column."""
    seen = fake_llm(monkeypatch, reply="I can help with that.")
    monkeypatch.setattr(harness.arms, "stt", lambda *a, **kw: ANSWERABLE)
    monkeypatch.setattr(harness.arms, "tts", lambda *a, **kw: types.SimpleNamespace(
        audio=[0.0] * 240, sample_rate=24_000))
    run = harness.fixture_turn(DEFAULTS, [], "fake.mp3", play=False, segment=FakeCapture())

    assert run.answer is None
    assert system_message(seen) == nlu.system_prompt()
    assert "t_retrieval_ms" not in run.record


def _floorless(fn):
    """`turn_reply` with floor=0.0 forced.

    The scores over six hand-written chunks are meaningless (test_retrieval.py says why), so the
    real floor would reject all of them and every loop test above would silently assert the plain
    path. The loop does not take a `floor` — it is config — so the seam has to be here.
    """
    def wrapped(*a, **kw):
        kw.setdefault("floor", 0.0)
        return fn(*a, **kw)
    return wrapped
