"""VOX-031. What the grounded answer promises, with no key and no network.

Every test here fakes the LLM at the same seam test_arms.py and test_fallback.py use — the stage
module's BACKENDS table — so what is exercised is the real `arms.llm` path: the cost logger runs,
the arm resolves, the record is written. Nothing is stubbed above that line, because "the call went
through arms.llm" is half of what this ticket bought and a monkeypatched `arms.llm` would assert the
other half only.

Three groups, and they fail for different reasons:

  the miss      no chunk clears the floor -> the refusal, and *no model call*. The control that
                makes this mean something is test_a_floor_miss_makes_no_model_call: it points the
                backend at something that raises, so a call would fail the test rather than pass it
                with a plausible answer.
  the context   the excerpts really reach the model, whole, with their provenance attached. A
                grounded answer prompt whose context silently arrived empty would still return a
                fluent sentence — that is exactly the failure mode worth a test.
  the citations what `sources` claims. Deduped, rank-ordered, and emptied on a refusal.

The scores here are meaningless (six hand-written chunks, see test_retrieval.py on why), so nothing
asserts a score value. The floor is crossed by passing `floor=` explicitly instead.
"""
import pytest

from src import answer as answer_mod, arms, nlu, retrieval
from src.answer import REFUSAL, Answer
from src.config import FALLBACKS, LLM_ARMS, resolve
from src.retrieval import Hit, Index

from test_retrieval import CHUNKS       # same directory; pytest puts tests/unit on sys.path


@pytest.fixture
def idx():
    return Index(CHUNKS)


@pytest.fixture
def hits(idx):
    """Three real Hits off the hand-written corpus, so provenance is a fact and not a fixture.

    "casual privilege" matches all three leave-policy chunks and neither travel-policy one — and two
    of the three are on page 2, which is the overlap case `cited()` has to dedupe. So the fixture
    that feeds the context tests is also the one that makes the citation tests non-trivial.

    Not the query the tests below ask as a *question*: the question the model sees is written per
    test, and this only has to produce three real chunks off two pages. (A bare "leave" would not —
    it is in exactly half these chunks, which is where BM25 floors an IDF to zero. See
    src/retrieval.py on why that matters far more on the real corpus than it does here.)
    """
    got = idx.search("casual privilege", k=3, floor=0.0)
    assert len(got) == 3, "the corpus in test_retrieval.py changed under this test"
    assert len({(h.doc_id, h.page) for h in got}) == 2, "two of these share a page — see cited()"
    return got


def fake_llm(monkeypatch, reply="Twelve days of casual leave a year.", capture=None):
    """Answer every LLM arm with `reply`, recording the msgs it was handed. -> the capture list.

    Patches BACKENDS rather than arms.llm: everything between `answer()` and the wire — resolve,
    log_call, the free-tier check — then still runs for real. `options` are the per-call backend
    parameters (the temperature), captured because the grounded path sets one and a spoken reply
    does not.
    """
    seen = capture if capture is not None else []

    def backend(arm, msgs, rec, **options):
        seen.append({"arm": arm, "msgs": msgs, "rec": rec, "options": options})
        if isinstance(reply, Exception):
            raise reply
        return reply

    for arm in LLM_ARMS:
        monkeypatch.setitem(nlu.BACKENDS, arm.backend, backend)
    return seen


# --- the prompt -------------------------------------------------------------------------------

def test_the_prompt_file_exists_and_its_front_matter_never_reaches_the_model():
    """Front matter is metadata *about* the prompt. A leak would send the model its own version."""
    prompt = answer_mod.system_prompt()
    assert answer_mod.PROMPT_FILE.is_file()
    assert not prompt.startswith("---")
    assert "version:" not in prompt and "VOX-031" not in prompt


def test_the_refusal_sentence_is_the_same_string_in_the_prompt_and_in_the_code():
    """The drift guard named in src/answer.py.

    The floor-miss path returns REFUSAL with no model call; the model returns its own copy from the
    prompt file. If the two wordings drifted, a listener could hear which path ran — and which path
    ran is an implementation detail of the score floor, not information about their leave.
    """
    assert REFUSAL in answer_mod.system_prompt()


def test_the_prompt_forbids_outside_knowledge():
    """Not a wording test — the one instruction the whole ticket rests on has to be in there."""
    prompt = answer_mod.system_prompt().lower()
    assert "only from the excerpts" in prompt
    assert "do not use anything else" in prompt


# --- the miss ---------------------------------------------------------------------------------

def test_a_floor_miss_refuses_with_no_sources(idx):
    got = answer_mod.answer("is there a sabbatical policy", "t", idx=idx)

    assert got == Answer(REFUSAL, [], [], grounded=False)
    assert got.labels == []


def test_a_floor_miss_makes_no_model_call(monkeypatch, idx, calls_log):
    """The control. A call here would be a call with nothing to be grounded in — see src/answer.py.

    The backend is pointed at a raise, so a model call fails the test loudly instead of passing it
    with a plausible sentence, and calls_log is asserted empty so a *logged* call cannot hide either.
    """
    fake_llm(monkeypatch, reply=AssertionError("the model must not be called on a floor miss"))

    assert answer_mod.answer("is there a sabbatical policy", "t", idx=idx).text == REFUSAL
    assert not calls_log.exists() or calls_log.read_text(encoding="utf-8") == ""


def test_the_models_own_refusal_is_reported_as_a_refusal_and_cites_nothing(monkeypatch, hits):
    """The case the floor cannot catch: the right document, scoring well, without the answer in it.

    Citing three chunks next to "I could not find that" would claim they support an answer that was
    never given.
    """
    fake_llm(monkeypatch, reply=REFUSAL)

    got = answer_mod.answer("how much casual leave", "t", hits=hits)

    assert got.text == REFUSAL and got.grounded is False
    assert got.sources == [] and got.labels == []
    assert got.hits == hits, "the chunks that were searched are still worth returning"


@pytest.mark.parametrize("reply", [
    pytest.param(REFUSAL, id="verbatim"),
    pytest.param(REFUSAL.rstrip("."), id="no-full-stop"),
    pytest.param(f"  {REFUSAL.upper()}  ", id="shouted-and-padded"),
    pytest.param("I could not find that in the policy documents I have!", id="other-punctuation"),
])
def test_the_refusal_is_recognised_through_case_and_punctuation(reply):
    assert answer_mod.is_refusal(reply)


@pytest.mark.parametrize("reply", [
    pytest.param("Twelve days of casual leave a year.", id="an-answer"),
    pytest.param("I could not find the casual leave number in the leave policy.", id="reworded"),
    pytest.param("", id="empty"),
    pytest.param(None, id="none"),
])
def test_anything_but_that_sentence_is_treated_as_an_answer(reply):
    """The documented direction of the mechanism: equality against one constant, never a parser.

    A reworded refusal keeps its citations. That is the safe way round — the alternative is silently
    dropping the provenance off a reply that did answer the question.
    """
    assert not answer_mod.is_refusal(reply)


# --- the context ------------------------------------------------------------------------------

def test_the_excerpts_reach_the_model_whole_with_their_provenance(monkeypatch, hits):
    """The failure mode a fluent answer hides: context that silently arrived empty."""
    seen = fake_llm(monkeypatch)

    answer_mod.answer("how much casual leave", "t", hits=hits)

    user = seen[0]["msgs"][1]["content"]
    assert answer_mod.CONTEXT_HEADER in user
    for h in hits:
        assert h.doc_id in user and f"page {h.page}" in user
        assert h.text in user, "chunks go in whole — see context_block()"


def test_the_system_message_is_the_answer_prompt_and_not_the_reply_prompt(monkeypatch, hits):
    """Two prompt files now load the same way, so mixing them up is a one-character mistake."""
    seen = fake_llm(monkeypatch)

    answer_mod.answer("how much casual leave", "t", hits=hits)

    system = seen[0]["msgs"][0]["content"]
    assert system == answer_mod.system_prompt()
    assert system != nlu.system_prompt()


def test_the_question_is_asked_before_and_after_the_excerpts(monkeypatch, hits):
    """Deliberate, and stated in messages(): ~1500 tokens of policy between the question and the
    answer is enough for a 3B fallback arm to start summarising instead of answering."""
    seen = fake_llm(monkeypatch)

    answer_mod.answer("how much casual leave", "t", hits=hits)

    user = seen[0]["msgs"][1]["content"]
    assert user.count("how much casual leave") == 2
    assert user.startswith("Question: how much casual leave")


def test_context_is_numbered_so_five_chunks_can_be_held_apart(hits):
    block = answer_mod.context_block(hits)
    for n in range(1, len(hits) + 1):
        assert f"[{n}] " in block


def test_retrieval_runs_here_when_the_caller_did_not_do_it(monkeypatch, idx):
    """VOX-032 passes its own hits so it can time retrieval; a plain caller should not have to."""
    seen = fake_llm(monkeypatch)

    got = answer_mod.answer("casual leave", "t", idx=idx, floor=0.0)

    assert got.grounded and got.hits, "answer() retrieved for itself"
    assert seen[0]["msgs"][1]["content"].count("casual leave") >= 2


# --- the citations ----------------------------------------------------------------------------

def test_sources_are_the_provenance_of_the_chunks_that_were_passed(monkeypatch, hits):
    """The mechanism this ticket chose: context provenance, not a model-emitted citation token.

    So `sources` is exactly the deduped doc:page of what went in — no more, and never less.
    """
    fake_llm(monkeypatch)

    got = answer_mod.answer("how much casual leave", "t", hits=hits)

    assert got.grounded is True
    assert got.sources == answer_mod.cited(hits)
    assert set(got.labels) == {h.source for h in hits}


def test_two_chunks_off_one_page_are_cited_once():
    """Chunks overlap by CHUNK_OVERLAP_TOKENS, so this is the ordinary case, not an edge one."""
    same_page = [Hit("leave-policy", 4, 7, 0.6, "first half"),
                 Hit("leave-policy", 4, 8, 0.5, "second half"),
                 Hit("travel-policy", 1, 0, 0.4, "elsewhere")]

    assert answer_mod.cited(same_page) == [{"doc_id": "leave-policy", "page": 4},
                                           {"doc_id": "travel-policy", "page": 1}]


def test_citations_keep_the_ranking(monkeypatch, hits):
    """Best first, because the first thing a reader checks is the first thing listed."""
    fake_llm(monkeypatch)

    got = answer_mod.answer("how much casual leave", "t", hits=hits)

    assert got.labels[0] == hits[0].source


def test_labels_spell_a_citation_the_same_way_retrieval_does(hits):
    """`leave-policy:p4` in one place only — Hit.source. A second spelling is a second format."""
    assert Answer("x", answer_mod.cited(hits), hits, True).labels[0] == hits[0].source


# --- it is a real arms.llm call ---------------------------------------------------------------

def test_the_call_goes_through_the_cost_logger_with_the_hf_repo_id(monkeypatch, hits, calls_log):
    """Half of what routing through arms.llm bought. The other half is the fallback, below."""
    import json

    fake_llm(monkeypatch)
    answer_mod.answer("how much casual leave", "t-42", hits=hits)

    lines = calls_log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["stage"] == "llm" and rec["turn_id"] == "t-42" and rec["ok"] is True
    assert rec["model_id"] == LLM_ARMS[0].repo_id      # the HF repo id, never the provider string
    assert rec["cost_usd"] == 0.0


def test_the_call_logs_what_grounded_it(monkeypatch, hits, calls_log):
    """The provenance on the *call* line, so calls.jsonl says which chunks an answer came from
    without having to re-run retrieval against a corpus that may have been re-indexed since."""
    import json

    fake_llm(monkeypatch)
    answer_mod.answer("how much casual leave", "t", hits=hits)

    rec = json.loads(calls_log.read_text(encoding="utf-8").splitlines()[0])
    assert rec["prompt_file"] == answer_mod.PROMPT_FILE.name
    assert rec["chunks"] == len(hits)
    assert rec["sources"] == [h.source for h in hits]


def test_a_rate_limited_arm_still_answers_from_the_local_one(monkeypatch, hits):
    """The grounded path inherits the VOX-006 fallback for free — that is why it goes through
    arms.llm rather than calling httpx itself. Asserted here because "for free" is a claim."""
    from tests.unit.test_rate_limit import rate_limited, serve

    serve(monkeypatch, nlu, rate_limited())
    fb = resolve("llm", FALLBACKS["llm"])
    monkeypatch.setitem(nlu.BACKENDS, fb.backend,
                        lambda arm, msgs, rec, **options: "The leave policy says twelve days.")

    got = answer_mod.answer("how much casual leave", "t", hits=hits)

    assert got.text == "The leave policy says twelve days."
    assert got.grounded is True
    assert got.sources == answer_mod.cited(hits), "a fallback answer is grounded in the same chunks"


def test_the_named_arm_is_the_arm_that_is_called(monkeypatch, hits):
    """--llm has to reach this stage too, or `make answer LLM=...` would print the wrong model id."""
    seen = fake_llm(monkeypatch)
    wanted = LLM_ARMS[1]

    answer_mod.answer("how much casual leave", "t", hits=hits, model_id=wanted.alias)

    assert seen[0]["arm"].id == wanted.id


def test_an_empty_reply_is_not_dressed_up_as_an_answer(monkeypatch, hits):
    """nlu raises on an empty reply from a reasoning arm (see test_fallback.py). Nothing here may
    catch that and return a blank Answer — a blank spoken turn is worse than a visible failure."""
    fake_llm(monkeypatch, reply=RuntimeError("returned an empty reply"))

    with pytest.raises(RuntimeError, match="empty reply"):
        answer_mod.answer("how much casual leave", "t", hits=hits, fallback=False)


# --- the numeric guard --------------------------------------------------------------------------
# `answer_from_source_v2.md` says "Numbers you may say are the ones written in the excerpts". These
# assert that sentence is enforced rather than requested — measured, it is not enough to ask. At
# temperature 0.3 the leave-encashment question returned "you will get 12 rupees" one run in three;
# at temperature 0 it returned a four-sentence answer built on "you have 20 privileged leave, which
# is more than the 24 days allowed". Both fluent, both cited, neither number from a document.

@pytest.mark.parametrize("text, expected", [
    pytest.param("twelve working days", {12}, id="word"),
    pytest.param("12 working days", {12}, id="digit"),
    pytest.param("25,000 rupees", {25_000}, id="grouped-digits"),
    pytest.param("twenty-five thousand rupees", {25_000}, id="word-phrase"),
    pytest.param("the fourteenth of April", {14}, id="ordinal"),
    pytest.param("no numbers at all here", set(), id="none"),
])
def test_a_reply_states_the_numbers_it_says_however_it_spells_them(text, expected):
    """The prompt asks for spoken numbers and the documents are written in digits, so a guard that
    compared surface forms would refuse every correct answer it was ever given."""
    assert answer_mod.numbers_in(text) == {float(v) for v in expected}


def test_a_word_phrase_asserts_its_value_and_not_its_pieces():
    """The asymmetry the guard rests on. "twenty-five thousand" claims 25000 — a reply held to 5 and
    20 as well would be refused for saying a number correctly."""
    assert answer_mod.numbers_in("twenty-five thousand") == {25_000.0}
    assert {5.0, 20.0} <= answer_mod.numbers_in("twenty-five thousand", parts=True)


def test_a_figure_in_no_excerpt_is_reported_as_ungrounded():
    hits = [Hit("leave-policy", 4, 2, 0.5, "Employees are entitled to twelve days of casual leave.")]

    assert answer_mod.ungrounded_numbers("You get twelve days.", hits) == []
    assert answer_mod.ungrounded_numbers("You get 4,500 rupees.", hits) == [4500.0]


def test_a_number_the_person_supplied_is_not_grounded_by_having_been_asked(monkeypatch, hits):
    """The shape of the failure, not an innocent restatement: a fabricated calculation is built out
    of the figures the caller gave, so the excerpts are the only thing that counts as a source."""
    fake_llm(monkeypatch, reply="Since you have 20 privilege leaves, you will get 16 days.")

    got = answer_mod.answer("I have 20 privilege leaves, what is my encashment", "t", hits=hits)

    assert got.text == REFUSAL and got.grounded is False
    assert got.sources == [], "a refusal cites nothing"


def test_the_suppressed_reply_is_printed_so_the_refusal_can_be_explained(monkeypatch, hits, capsys):
    """A turn that refuses for this reason must be debuggable. The reply is not spoken and not
    logged as an answer, but it is on stderr with the figure that killed it."""
    fake_llm(monkeypatch, reply="You will get 4,500 rupees in leave encashment.")

    answer_mod.answer("what is my encashment", "t", hits=hits)

    err = capsys.readouterr().err
    assert "UNGROUNDED NUMBER" in err and "4500" in err
    assert "4,500 rupees" in err, "the suppressed reply itself, so it can be read"


def test_an_answer_whose_numbers_are_all_in_the_excerpts_is_untouched(monkeypatch, hits):
    """The guard must not fire on the ordinary case, which is most of them."""
    text = " ".join(hits[0].text.split())
    fake_llm(monkeypatch, reply=text)

    got = answer_mod.answer("how much casual leave", "t", hits=hits)

    assert got.grounded is True and got.text == text


# --- sampling ------------------------------------------------------------------------------------

def test_the_grounded_path_asks_for_a_deterministic_answer(monkeypatch, hits):
    """Reading five policy excerpts is not a task where variety is a feature, and a gate that cannot
    reproduce its own number is not a gate."""
    seen = fake_llm(monkeypatch)

    answer_mod.answer("how much casual leave", "t", hits=hits)

    assert seen[0]["options"]["temperature"] == answer_mod.ANSWER_TEMPERATURE == 0.0


def test_the_spoken_reply_path_keeps_its_own_temperature(monkeypatch):
    """Not a global change: a plain reply still samples the way VOX-002 measured it."""
    seen = fake_llm(monkeypatch, reply="I can help with that.")

    nlu.reply("hello there", "t")

    assert seen[0]["options"] == {}, "no per-call override, so nlu.TEMPERATURE applies"


def test_the_temperature_that_was_used_is_on_the_call_record(monkeypatch, hits, calls_log):
    """Two turns sampled differently are not comparable, and a latency table has no way to know."""
    import json

    fake_llm(monkeypatch)
    answer_mod.answer("how much casual leave", "t", hits=hits)

    rec = json.loads(calls_log.read_text(encoding="utf-8").splitlines()[0])
    assert rec["temperature"] == 0.0
