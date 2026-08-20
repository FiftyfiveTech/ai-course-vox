"""Retrieved chunks to a spoken, grounded answer through the existing LLM arm (VOX-031).

    answer(transcript, turn_id) -> Answer(text, sources, hits, grounded)

The stage between `src/retrieval.py` and the speaker. It adds no arm and no provider: the call goes
out through `arms.llm()` like every other LLM call in this repo, so the cost logger, the `--llm`
flag, the rate-limit cooldown and the local ollama fallback all apply here without a line of code.
That is the whole reason the POC routes through `arms.llm` instead of calling httpx itself.

Four decisions worth knowing before reading the code:

**A floor-miss never reaches the model.** `retrieval.retrieve()` returning `[]` means no chunk was
vouched for by either half of retrieval — neither `config.RETRIEVAL_SCORE_FLOOR` nor
`config.DENSE_SCORE_FLOOR` — so there is nothing to be grounded *in*, so there is nothing
for a model to do but invent. `answer()` returns `REFUSAL` with an empty source list and makes no
call at all: no token spend, no round-trip, and no path on which a hallucination is even possible.
It also means the refusal is testable without a key, which is what lets VOX-033's gate measure a
refusal rate offline.

The model still has to refuse on the harder case, which the floor cannot catch: chunks that score
well because they come from the document that *ought* to answer the question, and then stop short of
the answer. "Is there a canteen subsidy" scoring 0.23 is a floor's job; the attendance policy
mentioning working days without mentioning lunch is a prompt's job.

That division of labour carries more weight since retrieval grew a dense half, and the shift is
measured: the dense cosine does not separate answerable questions from absent ones on the dev set
(see `config.DENSE_SCORE_FLOOR`), so four of six absent dev queries now reach the model instead of
being refused arithmetically. They are refused here, by the prompt, which is why the prompt is
versioned and measured rather than assumed.

**There is one refusal sentence, not two.** `REFUSAL` is the literal string in `PROMPT_FILE`
(currently `prompts/answer_from_source_v2.md`), and `tests/unit/test_answer.py` asserts it appears
there verbatim. If the deterministic refusal and the model's refusal were worded differently, a caller
could tell which path ran — and "the score floor rejected this" is an implementation detail of
retrieval, not something a person asking about leave should hear the shape of.

**Citations are the provenance of the context, not a choice the model made.** `sources` is the
deduped `doc_id`/`page` of the chunks that were passed in. The model is never asked to emit a
citation token, so there is no format for it to get wrong — which matters because the local
fallback arm is a 3B, and a parse failure on the fallback path would land on the turn that was
already going badly. The cost of that choice is stated plainly: `sources` says *what the answer was
grounded in*, up to five chunks, not which one sentence it came from. VOX-033's correct-source@3 is
a retrieval measurement and is unaffected; anything wanting per-sentence attribution needs a
different mechanism than this ticket bought.

The prompt does ask the model to name the document in passing — "the leave policy says twelve days"
— because that is what a colleague would say out loud. That naming is prose for the listener, and
nothing here parses it.

**A model refusal empties the source list.** If the reply is the refusal sentence, citing five
documents next to it would be a claim that they support an answer that was not given. The check is
normalised equality against one constant string, not a parser: punctuation and case are ignored,
anything else is treated as an answer. That direction is deliberate — a near-miss keeps its
citations, which is safer than silently dropping the provenance off a reply that did answer.

`turn_reply()` at the bottom is VOX-032: the same two paths as seen from inside a spoken turn —
retrieve, and route to `answer()` or to the plain `nlu.reply()` on what comes back. It lives here
rather than in `src/loop.py` because the mic loop and the recording harness both need it and a
second copy of the routing is how the two quietly stop running the same pipeline.
"""
import re
from collections import namedtuple
from contextlib import contextmanager

from src import nlu, retrieval
from src.config import PROMPTS_DIR

# v2 forbids the model from *computing* a figure from the person's own numbers. v1 did not, and
# answered "you will be paid 12,000" to a leave-encashment question whose excerpt gave a formula and
# no such number — cited, fluent and wrong. Versioned as a new file rather than edited in place
# (VOX-018): the old prompt is what the numbers in ARCHITECTURE.md were measured against, and a
# prompt you can no longer read is a measurement you can no longer reproduce.
PROMPT_FILE = PROMPTS_DIR / "answer_from_source_v2.md"

# The one refusal, shared by the two paths that can produce it: this module when no chunk clears the
# floor, and the model when the chunks that did clear it do not contain the answer. Written out
# verbatim in the prompt file too; the test asserts the two are the same string.
REFUSAL = "I could not find that in the policy documents I have."

# How the excerpts are introduced in the user message. Named so a test can assert the context really
# reached the model rather than only that a call happened.
CONTEXT_HEADER = "Excerpts from the policy documents:"


class Answer(namedtuple("Answer", "text sources hits grounded")):
    """One answered question.

    text      what to say. Either the model's answer or REFUSAL — a caller can hand it straight to
              TTS without asking which.
    sources   deduped [{doc_id, page}] for the chunks the answer was grounded in, best first.
              Empty on any refusal. See the module docstring on what this does and does not claim.
    hits      the retrieval.Hit objects behind those sources, scores included, so a caller that
              wants to print or log the evidence does not have to retrieve twice.
    grounded  did a model actually answer from the documents? False for both refusals — the
              floor-miss and the model's — because from a listener's side they are the same event.
    """

    __slots__ = ()

    @property
    def labels(self):
        """-> ["leave-policy:p4", ...] — the citation in the form retrieval.Hit.source uses."""
        return [f"{s['doc_id']}:p{s['page']}" for s in self.sources]


def system_prompt():
    """The versioned answer prompt, front matter stripped. Loaded from disk, never inlined."""
    return nlu.load_prompt(PROMPT_FILE)


def context_block(hits):
    """-> the excerpts as one string: `[n] doc_id, page N` then the chunk text, blank-line separated.

    The chunk text goes in whole. Truncating it here would mean the answer is grounded in something
    other than what retrieval scored, and the chunk geometry (config.CHUNK_TOKENS) is already the
    knob for how much context a hit is worth.
    """
    blocks = []
    for n, h in enumerate(hits, start=1):
        text = " ".join((h.text or "").split())
        blocks.append(f"[{n}] {h.doc_id}, page {h.page}\n{text}")
    return "\n\n".join(blocks)


def messages(transcript, hits):
    """-> the `msgs` list for arms.llm(). The only place the grounded prompt's shape is decided.

    Question first, excerpts second, question again last. The repeat is not padding: ~1500 tokens of
    policy text between the question and where the answer gets written is enough for a small arm to
    start summarising the excerpts instead of answering, and the local fallback arm is a 3B.
    """
    return [
        {"role": "system", "content": system_prompt()},
        {"role": "user", "content": (
            f"Question: {transcript}\n\n"
            f"{CONTEXT_HEADER}\n\n"
            f"{context_block(hits)}\n\n"
            f"Answer the question — {transcript} — from those excerpts alone."
        )},
    ]


def cited(hits):
    """-> deduped [{doc_id, page}] in rank order.

    Deduped because chunks overlap by config.CHUNK_OVERLAP_TOKENS, so two chunks off the same page
    routinely both make the top five, and citing "leave-policy:p4" twice tells a reader nothing.
    """
    seen, out = set(), []
    for h in hits:
        key = (h.doc_id, h.page)
        if key not in seen:
            seen.add(key)
            out.append({"doc_id": h.doc_id, "page": h.page})
    return out


def _normalise(text):
    """-> `text` with case and punctuation flattened, for comparing against REFUSAL and nothing else."""
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


_REFUSAL_NORM = _normalise(REFUSAL)


def is_refusal(text):
    """-> True if `text` is the refusal sentence. Equality against one constant, not a parser."""
    return _normalise(text) == _REFUSAL_NORM


def answer(transcript, turn_id, hits=None, k=None, floor=None, idx=None,
           model_id=None, on_fallback=None, fallback=True):
    """-> Answer for `transcript`, grounded in the retrieved chunks. Never raises on a miss.

    `hits` lets a caller that has already retrieved — VOX-032's turn loop, which needs the retrieval
    latency on the turn record — pass its chunks in rather than have them fetched twice. Left as None
    this retrieves them itself with `k`/`floor`/`idx` forwarded to retrieval.retrieve().

    A provider failure raises exactly as `nlu.reply` does. A rate limit or a dead network is handled
    a layer down by `arms.llm`'s fallback and never reaches here; a bad key still stops the run,
    which is the behaviour VOX-006 chose and this ticket does not get to soften.
    """
    if hits is None:
        hits = retrieval.retrieve(transcript, k=k, floor=floor, idx=idx, turn_id=turn_id)

    if not hits:
        # No model call: see the module docstring. Nothing cleared the floor, so there is no context
        # to be grounded in and no question of what the model might say instead.
        return Answer(REFUSAL, [], [], grounded=False)

    from src import arms                      # imported here: arms imports nlu, which this imports
    text = arms.llm(
        messages(transcript, hits), model_id, turn_id=turn_id,
        on_fallback=on_fallback, fallback=fallback,
        prompt_file=PROMPT_FILE.name, transcript_chars=len(transcript),
        chunks=len(hits), sources=[h.source for h in hits],
        top_score=round(hits[0].score, 4),
    )

    if is_refusal(text):
        return Answer(REFUSAL, [], hits, grounded=False)
    return Answer(text, cited(hits), hits, grounded=True)


# --- inside a turn (VOX-032) -------------------------------------------------------------------

# What the reply stage of one turn produced. `answer` is the Answer on the grounded path and None
# on the plain one, so a caller can tell *which path ran* without re-deriving it from `hits` —
# and `text` is what goes to TTS either way, which is the only thing the speaker needs to know.
Reply = namedtuple("Reply", "text answer hits")


def knowledge_base(echo=print):
    """-> the process-wide retrieval index, or None if this machine has nothing indexed.

    Called once at startup, not per turn: building the index is per-process work (see
    retrieval.index()), and a build inside the first turn would land in a stage number and make
    the latency split a lie.

    None is a supported state, not an error. `sources/` is gitignored — internal HR policies — so a
    clean clone has no corpus and no chunk file, and `make demo` still has to run for whoever is
    standing in front of it. What must not happen is that state being *silent*: a demo that quietly
    stopped being grounded because nobody ran `make index` looks exactly like one where retrieval
    found nothing, and only one of the two is a working system. So the reason is printed here,
    once, and every turn afterwards says nothing.
    """
    try:
        idx = retrieval.index()
    except RuntimeError as e:
        echo(f"no knowledge base: {e}\n"
             f"  turns will answer from the plain reply prompt — nothing will be grounded.")
        return None
    echo(f"knowledge base: {len(idx)} chunks over {len(idx.doc_ids)} documents "
         f"({', '.join(idx.doc_ids[:4])}{', …' if len(idx.doc_ids) > 4 else ''})")
    return idx


def turn_reply(transcript, turn_id, idx=None, turn=None, model_id=None, on_fallback=None,
               fallback=True, k=None, floor=None):
    """One turn's reply: grounded in the documents when they cover the question, plain when not.

    -> Reply(text, answer, hits). The whole of VOX-032's routing decision, in one place because
    `src/loop.py` and `src/harness.fixture_turn` both need it and a second copy is how a
    comparison ends up timing a pipeline the live loop does not run.

    The decision is retrieval's, not a classifier's: chunks that clear `RETRIEVAL_SCORE_FLOOR` go
    to `answer()`, an empty list goes to `nlu.reply()` exactly as every turn did before this
    ticket. The floor was measured (`scripts/ask.py --calibrate`), which is the reason it gets to
    be the router — an intent model in front of it would be a second, unmeasured decision, and it
    would fail in the expensive direction: a misrouted greeting costs a plain reply, a misrouted
    "how much casual leave" costs an invented policy number.

    `idx=None` means this machine has no knowledge base (see `knowledge_base()`) and skips
    retrieval altogether — that is not the same as retrieval returning nothing, and the two are
    distinguishable on the turn record: `t_retrieval_ms` is absent in the first case and measured
    in the second.

    `turn` is the TurnTimer. Retrieval is timed on it outside the llm stage, and the grounding is
    stamped on it — both so a turn line can be read afterwards without guessing which path it took.
    Left as None (a caller with no turn record, i.e. a test) nothing is timed and the routing is
    unchanged.
    """
    hits = []
    if idx is not None:
        with _timing(turn, "retrieval"):
            # turn_id goes down into retrieval because the dense half makes a model call now: the
            # query encoding is a line in runs/calls.jsonl, and a line with a null turn_id joins to
            # nothing, which is the one thing the two-log design exists to prevent.
            hits = retrieval.retrieve(transcript, k=k, floor=floor, idx=idx, turn_id=turn_id)

    with _timing(turn, "llm"):
        if hits:
            got = answer(transcript, turn_id, hits=hits, model_id=model_id,
                         on_fallback=on_fallback, fallback=fallback)
            text = got.text
        else:
            # Nothing cleared the floor, or there is no corpus at all. Either way there is nothing
            # to be grounded in, so the turn behaves as it did before this ticket existed.
            got = None
            text = nlu.reply(transcript, turn_id, model_id=model_id,
                             on_fallback=on_fallback, fallback=fallback)

    if turn is not None:
        turn.grounding(hits, grounded=bool(got and got.grounded),
                       sources=got.labels if got else [])
    return Reply(text, got, hits)


@contextmanager
def _timing(turn, what):
    """Time `what` on `turn` if there is one. `retrieval` is a turn field, `llm` a stage — see
    TurnTimer.retrieval() on why the two are recorded differently."""
    if turn is None:
        yield
    elif what == "retrieval":
        with turn.retrieval():
            yield
    else:
        with turn.stage(what):
            yield
