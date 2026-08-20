"""Retrieved chunks to a spoken, grounded answer through the existing LLM arm (VOX-031).

    answer(transcript, turn_id) -> Answer(text, sources, hits, grounded)

The stage between `src/retrieval.py` and the speaker. It adds no arm and no provider: the call goes
out through `arms.llm()` like every other LLM call in this repo, so the cost logger, the `--llm`
flag, the rate-limit cooldown and the local ollama fallback all apply here without a line of code.
That is the whole reason the POC routes through `arms.llm` instead of calling httpx itself.

Four decisions worth knowing before reading the code:

**A floor-miss never reaches the model.** `retrieval.retrieve()` returning `[]` means no chunk
cleared `config.RETRIEVAL_SCORE_FLOOR` — there is nothing to be grounded *in*, so there is nothing
for a model to do but invent. `answer()` returns `REFUSAL` with an empty source list and makes no
call at all: no token spend, no round-trip, and no path on which a hallucination is even possible.
It also means the refusal is testable without a key, which is what lets VOX-033's gate measure a
refusal rate offline.

The model still has to refuse on the harder case, which the floor cannot catch: chunks that score
well because they come from the document that *ought* to answer the question, and then stop short of
the answer. "Is there a canteen subsidy" scoring 0.23 is a floor's job; the attendance policy
mentioning working days without mentioning lunch is a prompt's job.

**There is one refusal sentence, not two.** `REFUSAL` is the literal string in
`prompts/answer_from_source_v1.md`, and `tests/unit/test_answer.py` asserts it appears there
verbatim. If the deterministic refusal and the model's refusal were worded differently, a caller
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
"""
import re
from collections import namedtuple

from src import nlu, retrieval
from src.config import PROMPTS_DIR

PROMPT_FILE = PROMPTS_DIR / "answer_from_source_v1.md"

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
        hits = retrieval.retrieve(transcript, k=k, floor=floor, idx=idx)

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
