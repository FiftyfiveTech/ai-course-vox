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
retrieve, and route to `answer()` or to the plain reply on what comes back (`nlu.reply` unless the
caller supplies its own; the live loop supplies VOX-019's `state.build`). It lives here
rather than in `src/loop.py` because the mic loop and the recording harness both need it and a
second copy of the routing is how the two quietly stop running the same pipeline.
"""
import re
import sys
from collections import namedtuple
from contextlib import contextmanager

from src import dates, figures, nlu, retrieval
from src.config import PROMPTS_DIR, RETRIEVAL_TOP_K

# v2 forbids the model from *computing* a figure from the person's own numbers. v1 did not, and
# answered "you will be paid 12,000" to a leave-encashment question whose excerpt gave a formula and
# no such number — cited, fluent and wrong. v3 (VOX-034) adds one thing the guard provably cannot
# do: correct a false premise. "Your 30 days of paternity leave" has every number traced — 30 is the
# advance-notice window in the chunk beside the five-day entitlement — so ungrounded_numbers() stays
# silent and only a prompt can tell a present number from an answering one. Arithmetic did not
# become allowed; it moved to prompts/compute_figure_v1.md and src/figures.py, where a figure has a
# checked derivation behind it. Versioned as new files rather than edited in place (VOX-018): v1 and
# v2 are what the numbers in ARCHITECTURE.md were measured against, and a prompt you can no longer
# read is a measurement you can no longer reproduce.
PROMPT_FILE = PROMPTS_DIR / "answer_from_source_v3.md"

# The one refusal, shared by the two paths that can produce it: this module when no chunk clears the
# floor, and the model when the chunks that did clear it do not contain the answer. Written out
# verbatim in the prompt file too; the test asserts the two are the same string.
REFUSAL = "I could not find that in the policy documents I have."

# How the excerpts are introduced in the user message. Named so a test can assert the context really
# reached the model rather than only that a call happened.
CONTEXT_HEADER = "Excerpts from the policy documents:"

# Sampling for the grounded path. Zero, where the spoken-reply path uses nlu.TEMPERATURE = 0.3.
#
# Measured, 2026-08-20: the same leave-encashment question asked three times at 0.3 returned two
# correct refusals and one invented figure ("you will get 12 rupees"). A prompt cannot fix that,
# because nothing was wrong with the prompt on the two runs where it worked — the answer was being
# sampled from a distribution that includes the bad one. There is nothing for temperature to buy
# here anyway: reading five policy excerpts is not a task where variety is a feature, and a gate
# that cannot reproduce its own number is not a gate.
ANSWER_TEMPERATURE = 0.0


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


# --- the numeric guard --------------------------------------------------------------------------
# `answer_from_source_v2.md` ends its rule with "Numbers you may say are the ones written in the
# excerpts". Everything below is that sentence enforced in code, because asking was measured and it
# is not enough: at temperature 0.3 the same leave-encashment question produced "you will get 12
# rupees" one run in three, and at temperature 0 it produced a four-sentence answer built on
# "you have 20 privileged leave, which is more than the 24 days allowed". Both are fluent, both
# carry citations, and neither number came from a document.
#
# The rule is deliberately about the *excerpts* and not the conversation. A figure the person
# themselves supplied is exactly what a fabricated calculation is built out of, so echoing it back
# inside an answer is the shape of the failure rather than an innocent restatement.

_DIGITS = re.compile(r"\d[\d,]*(?:\.\d+)?")

_UNITS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
          "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
          "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
          "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
          "seventy": 70, "eighty": 80, "ninety": 90,
          # Ordinals, because the prompt asks for dates the way a person says them — "the
          # fourteenth of April" has to match a "14" in the excerpt or every date answer refuses.
          "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7,
          "eighth": 8, "ninth": 9, "tenth": 10, "eleventh": 11, "twelfth": 12, "thirteenth": 13,
          "fourteenth": 14, "fifteenth": 15, "sixteenth": 16, "seventeenth": 17, "eighteenth": 18,
          "nineteenth": 19, "twentieth": 20, "thirtieth": 30}
# "hundred" is a MULTIPLIER inside a group; the rest are scales that close one.
#
# Getting that wrong was a real bug, found by VOX-034's figure gate: with "hundred" treated as a
# closing scale, "six hundred thousand" parsed as (6*100) + (1*1000) = 1600 rather than 600000, and
# "three hundred and sixty five thousand" as 65300 rather than 365000. Both are how a person says a
# salary out loud, so the guard was checking spoken currency figures against numbers nobody said —
# in both directions: a correctly grounded reply could be refused, and an invented figure could pass
# if the mis-parse happened to land on something in the excerpts.
_MULTIPLIER = {"hundred": 100}
_SCALES = {"thousand": 1_000, "lakh": 100_000, "lakhs": 100_000,
           "million": 1_000_000, "crore": 10_000_000, "crores": 10_000_000}
_NUMBER_WORD = re.compile(r"[a-z]+")


def numbers_in(text, parts=False):
    """-> the set of numeric values `text` states, digits and words alike.

    "twenty-five thousand" and "25,000" both come out as 25000, because the prompt asks the model to
    say numbers the way a person does while the excerpts are written the way a document does. A
    guard comparing surface forms would refuse every correct answer it was ever given.

    `parts` is the asymmetry that makes this usable. Off (the answer side) a word phrase yields only
    what it means: "twenty-five thousand" is 25000 and nothing else, so the reply is held to the
    figures it actually asserts. On (the excerpt side) the pieces come too — 25, 5, 20, 1000 — so an
    excerpt written "25 thousand", or one that happens to spell a component, still counts as having
    said it. Being generous about what a document contains and strict about what a reply claims is
    the safe direction: the guard then only ever fires on a number that appears nowhere at all.
    """
    found = set()
    for raw in _DIGITS.findall(text or ""):
        try:
            found.add(float(raw.replace(",", "")))
        except ValueError:
            continue

    words = _NUMBER_WORD.findall((text or "").lower().replace("-", " "))
    total = current = 0.0
    running = False

    def flush():
        nonlocal total, current, running
        if running and (total or current):
            found.add(total + current)
        total = current = 0.0
        running = False

    for w in words:
        if w in _UNITS:
            current += _UNITS[w]
            running = True
            if parts:
                found.add(float(_UNITS[w]))
        elif w in _MULTIPLIER and running:
            # Scales the group in progress and does NOT close it, so "six hundred thousand" is
            # (6 * 100) * 1000 and not 600 + 1000. See the comment on _MULTIPLIER.
            current = max(current, 1.0) * _MULTIPLIER[w]
            if parts:
                found.add(current)
        elif w in _SCALES and running:
            current = max(current, 1.0) * _SCALES[w]
            total += current
            if parts:
                found.add(current)
            current = 0.0
        elif w == "and" and running:
            continue
        else:
            flush()
    flush()
    return found


def ungrounded_numbers(text, hits):
    """-> sorted values stated in `text` that appear in none of the excerpts.

    Empty is the good case. A non-empty list means the reply asserts a figure that is not in the
    documents it cites — invented, or worse, calculated, which is the version that sounds most like
    a right answer.
    """
    context = set()
    for h in hits:
        context |= numbers_in(h.text, parts=True)
    return sorted(numbers_in(text) - context)


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

    # VOX-034 part D. A question that states a DATE and asks when goes to the date path first, for
    # the reason src/dates.py opens with: the figure path is right to call a deadline "not a formula",
    # so a duration question was answered with the sentence read back and no date. Routed before the
    # figure path and not after it, so a date question still costs ONE model call — the invariant the
    # figure path kept when it replaced the prose call rather than preceding it. A date question that
    # cannot be counted falls through to the paths below, which is a second call on a failure path.
    #
    # `asks_for_a_date` needs both halves — a date stated and a when-ish question — so a counting
    # question that happens to name a month ("leave from the nineteenth to the twenty third, how many
    # PLs is that") stays on the numeric path where it belongs.
    if dates.asks_for_a_date(transcript):
        dfig = dates.compute(transcript, hits, turn_id, model_id=model_id,
                             fallback=fallback, on_fallback=on_fallback)
        if dfig is not None and dfig.rule:
            spoken = dfig.spoken()
            if is_refusal(spoken):
                return Answer(REFUSAL, [], hits, grounded=False)

            # The guard applies here exactly as it does on the figure path, with the derivation's own
            # numbers added to what counts as grounded: the days and years of the dates Python
            # counted, the anchor the person gave, and the durations that traced to an excerpt. A
            # computed date is spoken as "29 September 2026", so without this the guard suppresses
            # every correct answer this path can give — 29 is in no excerpt, and that is the point.
            allowed = set()
            if dfig.computed:
                for d in (dfig.value, dfig.end, dfig.anchor):
                    if d is not None:
                        allowed |= {float(d.day), float(d.year)}
                for n in (dfig.offset, dfig.offset_end):
                    if n is not None:
                        allowed.add(float(n))
            invented = [v for v in ungrounded_numbers(spoken, hits)
                        if not any(abs(v - a) < 1e-6 for a in allowed)]
            if invented:
                print(f"UNGROUNDED NUMBER on the date path — {arms_repr(invented)} appears in no "
                      f"excerpt and in no counted date; refusing instead of speaking it.\n"
                      f"  suppressed reply: {' '.join(spoken.split())}", file=sys.stderr)
                return Answer(REFUSAL, [], hits, grounded=False)

            if dfig.computed:
                return Answer(spoken, cited(hits), hits, grounded=True)
            # Nothing counted. Fall through rather than state the rule here: the question may still
            # be a figure question ("I joined in March, how many leaves by June" states a date and
            # asks when-ish), and the paths below are the ones that have been measured on it.
            print(f"date not computed — {', '.join(dfig.missing)}", file=sys.stderr)

    # VOX-034 part B. A question that states a number may be asking for one back, and the prose
    # prompt is forbidden from doing arithmetic — so it goes to the figure path instead, where the
    # model names the operands and Python does the sum. See src/figures.py.
    #
    # Routed on `states_a_number` and not on a classifier, for the reason turn_reply's docstring
    # gives about the retrieval floor being the router: a cheap syntactic test that can be read off
    # the transcript beats a second unmeasured decision. It is also why every query in
    # evals/dev/pdf_queries.json is unaffected — none of them states a number, so `make gate-poc`
    # never enters this branch.
    #
    # This does NOT add a call to the turn: the figure path replaces the prose call rather than
    # preceding it, so a numeric question still costs one LLM call. That is the same constraint
    # turn_reply keeps for the two-path routing above it.
    if figures.states_a_number(transcript):
        fig = figures.compute(transcript, hits, turn_id, model_id=model_id,
                              fallback=fallback, on_fallback=on_fallback)
        # `None` is an unparseable extraction and `not fig.rule` is an extraction with nothing in it.
        # Both fall through to the prose prompt below rather than refusing, because v2 already
        # answers this question correctly — it just will not compute. Costs a second call on a
        # failure path, which is the right place to spend one.
        if fig is not None and fig.rule:
            spoken = fig.spoken()
            if is_refusal(spoken):
                return Answer(REFUSAL, [], hits, grounded=False)

            # THE GUARD STILL APPLIES, and this is the point of the whole design. The figure path
            # composes its sentence around a value Python computed, but `rule` is model-authored
            # prose and can carry a figure of its own — "you will get sixteen days" — which is the
            # v1 failure with extra steps. So the reply is checked exactly as the prose path is
            # checked, with the derivation's own numbers added to what counts as grounded: the
            # computed value, and every operand that traced.
            #
            # That is the numeric guard NARROWED, not relaxed. A number the person supplied is still
            # not grounded by having been asked — it is grounded only as an operand inside a
            # derivation that checked out, which is why `allowed` is empty when nothing was computed.
            allowed = set()
            if fig.computed:
                allowed.add(float(fig.value))
                for op in fig.operands:
                    try:
                        allowed.add(float(op.get("value")))
                    except (TypeError, ValueError):
                        continue
            invented = [v for v in ungrounded_numbers(spoken, hits)
                        if not any(abs(v - a) < 1e-6 for a in allowed)]
            if invented:
                print(f"UNGROUNDED NUMBER on the figure path — {arms_repr(invented)} appears in no "
                      f"excerpt and in no checked derivation; refusing instead of speaking it.\n"
                      f"  suppressed reply: {' '.join(spoken.split())}", file=sys.stderr)
                return Answer(REFUSAL, [], hits, grounded=False)

            if not fig.computed:
                # The rule, stated, with no figure — what v2 already asked for and what
                # evals/dev/figure_queries.json scores as `state_rule`. Grounded: a rule read out of
                # an excerpt is an answer, and `missing` says why no number came with it.
                print(f"figure not computed — {', '.join(fig.missing)}", file=sys.stderr)
            return Answer(spoken, cited(hits), hits, grounded=True)

    from src import arms                      # imported here: arms imports nlu, which this imports
    text = arms.llm(
        messages(transcript, hits), model_id, turn_id=turn_id,
        on_fallback=on_fallback, fallback=fallback, temperature=ANSWER_TEMPERATURE,
        prompt_file=PROMPT_FILE.name, transcript_chars=len(transcript),
        chunks=len(hits), sources=[h.source for h in hits],
        top_score=round(hits[0].score, 4),
    )

    if is_refusal(text):
        return Answer(REFUSAL, [], hits, grounded=False)

    # The numeric guard. A figure that is in no excerpt makes this reply ungrounded whatever else it
    # says, so it becomes the same refusal a listener would have heard if retrieval had missed —
    # they are not owed the distinction, and the alternative is a confident wrong number about their
    # own salary. Recorded on the call record so the rate is visible in runs/calls.jsonl rather than
    # only in whatever the caller decided to print.
    invented = ungrounded_numbers(text, hits)
    if invented:
        print(f"UNGROUNDED NUMBER — {arms_repr(invented)} appears in no excerpt; refusing instead "
              f"of speaking it.\n  suppressed reply: {' '.join(text.split())}", file=sys.stderr)
        return Answer(REFUSAL, [], hits, grounded=False)

    return Answer(text, cited(hits), hits, grounded=True)


def arms_repr(values):
    """-> "12, 4" — the invented figures, as a person would read them in a log line."""
    return ", ".join(f"{v:g}" for v in values)


# --- retrieval, with one rewritten retry (VOX-034) ---------------------------------------------


def retrieve_with_history(transcript, turn_id, idx, history=None, k=None, floor=None):
    """-> (hits, question_to_ask, rewrite) — retrieval for one turn, retried once on a miss.

    The single copy of VOX-034's retrieval decision. `turn_reply()` calls it and so does
    `tests/gates/gate_followup.py`, for the reason `turn_reply`'s own docstring gives about a second
    copy of the routing: a gate that scores a slightly different pipeline from the one the loop runs
    is measuring something nobody ships.

    The retry fires **only on a miss**, and that is the whole safety argument. A turn that already
    retrieved something is never rewritten, so this cannot change the answer to a question that
    already worked — which is what lets `make gate-poc` stand as a no-regression check instead of a
    number to re-measure. See `src/history.py` on why the rewritten query is built from previous
    QUESTIONS and never from previous ANSWERS.

    `question_to_ask` is the rewritten string when the retry succeeded, and `transcript` otherwise.
    It is what the grounded prompt should ask, because handing a model five paternity excerpts and
    the words "how far in advance do I have to plan it" is asking it to guess what "it" was. This
    keeps `messages()` single-shot and adds no previous *answer*, so it introduces no figure the
    numeric guard cannot see: the only numbers it can add are ones the person said themselves,
    which the guard already treats as ungrounded.

    `rewrite` is None when no retry was attempted, or `{"query", "used", "trigger"}` when one was —
    `used` False meaning the retry also missed. Both facts are worth recording: a rewrite that fires
    and fails is a different diagnosis from one that never fired.

    **Two triggers, and the first one was added after the first gate run disproved the design.**
    VOX-034 shipped with a miss-only trigger, on the argument that it was self-limiting and
    therefore safe. It is safe and it was measured to be nearly useless: `make gate-followup` at
    attempt 1 scored referential 2/6 -> 3/6, and three of the four failures never triggered a
    rewrite at all because they did not MISS — they confidently retrieved the wrong document.
    "how far in advance do I have to plan it" returned travel-policy (advance booking); "what about
    during a performance improvement plan" returned performance-management at 0.742, when the answer
    is one clause of leave-policy:p5. A fragment does not fail by finding nothing; it fails by
    finding whatever its few surviving terms happen to match.

    So an *elliptical* follow-up (see `history.elliptical`) has its query rewritten BEFORE retrieval
    and the fragment is not used at all — the fragment is not what the person asked, it is half of
    it. A non-elliptical transcript is untouched, which is what keeps `make gate-poc` a
    no-regression check: every query in that set is a whole question.

    The miss trigger is kept as a second chance for the referential turns ellipsis detection does
    not catch — "what happens to the extra ones" has an anaphor and is caught, but the general case
    of a fragment with neither an opener nor a pronoun still exists.
    """
    # Trigger 1: the transcript reads as a continuation, so retrieve on BOTH forms and fuse.
    #
    # Fused and not replaced, and that is measured rather than preferred. Attempt 2 replaced the
    # fragment with the concatenation and scored referential 3/6 — but a DIFFERENT 3: f01/f02/f04
    # were rescued and f05/f06 were lost, because the antecedent's terms swamp the follow-up's.
    # "and if I am still on probation" finds probation-period:p6 on its one high-IDF term; prepend
    # "what is the notice period when I resign" and separation-policy's many matching terms bury it.
    # Both queries carry real signal, so neither gets to win outright — which is the same argument
    # Index.search already makes about its lexical and dense halves, and the reason retrieval.fuse
    # exists to be shared rather than reimplemented here.
    if history and history.elliptical(transcript):
        rq = history.retrieval_query(transcript)
        if rq and rq != transcript:
            raw = retrieval.retrieve(transcript, k=k, floor=floor, idx=idx, turn_id=turn_id)
            rw = retrieval.retrieve(rq, k=k, floor=floor, idx=idx, turn_id=turn_id)
            hits = retrieval.fuse([raw, rw], k=k or RETRIEVAL_TOP_K)
            rewrite = {"query": rq, "used": bool(rw), "trigger": "elliptical"}
            # The question the model is asked is the resolved one only when the rewrite actually
            # contributed something the fragment did not find on its own. Otherwise the fragment
            # already retrieved its own answer and the antecedent is noise in the prompt.
            asked = rq if rw and not raw else transcript
            return hits, asked, rewrite

    hits = retrieval.retrieve(transcript, k=k, floor=floor, idx=idx, turn_id=turn_id)
    if hits or not history:
        return hits, transcript, None

    # Trigger 2: it missed, and there is an antecedent that might rescue it.
    rq = history.retrieval_query(transcript)
    if not rq or rq == transcript:
        return hits, transcript, None

    retry = retrieval.retrieve(rq, k=k, floor=floor, idx=idx, turn_id=turn_id)
    rewrite = {"query": rq, "used": bool(retry), "trigger": "miss"}
    if retry:
        return retry, rq, rewrite
    return hits, transcript, rewrite


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
               fallback=True, k=None, floor=None, plain=None, history=None):
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

    `history` is a `src.history.History` for this session, or None for a caller with no session
    (every test written before VOX-034, and `scripts/compare_arms.py`, which must keep timing the
    pre-history pipeline). It does two things and no third: on a retrieval **miss** it supplies the
    antecedent for one rewritten retry, and on the plain path it is passed through as prior
    messages. It never reaches the grounded prompt — `answer.messages()` is single-shot and stays
    so, because the numeric guard only ever sees `hits` and a previous answer in the prompt would be
    a figure it cannot check. `history=None` is byte-for-byte the pre-VOX-034 path.

    `plain` replaces what the un-retrieved path calls, with the same signature as `nlu.reply` and
    the same job: transcript in, spoken text out. It exists because VOX-019 gave the live loop a
    second thing to want from that call — the structured TurnState that VOX-020's confirmation gate
    reads — and the alternative was either a second LLM call per turn or a copy of this routing
    inside `src/loop.py`. The routing itself is not negotiable by a caller: what retrieval vouched
    for still goes to the grounded prompt, whatever `plain` is.
    """
    hits, asked, rewrite = [], transcript, None
    if idx is not None:
        with _timing(turn, "retrieval"):
            hits, asked, rewrite = retrieve_with_history(
                transcript, turn_id, idx=idx, history=history, k=k, floor=floor)
        if turn is not None and rewrite is not None:
            turn.extra["query_rewritten"] = rewrite["used"]
            turn.extra["rewritten_query"] = rewrite["query"]

    with _timing(turn, "llm"):
        if hits:
            got = answer(asked, turn_id, hits=hits, model_id=model_id,
                         on_fallback=on_fallback, fallback=fallback)
            text = got.text
        else:
            # Nothing cleared the floor, or there is no corpus at all. Either way there is nothing
            # to be grounded in, so the turn behaves as it did before this ticket existed.
            got = None
            text = (plain or nlu.reply)(transcript, turn_id, model_id=model_id,
                                        on_fallback=on_fallback, fallback=fallback,
                                        history=history)

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
