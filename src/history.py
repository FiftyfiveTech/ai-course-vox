"""Session-scoped conversation history (VOX-034).

Everything before this ticket treated a turn as the unit of work: `nlu.messages()` is
`[system, transcript]`, `answer.messages()` is one user message, `state.build()` sees one
transcript. That was not an oversight — a stateless turn has no history to be wrong about, which is
what made VOX-031's numeric guard and VOX-033's gate reproducible. This module spends that property
on purpose, and the shape of it is chosen to spend as little as possible.

**A follow-up fails at retrieval first.** *"And for paternity?"* after a question about casual leave
tokenizes to `["paternity"]` — one term for BM25, a two-word fragment for the encoder. It misses
`RETRIEVAL_SCORE_FLOOR`, drops to the plain-reply path, and that path has no history either. Two
failures, one cause, and the upstream one is retrieval. So the primary thing history does here is
build a *better retrieval query*, not a longer prompt.

Four decisions worth knowing before reading the code.

**The rewrite is triggered by a miss, not applied to every turn.** `retrieval_query()` exists to be
called after the raw transcript has already failed to clear the floor. That makes it self-limiting:
it can only ever rescue a turn that was already going to be answered from the plain prompt, so it
cannot change the answer to a question that already worked. That property is what lets
`make gate-poc` stand as a no-regression check instead of needing to be re-measured. The alternative
— rewriting unconditionally — costs a second model call per turn (measured 420–480 ms `t_llm` on the
NIM arm) against a retrieval stage whose median is 41.5 ms, and re-opens every VOX-031/032/033
number.

**The rewrite uses previous QUESTIONS and never previous ANSWERS.** An answer is policy prose: put
it in the retrieval query and its terms drag the ranking back toward whatever was already cited,
which is precisely wrong for a follow-up that has moved on. A question is what the person wanted.
`Turn.reply` is kept on the record anyway, because the plain path and the trick-question work both
need to know what was said — but `retrieval_query()` does not read it.

**Nothing here reaches the grounded prompt.** `answer.messages()` stays single-shot. The numeric
guard is a set difference against the excerpts retrieved for *this* question; put the previous
turn's answer in the prompt and every figure in it becomes ambient context the guard cannot see.
That is the one design line this ticket must not cross, and it is enforced by omission — this module
offers no method that returns anything shaped like a grounded prompt.

**The window is small on purpose.** `config.HISTORY_TURNS` is 3. A reference in speech reaches back
one or two turns; a query rewritten from a topic three turns dead is worse than no rewrite, because
it manufactures a plausible link where there is none. `evals/dev/followup_queries.json` scores
exactly that failure with its `absent_followup` group.
"""
import re
from collections import deque

from src.config import HISTORY_TURNS

# Discourse markers that open a fragment rather than a question. A turn starting with one of these
# is continuing the previous turn by construction — that is what the words are for.
_OPENERS = (
    "and ", "also ", "but ", "so ", "then ", "what about", "how about", "and what about",
    "what if", "ok and ", "okay and ", "plus ",
)

# Anaphors: words that stand in for something said earlier. `retrieval.tokenize()` drops all of
# these as stopwords, which is exactly why they have to be looked for in the RAW text — by the time
# the query reaches BM25 the evidence that it was referential has been thrown away.
#
# Deliberately excludes "this"/"that", which appear in plenty of self-contained questions ("what is
# the policy on this year's holidays"). The set is meant to be precise rather than complete: a
# missed ellipsis costs one unrescued turn, a false positive rewrites a question that was fine.
_ANAPHORS = frozenset("it its them they those these ones".split())

_WORDS = re.compile(r"[a-z']+")


def elliptical(text):
    """-> True if `text` reads as a continuation of an earlier turn rather than a whole question.

    Two signals, both measured against evals/dev/followup_queries.json rather than guessed:

      an opener   "and privilege leaves", "what about during a performance improvement plan"
      an anaphor  "how far in advance do I have to plan it", "when is it paid out"

    NOT a term count. The obvious rule — few content words means elliptical — was tried first and
    misfires: 'what is the notice period when I resign' tokenizes to three terms and is a complete
    question, so a count threshold rewrites something that needed nothing. The two signals above
    separate all ten dev cases correctly, and both are properties of the words the person chose
    rather than of how many survived stopword removal.
    """
    t = (text or "").strip().lower()
    if not t:
        return False
    if t.startswith(_OPENERS):
        return True
    return any(w in _ANAPHORS for w in _WORDS.findall(t))


class Turn:
    """One completed exchange, as history remembers it.

    Deliberately not the telemetry record: that one carries latencies and arm ids and is written to
    `runs/turns.jsonl` for measurement. This is the conversational content only, and the two have
    different lifetimes — a turn record is appended and forgotten, a Turn here is read by the next
    turn and then ages out of the window.

    `sources` is the citation labels the answer was grounded in, or an empty list for a plain reply
    or a refusal. Carried because "which document were we just talking about" is a question the
    trick-question work needs to be able to ask, and re-deriving it from the reply text is a parser
    nobody should write.
    """

    __slots__ = ("transcript", "reply", "sources", "grounded")

    def __init__(self, transcript, reply, sources=(), grounded=False):
        self.transcript = transcript or ""
        self.reply = reply or ""
        self.sources = list(sources)
        self.grounded = bool(grounded)

    def __repr__(self):
        return (f"Turn({self.transcript[:32]!r} -> {self.reply[:32]!r}, "
                f"grounded={self.grounded}, sources={self.sources})")


class History:
    """The last `maxlen` turns of one session. Bounded, ordered oldest to newest.

    One instance per session, owned by whatever runs the loop — `loop.main()` for a live run, the
    caller for a scripted one. Not a module-level singleton: two sessions in one process (the test
    suite, `scripts/compare_arms.py`) must not see each other's turns, and a global would make that
    a bug that only appears in the second test.
    """

    def __init__(self, maxlen=None, enabled=True):
        self.turns = deque(maxlen=HISTORY_TURNS if maxlen is None else maxlen)
        self.enabled = enabled

    def __len__(self):
        return len(self.turns)

    def __bool__(self):
        """Truthy only when it can actually contribute — empty or disabled is falsey.

        So `if history:` at a call site means "there is context to use", which is the question every
        caller actually has. `history is not None` would be true for a fresh session that has
        nothing to offer yet.
        """
        return bool(self.enabled and self.turns)

    def __iter__(self):
        return iter(self.turns)

    def add(self, transcript, reply, sources=(), grounded=False):
        """Record one completed exchange. -> the Turn, for a caller that wants to inspect it.

        Called after the reply has been produced, not before, and deliberately not called at all for
        a turn that failed: an empty transcript from STT or a turn that raised has no conversational
        content, and remembering it would let a broken turn poison the next query.
        """
        turn = Turn(transcript, reply, sources, grounded)
        self.turns.append(turn)
        return turn

    @property
    def last(self):
        """-> the most recent Turn, or None."""
        return self.turns[-1] if self.turns else None

    def questions(self, n=None):
        """-> the previous transcripts, oldest first, most recent last. Never the replies."""
        qs = [t.transcript for t in self.turns if t.transcript.strip()]
        return qs if n is None else qs[-n:]

    def retrieval_query(self, transcript, n=1):
        """-> a query for the retrieval retry: the last `n` questions, then this transcript.

        Called only after `transcript` alone has already missed the floor — see the module
        docstring on why that trigger is the whole safety argument for this function.

        Concatenation rather than a model rewrite. It is not elegant, and it is what BM25 and a
        sentence encoder both actually want: `tokenize()` unions the term sets, so the antecedent's
        high-IDF terms ("paternity", "encashment") re-enter the query, and the encoder sees a fuller
        sentence to embed than a two-word fragment. An LLM rewrite would produce a cleaner string
        and cost a second model call per rescued turn; that is the next thing to try if the measured
        column says this is not enough, and `evals/dev/followup_queries.json` is what would say so.

        Order is antecedent-first, current-last. The current question is the one being answered, and
        putting it last keeps it adjacent to nothing — this string is never shown to a model on the
        grounded path, so the ordering only matters for the encoder, which is not
        position-invariant.

        Returns `transcript` unchanged when there is nothing to add, so a caller can use the result
        without checking whether a retry is even possible.
        """
        if not self:
            return transcript
        prior = self.questions(n)
        if not prior:
            return transcript
        return " ".join(prior + [transcript or ""]).strip()

    def messages_prefix(self):
        """-> prior turns as OpenAI-shaped messages, for the PLAIN reply path only.

        Never for the grounded path. `answer.messages()` does not call this and must not: see the
        module docstring on why a previous answer inside the grounded prompt is a hole in the
        numeric guard rather than a nicety.

        Assistant turns are included here — unlike in `retrieval_query()` — because on the plain
        path the thing being fixed is conversational coherence, and a transcript of questions with
        no answers reads to a model as a person repeating themselves.
        """
        if not self:
            return []
        msgs = []
        for t in self.turns:
            if t.transcript.strip():
                msgs.append({"role": "user", "content": t.transcript})
            if t.reply.strip():
                msgs.append({"role": "assistant", "content": t.reply})
        return msgs

    def elliptical(self, transcript):
        """-> True if `transcript` needs this history to be a whole question.

        A method as well as a module function so a caller holding a History can ask the question
        without importing anything, and so an empty or disabled history answers False — a fragment
        with nothing to refer back to is not resolvable, and rewriting it against nothing would
        produce the fragment again.
        """
        return bool(self) and elliptical(transcript)

    def clear(self):
        """Forget everything. For a session boundary inside one process."""
        self.turns.clear()
