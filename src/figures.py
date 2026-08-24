"""Computing a figure from a formula the documents state (VOX-034 part B).

    compute(transcript, hits, turn_id) -> Figure(value, rule, expression, operands, missing)

`prompts/answer_from_source_v2.md` forbids arithmetic outright, and that rule is the residue of a
measurement: asked *"base pay of 10,000 and PL balance of 12, how much would my leave encashment
be"*, v1 answered *"you will be paid 12,000"* — cited, fluent, and not a number any reading of the
formula produces. Every *fact* it used came from a document; the *number it produced from them* did
not, and grounding had nothing to say about the difference.

So this module does not relax that rule. It replaces the enforcement.

**The model never does the arithmetic.** It is asked for the formula, its operands and an
expression, as JSON — extraction, which these arms are reliable at — and Python evaluates it. An
LLM's arithmetic is *sampled*, not computed; `answer.ANSWER_TEMPERATURE` is 0.0 because the same
question at 0.3 invented a figure one run in three, and determinism makes a wrong answer
reproducible without making it right.

**Every operand must trace, or nothing is computed.** An operand traces if its value appears in the
retrieved excerpts or in what the person themselves said. Anything else — a constant the model knew,
a number it inferred — means the derivation is incomplete and `compute()` returns no value. This is
the existing numeric guard narrowed rather than widened: `src/answer.py` holds that a number the
person supplied is not grounded *by having been asked*, and that stays true — a person's number is
legitimate only as an operand inside a checked derivation, never as an answer on its own.

**Exactly one constant is inferred, and it was a reversal.** The encashment formula in this corpus
divides by "number of days within a year" and the corpus never says what that number is
(`leave-policy:p7` #7), so under the rule above every encashment question refused unless the person
volunteered the figure. VOX-034 first decided to keep that refusal; the decision was then reversed —
assume 365. `config.DAYS_IN_YEAR` carries the reversal and what it costs, and
`evals/dev/figure_queries.json` records both decisions in order.

What it costs, stated here because nothing warns about it at run time: 365 is not in the documents,
so a figure computed with it carries an assumption the listener is never told about, and the
assumption is wrong one year in four. `allowed_constant()` is the boundary that keeps this from
spreading — it matches on the operand's NAME, so 365 traces as a days-in-year and nowhere else, and
every other constant the model might supply from general knowledge still fails to trace. That is what
stops "which constants" from quietly becoming "any constant". A second entry belongs on a ticket.

**`eval()` is not used.** The expression comes from a remote service, so it is untrusted input. It is
parsed and walked with an explicit whitelist of node types and operators; anything else raises.
"""
import ast
import json
import operator
import re
from collections import namedtuple

import httpx

from src import errors
from src.config import DAYS_IN_YEAR, FORMULA_OVERLAP, PROMPTS_DIR
from src.telemetry import log_call

PROMPT_FILE = PROMPTS_DIR / "compute_figure_v2.md"

# Extraction, not composition: temperature 0 for the same reason state.py uses it. There is nothing
# for variety to buy when the task is "name the operands".
TEMPERATURE = 0.0
MAX_TOKENS = 512

# Rounding for a currency-shaped result. Two decimals, and applied only at the end so intermediate
# division is not truncated. A result that is integral prints as an integer — "ten thousand", not
# "ten thousand point zero zero" — because a TTS voice reads this aloud.
ROUND_TO = 2


class Figure(namedtuple("Figure", "value rule expression operands missing raw")):
    """One attempt at a computed figure.

    value       the computed number, or None when nothing could be computed. None is the common
                case and is not an error: it means "state the rule instead", which is what
                prompts/answer_from_source_v2.md already asks for.
    rule        the rule in one sentence, as the model read it out of the excerpts. Spoken whether
                or not a value was computed, so a refusal to compute is still a useful answer.
    expression  the arithmetic Python evaluated, for the turn record. None when nothing ran.
    operands    [{name, value, source}] as extracted, before tracing.
    missing     operand names that did not trace to an excerpt or to the person's words. Non-empty
                means `value` is None by construction.
    raw         the model's JSON, kept so a wrong answer can be read afterwards rather than guessed
                at.
    """

    __slots__ = ()

    @property
    def computed(self):
        return self.value is not None

    def spoken(self):
        """-> what to say. The rule alone when nothing was computed, the rule then the figure when
        something was.

        Composed in Python rather than asked for, so the number in the sentence is definitely the
        number Python calculated. A model asked to restate its own computed figure is a model that
        can restate it wrong.
        """
        rule = (self.rule or "").strip()
        if not self.computed:
            return rule
        v = self.value
        num = f"{int(v)}" if float(v).is_integer() else f"{v}"
        joiner = " " if rule.endswith((".", "!", "?")) else ". "
        return f"{rule}{joiner}For the numbers you gave, that comes to {num}."


# --- is this even a figure question? ------------------------------------------------------------
# Cheap gate in front of the extraction call, so an ordinary policy question costs nothing extra.
# A question that states no number cannot have all its operands supplied, so it cannot compute —
# `evals/dev/figure_queries.json` case g06 is exactly that and its expected shape is "state the
# rule", which is the path this skips to.

_DIGITS = re.compile(r"\d")
_NUMBER_WORDS = frozenset("""
zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen
sixteen seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety
hundred thousand lakh lakhs million crore crores
""".split())
_WORD = re.compile(r"[a-z]+")


def states_a_number(text):
    """-> True if `text` asserts a number, in digits or in words.

    Word numbers count because the person is speaking: STT returns "three hundred and sixty five
    thousand", not "365000". `src/answer.py`'s `numbers_in()` already parses that form; this is only
    the cheaper question of whether any number is present at all.
    """
    t = (text or "").lower()
    if _DIGITS.search(t):
        return True
    return any(w in _NUMBER_WORDS for w in _WORD.findall(t))


# --- the safe evaluator -------------------------------------------------------------------------
# An explicit whitelist, because the expression is a string from a remote service. Division is
# included and division by zero is a failure rather than an exception that escapes: a formula whose
# denominator is zero has not been understood, and the right answer is to state the rule.

_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
_UNARYOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


class UnsafeExpression(Exception):
    """The expression used something outside the whitelist. Never surfaced to a listener."""


def safe_eval(expression, names):
    """-> the value of `expression` with `names` bound. Raises UnsafeExpression for anything else.

    Whitelisted: numeric literals, the four arithmetic operators, unary plus/minus, parentheses, and
    bare names present in `names`. Everything else — calls, attributes, subscripts, comprehensions,
    comparisons, `**` — raises. `**` is excluded deliberately as well as safely: no formula in this
    corpus needs it, and an unbounded exponent is a denial of service on a turn budget.
    """
    try:
        tree = ast.parse((expression or "").strip(), mode="eval")
    except SyntaxError as e:
        raise UnsafeExpression(f"not a parseable expression: {e}") from e

    def walk(node):
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise UnsafeExpression(f"non-numeric constant {node.value!r}")
            return float(node.value)
        if isinstance(node, ast.Name):
            if node.id not in names:
                raise UnsafeExpression(f"unknown name {node.id!r}")
            return float(names[node.id])
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            left, right = walk(node.left), walk(node.right)
            if isinstance(node.op, ast.Div) and right == 0:
                raise UnsafeExpression("division by zero")
            return _BINOPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARYOPS:
            return _UNARYOPS[type(node.op)](walk(node.operand))
        raise UnsafeExpression(f"{type(node).__name__} is not allowed in a formula")

    return walk(tree)


# --- tracing ------------------------------------------------------------------------------------

# The allowlist of inferred constants: operand-name pattern -> value. Exactly one entry, and the
# narrowness is the design. `days_in_year` is the operand leave-policy:p7's formula names and never
# values, so without this the encashment questions refuse; see config.DAYS_IN_YEAR for the decision
# and what it costs. A second entry here should be argued for on a ticket, not added in passing —
# every constant added is a number a listener is told without being told it was assumed.
# Names spelled out rather than matched by a predicate. The first version tested `"day" in n and
# "year" in n`, which is the kind of rule that looks careful and is not: `working_days_in_year` matches
# it and would have been handed 365, when working days in a year is about 250. A wrong denominator in
# a currency figure is precisely the failure this whole module exists to prevent, so the accepted
# spellings are enumerated and anything else gets nothing.
_DAYS_IN_YEAR_NAMES = frozenset((
    "days in year", "days in a year", "days within year", "days within a year",
    "number of days in year", "number of days in a year",
    "number of days within year", "number of days within a year",
    "calendar days in year", "calendar days in a year",
    "total days in year", "total days in a year", "days per year", "days in the year",
))

_CONSTANTS = (
    (lambda n: n in _DAYS_IN_YEAR_NAMES, lambda: DAYS_IN_YEAR),
)


# Phrases a rule uses when it is stating a calculation rather than a limit. Used only to catch a
# rule sentence that describes arithmetic the excerpts do not contain — see the comment in compute().
# Deliberately about OPERATIONS ("divided by", "multiplied by") and not about quantities: "up to
# 1500", "capped at 24 days" and "two days per week" are limits, and a limit read off an excerpt is a
# perfectly good answer.
_ARITHMETIC_PHRASES = (
    "divided by", "multiplied by", "times your", "times the", "minus the", "minus your",
    "plus the", "plus your", "subtracted from", "divided into", "pro-rated by", "prorated by",
)


def describes_arithmetic(text):
    """-> True if `text` reads as a formula rather than a rule or a limit."""
    low = " ".join((text or "").split()).lower()
    return any(p in low for p in _ARITHMETIC_PHRASES)


def formula_grounded(formula, hits, threshold=None):
    """-> True if the model's quoted `formula` is really in the excerpts. The check one level up.

    Operand tracing verifies where each NUMBER came from and says nothing about whether the SUM is
    the one the documents state. That gap produced the worst answer this module has given. Asked
    "eligible leave balance is 32 and my basic salary is 10,000, how much will I get" on a turn where
    retrieval returned the accrual tables instead of leave-policy:p7, it invented
    "(eligible_balance - 24) * basic_salary", computed 80000 and said it out loud. Every operand
    traced — 32 and 10000 from the person, 24 from an excerpt — so nothing objected. The real formula
    gives 876.71.

    So the quote is verified like any other provenance claim: the distinctive words of the formula
    have to appear in the text the model was given. Token overlap rather than substring, because the
    model reformats what it quotes (whitespace, "/" for "divided by", dropped articles) and a
    substring test would reject every honest quote. Stopwords are dropped for the same reason they
    are dropped in retrieval — "the" matching proves nothing.

    A high bar on purpose. A formula is a short, highly specific string, so an honest quote scores
    close to 1.0 and an invention scores well under it; the cost of being wrong here is a confident
    wrong number about someone's pay.
    """
    from src.retrieval import tokenize          # the same stopword list retrieval scores with

    threshold = FORMULA_OVERLAP if threshold is None else threshold
    terms = set(tokenize(formula))
    if not terms:
        return False
    corpus = set()
    for h in hits or ():
        corpus |= set(tokenize(h.text))
    return (len(terms & corpus) / len(terms)) >= threshold


def constant_for(name):
    """-> the constant this repo supplies for an operand called `name`, or None.

    THE ASSUMPTION IS OURS TO MAKE, not the model's to guess, and getting that backwards was a real
    bug. The first version only accepted a days-in-year if the model itself put 365 there; a careful
    extractor instead reports `{"value": null, "source": "missing"}` — which is what v1's prompt
    trained it to do — and the figure was then discarded for want of a number the config already
    held. A live turn asking "eligible leave balance is 32 and my basic salary is 10,000, how much
    will I get" refused with `days_in_year` missing for exactly that reason.
    """
    n = re.sub(r"[^a-z]+", " ", (name or "").lower())
    for matches, const in _CONSTANTS:
        if matches(n):
            return const()
    return None


def allowed_constant(name, value):
    """-> True if `value` is the constant this repo supplies for an operand called `name`.

    Matched on the operand's NAME, not on the number, so 365 traces as a days-in-year and nowhere
    else. An extractor that labels a salary 365 gets no help from this.
    """
    const = constant_for(name)
    if const is None:
        return False
    try:
        return abs(float(value) - const) < 1e-6
    except (TypeError, ValueError):
        return False


def bind_constants(operands, expression):
    """-> {name: value} for every allowlisted constant this derivation needs. Supplied, not asked for.

    Covers both shapes the extractor produces: an operand it named but could not value, and a bare
    name it used in the expression without listing as an operand at all. Either way the number comes
    from `config.DAYS_IN_YEAR` and never from the model, so the model cannot fail to guess it and
    cannot guess it wrong.
    """
    bound = {}
    for op in operands or ():
        name = op.get("name")
        const = constant_for(name)
        if const is not None:
            bound[name] = const
    for token in set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expression or "")):
        const = constant_for(token)
        if const is not None:
            bound.setdefault(token, const)
    return bound


_MONTHS = {m: i for i, m in enumerate(
    "january february march april may june july august september october november december".split(),
    start=1)}
_MONTHS.update({m[:3]: i for m, i in list(_MONTHS.items())})


def months_said(transcript):
    """-> month indices named in `transcript`, plus the inclusive span between the first and last.

    A narrow, deliberate widening of what "the person said this number" means, and it is worth being
    explicit about why it is not a loophole.

    An accrual question is "I joined in January and it is now June". The operand the formula needs is
    six, and nobody said "six" — so under a literal reading of the tracing rule every accrual
    question refuses, including `evals/dev/figure_queries.json` cases g01 and g02 which were written
    as computable before any of this existed. The alternative to this function was to move those
    cases out of `computable`, which is tuning the eval to the implementation and is exactly what the
    cases-first commit exists to prevent.

    Six *is* something the person said, in the same sense that 25000 is what they said when they said
    "twenty-five thousand": it is a parse of their words, not a constant the model supplied from
    general knowledge. That is the line — this function reads the transcript and nothing else. It
    cannot invent a days-in-year, because the person never named two dates a year apart; g09 still
    refuses, which is the case that proves the rule still bites.

    Inclusive, because "joined in January, how many by June" is six months of accrual and not five.
    """
    t = (transcript or "").lower()
    found = sorted({i for m, i in _MONTHS.items() if re.search(rf"\b{m}\b", t)})
    if not found:
        return set()
    out = {float(i) for i in found}
    if len(found) > 1:
        out.add(float(found[-1] - found[0] + 1))
    return out


def untraced(operands, hits, transcript):
    """-> the names of operands that do not trace to the source the model claimed for them.

    **Checked against the DECLARED source, not against the union of both.** Unioning was tried first
    and is too weak, and the gate caught it: asked "I joined on the first of January and I have taken
    three casual leaves, how many do I have left in June", the model returned
    `months_accrued = 5, source "person"` — a number nobody said, which traced anyway because 5
    appears in the excerpts (it is the number of leaves Ram takes in the worked example). It then
    computed 12 * 5 - 3 = 57 and every operand "traced".

    That is the same coarseness the false-premise finding is about: value membership in a bundle says
    nothing about whether the number is attached to the fact being asserted. The model already tells
    us where it thinks each number came from, so the cheapest real check is to hold it to that claim.

    `parts=True` on both sides, for the reason `answer.numbers_in` documents: be generous about how a
    number is spelled, strict about whether it is there.
    """
    from src.answer import numbers_in          # local: answer imports this module

    in_excerpts = set()
    for h in hits or ():
        in_excerpts |= numbers_in(h.text, parts=True)
    said = numbers_in(transcript, parts=True) | months_said(transcript)

    missing = []
    for op in operands or ():
        name = op.get("name") or "?"
        # Ours to supply, so it is never missing however the model valued it (or failed to).
        if constant_for(name) is not None:
            continue
        try:
            v = float(op.get("value"))
        except (TypeError, ValueError):
            missing.append(name)
            continue
        source = (op.get("source") or "").strip().lower()
        # The one inferred constant, checked before the declared source is consulted: the model may
        # label days_in_year "excerpt" (it is named in the formula) or "constant", and neither claim
        # is what makes it acceptable — the allowlist is.
        if allowed_constant(name, v):
            continue
        # A value the PERSON demonstrably said traces whatever the model labelled it. The claimed
        # source is only load-bearing in the other direction.
        #
        # Measured, both directions. The label check exists because the extractor returned
        # months_accrued = 5 claiming "person" for a question that never said five — it traced under
        # a plain union because 5 is in the excerpts as Ram's worked example, and produced
        # 12*5-3 = 57. But holding the label strictly then broke the same case the other way: on a
        # later run the extractor labelled months_accrued = 6 "excerpt", and six is a month span the
        # person named, so a correct derivation was thrown away over a mislabel.
        #
        # The asymmetry resolves both. What the person said is independently checkable, so a wrong
        # label about it costs nothing. A number found only in the EXCERPTS still has to be claimed
        # as an excerpt value — otherwise "person" becomes a way to launder any figure that happens
        # to appear somewhere in the bundle, which is exactly the 57.
        if any(abs(v - k) < 1e-6 for k in said):
            continue
        if source == "excerpt":
            known = in_excerpts
        elif source == "person":
            known = said               # already checked above, so this fails and says why
        else:
            missing.append(name)       # "missing", or a source it did not name
            continue
        # A tolerance, not equality: numbers_in returns floats, and 365000.0 is 365000.
        if not any(abs(v - k) < 1e-6 for k in known):
            missing.append(name)
    return missing


# --- the extraction call ------------------------------------------------------------------------


def _system_prompt():
    from src import nlu                        # load_prompt strips the YAML front matter
    return nlu.load_prompt(PROMPT_FILE)


def extract(transcript, hits, turn_id, model_id=None, fallback=True, on_fallback=None):
    """-> the model's JSON as a dict: {rule, formula, operands, expression}. Raises on a bad call.

    One call, through `arms.llm` like every other model call in this repo, so the cost logger, the
    cooldown and the ollama fallback all apply. JSON mode, the same way `state.build()` gets it.
    """
    from src import answer as answer_mod, arms

    msgs = [
        {"role": "system", "content": _system_prompt()},
        {"role": "user", "content": (
            f"Question: {transcript}\n\n"
            f"{answer_mod.CONTEXT_HEADER}\n\n"
            f"{answer_mod.context_block(hits)}\n\n"
            f"Extract the rule and the arithmetic for: {transcript}"
        )},
    ]
    text = arms.llm(msgs, model_id, turn_id=turn_id, on_fallback=on_fallback, fallback=fallback,
                    temperature=TEMPERATURE, json_mode=True, max_tokens=MAX_TOKENS,
                    prompt_file=PROMPT_FILE.name, transcript_chars=len(transcript),
                    chunks=len(hits or ()), stage_kind="figure")
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", (text or "").strip(), flags=re.MULTILINE)
    try:
        data = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        # Not JSON. Returning None rather than raising, because the caller's next move is the same
        # either way: give up on the figure path and let the ordinary prose prompt answer. The local
        # fallback arm is a 3B and JSON mode is a request, not a guarantee — a parse failure on the
        # fallback path would otherwise land on the turn that was already going badly. `state.build`
        # raises here instead, and that difference is deliberate: a malformed TurnState has nowhere
        # to go, a malformed figure has v2.
        return None
    return data if isinstance(data, dict) else None


def compute(transcript, hits, turn_id, model_id=None, fallback=True, on_fallback=None):
    """-> Figure. Never raises for a question it cannot compute; that is what `value=None` means.

    The three ways this ends with no value, all of them correct answers rather than errors:
      the excerpts state no formula        -> the model returns none, rule is spoken alone
      an operand did not trace            -> `missing` names it (this is g05, g07, g09)
      the expression was not safe or sane -> `missing` carries the reason

    A provider failure still propagates, exactly as `answer.answer()` lets one propagate: a dead key
    is not a question about arithmetic.
    """
    data = extract(transcript, hits, turn_id, model_id=model_id, fallback=fallback,
                   on_fallback=on_fallback)
    if data is None:
        return None                     # unparseable: the caller falls through to the prose prompt

    rule = (data.get("rule") or "").strip()
    operands = data.get("operands") or []
    expression = (data.get("expression") or "").strip()

    if not expression or not operands:
        # No formula in the excerpts, or nothing to put in it. g08's courier cap lands here: a cap is
        # a rule to state, not arithmetic to do.
        #
        # But a rule sentence that DESCRIBES arithmetic when no formula was found is a formula the
        # model supplied itself, and it is spoken aloud as though the documents said it. Observed on
        # a live turn where retrieval returned the accrual tables and not leave-policy:p7: asked
        # "eligible leave balance is 32 and my basic salary is 10,000, how much will I get", it said
        # "encashment is your eligible leave balance divided by the number of days in the year,
        # multiplied by your basic salary" — the real formula with two operands swapped — and on the
        # next turn "your eligible leave balance minus 24, multiplied by your basic salary", which is
        # not a rule in any document. Fluent, cited, and invented.
        #
        # So: no formula found and a rule that talks like one means refuse. Stating the rule is only
        # an answer when the rule was actually read off an excerpt.
        if describes_arithmetic(rule):
            return Figure(None, "", None, operands,
                          ["the rule describes a calculation that is in no excerpt"], data)
        return Figure(None, rule, None, operands, ["no formula in the excerpts"], data)

    # The quote is verified before the operands are, because a formula that is not in the excerpts
    # makes the operands irrelevant — see formula_grounded() for the 80000 this exists to stop.
    if not formula_grounded(data.get("formula") or rule, hits):
        return Figure(None, "", expression, operands,
                      ["the quoted formula is not in the excerpts"], data)

    missing = untraced(operands, hits, transcript)
    if missing:
        return Figure(None, rule, expression, operands, missing, data)

    names = bind_constants(operands, expression)
    for op in operands:
        name = op.get("name")
        if not name or name in names:
            continue                    # a constant we supplied; the model's value is irrelevant
        try:
            names[name] = float(op.get("value"))
        except (TypeError, ValueError):
            return Figure(None, rule, expression, operands, [f"{name} is not a number"], data)

    try:
        value = safe_eval(expression, names)
    except UnsafeExpression as e:
        return Figure(None, rule, expression, operands, [str(e)], data)

    value = round(float(value), ROUND_TO)
    return Figure(value, rule, expression, operands, [], data)
