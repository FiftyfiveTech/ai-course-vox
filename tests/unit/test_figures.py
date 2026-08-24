"""Computed figures: the safe evaluator, operand tracing, and the guard that still applies (VOX-034).

No model is called. Everything here is the part of the figure path that has to be right regardless of
what the extractor returns — which is the whole point of moving the arithmetic out of the model.

The tests that matter most are the ones asserting a figure is NOT produced. A computed figure that is
wrong is a person told the wrong number about their own pay, so every way the derivation can be
incomplete has a test saying it produces nothing.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src import answer as answer_mod, figures                      # noqa: E402
from src.figures import UnsafeExpression, safe_eval                # noqa: E402


class _Hit:
    def __init__(self, text, doc_id="leave-policy", page=7, chunk_idx=7):
        self.text, self.doc_id, self.page, self.chunk_idx = text, doc_id, page, chunk_idx
        self.score = 0.5

    @property
    def source(self):
        return f"{self.doc_id}:p{self.page}"


FORMULA = _Hit("The encashment formula is followed as mentioned below: Encashment amount = "
               "(last drawn basic salary of the last calendar year /number of days within a year) "
               "* eligible leave balance.")


# --- the safe evaluator -------------------------------------------------------------------------

def test_the_four_operators_and_parentheses_work():
    assert safe_eval("(a / b) * c", {"a": 365000, "b": 365, "c": 10}) == 10000.0
    assert safe_eval("a - b", {"a": 36, "b": 24}) == 12.0
    assert safe_eval("(1 * m) - t", {"m": 8, "t": 2}) == 6.0
    assert safe_eval("-a + b", {"a": 1, "b": 5}) == 4.0


@pytest.mark.parametrize("expr", [
    '__import__("os").system("echo hi")',
    'open("/etc/passwd").read()',
    "a.__class__",
    "a[0]",
    "[x for x in (1, 2)]",
    "a if a else b",
    "a == b",
    "a and b",
    "lambda: 1",
    "a ** b",
])
def test_anything_outside_the_whitelist_raises(expr):
    """The expression is a string from a remote service, so it is untrusted input.

    `**` is in this list as well as the dangerous ones: no formula in this corpus needs it, and an
    unbounded exponent is a denial of service on a turn with a latency budget.
    """
    with pytest.raises(UnsafeExpression):
        safe_eval(expr, {"a": 2, "b": 3})


def test_an_unknown_name_raises_rather_than_defaulting_to_zero():
    """Defaulting would silently produce a number from an operand nobody supplied."""
    with pytest.raises(UnsafeExpression):
        safe_eval("a * days_in_year", {"a": 1})


def test_division_by_zero_is_a_failure_and_not_an_exception_that_escapes():
    with pytest.raises(UnsafeExpression):
        safe_eval("a / b", {"a": 1, "b": 0})


def test_nonsense_is_not_an_expression():
    with pytest.raises(UnsafeExpression):
        safe_eval("courier_cost <= budget", {"courier_cost": 1800, "budget": 1500})


# --- what counts as a figure question -----------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("how much will my leave encashment come to", False),
    ("what is the notice period when I resign", False),
    ("I have twelve privilege leaves left, how much encashment will I get", True),
    ("my basic salary is 600000", True),
    ("I joined in January and have taken two casual leaves", True),
])
def test_only_a_question_stating_a_number_reaches_the_figure_path(text, expected):
    """The cheap router. A question with no number cannot have all its operands supplied, so it
    cannot compute — and every query in evals/dev/pdf_queries.json states none, which is why
    `make gate-poc` never enters this branch."""
    assert figures.states_a_number(text) is expected


# --- tracing ------------------------------------------------------------------------------------

def test_an_operand_must_trace_to_the_source_the_model_claimed():
    """The union of both sources was too weak, and the gate caught it.

    Asked about leaves taken by June, the extractor returned months_accrued=5 sourced "person" — a
    number nobody said, which traced anyway because 5 appears in the excerpts as the leaves Ram takes
    in the worked example. It then computed 12 * 5 - 3 = 57 with every operand "traced".
    """
    hits = [_Hit("Ram joined on 1st January and avails 5 leaves till September. 1*9 = 9")]
    ops = [{"name": "months", "value": 5, "source": "person"}]
    assert figures.untraced(ops, hits, "I have taken three leaves, what about June") == ["months"]
    # The same value, honestly sourced, traces.
    ops = [{"name": "leaves_in_example", "value": 5, "source": "excerpt"}]
    assert figures.untraced(ops, hits, "anything") == []


def test_a_number_the_person_said_traces_however_they_said_it():
    ops = [{"name": "basic", "value": 365000, "source": "person"}]
    said = "my last drawn basic salary was three hundred and sixty five thousand"
    assert figures.untraced(ops, [], said) == []


def test_an_operand_sourced_as_missing_never_traces():
    """`days_in_year` is deliberately excluded here — that one we supply ourselves; see
    test_a_days_in_year_operand_is_overridden_whatever_the_model_put_there. Every other unsourced
    operand still kills the derivation, which is the case g05/g07 score."""
    ops = [{"name": "eligible_balance", "value": None, "source": "missing"}]
    assert figures.untraced(ops, [FORMULA], "anything") == ["eligible_balance"]
    ops = [{"name": "last_drawn_basic", "value": None, "source": "missing"}]
    assert figures.untraced(ops, [FORMULA], "my balance is ten") == ["last_drawn_basic"]


def test_days_in_year_is_the_one_constant_that_may_be_assumed():
    """DECISION REVERSED. g09 first refused because 365 is not in the corpus; it now computes.

    config.DAYS_IN_YEAR carries the reversal and what it costs. The test is here rather than only in
    the gate because the allowlist is the thing a future change is most likely to widen by accident.
    """
    ops = [{"name": "days_in_year", "value": 365, "source": "constant"}]
    assert figures.untraced(ops, [FORMULA], "my basic was six hundred thousand") == []
    # The claimed source is irrelevant — the allowlist is what permits it, not the model's label.
    ops = [{"name": "number_of_days_in_year", "value": 365, "source": "excerpt"}]
    assert figures.untraced(ops, [FORMULA], "anything") == []


@pytest.mark.parametrize("name,value", [
    ("last_drawn_basic", 365),        # the number alone earns nothing
    ("working_days_in_year", 250),    # a different constant the model might know
    ("business_days_in_year", 365),   # nor does looking almost like the allowlisted name
    ("tax_rate", 30),
    ("months_elapsed", 6),
])
def test_no_other_constant_is_supplied(name, value):
    """The boundary. "Which constants" must not quietly become "any constant".

    `working_days_in_year` is the case that matters and the one the first implementation got wrong:
    it tested `"day" in name and "year" in name`, which looks careful and would have handed 365 to an
    operand counting working days — about 250. A wrong denominator inside a currency figure is
    precisely what this module exists to prevent, so the accepted spellings are enumerated.
    """
    assert figures.constant_for(name) is None
    ops = [{"name": name, "value": value, "source": "constant"}]
    assert figures.untraced(ops, [FORMULA], "no numbers here") == [name]


@pytest.mark.parametrize("value", [None, 360, 366, "unknown"])
def test_a_days_in_year_operand_is_overridden_whatever_the_model_put_there(value):
    """We supply this number; the model's value for it is never consulted.

    The bug this pins: the first version only accepted a days-in-year the MODEL valued at 365, so a
    careful extractor reporting {"value": null, "source": "missing"} had its whole derivation thrown
    away for want of a number config already held. A live turn refused for exactly that reason.
    """
    ops = [{"name": "days_in_year", "value": value, "source": "missing"}]
    assert figures.untraced(ops, [FORMULA], "no numbers here") == []
    assert figures.bind_constants(ops, "") == {"days_in_year": 365.0}


def test_a_constant_is_bound_from_a_bare_name_in_the_expression():
    """The extractor sometimes uses the name without listing it as an operand at all."""
    got = figures.bind_constants([{"name": "basic", "value": 1}],
                                 "(basic / number_of_days_in_year) * balance")
    assert got == {"number_of_days_in_year": 365.0}


def test_the_constant_is_configurable_so_a_leap_year_is_a_flag_not_an_edit(monkeypatch):
    monkeypatch.setattr(figures, "DAYS_IN_YEAR", 366.0)
    assert figures.allowed_constant("days_in_year", 366) is True
    assert figures.allowed_constant("days_in_year", 365) is False


# --- month spans --------------------------------------------------------------------------------

def test_a_month_span_is_a_parse_of_what_was_said():
    """January to June is six months of accrual, inclusive, and six is not a constant the model knew."""
    assert 6.0 in figures.months_said("I joined in January, how many by June")
    assert 8.0 in figures.months_said("joined January, what is my balance in August")


def test_month_parsing_cannot_manufacture_a_days_in_year():
    """The line this widening must not cross — g09 still refuses."""
    said = "my last drawn basic salary was six hundred thousand and I have ten privilege leaves"
    assert figures.months_said(said) == set()
    assert 365.0 not in figures.months_said("I joined in January and it is now December")


# --- the guard still applies to the figure path --------------------------------------------------

def test_the_number_word_parser_handles_a_spoken_salary():
    """A pre-existing bug in VOX-031's guard, found by the figure gate.

    "hundred" is a multiplier inside a group, not a scale that closes one. Treated as a closing
    scale, "six hundred thousand" parsed as (6*100) + (1*1000) = 1600 — so the guard was checking
    spoken currency against numbers nobody said, in both directions.
    """
    assert 600000.0 in answer_mod.numbers_in("six hundred thousand")
    assert 365000.0 in answer_mod.numbers_in("three hundred and sixty five thousand")
    assert 150000.0 in answer_mod.numbers_in("one lakh fifty thousand")
    assert 25000.0 in answer_mod.numbers_in("twenty-five thousand")
    assert answer_mod.numbers_in("two hundred") == {200.0}


def test_the_spoken_sentence_carries_the_computed_value():
    fig = figures.Figure(10000.0, "The encashment is basic over days times balance.",
                         "(a/b)*c", [], [], {})
    said = fig.spoken()
    assert "10000" in said and said.startswith("The encashment is")


def test_a_figure_that_was_not_computed_speaks_only_the_rule():
    fig = figures.Figure(None, "The encashment is basic over days times balance.", None, [],
                         ["days_in_year"], {})
    assert fig.spoken() == "The encashment is basic over days times balance."
    assert not fig.computed


def test_an_integral_value_is_not_spoken_with_decimals():
    """A TTS voice reads this aloud; "ten thousand point zero" is not what a person says."""
    fig = figures.Figure(10000.0, "Rule.", "(a/b)*c", [], [], {})
    assert "comes to 10000." in fig.spoken(), "the integer, then the sentence's full stop"
    assert "10000.0" not in fig.spoken()
    # A genuinely fractional result keeps its decimals rather than being silently rounded to an int.
    assert "comes to 1234.5" in figures.Figure(1234.5, "Rule.", "e", [], [], {}).spoken()
