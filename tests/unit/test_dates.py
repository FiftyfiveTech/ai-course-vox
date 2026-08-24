"""Computed dates: the calendar arithmetic, what has to trace, and the routing gate (VOX-034 part D).

No model is called. Everything here is the part of the date path that has to be right whatever the
extractor returns, which is the reason the counting was moved out of the model in the first place.

Same posture as test_figures.py: the tests that matter most assert a date is NOT produced. A wrong
settlement date is a person planning around a number the documents do not support, and every way the
derivation can be incomplete has a test saying it yields nothing.

The expected dates here were computed independently — by hand off a 2026 calendar and cross-checked
with a throwaway script — before the module existed, so a test agreeing with the implementation is
not the implementation agreeing with itself.
"""
import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src import dates                                              # noqa: E402


class _Hit:
    def __init__(self, text, doc_id="separation-policy", page=10, chunk_idx=13):
        self.text, self.doc_id, self.page, self.chunk_idx = text, doc_id, page, chunk_idx
        self.score = 0.5

    @property
    def source(self):
        return f"{self.doc_id}:p{self.page}"


FNF = _Hit("iii) Full & Final Settlement Upon formal acceptance of resignation, the employee's "
           "salary for the notice period will be included in the full and final settlement, which "
           "shall be processed and cleared within 30–45 working days from the employee's last "
           "working day.")

# The fixed rows of holiday-list-2026:p1 #0, copied verbatim from runs/chunks.jsonl so the parser is
# tested against the string the corpus actually produces rather than a tidied version of it.
CALENDAR = _Hit(
    "No Holiday Month Date Day Remarks 1 New Year - 2025 January 1-Jan-26 Thursday Fixed 2 Republic "
    "Day January 26-Jan-26 Monday Fixed 3 Holi March 4-Mar-26 Wednesday Fixed 4 Independence Day "
    "August 15-Aug-26 Saturday Weekend 5 Mahatama Gandhi Jayanti October 2-Oct-26 Friday Fixed 6 "
    "Dusshera October 20-Oct-26 Tuesday Fixed 7 Diwali November 8-Nov-26 Sunday Weekend 8 Christmas "
    "Day December 25-Dec-26 Fri", doc_id="holiday-list-2026", page=1, chunk_idx=0)

FLOATERS = _Hit(
    "26-Aug-26 Wednesday Floater Leave 10 Raksha Bandhan August 28-Aug-26 Friday Floater Leave 11 "
    "Janmashtami Sept 4-Sep-26 Friday Floater Leave 12 Ganesh Chaturthi Sept 14-Sep-26 Monday "
    "Floater Leave 13 Mahav Navmi/ Durga Pooja October 19-Oct-26 Monday Floater Leave",
    doc_id="holiday-list-2026", page=1, chunk_idx=1)


# --- calendar arithmetic ------------------------------------------------------------------------

def test_working_days_skip_weekends_and_start_the_day_after_the_anchor():
    # 30 working days from Tuesday 18 August 2026, weekends only.
    assert dates.add_working_days(dt.date(2026, 8, 18), 30) == dt.date(2026, 9, 29)
    assert dates.add_working_days(dt.date(2026, 8, 18), 45) == dt.date(2026, 10, 20)
    # One working day after a Friday is the Monday.
    assert dates.add_working_days(dt.date(2026, 8, 21), 1) == dt.date(2026, 8, 24)


def test_working_days_also_skip_the_holidays_they_are_given():
    holidays = {dt.date(2026, 10, 2), dt.date(2026, 10, 20)}
    # The two fixed holidays inside the window push the far end out by exactly two days. The near
    # end does not move: no holiday falls within 30 working days of 18 August.
    assert dates.add_working_days(dt.date(2026, 8, 18), 45, holidays) == dt.date(2026, 10, 22)
    assert dates.add_working_days(dt.date(2026, 8, 18), 30, holidays) == dt.date(2026, 9, 29)


def test_a_holiday_that_falls_on_a_weekend_changes_nothing():
    # Independence Day 2026 is a Saturday, which is why the corpus marks that row "Weekend".
    assert (dates.add_working_days(dt.date(2026, 8, 10), 10, {dt.date(2026, 8, 15)})
            == dates.add_working_days(dt.date(2026, 8, 10), 10))


def test_months_clamp_to_the_length_of_the_target_month():
    assert dates.add_months(dt.date(2026, 1, 31), 1) == dt.date(2026, 2, 28)
    assert dates.add_months(dt.date(2026, 5, 1), 2) == dt.date(2026, 7, 1)
    assert dates.add_months(dt.date(2026, 3, 15), 3) == dt.date(2026, 6, 15)
    assert dates.add_months(dt.date(2026, 12, 15), 1) == dt.date(2027, 1, 15)


def test_years_are_months_times_twelve():
    assert dates.offset_by(dt.date(2026, 6, 15), 2, "years") == dt.date(2028, 6, 15)


def test_calendar_days_are_not_working_days():
    # The unit is worth a fortnight, which is why trace_unit demands evidence for it.
    assert dates.offset_by(dt.date(2026, 8, 18), 30, "calendar_days") == dt.date(2026, 9, 17)
    assert dates.offset_by(dt.date(2026, 8, 18), 30, "working_days") == dt.date(2026, 9, 29)


# --- reading dates out of speech ----------------------------------------------------------------

def test_a_spoken_ordinal_is_a_date():
    assert dt.date(2026, 8, 18) in dates.dates_in("my last working day is the eighteenth of August")
    assert dt.date(2026, 3, 2) in dates.dates_in("I'm joining from home on the second of March")
    assert dt.date(2026, 2, 26) in dates.dates_in("it happened on the twenty sixth of February")


def test_stt_digits_with_an_ordinal_suffix_are_a_date():
    # The live transcript form: "My last working day is 18th of August."
    assert dt.date(2026, 8, 18) in dates.dates_in("My last working day is 18th of August.")
    assert dt.date(2026, 3, 2) in dates.dates_in("I'm joining from home on 2nd of March.")


def test_a_stated_year_wins_over_the_configured_one():
    assert dt.date(2027, 8, 18) in dates.dates_in("my last day is the eighteenth of August 2027")


def test_a_date_that_does_not_exist_is_not_a_date():
    assert dates.dates_in("the thirty first of February") == set()


def test_calendar_table_dates_parse_with_their_own_two_digit_year():
    assert dt.date(2026, 10, 20) in dates.dates_in("6 Dusshera October 20-Oct-26 Tuesday Fixed")


# --- the published holiday rows -----------------------------------------------------------------

def test_fixed_and_weekend_rows_are_company_holidays():
    found = dates.published_holidays([CALENDAR])
    assert dt.date(2026, 1, 1) in found and dt.date(2026, 10, 20) in found
    assert dt.date(2026, 8, 15) in found          # labelled Weekend in the corpus
    # Seven and not eight: the chunk this fixture is copied from ends mid-row at "25-Dec-26 Fri", so
    # Christmas has no label inside it and is dropped. A known limit of reading a table that a
    # chunker split — the row is complete in a later chunk, so a turn that retrieves that one gets
    # it. An unlabelled row is never assumed to be a holiday, which is the safe direction: it makes
    # the count depend on what was retrieved, and dates.compute() says so out loud.
    assert dt.date(2026, 12, 25) not in found
    assert len(found) == 7


def test_floater_rows_are_not_company_holidays():
    # holiday-policy p4: two floaters may be availed, on application, 15 days ahead. That is leave a
    # person takes, not a day the company is shut — counting it would push every settlement date out
    # for everyone who never applied for it.
    assert dates.published_holidays([FLOATERS]) == {}


# --- what has to trace --------------------------------------------------------------------------

def test_the_anchor_has_to_be_a_date_somebody_stated():
    d, note, problem = dates.trace_anchor("2026-08-18", "my last working day is the eighteenth of "
                                          "August", [FNF])
    assert (d, problem) == (dt.date(2026, 8, 18), None)
    assert note == "I took the year as 2026."


def test_an_anchor_nobody_stated_is_refused():
    d, _, problem = dates.trace_anchor("2026-01-01", "when is my full and final settled", [FNF])
    assert d is None and "not stated" in problem


def test_the_models_year_is_never_load_bearing():
    # The extractor returned 2023-03-02 for "the second of March" on a live run and the strict check
    # refused a question the corpus answers. The day and month have to be stated; the year is ours.
    d, note, problem = dates.trace_anchor("2023-03-02", "I'm joining from home on the second of "
                                          "March", [FNF])
    assert (d, problem) == (dt.date(2026, 3, 2), None)
    assert note == "I took the year as 2026."


def test_no_year_note_when_the_person_gave_the_year():
    _, note, _ = dates.trace_anchor("2027-08-18", "my last day is the eighteenth of August 2027",
                                    [FNF])
    assert note is None


def test_a_duration_that_is_in_no_excerpt_is_refused():
    n, problem = dates.trace_offset(60, [FNF])
    assert n is None and "in no excerpt" in problem


def test_a_duration_written_in_the_excerpt_traces():
    assert dates.trace_offset(30, [FNF]) == (30.0, None)
    assert dates.trace_offset(45, [FNF]) == (45.0, None)


def test_the_unit_has_to_be_evidenced_too():
    assert dates.trace_unit("working_days", [FNF]) == ("working_days", None)
    unit, problem = dates.trace_unit("months", [FNF])
    assert unit is None and "no excerpt says" in problem


def test_an_unknown_unit_is_refused():
    unit, problem = dates.trace_unit("fortnights", [FNF])
    assert unit is None and "not one of" in problem


# --- routing ------------------------------------------------------------------------------------
# Both halves are required. Getting this wrong in either direction costs a measured number: too wide
# and a counting question takes a path never scored on it, too narrow and the date question that
# started this ticket still gets its policy read back.

def test_a_stated_date_plus_a_when_question_routes():
    assert dates.asks_for_a_date("my last working day is the eighteenth of August. by when should "
                                 "my full and final be settled")
    assert dates.asks_for_a_date("I'm joining from home on the second of March. when should my "
                                 "laptop turn up")
    assert dates.asks_for_a_date("I joined on the first of March and I want to resign on the first "
                                 "of May. how long is my notice")


def test_a_counting_question_that_names_a_month_does_not_route():
    # x03 in evals/dev/cross_figure_queries.json: two dates, and the answer is a number of leaves.
    assert not dates.asks_for_a_date("if I take leave from Monday the nineteenth of October to "
                                     "Friday the twenty third, how many privilege leaves come out "
                                     "of my balance")
    # g01 in evals/dev/figure_queries.json, which the figures gate is scored on.
    assert not dates.asks_for_a_date("I joined on the first of January and I have taken three "
                                     "casual leaves, how many do I have left in June")


def test_a_when_question_with_no_date_does_not_route():
    assert not dates.asks_for_a_date("when is my leave encashment paid out")


def test_no_clock_is_read_anywhere():
    """The module must not answer from today's date.

    A duration is counted from a date the person named, or not at all. Asserted against the parsed
    source rather than a stubbed clock, because stubbing proves the call was intercepted and not that
    it is absent — and parsed rather than grepped, because the module's own docstring explains why
    `date.today()` is not used and a substring test trips over the explanation.
    """
    import ast

    src = (Path(__file__).resolve().parent.parent.parent / "src" / "dates.py").read_text(
        encoding="utf-8")
    called = {node.func.attr for node in ast.walk(ast.parse(src))
              if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
    assert "today" not in called and "now" not in called


# --- what gets said ------------------------------------------------------------------------------

def test_a_range_is_spoken_as_a_range_with_its_caveats():
    fig = dates.DateFigure(dt.date(2026, 9, 29), dt.date(2026, 10, 20), "Settlement is within 30 to "
                           "45 working days.", dt.date(2026, 8, 18), 30, 45, "working_days", [], {},
                           ["I took the year as 2026."])
    said = fig.spoken()
    assert "between 29 September 2026 and 20 October 2026" in said
    assert "Counting from 18 August 2026" in said
    assert said.endswith("I took the year as 2026.")


def test_nothing_computed_speaks_the_rule_alone():
    fig = dates.DateFigure(None, None, "Settlement is within 30 to 45 working days.", None, None,
                           None, None, ["no duration in the excerpts"], {}, [])
    assert fig.spoken() == "Settlement is within 30 to 45 working days."
    assert not fig.computed
