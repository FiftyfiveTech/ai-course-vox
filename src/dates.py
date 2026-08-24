"""Computing a DATE from a duration the documents state (VOX-034 part D).

`figures.py` computes a number from a formula. This computes a date from a duration, and it exists
because the two are not the same problem and one module cannot be both.

The failure it fixes, from a live turn: *"My last working day is the eighteenth of August. By when
should my full and final be settled?"* The corpus answers this — separation-policy:p10 says the F&F
is "processed and cleared within 30–45 working days from the employee's last working day" — and the
reply was the sentence, read back. `figures.compute()` printed `no formula in the excerpts`, which is
correct on its own terms: `prompts/compute_figure_v2.md` says in as many words that a deadline is not
a formula. A person who has given a date and been told a duration wants the date.

**Why a separate module and a separate prompt, rather than a wider `figures.py`.** Three reasons, in
order of how much they cost to ignore:

  1. `safe_eval` is a whitelist over numbers. A date is not a number and month arithmetic is not
     addition — 31 January plus one month is 28 February, and the clamp is a rule, not a sum. Widening
     the evaluator to carry dates means widening what an untrusted expression string can ask for,
     which is the one thing that module refuses to do.
  2. `evals/dev/figure_queries.json` scores `figures.compute()` at floor 1.0 on both columns. Every
     line changed in that module re-opens those numbers. This module cannot regress them because it
     cannot run: `answer.py` routes to it only for a question that states a date AND asks when.
  3. The two paths refuse for different reasons and a listener should hear which. "No formula" and
     "no duration" are different missing things.

**What still has to trace, and it is stricter here than for numbers.** A date figure has three
inputs and all three are checked:

    anchor      the date to count from. Must be a date the PERSON said, or one written in an
                excerpt. Never the model's idea of today — see below.
    offset      how long. Must appear as a number IN THE EXCERPTS. A duration the model supplies
                from general knowledge ("notice is usually a month") is the whole failure mode.
    unit        working days, calendar days or months. The phrase must appear in an excerpt too,
                because the unit changes the answer by weeks: 30 working days from 18 August is
                29 September, 30 calendar days is 17 September.

**There is no clock in this module.** Nothing here reads the system date, and that is deliberate
rather than incidental. A duration is counted from a date the person named; if they named none, there
is nothing to count from and the rule is stated instead. `datetime.date.today()` would make every
answer depend on when it was asked, make the gate unreproducible, and let "when am I eligible" be
answered from a fact in no document. `config.ANCHOR_YEAR` is the one concession — a person says "the
eighteenth of August" and not the year — and it is spoken aloud, never assumed silently.

**Weekends and holidays.** Weekends come out of a working-day count under any reading. The company's
own published holidays come out too when `config.WORKING_DAYS_SKIP_HOLIDAYS` is on AND the calendar
rows were actually retrieved for this turn — the dates are parsed out of the excerpts, so a holiday
that changes the answer is a holiday the reply can cite. A turn that never retrieved the calendar
counts weekends only and says so out loud, which is the honest version of an answer that would
otherwise silently be two days early.
"""
import calendar
import datetime as dt
import json
import re
from collections import namedtuple

from src.config import ANCHOR_YEAR, PROMPTS_DIR, WORKING_DAYS_SKIP_HOLIDAYS

PROMPT_FILE = PROMPTS_DIR / "compute_date_v1.md"

# Extraction, not composition — the same reason figures.py pins it there.
TEMPERATURE = 0.0
MAX_TOKENS = 512

# The units this module can count in, and nothing else. A unit outside this set is an extraction that
# was not understood, so the rule is stated rather than counted in a unit we are guessing at.
UNITS = ("working_days", "calendar_days", "months", "years")


class DateFigure(namedtuple("DateFigure",
                            "value end rule anchor offset offset_end unit missing raw notes")):
    """One attempt at a computed date.

    value       the computed date, or None when nothing could be counted. None is not an error: it
                means "state the rule instead", exactly as `figures.Figure.value` does.
    end         the far end when the duration is a range ("within 30-45 working days"). None for a
                single date. Both ends are computed because both are in the document — collapsing a
                stated range to one date is picking a number the policy did not.
    rule        the rule in one sentence as the model read it out of the excerpts. Spoken either way.
    anchor      the date counted from, after tracing.
    offset      the duration, as read out of the excerpts.
    unit        one of UNITS.
    missing     why nothing was computed. Non-empty means `value` is None by construction.
    raw         the model's JSON, kept so a wrong answer can be read afterwards rather than guessed at.
    notes       assumptions this answer carries, in spoken English. Appended to what is said, never
                dropped: an assumed year and a holiday list that was or was not applied are the two
                things that move this figure, and a listener who is not told cannot correct it.
    """

    __slots__ = ()

    @property
    def computed(self):
        return self.value is not None

    def spoken(self):
        """-> what to say. The rule alone when nothing was counted, then the date, then the caveats.

        Composed in Python for the same reason `figures.Figure.spoken` is: the date in the sentence
        has to be the date Python counted, and a model asked to restate its own arithmetic is a model
        that can restate it wrong.
        """
        rule = (self.rule or "").strip()
        if not self.computed:
            return rule
        joiner = " " if rule.endswith((".", "!", "?")) else ". "
        if self.end is not None and self.end != self.value:
            body = (f"Counting from {say_date(self.anchor)}, that is between "
                    f"{say_date(self.value)} and {say_date(self.end)}.")
        else:
            body = f"Counting from {say_date(self.anchor)}, that is {say_date(self.value)}."
        notes = (" " + " ".join(self.notes)) if self.notes else ""
        return f"{rule}{joiner}{body}{notes}"


def say_date(d):
    """-> '18 August 2026'. Spelled for a TTS voice, so no zero padding and no ISO dashes."""
    return f"{d.day} {calendar.month_name[d.month]} {d.year}"


# --- date arithmetic ----------------------------------------------------------------------------
# Plain stdlib. `dateutil` would give `relativedelta` for the month case and is not worth a
# dependency for eleven lines, and `pandas` business-day offsets carry a holiday calendar concept
# that would compete with the one parsed out of the excerpts.


def add_calendar_days(start, n):
    return start + dt.timedelta(days=int(n))


def add_working_days(start, n, holidays=()):
    """-> the date `n` working days after `start`, skipping weekends and `holidays`.

    Counts forward from the day AFTER `start`: "within 30 working days from the last working day"
    does not count the last working day itself. Exclusive of the anchor, inclusive of the day landed
    on, which is how a notice period or a settlement window reads in every policy in this corpus.
    """
    holidays = set(holidays or ())
    d, counted, step = start, 0, (1 if n >= 0 else -1)
    while counted < abs(int(n)):
        d += dt.timedelta(days=step)
        if d.weekday() < 5 and d not in holidays:
            counted += 1
    return d


def add_months(start, n):
    """-> the same day-of-month `n` months on, clamped to the length of the target month.

    Clamped rather than rolled over: 31 January plus one month is 28 February and not 3 March. A
    notice period that ends on the 3rd because February is short is a wrong last working day.
    """
    n = int(n)
    month0 = start.month - 1 + n
    year = start.year + month0 // 12
    month = month0 % 12 + 1
    return dt.date(year, month, min(start.day, calendar.monthrange(year, month)[1]))


def offset_by(anchor, n, unit, holidays=()):
    if unit == "working_days":
        return add_working_days(anchor, n, holidays)
    if unit == "calendar_days":
        return add_calendar_days(anchor, n)
    if unit == "months":
        return add_months(anchor, n)
    if unit == "years":
        return add_months(anchor, float(n) * 12)
    raise ValueError(f"unit {unit!r} is not one of {UNITS}")


# --- reading dates out of speech and out of excerpts --------------------------------------------
# Two spellings matter and they are not the same. A person speaking gives "the eighteenth of August"
# or "18th of August" — STT returns both, and which one depends on the provider. A document gives
# "15-Aug-26" in a calendar table, or "01 January 2026" in a release notice.

_MONTHS = {}
for _i, _m in enumerate(calendar.month_name[1:], start=1):
    _MONTHS[_m.lower()] = _i
    _MONTHS[_m.lower()[:3]] = _i
_MONTHS["sept"] = 9                              # the corpus spells it this way in the 2026 list

_ORDINAL_WORDS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7,
    "eighth": 8, "ninth": 9, "tenth": 10, "eleventh": 11, "twelfth": 12, "thirteenth": 13,
    "fourteenth": 14, "fifteenth": 15, "sixteenth": 16, "seventeenth": 17, "eighteenth": 18,
    "nineteenth": 19, "twentieth": 20, "twenty first": 21, "twenty second": 22, "twenty third": 23,
    "twenty fourth": 24, "twenty fifth": 25, "twenty sixth": 26, "twenty seventh": 27,
    "twenty eighth": 28, "twenty ninth": 29, "thirtieth": 30, "thirty first": 31,
}

_DAY_WORD = "|".join(sorted(_ORDINAL_WORDS, key=len, reverse=True))
_MONTH_WORD = "|".join(sorted(_MONTHS, key=len, reverse=True))

# "the eighteenth of August", "18th of August 2026", "1 March"
_DAY_MONTH = re.compile(
    rf"\b(?:the\s+)?(?P<day>{_DAY_WORD}|\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?"
    rf"(?P<month>{_MONTH_WORD})\b(?:,?\s+(?P<year>\d{{4}}))?", re.I)
# "August the eighteenth", "August 18, 2026"
_MONTH_DAY = re.compile(
    rf"\b(?P<month>{_MONTH_WORD})\s+(?:the\s+)?(?P<day>{_DAY_WORD}|\d{{1,2}})(?:st|nd|rd|th)?"
    rf"\b(?:,?\s+(?P<year>\d{{4}}))?", re.I)
# "15-Aug-26" — the calendar tables, and the one form that carries its own year
_TABLE = re.compile(rf"\b(?P<day>\d{{1,2}})-(?P<month>{_MONTH_WORD})-(?P<year>\d{{2,4}})\b", re.I)


def _day_number(raw):
    raw = " ".join(raw.lower().replace("-", " ").split())
    if raw in _ORDINAL_WORDS:
        return _ORDINAL_WORDS[raw]
    try:
        return int(raw)
    except ValueError:
        return None


def _make(day, month, year):
    try:
        return dt.date(int(year), int(month), int(day))
    except (TypeError, ValueError):
        return None                              # 31 February and friends: not a date, not an error


def dates_in(text, anchor_year=None):
    """-> the set of dates `text` states. A bare date takes `config.ANCHOR_YEAR`.

    Generous about spelling, strict about existence — the same posture `answer.numbers_in` takes, and
    for the same reason: the person is speaking, so the spelling is the provider's choice, but a date
    that does not exist is not a date.
    """
    year_default = ANCHOR_YEAR if anchor_year is None else anchor_year
    found = set()
    for pattern in (_TABLE, _DAY_MONTH, _MONTH_DAY):
        for m in pattern.finditer(text or ""):
            day = _day_number(m.group("day"))
            month = _MONTHS.get(m.group("month").lower())
            year = m.group("year")
            if year is None:
                year = year_default
            else:
                year = int(year)
                if year < 100:                   # "15-Aug-26"
                    year += 2000
            d = _make(day, month, year)
            if d is not None:
                found.add(d)
    return found


def dates_in_excerpts(hits):
    found = set()
    for h in hits or ():
        found |= dates_in(h.text)
    return found


# Calendar rows carry a label, and the label decides whether the day is a company holiday at all.
# "Fixed" and "Weekend" rows are holidays for everyone. A "Floater Leave" row is NOT: holiday-policy
# p4 says two of them may be availed, on application, 15 days ahead — so a floater is leave a person
# takes, not a day the company is shut, and counting it as a non-working day would push every
# settlement date later for everyone who never applied for it.
_HOLIDAY_ROW = re.compile(
    rf"\b(?P<day>\d{{1,2}})-(?P<month>{_MONTH_WORD})-(?P<year>\d{{2,4}})\s+"
    rf"(?P<weekday>[A-Za-z]+day)?\s*(?P<label>Fixed|Weekend|Floater Leave)?", re.I)


def published_holidays(hits):
    """-> {date: label} for the company-holiday rows written in these excerpts.

    Parsed from the excerpts and never from a table in this repo. A holiday that moves the answer is
    then a holiday the reply is grounded in, which is the same rule every other number in a VOX reply
    follows. Unlabelled rows are ignored rather than assumed to be holidays.
    """
    out = {}
    for h in hits or ():
        for m in _HOLIDAY_ROW.finditer(h.text or ""):
            label = (m.group("label") or "").strip().lower()
            if label not in ("fixed", "weekend"):
                continue
            year = int(m.group("year"))
            if year < 100:
                year += 2000
            d = _make(_day_number(m.group("day")), _MONTHS.get(m.group("month").lower()), year)
            if d is not None:
                out[d] = label
    return out


# --- is this even a date question? --------------------------------------------------------------
# Cheap gate in front of the extraction call, the same shape as `figures.states_a_number`. BOTH halves
# are required: a date the person stated, and a question that asks when. "If I take leave from the
# nineteenth of October to the twenty third, how many privilege leaves come out of my balance" states
# two dates and is a counting question — it must stay on the numeric path.

_WHEN_CUES = (
    "when", "by when", "what date", "which date", "deadline", "last date", "due", "turn up",
    "arrive", "dispatched", "settled", "eligible", "last working day", "how much time", "by what",
    # "how long is my notice" is a date question and "how many leaves do I have" is not, so the
    # duration cues are spelled with their verb. Bare "how long" would be wide enough to pull in
    # "how long can I carry leave forward", which is a count and belongs on the numeric path.
    "how long do i", "how long have i", "how long is", "how long will", "how long would",
    "how long does", "notice period",
)


def asks_for_a_date(text):
    """-> True if `text` states a date and asks for one back."""
    low = " ".join((text or "").lower().split())
    if not any(cue in low for cue in _WHEN_CUES):
        return False
    return bool(dates_in(low))


# --- the extraction call ------------------------------------------------------------------------


def _system_prompt():
    from src import nlu                          # load_prompt strips the YAML front matter
    return nlu.load_prompt(PROMPT_FILE)


def extract(transcript, hits, turn_id, model_id=None, fallback=True, on_fallback=None):
    """-> the model's JSON as a dict, or None if it did not return JSON.

    One call, through `arms.llm` like every other model call here, so the cost logger, the cooldown
    and the ollama fallback all apply. None rather than a raise for an unparseable reply, for the
    reason `figures.extract` gives: the caller's next move is the same either way.
    """
    from src import answer as answer_mod, arms

    msgs = [
        {"role": "system", "content": _system_prompt()},
        {"role": "user", "content": (
            f"Question: {transcript}\n\n"
            f"{answer_mod.CONTEXT_HEADER}\n\n"
            f"{answer_mod.context_block(hits)}\n\n"
            f"Extract the rule and the duration for: {transcript}"
        )},
    ]
    text = arms.llm(msgs, model_id, turn_id=turn_id, on_fallback=on_fallback, fallback=fallback,
                    temperature=TEMPERATURE, json_mode=True, max_tokens=MAX_TOKENS,
                    prompt_file=PROMPT_FILE.name, transcript_chars=len(transcript),
                    chunks=len(hits or ()), stage_kind="date")
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", (text or "").strip(), flags=re.MULTILINE)
    try:
        data = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        return None
    return data if isinstance(data, dict) else None


# --- tracing ------------------------------------------------------------------------------------


def trace_anchor(raw, transcript, hits):
    """-> (date, note, problem). The anchor has to be a date somebody actually stated.

    Accepted from the person's words first and the excerpts second, and the ORDER matters: a question
    that names a date is asking about that date, and an excerpt date ("with effect from 01 January
    2026", on the release notice of every policy in this corpus) is a plausible-looking anchor that
    would produce a confident answer to a question nobody asked.
    """
    said = dates_in(transcript)
    if not raw:
        return None, None, "no anchor date"
    parsed = dates_in(str(raw)) | ({_iso(raw)} - {None})
    parsed = {d for d in parsed if d is not None}
    if not parsed:
        return None, None, f"anchor {raw!r} is not a date"

    for d in sorted(parsed):
        if d in said:
            return d, _year_note(d, transcript), None

    # The model's YEAR is never load-bearing, and this is not a courtesy — it is the fix for a real
    # refusal. Asked "I'm joining from home on the second of March", the extractor returned
    # anchor_date 2023-03-02: the right day, a year from nowhere. Checked strictly that is an anchor
    # the person never stated, so a question the corpus answers refused over a digit the person did
    # not say and the config already holds.
    #
    # So a stated day-and-month is matched on day and month, and the date used is the PERSON'S — our
    # year, from config.ANCHOR_YEAR, not the model's. Strictly narrower than trusting what it
    # returned: an anchor still has to be a date somebody said, and the one component it is allowed
    # to be wrong about is the one component we supply ourselves.
    said_by_day = {(d.month, d.day): d for d in said}
    for d in sorted(parsed):
        stated = said_by_day.get((d.month, d.day))
        if stated is not None:
            return stated, _year_note(stated, transcript), None

    for d in sorted(parsed):
        if d in dates_in_excerpts(hits):
            return d, None, None
    return None, None, f"the anchor {say_date(sorted(parsed)[0])} was not stated"


def _year_note(d, transcript):
    """-> the spoken caveat when the year in `d` is ours rather than the person's, else None.

    The DAYS_IN_YEAR trade in a different unit: needed to answer at all, absent from what was said,
    and therefore said out loud instead of assumed silently.
    """
    if re.search(rf"\b{d.year}\b", transcript or ""):
        return None
    return f"I took the year as {d.year}."


def _iso(raw):
    try:
        return dt.date.fromisoformat(str(raw).strip()[:10])
    except (TypeError, ValueError):
        return None


def trace_offset(value, hits):
    """-> (float, problem). A duration has to be written in the excerpts.

    The one check that does the most work in this module. "Notice is three months", "the laptop comes
    in 21 working days", "the settlement takes 30 to 45 days" are all facts a model knows about
    companies in general, and a date computed from a remembered duration is the single most
    convincing wrong answer this path can give.
    """
    from src.answer import numbers_in            # local: answer imports this module

    if value in (None, ""):
        return None, "no duration in the excerpts"
    try:
        n = float(value)
    except (TypeError, ValueError):
        return None, f"duration {value!r} is not a number"
    in_excerpts = set()
    for h in hits or ():
        in_excerpts |= numbers_in(h.text, parts=True)
    if not any(abs(n - k) < 1e-6 for k in in_excerpts):
        return None, f"the duration {n:g} is in no excerpt"
    return n, None


_UNIT_EVIDENCE = {
    "working_days": ("working day",),
    "calendar_days": ("calendar day", "days", "day"),
    "months": ("month",),
    "years": ("year",),
}


def trace_unit(unit, hits):
    """-> (unit, problem). The unit has to be evidenced too, because it is worth weeks.

    30 working days from 18 August 2026 is 29 September; 30 calendar days is 17 September. Reading
    "working" into a policy that said "days", or out of one that said "working days", is a fortnight
    either way — so the phrase has to be in the text the model was given.
    """
    unit = (unit or "").strip().lower().replace(" ", "_").replace("-", "_")
    if unit not in UNITS:
        return None, f"unit {unit!r} is not one of {', '.join(UNITS)}"
    blob = " ".join((h.text or "") for h in hits or ()).lower()
    if not any(phrase in blob for phrase in _UNIT_EVIDENCE[unit]):
        return None, f"no excerpt says {unit.replace('_', ' ')}"
    return unit, None


# --- the whole thing ----------------------------------------------------------------------------


def compute(transcript, hits, turn_id, model_id=None, fallback=True, on_fallback=None):
    """-> DateFigure, or None when the extraction was unparseable.

    The ways this ends with no date, all of them answers rather than errors:
      the excerpts state no duration     -> the rule is spoken alone
      the anchor was never stated        -> ditto, and `missing` says which
      the duration or unit did not trace -> ditto

    A provider failure propagates, exactly as `figures.compute` lets one propagate.
    """
    data = extract(transcript, hits, turn_id, model_id=model_id, fallback=fallback,
                   on_fallback=on_fallback)
    if data is None:
        return None

    rule = (data.get("rule") or "").strip()
    spec = data.get("date_rule") or {}
    if not isinstance(spec, dict):
        spec = {}
    blank = DateFigure(None, None, rule, None, None, None, None, ["no duration in the excerpts"],
                       data, [])
    if not spec:
        return blank

    anchor, year_note, problem = trace_anchor(spec.get("anchor_date"), transcript, hits)
    if problem:
        return blank._replace(missing=[problem])

    offset, problem = trace_offset(spec.get("offset"), hits)
    if problem:
        return blank._replace(anchor=anchor, missing=[problem])

    unit, problem = trace_unit(spec.get("unit"), hits)
    if problem:
        return blank._replace(anchor=anchor, offset=offset, missing=[problem])

    # The far end of a stated range. Optional, and a range whose end does not trace is answered as a
    # single date rather than discarded — "within 30 working days" is still true when the 45 could
    # not be sourced.
    end_offset = None
    if spec.get("offset_end") not in (None, ""):
        end_offset, end_problem = trace_offset(spec.get("offset_end"), hits)
        if end_problem:
            end_offset = None

    notes = [n for n in (year_note,) if n]
    holidays = {}
    if unit == "working_days":
        if WORKING_DAYS_SKIP_HOLIDAYS:
            holidays = published_holidays(hits)
        if holidays:
            notes.append("I counted weekends and the company holidays in the excerpts as "
                         "non-working days.")
        else:
            # Precise about WHY, because "no holiday list was retrieved" would be wrong in the case
            # that produced this wording: the F&F question retrieved holiday-list-2026 p1 #1, which
            # holds only Floater Leave rows. A floater is leave a person applies for, not a day the
            # company is shut, so it is correctly not a company holiday — and the reply has to say
            # that no HOLIDAY DATES were found rather than that no calendar was.
            notes.append("I counted weekends as non-working days; no company holiday dates were in "
                         "the excerpts I retrieved, so none are taken out.")

    try:
        value = offset_by(anchor, offset, unit, holidays)
        end = offset_by(anchor, end_offset, unit, holidays) if end_offset is not None else None
    except (ValueError, OverflowError) as e:
        return blank._replace(anchor=anchor, offset=offset, unit=unit, missing=[str(e)])

    return DateFigure(value, end, rule, anchor, offset, end_offset, unit, [], data, notes)
