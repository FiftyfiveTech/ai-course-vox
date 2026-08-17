# VOX Entity Spec — Critical Entities and Exact-Match Rule

**Version:** 1.0
**Date:** 2026-08-17
**Author:** Ritika (Builder)
**Status:** Awaiting Evaluator sign-off (verification: Evaluator predicts 3 labels from this spec and matches)

---

## 1. What is a critical entity?

A **critical entity** is any slot whose value, if wrong, would cause the system to take the wrong
action or ask for confirmation about the wrong thing. Missing or incorrect critical entities must
be caught by the scorer — non-critical fields (e.g. meeting title when not spoken) are not scored.

| Entity type | Label key | Examples |
|---|---|---|
| Person name | `person` | Priya, Rahul, Ananya, Kiran, Vikram |
| Team / group name | `team` | design team, backend team |
| Duration | `duration` | 1h, 30m, 2h, 2.5h, 6h |
| Date | `date` | tomorrow, Friday, 2026-09-15, next week, Monday, last Friday, August 20 |
| Time | `time` | 15:00, 10:00, 09:00, 17:00, 12:30, 16:00 |
| Recurrence | `recurrence` | daily, weekly, every Monday |
| Project / task name | `project` | VOX project, AI course track, infrastructure project, design sprint, client onboarding task |
| Action intent | `intent` | book_meeting, log_hours, set_reminder, query_calendar, escalate, refuse, greet, ambiguous |

---

## 2. Exact-match rule

The scorer compares the NLU output to the gold label using **exact string match after
normalisation**. A label is correct if and only if the normalised extracted value equals the
normalised gold value.

### 2.1 Normalisation steps (apply in order)

1. **Lowercase** the entire string.
2. **Strip** leading and trailing whitespace.
3. **Collapse** internal whitespace runs to a single space.
4. **Expand spoken numbers to digits** for durations and times (see §2.2).
5. **Resolve relative dates** to ISO 8601 (`YYYY-MM-DD`) using the utterance recording date as
   the reference point (see §2.3).
6. **Remove punctuation** that is not part of the value (trailing full stops, commas between
   list items). Keep hyphens inside hyphenated names (e.g. `stand-up`).

### 2.2 Duration and time format

| Spoken form | Normalised gold value |
|---|---|
| one hour | `1h` |
| thirty minutes / thirty minute | `30m` |
| two hours | `2h` |
| two and a half hours | `2.5h` |
| four hours | `4h` |
| six hours | `6h` |
| three hours | `3h` |
| one hour | `1h` |
| three p.m. / 3 pm | `15:00` |
| ten a.m. / 10 am | `10:00` |
| nine a.m. / 9 am | `09:00` |
| five p.m. / 5 pm | `17:00` |
| twelve thirty / 12:30 | `12:30` |
| two p.m. / 2 pm | `14:00` |
| four p.m. / 4 pm | `16:00` |
| end of day | `17:00` |

### 2.3 Date resolution

Relative dates are resolved against the **recording date** stored in `manifest_all.json` (field
`recorded_on`; if absent, use 2026-08-17 for all dev-set utterances).

| Spoken form | Resolution rule | Example (ref = 2026-08-17, Monday) |
|---|---|---|
| today | ref date | `2026-08-17` |
| tomorrow | ref + 1 day | `2026-08-18` |
| yesterday | ref − 1 day | `2026-08-16` |
| last Friday | most recent Friday before ref | `2026-08-14` |
| this week | ISO week containing ref | week of `2026-08-17` |
| next week | ISO week + 1 | week of `2026-08-24` |
| Friday | next Friday on or after ref | `2026-08-21` |
| Monday | next Monday on or after ref | `2026-08-17` |
| Thursday | next Thursday on or after ref | `2026-08-20` |
| the fifteenth of September | explicit month/day, current year | `2026-09-15` |
| August twentieth / August 20 | explicit month/day, current year | `2026-08-20` |
| every Monday | recurrence, not a date → `recurrence: weekly_monday` | — |

### 2.4 Person and team names

- Use the casing spoken in the utterance text (after lowercasing): `priya`, `rahul`, `ananya`.
- Multiple people are a list sorted alphabetically: `["rahul", "sneha"]`.
- Teams are lowercased, trimmed: `design team`, `backend team`.

### 2.5 Project names

Lowercased, trimmed. Use the full spoken name: `vox project`, `ai course track`,
`infrastructure project`, `client onboarding task`, `design sprint`.

### 2.6 Intent labels

| Situation | `intent` value |
|---|---|
| User books / schedules a meeting | `book_meeting` |
| User logs hours on a project | `log_hours` |
| User sets a reminder | `set_reminder` |
| User queries calendar / availability | `query_calendar` |
| User says something with unclear referent (e.g. "cancel it") | `ambiguous` |
| User triggers escalation (security, HR, urgent) | `escalate` |
| User asks something the system must decline | `refuse` |
| User greeting with no action | `greet` |

---

## 3. Scoring

A response is **correct** for entity `e` if `normalise(extracted[e]) == normalise(gold[e])`.

- An entity present in gold but absent in extraction → **miss** (score 0 for that slot).
- An entity absent in gold but present in extraction → **false positive** (not penalised at
  utterance level; tracked separately).
- Overall slot accuracy = `correct_slots / total_gold_slots` across the eval set.

The gate threshold is defined in `tests/gates/gate_phase0.py`.

---

## 4. Worked examples (dev set)

### utt_006 — entity
**Text:** "book a one hour meeting with Priya tomorrow at three p.m."
**Gold:**
```json
{
  "intent": "book_meeting",
  "person": ["priya"],
  "duration": "1h",
  "date": "2026-08-18",
  "time": "15:00"
}
```

### utt_009 — entity
**Text:** "book a stand-up with the backend team every day at nine a.m."
**Gold:**
```json
{
  "intent": "book_meeting",
  "team": "backend team",
  "recurrence": "daily",
  "time": "09:00"
}
```

### utt_031 — ambig
**Text:** "cancel it"
**Gold:**
```json
{
  "intent": "ambiguous"
}
```

### utt_036 — escalate
**Text:** "I need to report a security incident"
**Gold:**
```json
{
  "intent": "escalate"
}
```

---

## 5. Edge cases and tie-breakers

- If a duration is implicit (e.g. "stand-up" conventionally means 15m), do **not** infer it —
  only extract what was spoken. Gold will be absent for `duration` in that case.
- "Monday morning" → `date: 2026-08-17`, `time` absent (morning is not precise enough to normalise).
- "between two and four p.m." → `time_start: 14:00`, `time_end: 16:00` (two separate fields).
- "end of day Friday" → `date: 2026-08-21`, `time: 17:00`.
- Recurrence takes priority over a single date: "every Monday at four p.m." → `recurrence:
  weekly_monday`, `time: 16:00`, no `date` field.
