# VOX-026 — the end-to-end execution run

**Ticket:** a scripted 10-turn session on the demo machine including a barge-in and a confirmation,
recorded and dry-run twice. *Done when the run completes clean twice with timings captured. No live
debugging during the demo.*

**Date:** 2026-08-21 · **Machine:** the demo machine (Windows 11, WH-CH520 headphones)
**Script:** `evals/demo/session_v1.json` · **Runner:** `scripts/dry_run.py` (`make dry-run`)

Every number below appeared in a terminal on this machine and the command that produced it is next
to it. The two runs' full stdout is in the run reports named under each table.

---

## 1 · The command, and what it does

```bash
make dry-run                  # == uv run --no-sync python scripts/dry_run.py
```

Ten turns from `evals/demo/session_v1.json`, driven through the real stages — silero endpointing,
`openai/whisper-large-v3` at Groq, hybrid retrieval over the internal corpus,
`meta-llama/Llama-3.1-8B-Instruct` at NVIDIA NIM, `rhasspy/piper-voices` locally — with no
microphone. The user's ten lines are synthesised locally by `hexgrad/Kokoro-82M` (a different voice
from the reply's, so a recording of the session has two speakers in it), cached under
`runs/rehearsal/audio/`, and handed to the real STT stage as samples.

**The frames are paced at one every 32 ms of wall clock.** That is the difference between this and
`make turn`: unpaced, the `VAD_SILENCE_MS` hangover a person actually waits out collapses to silero
compute (~4 ms against ~1120 ms), and every `time_to_first_audio` in this repo measured from a
fixture is about a second optimistic. Paced, `t_vad` is 1120 ms on every turn below — the live
number, with the microphone removed.

Each turn's `expect` block is asserted against the **turn record**, never against the reply's
wording: `grounded`, `sources`, `retrieved`, `barged_in`, `confirmation`, the extractor's `intent`,
and the harness's own `ok` (all five VOX-003 fields measured). A reply is a sampled model output;
asserting on its words would fail the rehearsal for the wrong reason. An `expect` key the checker
does not implement is itself a failure — an expectation nobody asserts prints PASS.

---

## 2 · The two runs

Consecutive, on the same machine, with **no edit between them** — which is the only pair that means
anything. A "second run" after a fix is a first run again.

### Run 1 — `runs/rehearsal/run-20260821T164822.json`, exit 0

```
  turn path           vad     stt     llm     tts    retr    ttfa  notes
  t01  plain        1120    340    586    275     77   2464  intent=greet
  t02  grounded     1120    340    428    194     28   2170  leave-policy:p15,leave-policy:p17
  t03  grounded     1120    294    422    337     35   2265  attendance-policy:p4,attendance-policy:p15
  t04  grounded     1120    310    458    351     47   2354  assets-policy:p13,assets-policy:p14; barge 226ms cut 5.3s
  t05  grounded     1120    305    468    451     31   2442  code-of-ethics-business-conduct:p12,annual-event-policy:p3
  t06  plain        1121    396    479    128     48   2246  intent=unknown
  t07  refused      1120    339    348    180     24   2106
  t08  plain        1119    326    652    193     23   2380  intent=capture; confirm=yes
  t09  plain        1120    309    741    170     26   2439  intent=capture; confirm=no
  t10  plain        1120    291    514    129     33   2155  intent=escalate

  time_to_first_audio over 10/10 turns: median 2310ms, min 2106ms, max 2464ms
  llm        median     474ms  band 348-741ms  n=10
  tts        median     194ms  band 128-451ms  n=10
  retrieval  median      32ms  band 23-77ms  n=10
  fell back on 0/10 turns
  grounded 4/10 turns; the script asks for grounding on 4 of them
  wall clock 103s for 10 turns
CLEAN — 10/10 turns met their expectations.
```

### Run 2 — `runs/rehearsal/run-20260821T165040.json`, exit 0

```
  turn path           vad     stt     llm     tts    retr    ttfa  notes
  t01  plain        1120    308    540    266     76   2926  intent=greet
  t02  grounded     1120    329    377    191     26   2137  leave-policy:p15,leave-policy:p17
  t03  grounded     1120    284   1315    305     29   3125  attendance-policy:p4,attendance-policy:p15
  t04  grounded     1120    326   1456    323     39   3317  assets-policy:p13,assets-policy:p14; barge 228ms cut 5.3s
  t05  grounded     1120    279    405    436     22   2341  code-of-ethics-business-conduct:p12,annual-event-policy:p3
  t06  plain        1120    329    510     94     31   2158  intent=unknown
  t07  refused      1120    308    364    129     47   2025
  t08  plain        1120    354    674    201     27   2448  intent=capture; confirm=yes
  t09  plain        1120    292    663    192     45   2374  intent=capture; confirm=no
  t10  plain        1120    315    502    124     28   2144  intent=escalate

  time_to_first_audio over 10/10 turns: median 2357ms, min 2025ms, max 3317ms
  llm        median     525ms  band 364-1456ms  n=10
  tts        median     196ms  band 94-436ms  n=10
  retrieval  median      30ms  band 22-76ms  n=10
  fell back on 0/10 turns
  grounded 4/10 turns; the script asks for grounding on 4 of them
  wall clock 106s for 10 turns
CLEAN — 10/10 turns met their expectations.
```

### What the pair says that one run cannot

| | run 1 | run 2 | read |
|---|---|---|---|
| median `time_to_first_audio` | 2310 ms | 2357 ms | **stable to ~2%** |
| worst turn | 2464 ms | 3317 ms | the *tail* is what moves, and it is the LLM |
| `t_llm` band | 348–741 ms | 364–**1456** ms | two turns in run 2 took ~3× the median |
| `t_vad` | 1119–1121 ms | 1120 ms | pacing is doing its job; this is the hangover |
| barge stop latency | 226 ms | 228 ms | the most repeatable number in the run |
| fallbacks | 0/10 | 0/10 | both free tiers held for ~90 s twice |
| spend | 0.00 USD | 0.00 USD | 86 calls across the two runs, `max(cost_usd) = 0.0` |

The demo's felt latency is therefore **~2.3 s to first syllable**, and the risk is not the median —
it is that any one turn can take a second longer because NIM decided to. `VOX_REMOTE_TIMEOUT_S=5` in
the demo profile means a turn that goes past 5 s falls back to the local 3B and costs ~4.3 s
instead; that did not happen in either run.

Zero-spend evidence, over the 86 calls the two runs made:

```
$ (report reader over runs/calls.jsonl, joined on the two runs' turn_ids)
calls belonging to the two runs: 86
max cost_usd: 0.0
providers/tiers: [('groq', 'free-tier'), ('local', 'local-weights'), ('nvidia-nim', 'free-tier')]
models: ['BAAI/bge-small-en-v1.5', 'meta-llama/Llama-3.1-8B-Instruct', 'openai/whisper-large-v3', 'rhasspy/piper-voices']
all ok: True
```

### The barge-in, and the confirmation, on the record

```
t04  barge_stop_ms 226.2  played_s 2.534  reply_s 7.848  cut_s 5.314  out_latency_s 0.183
t05  input=carried-in     grounded from code-of-ethics-business-conduct:p12
t08  t_stt_ms 325.5  t_tts_ms 192.8  t_confirm_stt_ms 277.6                   confirmation=yes  'Yes, go ahead.'
t09  t_stt_ms 308.7  t_tts_ms 170.5  t_confirm_stt_ms 275.5  t_confirm_tts_ms 50.6  confirmation=no   'No, cancel that.'
```

t04 was interrupted 2.5 s into a 7.8 s reply, 226 ms after the interrupting speech began, with
183 ms of output buffer still to drain behind the cut. The utterance that stopped it became t05's
input — `input=carried-in`, no second listener, exactly the live path. t09's two TTS calls are the
read-back (170 ms) and the cancel line (51 ms), in **separate** fields; see finding 2.

---

## 3 · What the dry run found, and what was done about it

Five things. All five would have happened in front of an audience.

### 1 · The barge-in print killed the turn — `src/loop.py`

`speak_and_watch` opened its barge-in line with `──` (U+2500), which **cp1252 cannot encode**, and a
Windows console encodes to cp1252 unless something has changed it. The first scripted barge produced:

```
    speaking… (interruptible)
    RAISED UnicodeEncodeError: 'charmap' codec can't encode characters in position 2-3
```

`abort()` had already cut the reply. The interruption *worked*, and then the print announcing it
raised out of `on_speech`, out of the endpointer, and killed the turn — at the single most visible
moment in the demo. Fixed two ways, because the character was only the instance:

- the box-drawing dash is now an em dash (`—`, which cp1252 does encode), and
- `config.utf8_console()` reconfigures stdout/stderr to UTF-8 with `errors="replace"`, called from
  `src/loop.py:main`, `scripts/dry_run.py:main` and `scripts/turn_from_fixture.py:main`.

The second one is the real fix: a turn prints a transcript and a reply, and this process chooses the
characters of neither. The corpus itself carries `●` bullets — `print`ing a chunk raised the same
error while this report was being written — so "choose nicer glyphs" was never going to be enough.
`tests/unit/test_rehearsal.py` pins it: the barge-in line is printed into a strict cp1252 stream and
the capture still comes back.

### 2 · A confirmation turn was reporting the wrong latency — `src/loop.py`

The yes/no leg ran `arms.stt` and `arms.tts` a second time inside `turn.stage("stt")` and
`turn.stage("tts")` — the same two slots the turn's own utterance and read-back had already written,
and `TurnTimer.stage` overwrites. So on a confirmation turn `t_stt_ms` was the latency of
transcribing the word "yes" and `t_tts_ms` was the cancel sentence. Measured on t09: the record
would have said `t_tts_ms = 50.6` for a turn whose read-back actually took 170.5 ms.

Those are two of the five VOX-003 fields the phase gates percentile, and a confirmation turn is the
turn whose latency matters most. The leg is now `loop.confirmation_leg`, a function with an
injectable listener, and it writes `t_confirm_stt_ms` / `t_confirm_tts_ms`. `stage_sum_ms` is still
the five-field sum it always was.

This is the same shape as the reused-`Capture` clock (VOX-013) and `grounded = 8/8` next to
correct-source@3 `= 7/8` (VOX-033): **a metric slot written twice per turn does not error, it
reports the last writer.**

### 3 · The unit suite was reading the operator's demo profile — `tests/unit/conftest.py`

`uv run pytest tests/unit -q` on this machine, before any of this ticket's code:

```
FAILED tests/unit/test_session.py::test_a_pause_does_not_end_a_timed_session
FAILED tests/unit/test_session.py::test_a_dead_mic_gives_up_instead_of_spinning_to_the_deadline
FAILED tests/unit/test_session.py::test_a_spoken_turn_clears_the_quiet_streak
3 failed, 310 passed in 4.67s
```

The week report records **313 passed** on 2026-08-21. The difference is `.env`: the demo profile sets
`VOX_SESSION_QUIET_LIMIT=1` (one silent listen ends a session instead of two, so a muted headset
mid-demo does not cost a minute of dead air), and `test_session.py` asserts against the code default
of 2. `.env` is gitignored and per-machine, so the suite's result depended on whose machine ran it.

The profile is correct; the *test* was the bug. `conftest.py` already guards this class of problem
for `VOX_*_MODEL` (`no_env_override`) and could not for these, because `config.py` reads them once at
import into constants `src/loop.py` then imports by name. New autouse fixture `no_ambient_tunables`
pins them, and `test_the_pinned_session_defaults_match_config_py` fails if the pinned value and
`config.py` ever disagree. **337 passed in 9.09s** now.

### 4 · `make dry-run` uses `uv run --no-sync`, and that is not decoration

While writing this ticket:

```
error: Failed to generate package metadata for `en-core-web-sm==3.8.0 @ direct+https://github.com/...`
  Caused by: HTTP status server error (504 Gateway Timeout)
```

`uv run` re-resolves the environment on every invocation and `en-core-web-sm` is pinned to a GitHub
release URL, so GitHub having a bad minute is a failed run before a line of VOX executes. `--no-sync`
skips resolution. It is in the target and in the runbook below, not in someone's memory.

### 5 · A knob that does nothing — reported, not fixed

`.env` sets `VOX_STT_FIXUPS=1` with a nine-line comment explaining that it makes "flutter" read as
"floater" whatever Whisper decides. **Nothing on `dev` reads it** — the mechanism lives on
`experimental/demo-followup-retrieval`, which is not merged. The knob is a belief, not a setting.

Not fixed here: merging that branch is not this ticket's, and editing someone's `.env` is not either.
Instead the preflight prints it, every run:

```
  WARN  set in .env and read by no module in src/: VOX_STT_FIXUPS — a knob that does nothing, not a setting
```

`scripts/dry_run.py:env_names_no_module_reads()` compares the `VOX_*` names assigned in `.env`
against every `VOX_*` name appearing anywhere in `src/`, `scripts/`, `tests/` or `tools/`.

---

## 4 · Two changes to the script itself, and why

Both are cases where the rehearsal changed the **demo**, not the code. Written down because a
question chosen for reliability is a form of tuning and has to be visible.

### q02 was cut from the running order

t04 was written as `pdf_queries` q02, *"how is my privilege leave encashment calculated when I leave
the company"* — the longest answer in the dev set, which is what makes it worth interrupting. From a
clean synthetic voice, Whisper returned:

```
you said : 'How is my privilege leaving CashMint calculated when I leave the company?'
vox says : "Privilege Leave encashment shall be calculated based on the employee's last drawn basic salary at the time of separation."
  grounded in leave-policy:p9, leave-policy:p11
```

The one high-information term is destroyed before retrieval runs, so the answer came fluently and
citedly off the **wrong document**. This is not new: `ARCHITECTURE.md` records the same damage from a
live turn ("leave encashment" → "leaving cashment"), and `pdf_queries.json` carries it as q03, the
one query its 7/8 floor is set to allow to fail.

It is therefore a known-fragile question, and with `VOX_STT_BIAS=0` there is nothing on this branch
that recovers the word. t04 is now q07 (*"what happens to my laptop if I resign before two years of
allotment"*) — a conditional answer, long enough to interrupt, grounded in `assets-policy:p14` in
both runs. The reason is recorded in the script as the turn's `not_q02` field.

### t05 asserts p12, where `pdf_queries` q06 labels p13

t05 is "what is the dress code in the office". The label in `evals/dev/pdf_queries.json` is
`code-of-ethics-business-conduct:p13`; this script asserts **p12**, because that is where the answer
is. Read off `runs/chunks.jsonl` before the label was written, which is the rule that file states for
its own labels:

- **p12**, chunk 16: *"During office hours, even when working remotely, one must be as presentable
  and dressed as if attending a physical office. The same dress code applies to remote working in
  office."* — chunk 17 is the men's and women's wear lists.
- **p13**, chunk 18: *"Note: 1. Managers or supervisors are expected to inform employees when they
  are violating the dress code…"* — enforcement, not the code itself.

**This is a finding for the Evaluator, not a fix.** q06's label is scored by VOX-033's
correct-source@3, and changing an eval label changes a gate number — that is not VOX-026's to change.
Handed over as: *q06 may be labelled one page late, in which case the gate has been asking retrieval
for the enforcement note rather than the dress code.*

Worth noting alongside it: t05's interrupting line is *"actually, what is the dress code in the
office"*, and that leading "actually," is charged into the query's information ceiling — the same
mechanism as finding 2 in the week report. Retrieval returned two pages, and p13 was not among them
at all.

---

## 5 · The demo runbook

The point of the ticket: **no live debugging during the demo.** Everything below is decided in
advance.

### Before the audience arrives

```bash
git status                 # clean, on a reviewed commit
make test                  # 337 passed
ollama list                # the 3B is pulled — this is what a rate limit costs you
make index                 # only if sources/ changed; never during the demo
make dry-run               # ~2 min, must print CLEAN 10/10 and exit 0
```

Headphones **on the machine, plugged in, before starting**. There is no acoustic echo cancellation:
on open speakers silero hears the reply and every turn interrupts itself.

Read the preflight block. It prints the arms, the retrieval floors, `REMOTE_TIMEOUT_S`,
`SESSION_QUIET_LIMIT`, whether STT biasing is on, the output device, and any `.env` knob no module
reads. If it says `STOP`, nothing ran — fix that, then run the pair again from the top.

### The running order

Live, with a microphone, `make demo` — the script is the running order, in `evals/demo/session_v1.json`:

| # | say | what it shows |
|---|---|---|
| 1 | "hello Vox, are you there?" | the chain works, ~2.3 s to first syllable |
| 2 | "how many maternity leaves does a woman employee get" | grounded, cites `leave-policy:p15` |
| 3 | "what are the standard office working hours and shift timings" | an answer that is all numbers, and the numeric guard behind it |
| 4 | "what happens to my laptop if I resign before two years of allotment" | a long answer — **talk over it at ~2 s** |
| 5 | *(the interruption)* "actually, what is the dress code in the office" | barge-in: the reply stops, and the words that stopped it are the next turn's input |
| 6 | "what is the health insurance coverage amount for my family" | not in the corpus, and it says so instead of inventing |
| 7 | "how many days of paid menstrual leave does the company give" | the harder refusal: relevant-looking excerpts, refused anyway |
| 8 | "book a one hour meeting with Priya tomorrow at three p.m." → **"yes, go ahead"** | read-back and confirm before anything is written |
| 9 | "log four hours on the VOX project for today" → **"no, cancel that"** | the same gate answered the other way |
| 10 | "I need to escalate this to the engineering lead" | hands a person to a person |

### If it goes wrong anyway

| symptom | what it is | what to do, out loud |
|---|---|---|
| a turn takes ~5 s longer and prints `[fell back: llm]` | the free tier refused; the local 3B answered | "that one ran on the local model — the free tier throttled us." It is the design working |
| `RATE LIMITED` and the session ends | both the arm and its cooldown are exhausted | stop the session, say so, show `runs/calls.jsonl`. Do not restart into the same limit |
| the reply interrupts itself immediately | speakers, not headphones | plug the headphones in. Do not debug the VAD threshold |
| a grounded question answers plainly | nothing cleared the floor — the printed line says so | move on; question 6 is that behaviour on purpose |
| nothing is heard | mic muted; with `SESSION_QUIET_LIMIT=1` the session ends after one silent listen | restart `make demo`; it costs 10 s, not a minute |

**Do not**, mid-demo: edit `.env`, re-index, change an arm flag, or ask a question that is not on the
list. Question 4's original form (q02, leave encashment) is on the list of things not to ask.

---

## 6 · What this rehearsal still cannot tell you

- **Acoustics.** The user's audio is synthesised and handed to STT as samples. It never crosses a
  room, a microphone, or the speaker/mic coupling that makes echo cancellation a problem. Whisper
  finds a Kokoro voice easier than a person in an open-plan office, so every transcript above is an
  **upper bound**. The live rehearsal on the demo machine, with a person and headphones, is still
  worth doing — this ticket makes it reproducible, not unnecessary.
- **A person's timing.** The scripted barge-in lands at 2.0 s every time, and 226/228 ms is the stop
  latency for *that* offset. A human interrupts when they lose patience, and the interesting tail is
  "just as the reply ends", where `abort()` returns `None` and there is no barge-in at all.
- **The free tier's mood.** Two clean runs at 16:48 say nothing about 16:48 tomorrow. Run 2's
  1456 ms LLM turn is the visible edge of that, and the fallback path — unexercised in both runs —
  is what covers the rest.
- **Anything about held-out data.** `evals/heldout/` is untouched by this ticket. Four of the ten
  questions are copies of `evals/dev/pdf_queries.json`, chosen *because* they are known-good, which
  is what a demo wants and what an eval must never be. Every number here is a dev number.
