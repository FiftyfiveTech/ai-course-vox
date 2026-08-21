# VOX-028 — concept primer: what a freeze is for

The ticket: a README carrying the HF repo ids, the gating table and the latency-budget table; the
tag `mvp-v1`; a week report in `notes/build-log/VOX/week-report.md`; a retro that names the gate
debt. Written before the implementation. Where this and the code disagree, the code and
`ARCHITECTURE.md` win and this file is wrong.

Nothing in this ticket makes VOX faster or more accurate. Every previous ticket added a capability
and measured it; this one adds a **claim about the repo** — that someone who has never seen it can
clone it and get the same system. That claim is falsifiable, and the whole ticket is about writing
it down in a form that can fail.

## 1 · A freeze is a claim about a clean clone, not about the code

"MVP freeze" sounds like "stop changing things". The Done condition says something else: *a clean
clone reproduces the demo*. So the artifact under test is not `src/` — it is the union of what is
**tracked in git** plus what the README tells a stranger to do.

**Why here:** this repo deliberately does not track several things it needs. `sources/` is
gitignored (internal HR policies), `runs/` is gitignored, `evals/dev/*.wav` are gitignored by
`*.wav`, and `uv.lock` is gitignored too. Each of those was the right call for its own reason, and
each one is a hole in the clean-clone claim that only a README can fill.

**Why it matters:** the difference between "works on my machine" and "reproduces" is entirely in
the gaps — the system package nobody remembers installing, the corpus that lives in someone's
Downloads folder, the key in `~/.config/`. A freeze that only says "run `make setup`" is not a
freeze, it is a memory.

**Pitfall:** documenting the happy path and omitting what a clean clone *cannot* do. If `make
gate-poc` needs a corpus that is not in the repo, the README has to say so on the line that names
the command, not in a caveat three sections down.

## 2 · Gate debt is a number that was never printed

CLAUDE.md: *a phase is done when its number is computed and printed*. The corollary is that a gate
which does not run has no number, and a phase built on it is not done — it is **owed**.

Three kinds of debt, and they are not equally bad:

| kind | example here | cost |
|---|---|---|
| a gate that does not exist | `gate_phase1b.py` (VOX-024, still inProgress) | a phase closed on inspection |
| a gate that exists and cannot be invoked | `make gate` collects nothing — the gates expose `main()`, not `test_*` | the number exists but nobody re-runs it |
| a rule with no enforcement | `tests/gates/test_no_leakage.py` was never written | blind labelling is a promise, not a check |

**Why it matters:** debt of the third kind is the dangerous one, because it looks like a rule that
is being followed. `dev ∩ heldout = ∅` is asserted nowhere in this repo; it is true because two
people were careful, which is a different guarantee from a hash comparison.

**Pitfall:** listing the debt in a retro and calling it discharged. Naming it is the minimum; the
retro is honest only if it also says what the debt would have caught.

## 3 · A tag is the only reproducible way to cite a number

Every number in `ARCHITECTURE.md` is stamped with a date and a command. Neither pins the *code*.
`heldout-v1` already exists for exactly this reason — the sealed labels are a tag, not a sentence
in a doc — and `mvp-v1` does the same job for the demo.

**Why here:** the numbers in the README (correct-source@3 = 7/8, ttfa 5.6–20.0 s, 313 unit tests)
were measured against a specific tree. A reader who gets a different number needs to know whether
they are looking at a regression or at a later commit. `git diff mvp-v1` answers that; "measured on
2026-08-21" does not.

**Pitfall:** tagging a branch tip that has not been merged and reviewed. A tag on an unreviewed
commit is a self-merge with extra steps, and the Friday retro checks for exactly that.

## 4 · A README has two readers and they want opposite things

The reader who wants to *run* it needs one command and the truth about what it costs. The reader
who wants to *understand* it needs the tables. Putting both in one document is how a README becomes
a second, worse `ARCHITECTURE.md`.

**Why here:** the split this repo settled on — README says **what to run, what you need, and what
to expect**; `ARCHITECTURE.md` says **why, with the measurements**. The README's tables are
therefore summaries with pointers, and every number in them is a copy. A copied number is a number
that can go stale, so each one names its source section.

**Pitfall:** duplicating the reasoning. If the README explains *why* TTS is local, that paragraph
and the one in `ARCHITECTURE.md` will disagree within a week and neither will be marked wrong.

## 5 · "Optional" in a dependency list is a measurement, not a hedge

Software requirements split three ways and the split is worth being exact about, because a reader
uses it to decide whether to keep going:

- **Required** — without it nothing runs (`uv`, Python 3.12, a working audio device for anything
  end to end).
- **Degrades a stage** — without it the system runs a worse arm and says so (`ollama`: no local LLM
  fallback, so a 429 costs the turn instead of its quality; the sentence encoder: retrieval falls
  back to BM25 alone, which is VOX-030's measured behaviour).
- **Blocks one command only** (`libespeak-ng`, needed to regenerate the gitignored dev WAVs for
  `gate_phase0`/`gate_phase1`; the PDF corpus, needed for anything grounded).

**Why it matters:** `make setup` already encodes this distinction — it *fails* without `uv` and the
tokenizer, and *warns* without ollama and the encoder. The README's requirements table should be
readable as the same statement, or one of the two is lying.

**Pitfall:** marking something optional because the happy path does not hit it. The local LLM
fallback is optional right up to the first rate limit, which in a live demo is the moment it
matters most.

## What this ticket cannot establish

The freeze certifies that the tracked tree plus the README reproduce the demo **on hardware like
this one**. It cannot certify latency: every number in the budget table varies 2–4× run to run on
identical input, and `ARCHITECTURE.md` says so in three places. A clean clone reproducing the demo
means the turns complete and the stages are the ones named — not that anyone else sees 6.7 s.
