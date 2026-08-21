# VOX — Week 3 report and MVP freeze

**Written:** 2026-08-21 at MVP freeze (VOX-028) · **Tag:** `mvp-v1`
**Team:** Ritika and Vimal, swapping Builder / Evaluator · Saurabh reviewing early PRs
**Board:** Odoo project *AI Dev Learning* (60) · **Repo:** `FiftyfiveTech/ai-course-vox`

Every number below appeared in a terminal, and the command that produced it is next to it. Where a
number is missing, this file says *not measured* rather than estimating.

---

## 1 · What was built

A working internal voice assistant: microphone → endpointing → speech-to-text → hybrid retrieval
over a folder of company PDFs → an LLM reply grounded in what was retrieved → spoken answer, with
the `document:page` cited, and then round again until the session clock runs out. Every model call
goes through one interface, is named by its Hugging Face repo id, is logged with its cost and its
latency, and degrades to a **local** arm when its free tier refuses.

Twenty tickets closed across five days, in four arcs:

| arc | tickets | what it left behind |
|---|---|---|
| Phase 0 — the loop exists and is measured | VOX-001, 002, 003, 008/010 | the chained turn, the five-field latency split, `runs/calls.jsonl` + `runs/turns.jsonl` joined by `turn_id`, the Phase 0 gate |
| Phase 1 — the loop is swappable and survives failure | VOX-006, 007, 009, 011, 012, 013, 014, 016/017 | arms behind one interface, remote→local fallback with cooldown, barge-in, endpointing as config, a whole-architecture comparison, the Phase 1 gate |
| Phase 1B — the turn has structure | VOX-015, 018, 019, 020, 021, 022, 023 | versioned prompts, Pydantic turn state, read-back confirmation, vocabulary biasing, the entity scorer, the Phase 1B report script |
| POC — it answers from the documents | VOX-029, 030, 031, 032, 033 + hybrid | PDF chunking with exact provenance, BM25 fused by rank with a sentence encoder, a grounded answer prompt, retrieval inside the turn loop, and the POC gate |

Architecture, with the reasoning and the full tables: `ARCHITECTURE.md`. Per-ticket retros:
`docs/learning/retros.md`. Concept primers: `docs/learning/vox-*-concepts.md`.

---

## 2 · The numbers, with their commands

### Spend

**0.00 USD.** Every gate asserts `cost_usd == 0.0` and a HF repo id on every call it made, and every
gate that has run has passed that assertion. Groq and NVIDIA NIM free tiers, plus local weights.
No paid API was called, and no STOP-and-ask was needed for spend.

### Unit suite

```
$ uv run pytest tests/unit -q
313 passed in 6.63s
```
2026-08-21. No network, no key, no microphone.

### The POC gate (the number the week closes on)

`make gate-poc` over `evals/dev/pdf_queries.json`, 2026-08-21, `meta-llama/Llama-3.1-8B-Instruct` @
NVIDIA NIM, identical across two consecutive runs:

| number | measured | floor | asserted? |
|---|---|---|---|
| correct-source@3 | **7/8 = 0.875** | 0.875 | yes |
| refusal rate on the 2 absent queries | **2/2 = 1.000** | 1.000 | yes |
| grounded-answer rate | **8/8 = 1.000** | — | no |
| …grounded *on an expected source* | **7/8 = 0.875** | — | no |

19 model calls, `cost_usd = 0.0` on all 19. The finding is the gap between the last two rows:
`grounded` only says the model answered from the excerpts it was handed, and stays true when those
excerpts came off the wrong pages. q03 answered fluently and citedly from `leave-policy` p7/p10 about
a rule that lives on `separation-policy` p13.

### Latency — the target is missed, and by how much

| | budget | measured |
|---|---|---|
| `time_to_first_audio`, fixture-driven, n=4 | ~1.3 s | **5.6 – 20.0 s** |
| best single turn of six, `make compare`, n=3 per arm | — | **6.2 s** |
| live microphone | 2 s | *measured + ~1.1 s* — the VAD hangover a fixture run collapses to ~4 ms |

`scripts/turn_from_fixture.py` (2026-08-17) and `make compare` (2026-08-20). TTS is the largest
stage in every turn measured all week. The cheapest available lever is measured and is **not** the
one the architecture doc originally proposed: `rhasspy/piper-voices` costs 7.5–9.5 ms per reply
character against Kokoro's 87–103, i.e. **9–13× cheaper**, for a config change instead of a
streaming rework.

### Fallback — measured, n=1, 2026-08-19

Both free tiers pointed at a dead port, same clip, same machine:

| | clean | both free tiers unreachable |
|---|---|---|
| STT | 345 ms remote | 3430 ms = 2064 dead round-trip + 1357 local |
| LLM | 684 ms remote | 7513 ms = 2087 dead round-trip + 5415 local |
| transcript | `'Hello. So this is testing.'` | `'So this is testing.'` — the first word is gone |

The turn survives and the reply is real; a fallback is not free and **not identical**. On the
grounded path it is worse still: 1381 prompt tokens took the local 3B **72.6 s** against 0.94 s
remote (VOX-031). A rate limit does not cost the grounded turn's quality, it costs the turn.

### Retrieval

`make index` on the internal corpus, 2026-08-20: 15 files, 184 pages (163 with text, **21 with
none**, each named in the output), 215 chunks, 40,750 tokens. All 215 chunks verified as verbatim
substrings of their page; the overlap between consecutive chunks re-tokenizes to 49–52 tokens.
Encoding the 215 chunks: 25.1 s once per re-index, cached by fingerprint.

`make floors`, 13 dev queries (7 answerable / 6 absent):

| | answerable min | absent max | separable? |
|---|---|---|---|
| lexical (BM25 ÷ query ceiling) | **0.320** | **0.234** | yes → floor 0.28 |
| dense (cosine) | **0.676** | **0.738** | **no** — 0.65 is chosen, not derived |

Retrieval alone routes 9/13. End to end, with the v2 prompt and the numeric guard, **14/14**.
Per-turn cost of the dense half: `t_retrieval_ms` went from ~1 ms to **83–117 ms**, against
`t_tts_ms` of 4–5.8 s.

---

## 3 · The three findings worth carrying out of the week

**1 · Grounding constrains which facts, not what is done with them.** Given the right chunk — which
states the encashment *formula* and no per-day rate — the arm answered "you will be paid 12,000",
cited. Three fixes in order, each because the previous one was not enough: a v2 prompt forbidding a
computed figure; then `ANSWER_TEMPERATURE = 0` on the grounded path, because at 0.3 the same question
returned two correct refusals and one fabrication; then `answer.ungrounded_numbers()`, which refuses
any reply stating a figure that appears in no excerpt. A number the *caller* supplied does not count
as grounded — that asymmetry is what caught every wrong variant. Result: 14/14 with no answerable
query falsely refused, including the ones whose answers really do contain numbers.

**2 · A spoken question cannot reach a floor calibrated on typed keywords.** "How many paternal
leaves am I entitled to according to policy" put the answering chunk at **rank 110 of 137**, and
even with the exact corpus word "paternity" the six-word spoken form scored 0.276 against a floor of
0.280. The score divides BM25 by the query's own information ceiling, so every filler word a person
actually says is charged into the denominator. The dense half fixed the retrieval; the part that was
not obvious was that **the half with no evidence must abstain from the ordering** — with equal-weight
RRF, two lukewarm ranks on irrelevant chunks outvoted one excellent rank plus one terrible one.

**3 · A rate can be true and useless.** `grounded` = 8/8 next to correct-source@3 = 7/8 is the same
system described two ways, and only one of them is honest about the wrong-document answer. Any rate
printed from now on gets its denominator and its intersection printed under it.

---

## 4 · Gate debt

CLAUDE.md: *a phase is done when its number is computed and printed*. The corollary is that a gate
which does not run has no number, and a phase resting on it is **owed**. Six items, worst first.

| # | debt | what it would have caught | cost of leaving it |
|---|---|---|---|
| 1 | **`tests/gates/test_no_leakage.py` was never written** (task 0.7, named in CLAUDE.md and in the repo README as deliberately missing since day 1) | any content overlap between `evals/dev` and `evals/heldout` | blind labelling is currently guaranteed by two people being careful, not by a hash comparison. It is the cheapest gate in the repo — no key, no audio, no corpus — and it is the only one that guards a rule the whole evaluation contract rests on |
| 2 | **`gate_phase1b.py` asserts no threshold.** It prints entity capture, confirmation rate and state validity over `heldout-v1` and then exits `PASS — numbers printed`. VOX-024 is the ticket that sets the floor and is still `inProgress` | a regression in entity capture or confirmation coverage | Phase 1B has a **report**, not a gate. Nothing fails if the numbers get worse |
| 3 | **`make gate` collects nothing.** It is `pytest tests/gates`, and all four gates expose `main()` rather than `test_*`, so pytest exits 5 with "no tests ran" | nothing — it is a wiring bug, not a missing measurement | the four gates that exist have no single command, so they are re-run only when someone remembers the per-file invocation. A gate nobody re-runs is a number with a date on it |
| 4 | **17 of 23 merged PRs were self-merged.** `gh pr list --state merged --json number,author,mergedBy` — every PR from #8 onward except #14's reviewer, against 6 of 6 properly reviewed in the first two days | a second pair of eyes on every ticket after Tuesday | the no-self-merge rule held while a third person was pressing merge and stopped when the pair were merging their own work. `main` is protected, `dev` is not. This is the one item on this list that is a *process* failure rather than a missing file |
| 5 | **`uv.lock` is gitignored**, and so are `evals/**/*.wav` (by `*.wav`) | a dependency resolving differently on another machine; the three WAV-driven gates being unrunnable on a clean clone | "a clean clone reproduces the demo" is true of the code and not of the dependency graph. The utterance corpus survives in `evals/dev/manifest.json`, but regenerating it needs `libespeak-ng`, which is not in `make setup` |
| 6 | **No held-out number for anything the POC does.** `heldout-v1` holds 30 utterance labels and **zero** document queries — its categories are greet / entity / ambig / escalate / refuse — and it was not reopened to add any | how much the retrieval floors, the v2 prompt and the numeric guard were shaped by the ten dev queries they were built against | every POC number in this report is a dev number. Reopening a sealed set to add a category is how a held-out set becomes a dev set with extra ceremony, so the right fix is a *second* sealed set, not an edit to this one |

Also open on the board and not debt so much as unfinished: **VOX-026** (end-to-end execution run) and
**VOX-027** (demo) are still in Plan Backlog, and VOX-028 depends on both. This freeze therefore
certifies the repo and the documentation, not a rehearsed demo. VOX-026's dry-run has to happen on
the demo hardware, because there is no acoustic echo cancellation and on open speakers the reply
interrupts itself every time.

---

## 5 · The 45-minute retro

### Worked, and would do again

- **A gate per phase, and the number printed before it was discussed.** Every argument about latency
  this week was settled by a table rather than by an opinion, and twice the table said the opposite
  of what the architecture doc had proposed — TTS being the bottleneck rather than the model calls,
  and piper rather than streaming Kokoro being the fix.
- **`ARCHITECTURE.md` as a ledger, not a plan.** Budget columns were kept as originally written next
  to the measured ones, so the size of every miss stays visible. Superseded plans are marked
  superseded rather than deleted. It is now the most useful file in the repo.
- **One dispatch point for every model call.** VOX-006's `arms.py` meant that fallback, cooldown,
  cost logging, and later the *whole grounded POC*, were written once and applied everywhere.
  VOX-031 got the cost logger, the `--llm` flag, the rate-limit cooldown and the local fallback for
  free, because it called `arms.llm()` instead of `httpx`.
- **Versioning prompts as files.** `answer_from_source_v1.md` still exists, so the VOX-031 numbers
  measured against it remain reproducible after v2 changed the behaviour.
- **Writing the pitfall down before the implementation.** The concept primers repeatedly named the
  failure that then actually happened — most usefully "a paraphrased refusal counts as an answer",
  which is why the POC gate counts refusal *shapes* itself.

### Did not work

- **The review rule decayed the moment nobody outside the pair was merging** (debt #4). Enforcement
  was a sentence in `CLAUDE.md` and a promise to check `git log` on Friday; checking it on Friday is
  exactly what surfaced 17 self-merges, five days too late to change any of them.
- **Cheap gates were deferred behind expensive ones.** `test_no_leakage.py` needs no key, no audio
  and no corpus, and would have taken twenty minutes on day 1. It is still not written on day 5 —
  because it was never blocking anything, which is precisely the property that made it get skipped.
- **n=1 and n=3 measurements were made on stages that vary 2–4×.** Every table in
  `ARCHITECTURE.md` now carries a caveat about this, which is honest but is not the same as having
  taken more samples. The one case where it actually cost time: `time_to_first_audio` reading 6.5 s /
  39.4 s / 63.6 s on three identical turns turned out to be a reused `Capture` clock, and a bigger
  sample would have surfaced it as obviously-broken rather than as plausibly-variable.
- **Tuning against ad-hoc queries until VOX-033.** The retrieval floors were fitted to 13 queries
  written before anything about spoken questions was known, and the four failures that mattered were
  all found by typing something into `make demo` rather than by a set. A written set arrived on the
  last day.

### Do differently next week

1. **Branch protection on `dev`**, requiring one approving review. A rule a retro checks by hand is
   a rule that decays; make GitHub check it.
2. **Write the free gate first.** `test_no_leakage.py` on Monday morning, before anything with a key.
3. **`make gate` runs the gates.** One command, four scripts, and the ones needing a key or a corpus
   skip loudly rather than being absent.
4. **A second sealed set for the document path**, with its own tag, so the POC has a held-out
   number instead of a paragraph explaining why it does not.
5. **n≥5 before a number goes in a table**, or the table says n=1 in the cell rather than in a note
   underneath it.

---

## 6 · What the freeze certifies, and what it does not

`mvp-v1` certifies that the tracked tree plus the README get a stranger from `git clone` to a spoken
turn, on hardware like this: `make setup` → `make test` (313 passed) → `make turn` (no mic needed) →
`make demo`.

It does **not** certify:

- **The grounded path on a clean clone.** `sources/` is gitignored — internal HR policies — so a
  stranger has no knowledge base and none of the POC numbers in this report is reproducible by them
  until they supply their own corpus. This is the right call for the documents and a real hole in the
  claim, and it is why a redistributable sample corpus is on the list.
- **Latency.** Every stage varies 2–4× run to run on identical input. Reproducing the demo means the
  turns complete and the stages are the ones named — not that anyone else sees 6.7 s.
- **A rehearsed demo.** VOX-026 and VOX-027 are still open (see §4).
