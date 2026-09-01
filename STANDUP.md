# Standup log

Append at the end of each working session. Two minutes. Newest at the bottom.

Format — one block per session:

```
## YYYY-MM-DD — <name> (Builder|Evaluator)
Did:      what actually landed, with the task id
Number:   any measured number + the command that produced it (or "none today")
Blocked:  what is in the way, or "nothing"
Next:     the single next task id
```

Rule: a number goes in this file only if it appeared in your terminal. No number is better than a
remembered one.

## 2026-08-20 — Vimal (Builder)
Did:      VOX-013. `make compare` runs two whole architectures as real turns and prints the
          five-field VOX-003 split for both. Added the piper TTS arm (`rhasspy/piper-voices`,
          22.05 kHz) because the ticket names it and it is the only TTS arm faster than Kokoro.
          Extracted `src/harness.py` so the comparison and `make turn` run the same turn code
          instead of two copies. Fallback is off in compare, so a refused stage reads FAILED
          rather than borrowing the local arm's latency.
Number:   `make compare` — n=3 per arm, interleaved, hello_testing_voice.mp3, exit 0.
          median time_to_first_audio: fast 6695 ms (band 6202-6717), quality 9291 ms (band
          8914-21269). tts per reply char: piper 7.53-9.47 ms, Kokoro 87.36-102.65 ms.
          t_llm: 3B/ollama 4267-4436 ms, 70B/NIM 1726-15428 ms.
          One of three 70B calls took 15428 ms against REMOTE_TIMEOUT_S of 10 s, so a third of
          turns would fall back on the shipped config. Full table in ARCHITECTURE.md.
Blocked:  nothing. Found and fixed a real bug on the way: reusing one endpointed Capture across
          turns freezes speech_end_t, so time_to_first_audio read 6.5 s / 39.4 s / 63.6 s on three
          identical turns. compare now endpoints per turn and asserts the segment is unchanged;
          tests/unit/test_compare.py pins it.
Next:     VOX-014 — write the phase-1 finding.

## 2026-08-21 — Ritika (Builder)
Did:      VOX-028, the documentation half of the MVP freeze. README rewritten from the template
          scaffold into the repo's front door — developer note, what to expect, system requirements,
          required/optional software, getting started for both the plain and the grounded path, the
          full env sample, the HF repo-id arm table, the gating table, the latency-budget table, and
          a known-gaps table. `.env.example` grown from 3 names to every knob in `src/config.py`,
          tunables commented out so config.py stays the single source of defaults. Week report at
          `notes/build-log/VOX/week-report.md` with the numbers, three findings, six items of gate
          debt and the retro. Concept primer written first. Merged `origin/dev` in — `gate_phase1b.py`
          (PR #24) was not on this branch and the first draft of the gating table was wrong about it.
Number:   `uv run pytest tests/unit -q` -> 313 passed in 6.63s.
          `make gate` -> "no tests ran in 0.01s", Error 5 — all four gates expose main(), not test_*.
          `gh pr list --state merged --limit 40 --json number,author,mergedBy` -> 17 of 23 merged PRs
          have author == mergedBy. 6 of the first 6 were reviewed by a third person; every PR from
          #8 onward except #14 was self-merged.
          `git ls-files uv.lock` -> not tracked. `ls tests/gates/` -> no test_no_leakage.py.
Blocked:  `mvp-v1` is NOT pushed. The tag has to point at a reviewed commit on `dev`, and this is on
          `feat/vox-028` awaiting review — tagging a branch tip is a self-merge with extra steps.
          VOX-028 also depends on VOX-026 (end-to-end execution run) and VOX-027 (demo), both still
          in Plan Backlog, so this freeze certifies the repo and the docs, not a rehearsed demo.
Next:     VOX-026 — the end-to-end execution run, on the demo hardware, on headphones.

## 2026-08-21 — Vimal (Builder)
Did:      VOX-026. `make dry-run` runs the scripted ten-turn demo session end to end with no mic:
          `evals/demo/session_v1.json` holds the running order as data, each turn carries an
          `expect` block asserted against the turn record, and the run exits non-zero if any turn
          missed it. Includes the barge-in whose interrupting utterance becomes the next turn's
          input, and two confirmations — one confirmed, one cancelled. Frames are paced at one every
          32 ms, so `t_vad` is the live hangover and not the ~4 ms a fixture run collapses it to.
          Three bugs fixed on the way, all found by running it: the barge-in print used a character
          cp1252 cannot encode and killed the turn *after* the interruption worked; the confirmation
          leg timed its second STT/TTS over the turn's own `t_stt_ms`/`t_tts_ms`; and the unit suite
          was reading `.env`'s demo profile. Report and demo runbook:
          `notes/build-log/VOX/vox-026-dry-run.md`.
Number:   `make dry-run` twice, consecutive, no edit between: CLEAN 10/10, exit 0 both.
          time_to_first_audio median 2310 ms (2106-2464) and 2357 ms (2025-3317), paced.
          barge-in stopped 226.2 / 227.6 ms after speech began, cutting 5.3 s of a 7.8 s reply.
          0/10 fallbacks both runs. 86 calls, max(cost_usd) = 0.0.
          `uv run pytest tests/unit -q` -> 337 passed in 9.09s (was 3 failed / 310 passed on this
          machine before the conftest fix; the week report's 313 was measured without the demo
          profile in `.env`).
Blocked:  nothing. Two things handed on rather than fixed here: `evals/dev/pdf_queries.json` q06 may
          be labelled a page late (the dress code is on code-of-ethics p12; p13 is the enforcement
          note) — that is a gate number and belongs to the Evaluator; and `VOX_STT_FIXUPS=1` is set
          in `.env` and read by nothing on `dev`, so the preflight warns about it every run.
Next:     VOX-027 — the demo itself, off this script.
