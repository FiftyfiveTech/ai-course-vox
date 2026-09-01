# <TRACK> — AI Engineering Course

Hand-built. The point is that we build it, so **never** generate a whole phase in one shot,
and never inherit code you cannot explain.

## The board is the source of truth for what to do next

Tasks live in Odoo — project **AI Dev Learning**. The `odoo-board` MCP server is wired in.

Start every session with `my_tasks()`. Read `task(<id>)` before writing code: its description is
the **acceptance criterion**, and it links the week-task file and the PRD as PDFs.

`start(<id>)` → write code → `note(<id>, "...")` as you go → `request_review(<id>, "<real
output>")`. **You cannot set Done.** The supervisor re-runs the gate command and compares output.

## Roles this week

One of us is **Builder**, the other **Evaluator**; they swap next week. Read the team build plan
(linked from any task) for the full contract. The two rules that matter most:

- **Blind labelling.** `evals/heldout/` belongs to the Evaluator. It is sealed Wednesday and
  tagged `heldout-v1`. The Builder tunes on `evals/dev/` **only** and never reads held-out cases.
  `tests/gates/test_no_leakage.py` asserts `dev ∩ heldout = ∅` by content hash — if it fails, stop.
- **No self-merges.** Every PR is reviewed by the other person. The Friday retro checks
  `git log` for zero self-merged PRs.

## Measured gates, not vibes

A phase is done when its number is **computed and printed**, not when it looks right. Never
advance past a failed gate — fix it, max 3 attempts, each tuned on `dev`, then escalate.

Report numbers with the command that produced them. If a number is not in this session's output,
say so instead of quoting it.

## Models and datasets: Hugging Face repo ids only

Every model and dataset is named by its **HF repo id** (`openai/whisper-large-v3-turbo`, not
"Groq Whisper"). The provider is only *where it runs* — Groq / NVIDIA NIM free tiers, or Ollama
via `ollama pull hf.co/<repo>`. **Zero spend**: any paid call is a STOP-and-ask, not a judgement
call. Gemini and `groq/compound*` are out — no HF repo id.

## Learning loop and the web coach

Every ticket doubles as course material, and the artifacts are developer-agnostic on purpose —
clone the repo, run one command, the session works on anyone's machine.

- **Ticket start:** write `docs/learning/vox-<nnn>-concepts.md` before implementation — the
  concepts the ticket exercises, why each matters here, the pitfall. Then offer a coach
  session on it. Skippable, never silent.
- **The coach** is a static lesson page plus a stdlib server: `make coach`, open
  `http://127.0.0.1:8765/vox-day1.html`, then say **"start web coach session"** in Claude
  Code. The page bundles the quiz into one submission, Claude grades it and replies into the
  page's chat panel. Protocol: `.claude/skills/web-coach/SKILL.md`.
- **NotebookLM** hosts a shared notebook over the same tracked docs — plain-English Q&A with
  citations, and the short Video Overviews the coach pages embed. Sync with
  `uv run --script tools/coach/notebooklm_sync.py`; the notebook id comes from
  `--notebook`/`$VOX_NOTEBOOK_ID`, never hardcoded. Only tracked docs are uploaded —
  recordings, `runs/` and `evals/heldout/` must never become a notebook source.
- **Ticket close:** append to `docs/learning/retros.md` — executed, deviations + why, numbers
  with the command that produced them, lessons.

Setup, source hygiene and the video index: `docs/learning/README.md`.

## Pull requests

Full procedure — branch naming, the PR body template, the review checklist, merge commands
and how to recover from the usual mistakes — is `docs/CONTRIBUTING.md`. The short version:
branch off `dev`, PR against `dev`, the **other** developer reviews and presses merge
(`gh pr merge <n> --merge --delete-branch`). Nothing is committed straight to `dev` or `main`.

## Conventions

- `uv` for the env. `make setup` and `make demo` must work from a clean clone.
- Secrets in `~/.config/`, never in the repo. `.env.example` lists the names only.
- Every model call goes through the shared cost/latency logger.
- Commit messages describe the change. No AI attribution, no co-author trailers.
