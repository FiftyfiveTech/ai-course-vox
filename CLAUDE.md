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

## Conventions

- `uv` for the env. `make setup` and `make demo` must work from a clean clone.
- Secrets in `~/.config/`, never in the repo. `.env.example` lists the names only.
- Every model call goes through the shared cost/latency logger.
- Commit messages describe the change. No AI attribution, no co-author trailers.
