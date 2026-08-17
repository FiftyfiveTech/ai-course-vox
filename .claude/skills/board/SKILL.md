---
name: board
description: Use when starting a working session, picking up the next task, reporting progress, or finishing a task on the AI Dev Learning course board — drives the odoo-board MCP tools in the right order and enforces the evidence rule.
---

# Working the course board

The Odoo board decides what you do next. Do not invent work, and do not start the next thing
while a started task sits unreported.

## Start of session

1. `my_tasks()` — what is on you, soonest deadline first.
2. Pick the **earliest-deadline** ToDo task. Not the easiest one.
3. `task(<id>)` — read it fully. The description is the acceptance criterion; the Reference links
   are the week-task file and the PRD. If the criterion is ambiguous, `note()` the ambiguity and
   ask the supervisor — do not guess and build the wrong thing.
4. `start(<id>)`.

## While working

- `note(<id>, "...")` when something is measured, blocked, or decided. Paste the command and its
  real output, not a summary of it.
- Blocked >30 min → `note()` the blocker and move to another task. A silent stall is the one
  failure mode the board cannot see.
- Never touch `evals/heldout/` if you are the Builder this week.

## Finishing

1. Re-read the acceptance criterion in `task(<id>)`.
2. Run the check. Look at the output.
3. `request_review(<id>, "<the command and its actual output>")`.

The task stays In Progress. **Only the supervisor sets Done**, after re-running the gate and
getting the same number. `set_done()` exists only to refuse — do not look for a way around it.

## Hard rules

| Never | Why |
|---|---|
| Claim a number you did not just see printed | The discarded run failed exactly here — 1.0000 on self-written labels, 0.5195 on real ones |
| Work a task not assigned to you | The server refuses it; ask for reassignment instead |
| Touch another Odoo project | The server is pinned to project 72 and refuses. Do not route around it |
| Mark your own work Done | A gate verified by its author is not verified |
| Skip a failed gate | Fix it (max 3 attempts, tuned on `dev`), then escalate |
