# Phase gates

One script per gate. A gate **computes and prints its number**, then asserts the threshold. If the
number is not printed, it is not a gate.

Naming: `gate_phase0.py`, `gate_phase1.py`, `gate_phase1b.py`, …

Written by the week's **Evaluator**, before the Builder finishes the phase. The Evaluator runs it
**once** on `evals/heldout/` — running it repeatedly and tuning against the result is the failure
this course exists to avoid.

`test_no_leakage.py` (task 0.7) belongs here too: it asserts `evals/dev ∩ evals/heldout = ∅` by
content hash and must pass before any gate result counts.
