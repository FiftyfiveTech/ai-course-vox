# ai-course-template

Starting scaffold for the FiftyFive AI engineering course. One repo per track, grown phase by
phase by hand.

## Use it

```bash
gh repo create FiftyfiveTech/ai-course-<track> --template FiftyfiveTech/ai-course-template --public --clone
cd ai-course-<track>
make setup
```

Then, in order: replace `<TRACK>` in `CLAUDE.md` and `pyproject.toml`, protect `main`, add the
other person as a collaborator with push access.

## Layout

| Path | Holds |
|---|---|
| `src/` | The system. Small modules, one job each. |
| `prompts/` | Versioned prompt files (`extract_v1.md`, `extract_v2.md`, …). Never inline a prompt in code. |
| `schemas/` | Pydantic models. Structured output is validated, not parsed by hand. |
| `evals/dev/` | **Builder** tunes here. 15 cases. |
| `evals/heldout/` | **Evaluator** only. Sealed Wednesday, tagged `heldout-v1`. The Builder never reads it. |
| `tests/gates/` | One script per phase gate. It prints the number; the number decides. |
| `sources/` | The PDF corpus the POC answers from. **Gitignored** — internal documents, see below. |
| `STANDUP.md` | Daily log. Two minutes, append-only. |

## Deliberately missing

One file is absent because it is a Week 0 task, not scaffolding:

- `tests/gates/test_no_leakage.py` — asserts `evals/dev ∩ evals/heldout = ∅` by content hash
  (task **0.7**). Until it exists, the blind-labelling rule is unenforced.

Write it. Do not import it from somewhere else.

`src/telemetry.py` (task **0.8**) is now written: every model call goes through it, and it emits
two logs joined by `turn_id` — `runs/calls.jsonl` (one line per model call, cost and provider
latency) and `runs/turns.jsonl` (one line per turn, the five-field latency split). See the
telemetry section of `ARCHITECTURE.md`.

## Swapping models (VOX-006)

Each stage is an arm chosen at run time. `src/arms.py` is the only interface —
`stt(audio, model_id)`, `llm(msgs, model_id)`, `tts(text, model_id)` — and `src/config.py` is the
only table. Arms are named by **HF repo id**, with a short alias for typing and
`repo/id@provider` when two providers serve the same weights.

```bash
make arms                                    # call all 10 arms once, print the calls.jsonl lines
uv run python scripts/check_arms.py --list   # just the table
uv run python -m src.loop --stt openai/whisper-base --tts microsoft/speecht5_tts
uv run python scripts/turn_from_fixture.py --llm gpt-oss
VOX_STT_MODEL=faster-base make turn          # env sets the default; an explicit flag wins
```

Currently 4 STT / 4 LLM / 2 TTS arms, hosted (Groq, NVIDIA NIM free tiers) and local. Defaults are
unchanged from VOX-002, so `make demo` and `make turn` still reproduce those numbers. The measured
cost of each arm, and what the local ones get wrong, is in the models section of `ARCHITECTURE.md`.
`make arms` calls each arm with fallback disabled — it is measuring the arms, so a refusal has to
show up on the row it belongs to rather than being quietly covered.

## Where each stage runs

```
Mic -> [VAD] local -> [STT] remote -> [LLM] remote -> [TTS] local -> Speaker
                          |                |
                          +-- on 429, timeout or 5xx --> a local arm
```

The two expensive stages run on a free tier and the two cheap ones run here. `config.PIPELINE`
declares that and a test asserts each stage's default arm against it, so reordering the arm tables
can no longer move a stage across the network boundary by accident.

When a free tier rate-limits, times out or 5xxs, that stage runs its local arm instead of losing
the turn — loudly, with both attempts in `runs/calls.jsonl` and the arm that actually ran named on
the turn record. A **bad key or a missing credential deliberately does not fall back**: that is a
config bug, and hiding it behind a worse transcript is worse than stopping. A 429 also parks the
arm for its `Retry-After` window so the next turn does not pay another doomed round-trip.

The LLM fallback needs ollama and one ~2 GB pull; `make setup` does it, and warns rather than fails
if ollama is absent. Full rules, trigger table and measured numbers: the fallback section of
`ARCHITECTURE.md`.

## Answering from a folder of PDFs (POC)

```bash
make tokenizer   # once — caches the tokenizer the chunker counts with (~9 MB)
make index       # sources/*.pdf -> runs/chunks.jsonl, and prints the counts
```

`make index` extracts every PDF page by page and cuts it into 300-token chunks with 50 tokens of
overlap, each carrying the `doc_id` and `page` it came from so a spoken answer can say where it
came from. No network, no model call, no key. Pages that yield **no** text are printed by name:
on a scanned PDF that is the whole corpus, and it has to be visible at load rather than as an
empty answer later.

`sources/` and the chunk file are gitignored — the corpus is internal company documentation, and
the extracted text is the same disclosure as the PDFs. A clean clone has nothing to index until
someone puts documents there.

`make ask Q="..."` retrieves over those chunks: BM25, the top 5 with `doc_id:page`, `chunk_idx` and
a score, or **"not in the documents"** when the best score does not clear the measured floor. Still
no network, no model call, no key. The grounded answer on top of it is VOX-031. Details and the
measured numbers: the source-folder and retrieval sections of `ARCHITECTURE.md`.

## Rules that live in this repo

`CLAUDE.md` carries the full contract. The short version:

- Models and datasets are named by **Hugging Face repo id**. The provider is only where it runs.
- **Zero spend.** A paid call is a STOP-and-ask, never a judgement call.
- A phase is done when its gate **prints the number**, not when the code looks right.
- Every PR is reviewed by the other person. `main` is protected; self-merges are the one thing
  the Friday retro always checks.
- Tasks come from the Odoo board via the `odoo-board` MCP server, not from this README.
