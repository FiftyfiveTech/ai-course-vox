# POC: query a PDF source folder — proposed tickets

Drafted 2026-08-20 by Vimal (Builder). **Not yet on the board** — the MCP grant is
read/start/note/request_review, so these must be created by hand in Odoo (project 60).

Scope: a POC. VOX answers a spoken question from a folder of PDFs. Nothing more.

## Three decisions that make this fast

Each one removes a dependency the obvious design would have had:

1. **No embeddings.** Lexical scoring in pure Python. An embedding model would be a fourth verb in
   `src/arms.py` (its contract is `stt/llm/tts`, and `_MODULES` has three entries), a new row in
   `config.ARMS`, and a zero-spend question. Retrieval is not a model call, so it needs a latency
   stamp but not the cost logger. **Drops the dependency on VOX-006.**
2. **The answer prompt is its own versioned file.** `src/nlu.py` already loads `prompts/reply_v1.md`
   from disk and strips its front matter, so a second prompt file follows a pattern that exists
   today. **Drops the dependency on VOX-018** (whose six-prompt scope does not cover this anyway).
3. **Routing is a score floor, not structured state.** If the best chunk clears a threshold it is a
   document question; otherwise the existing reply path runs unchanged. No intent enum, no schema.
   **Drops the dependency on VOX-019** — and therefore on the Phase 1 / 1B gate chain.

Net: the POC has **no blocker on the board**. Only `pypdf` is added to `pyproject.toml`, and it is
a parser, not a model, so the HF-repo-id rule does not apply to it.

Standard constraints apply to all five (copy the board's boilerplate line): HF repo ids only,
zero spend, internal employee-facing, held-out set stays sealed.

---

## VOX-029 · PDF source folder to text chunks

    Estimate: 1.5h   Depends: -   Phase: POC   Day: 2026-08-21

A `sources/` root folder read at startup. Each PDF's text extracted per page with `pypdf` and cut
into chunks that carry `(file, page)` provenance. A page that yields no text is reported by name
rather than skipped silently — a scanned PDF is the most likely reason this POC returns nothing,
and it must be visible at load, not at query time.

    Done when: `make index` prints file count, page count, chunk count, and names every page
               that produced no text.
    Verification: `make index` output on the real sources folder.

## VOX-030 · Lexical retrieval, top-k with provenance

    Estimate: 1.5h   Depends: VOX-029   Phase: POC   Day: 2026-08-21

`retrieve(query, k=5)` returns chunks ranked by BM25, each with its score and `file:page`.
No model, no network, no key. A floor below which nothing is returned, so "not in the documents" is
a real answer state and not an empty string.

k is 5, not the 3 drafted here: the board's own text for VOX-030 (task 1768) says "top-k chunks
(default k=5)", and the board is the source of truth. VOX-031's prompt budget is what pays for the
extra two chunks, so if it turns out not to fit, that is where the number moves.

    Done when: `make ask Q="..."` prints the top 5 chunks with file:page and score, and prints
               nothing-found for a query about something absent from the corpus.
    Verification: two `make ask` runs - one hit, one miss.

## VOX-031 · Grounded answer through the existing LLM arm

    Estimate: 2.0h   Depends: VOX-030   Phase: POC   Day: 2026-08-21

`prompts/answer_from_source_v1.md`, loaded from file the way `reply_v1.md` already is. Retrieved
snippets go in the user message; the system prompt says answer only from them and say so plainly
when they do not contain it. Goes through `arms.llm`, so the cost logger, the arm flags and the
local fallback all apply for free.

    Done when: `make ask` prints a spoken-length answer plus the file:page it came from, and
               refuses rather than guessing on an out-of-corpus query.
    Verification: `make ask` transcript for one answerable and one unanswerable query, plus the
                  runs/calls.jsonl line showing the model id.

## VOX-032 · Wire it into the turn loop

    Estimate: 2.0h   Depends: VOX-031   Phase: POC   Day: 2026-08-21

Retrieval runs after STT. Best score over the floor routes the turn to the grounded answer;
otherwise the turn is unchanged. `t_retrieval_ms` and the sources used are added to the turn record
in `runs/turns.jsonl`, and a row goes in the ARCHITECTURE.md latency table - this adds a stage in
front of the LLM and the budget already misses by 3-10x, so the cost is recorded, not assumed.

    Done when: One spoken question about a PDF returns a spoken answer end to end, and the turn
               record carries t_retrieval_ms and the source list.
    Verification: turns.jsonl line for the turn, and the new ARCHITECTURE.md row.

## VOX-033 · POC gate: grounded-answer rate printed

    Estimate: 2.0h   Depends: VOX-031   Phase: POC   Day: 2026-08-21
    Owner: Evaluator (Ritika) writes the queries; Builder writes the gate script.

`tests/gates/gate_poc_pdf.py` over 10 written queries in `evals/dev/pdf_queries.json` - 8 with an
expected source `file:page`, 2 deliberately absent from the corpus. Prints correct-source@3 and the
refusal rate on the 2.

**heldout-v1 is sealed and tagged and contains zero document queries** (its categories are greet /
entity / ambig / escalate / refuse). It is not reopened for this. These queries are dev-only, and
the POC therefore has no held-out number - that limit is stated in the gate output, not glossed.

    Done when: The gate prints both numbers and exits non-zero below the agreed floor.
    Verification: gate output pasted into the task.

---

## Critical path

    VOX-029 -> VOX-030 -> VOX-031    5h   a real answer from a real PDF, text in / text out
                       -> VOX-032    +2h  the same answer, spoken
                       -> VOX-033    +2h  the number

VOX-033's queries can be authored in parallel with 029-031.

## Two STOP-and-asks before a single PDF lands in `sources/`

1. **PII.** ARCHITECTURE bans customer PII outright. A folder of internal PDFs is the likeliest way
   for it to arrive. The rule for what may go in `sources/` is needed before the folder exists, and
   `sources/` should be gitignored unless every document in it is repo-shareable.
2. **Scanned PDFs.** If the real corpus is images rather than text, `pypdf` returns nothing and this
   plan needs OCR - a different POC with a different estimate. VOX-029 surfaces this on day one,
   which is why it prints the empty pages by name.
