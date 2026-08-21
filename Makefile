.PHONY: setup fallback-model tokenizer encoder test gate demo barge turn ground arms compare index ask answer gate-poc board coach clean
.DEFAULT_GOAL := help

help:
	@echo "make setup   create the venv and install deps (uv), and pull the local fallback model"
	@echo "make test    unit tests"
	@echo "make gate    NOTE: collects nothing — the gates expose main(), not test_*."
	@echo "             Run them one by one; the commands are in README.md 'The gating table'"
	@echo "make demo    talk to it end to end for VOX_SESSION_MINUTES (default 3; needs a mic)"
	@echo "make barge   three turns with interruptible replies — talk over it (VOX-011)"
	@echo "make turn    one instrumented turn from a recording — no mic needed"
	@echo "make ground  one grounded turn from a recording — retrieval in the loop (VOX-032)"
	@echo "make arms    call every registered model arm once and print the log (VOX-006)"
	@echo "make compare two whole architectures end to end, five-stage split for both (VOX-013)"
	@echo "make index   extract the sources/ PDFs to text chunks and print the counts (VOX-029)"
	@echo "make ask     Q=\"...\" retrieve the top chunks for a question, with file:page (VOX-030)"
	@echo "make floors  re-measure both retrieval floors on evals/dev and print the gaps"
	@echo "make answer  Q=\"...\" the same chunks through the LLM arm, as a spoken answer (VOX-031)"
	@echo "make board   verify the Odoo board MCP connection (auth + project pin)"
	@echo "make coach   serve the interactive learning pages on 127.0.0.1:8765"

OLLAMA_MODEL := hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M

# How long `make demo` listens for. Three minutes is enough to ask a few things, interrupt one of
# them and hear the session end on its own — override it rather than editing the recipe.
MINUTES ?= 3

setup:
	@command -v uv >/dev/null || { echo "uv not installed: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }
	uv sync
	@test -f .env || { cp .env.example .env; echo "wrote .env from .env.example — fill it in"; }
	@$(MAKE) --no-print-directory fallback-model
	@$(MAKE) --no-print-directory tokenizer
	@$(MAKE) --no-print-directory encoder
	@echo "ok. next: make test"

# The local LLM the turn falls back to when NIM rate-limits or the network is gone. A warning and
# not a failure: without it the remote path still works, you just lose the turn instead of the
# quality when the free tier refuses. ~2 GB, and `make demo` checks it is pulled before timing
# anything rather than downloading it inside a turn that has already gone wrong.
fallback-model:
	@command -v ollama >/dev/null || { \
		echo "warning: ollama not installed — the LLM stage will have no local fallback."; \
		echo "         https://ollama.com/download, then: make fallback-model"; exit 0; }
	@ollama list 2>/dev/null | grep -q "$(OLLAMA_MODEL)" \
		|| ollama pull "$(OLLAMA_MODEL)" \
		|| echo "warning: ollama pull failed — the LLM stage will have no local fallback."

# The tokenizer the chunker counts with (config.TOKENIZER_REPO). `make index` loads it with
# local_files_only, so it has to be cached before an index build — that is what keeps "no network
# calls" true of the build itself rather than only of the second build onwards. ~9 MB, and unlike
# the fallback model this one is a hard requirement: without it there is no index.
tokenizer:
	uv run python -c "from src.sources import fetch_tokenizer; fetch_tokenizer()"

# The sentence encoder for the dense half of retrieval (config.EMBED_ARMS[0]). ~130 MB, fetched
# here so that neither `make index` nor a turn ever downloads weights while something is being
# timed. A warning and not a failure: without it retrieval still runs on BM25 alone, which is
# VOX-030's behaviour — a worse ranking, not a broken one.
encoder:
	@uv run python -c "from src.embeddings import fetch_encoder; fetch_encoder()" 		|| echo "warning: could not fetch the sentence encoder — retrieval will run on BM25 alone."

# The whole unit suite lives under tests/unit/, which is what VOX-007's gate command names. Gates
# are a separate target because they make real calls and need the dev set on disk.
test:
	uv run pytest tests/unit -q

# KNOWN BROKEN, and recorded as gate debt at the MVP freeze rather than papered over: all four
# gates expose main() rather than test_* functions, so pytest collects nothing here and exits 5.
# The four commands are in README.md § The gating table. Fixing this target means first deciding
# what a gate does when it has no key, no audio or no corpus — see notes/build-log/VOX/week-report.md,
# gate debt item 3.
gate:
	@test -n "$$(ls tests/gates/*.py 2>/dev/null)" || { echo "no gates written yet — see tests/gates/README.md"; exit 1; }
	uv run pytest tests/gates -q

# A conversation, not a turn: mic -> silero-vad -> whisper-large-v3-turbo -> retrieval ->
# Llama-3.1-8B -> Kokoro-82M, and then round again until the session clock runs out. Every reply
# plays with the mic still open, so you can talk over it (VOX-011) and a pause is just a pause — the
# session ends on the clock, on Ctrl-C, or after a minute of silence, and prints what it completed.
#
# The two model stages are remote and fall back to local arms if their free tier refuses; startup
# warms those fallbacks and warns if one is not ready. Needs a working microphone and speakers, and
# headphones if you want barge-in to work — there is no echo cancellation, so on open speakers
# silero hears Kokoro and the reply interrupts itself. First run downloads the Kokoro weights
# (~350 MB) and the faster-whisper-base fallback (~150 MB).
#
# How long it runs is config.SESSION_MINUTES, set where every other tunable is set — in the env:
#
#   VOX_SESSION_MINUTES=10 make demo     a longer session
#   VOX_SESSION_MINUTES=0.5 make demo    one or two turns, while iterating on a stage
#   uv run python -m src.loop            the old single turn, for a clean VOX-003 latency split
#
# A question the policy documents cover is answered *from* them, with the doc:page printed under
# the answer and logged on the turn line (VOX-032) — so `make index` first, or startup says why
# nothing will be grounded and every turn takes the plain reply path. `--no-kb` forces that older
# path for a whole run:  uv run python -m src.loop --no-kb
demo:
	uv run python -m src.loop --minutes

# Barge-in (VOX-011). Every turn but the last plays its reply with the mic still open, so talking
# over VOX stops it mid-sentence, prints the stop latency in ms, and feeds the words that stopped it
# into the next turn. Wear headphones: there is no echo cancellation, so on open speakers silero
# hears Kokoro and the reply interrupts itself. See ARCHITECTURE.md § Barge-in.
barge:
	uv run python -m src.loop --turns 3

# The same chain driven from a recording instead of the mic, so the VOX-003 latency split can be
# reproduced without a person at the keyboard. Still plays the reply — time_to_first_audio is not
# measurable without a speaker actually pulling samples. See the script's docstring for why its
# t_vad is not the live number.
turn:
	uv run python scripts/turn_from_fixture.py tests/fixtures/hello_testing_voice.mp3

# VOX-032, without a microphone: the same loop, driven from a spoken question the corpus answers.
# Retrieval runs after STT and the reply is written from the chunks that cleared the floor, so the
# printed line names the doc:page and `runs/turns.jsonl` carries grounded/sources/t_retrieval_ms.
# Needs `make index` and a speaker. See tests/fixtures/README.md on where the recording came from.
ground:
	uv run python scripts/turn_from_fixture.py tests/fixtures/casual_leave_question.mp3 --kb

# Every arm in the registry, one real call each, then the runs/calls.jsonl lines those calls wrote.
# `--list` alone prints the table without calling anything. First run downloads the local weights
# (~1.1 GB); `make demo` does not, because the defaults are unchanged.
arms:
	uv run python scripts/check_arms.py

# VOX-013. Two complete pipelines as real turns, three times each, interleaved, then the five-field
# VOX-003 split for both. Not the same question as `make arms`: that times stages, this times
# architectures, and the gaps between the calls belong to no call.
#
# Needs, or a column comes back FAILED:
#   arm "fast"     the ollama daemon up with the 3B pulled — `make fallback-model`
#   arm "quality"  GROQ_API_KEY and NVIDIA_API_KEY in .env
#   both           a working speaker. time_to_first_audio_ms is not measurable without one, and
#                  five stages is the criterion — `--silent` deliberately fails it.
# First run downloads the piper voice (~65 MB).
compare:
	uv run python scripts/compare_arms.py

# VOX-029 + the dense half. Every PDF in sources/ to runs/chunks.jsonl, then the counts read back
# off the file: files, pages, chunks, and every page that produced no text by name. pypdf parses and
# the tokenizer only counts, so that part needs no network and no model call.
#
# Then the chunks are encoded and the vectors cached to runs/embeddings.npz (~25 s for 215 chunks on
# CPU, once per re-index) — hybrid retrieval needs both halves built from the same text, and the
# cache is keyed to it by fingerprint so a stale vector file is rejected rather than used to cite
# the wrong page. `--no-embed` skips it and leaves retrieval on BM25 alone.
#
# The folder is gitignored (internal HR policies), so a clean clone has nothing to index until
# someone puts the corpus there.
index:
	uv run python scripts/build_index.py

# VOX-030. BM25 over runs/chunks.jsonl: the top 5 chunks for Q, each with doc:page, chunk_idx and
# score, or "not in the documents" when the best score does not clear config.RETRIEVAL_SCORE_FLOOR.
# Needs `make index` to have run. No model, no network, no key — this stage writes no cost log line
# because there is no provider to name.
#
#   make ask Q="how many casual leaves am I entitled to in a year"
#
# `--calibrate` instead of a Q re-measures the floor over evals/dev/retrieval_floor_queries.json.
ask:
	@test -n "$(Q)" || { echo 'usage: make ask Q="how many casual leaves do I get"'; exit 1; }
	uv run python scripts/ask.py "$(Q)"

# Both floors, re-measured against evals/dev/retrieval_floor_queries.json: the lexical one (a
# fraction of the query's information content) and the dense one (a cosine). Prints each column,
# whether it separates answerable from absent, and how the union rule that actually decides a
# refusal routes all 13. No model call for the lexical half; one encoder pass per query for the
# dense half. Run it after changing the corpus, the chunk geometry, the stopword list or the encoder.
floors:
	uv run python scripts/ask.py --calibrate

# VOX-033. The POC gate: ten written queries from evals/dev/pdf_queries.json through the same
# retrieval and answer path the turn loop runs. Prints correct-source@3 over the 8 answerable, the
# refusal rate over the 2 the corpus does not cover, and the grounded-answer rate with its
# denominator; exits non-zero below the floors named in the script. Up to ten free-tier LLM calls,
# so it needs a key and falls back locally like every other model call. Dev-only by construction —
# heldout-v1 holds zero document queries, and the gate says so in its own output.
#
# Not reachable through `make gate`: that is `pytest tests/gates`, which collects nothing here
# because these gates expose main() rather than test_* functions. Run directly, as their docstrings say.
gate-poc:
	uv run python tests/gates/gate_poc_pdf.py

# VOX-031. The same retrieval, then the chunks that cleared the floor go to the LLM arm with
# prompts/answer_from_source_v1.md: answer only from these excerpts, or say you could not find it.
# Prints the spoken answer, the doc:page it was grounded in, and the turn_id that joins this run to
# its runs/calls.jsonl line. The one path in this script that makes a model call, so it needs a key
# — NVIDIA_API_KEY for the default arm, and it falls back to the local ollama arm if the free tier
# refuses. A query that clears no chunk is refused here with no call at all.
#
#   make answer Q="how many casual leaves am I entitled to in a year"
#   LLM=gpt-oss make answer Q="..."      answer with a different arm
answer:
	@test -n "$(Q)" || { echo 'usage: make answer Q="how many casual leaves do I get"'; exit 1; }
	uv run python scripts/ask.py "$(Q)" --answer $(if $(LLM),--llm $(LLM),)

# Creds come from ~/.config/ai-course-board.env (ODOO_USER + ODOO_KEY), never from the repo.
# Board coordinates come from .mcp.json env, so this checks the same config Claude Code uses.
board:
	ODOO_URL=https://odoo.fiftyfivetech.io ODOO_DB=odoo-db ODOO_PROJECT_ID=60 \
		python tools/board_mcp/odoo_board_mcp.py --selftest

# The web-coach bridge: static lesson pages plus a chat file Claude watches. Stdlib only, so
# no venv and no deps — it runs on any machine with python3. Bound to 127.0.0.1, never
# exposed. Protocol and page contract: docs/learning/README.md and tools/coach/README.md.
coach:
	python3 tools/coach/server.py --dir docs/learning/coach --port 8765

clean:
	rm -rf .venv .pytest_cache **/__pycache__
