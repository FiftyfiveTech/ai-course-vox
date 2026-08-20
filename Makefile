.PHONY: setup fallback-model tokenizer test gate demo barge turn arms compare index ask answer board clean
.DEFAULT_GOAL := help

help:
	@echo "make setup   create the venv and install deps (uv), and pull the local fallback model"
	@echo "make test    unit tests"
	@echo "make gate    run every phase gate in tests/gates/"
	@echo "make demo    run the thing end to end (needs a mic)"
	@echo "make barge   three turns with interruptible replies — talk over it (VOX-011)"
	@echo "make turn    one instrumented turn from a recording — no mic needed"
	@echo "make arms    call every registered model arm once and print the log (VOX-006)"
	@echo "make compare two whole architectures end to end, five-stage split for both (VOX-013)"
	@echo "make index   extract the sources/ PDFs to text chunks and print the counts (VOX-029)"
	@echo "make ask     Q=\"...\" retrieve the top chunks for a question, with file:page (VOX-030)"
	@echo "make answer  Q=\"...\" the same chunks through the LLM arm, as a spoken answer (VOX-031)"
	@echo "make board   verify the Odoo board MCP connection (auth + project pin)"

OLLAMA_MODEL := hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M

setup:
	@command -v uv >/dev/null || { echo "uv not installed: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }
	uv sync
	@test -f .env || { cp .env.example .env; echo "wrote .env from .env.example — fill it in"; }
	@$(MAKE) --no-print-directory fallback-model
	@$(MAKE) --no-print-directory tokenizer
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

# The whole unit suite lives under tests/unit/, which is what VOX-007's gate command names. Gates
# are a separate target because they make real calls and need the dev set on disk.
test:
	uv run pytest tests/unit -q

gate:
	@test -n "$$(ls tests/gates/*.py 2>/dev/null)" || { echo "no gates written yet — see tests/gates/README.md"; exit 1; }
	uv run pytest tests/gates -q

# One chained turn: mic -> silero-vad -> whisper-large-v3-turbo -> Llama-3.1-8B -> Kokoro-82M.
# The two middle stages are remote and fall back to local arms if their free tier refuses; startup
# warms those fallbacks and warns if one is not ready. Needs a working microphone and speakers.
# First run downloads the Kokoro weights (~350 MB) and the faster-whisper-base fallback (~150 MB).
demo:
	uv run python -m src.loop

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

# VOX-029. Every PDF in sources/ to runs/chunks.jsonl, then the counts read back off the file:
# files, pages, chunks, and every page that produced no text by name. No network and no model call
# — pypdf parses, and the tokenizer only counts. The folder is gitignored (internal HR policies),
# so a clean clone has nothing to index until someone puts the corpus there.
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

clean:
	rm -rf .venv .pytest_cache **/__pycache__
