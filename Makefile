.PHONY: setup fallback-model test gate demo barge turn arms board coach clean
.DEFAULT_GOAL := help

help:
	@echo "make setup   create the venv and install deps (uv), and pull the local fallback model"
	@echo "make test    unit tests"
	@echo "make gate    run every phase gate in tests/gates/"
	@echo "make demo    run the thing end to end (needs a mic)"
	@echo "make barge   three turns with interruptible replies — talk over it (VOX-011)"
	@echo "make turn    one instrumented turn from a recording — no mic needed"
	@echo "make arms    call every registered model arm once and print the log (VOX-006)"
	@echo "make board   verify the Odoo board MCP connection (auth + project pin)"
	@echo "make coach   serve the interactive learning pages on 127.0.0.1:8765"

OLLAMA_MODEL := hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M

setup:
	@command -v uv >/dev/null || { echo "uv not installed: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }
	uv sync
	@test -f .env || { cp .env.example .env; echo "wrote .env from .env.example — fill it in"; }
	@$(MAKE) --no-print-directory fallback-model
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
