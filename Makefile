.PHONY: setup test gate demo turn arms board clean
.DEFAULT_GOAL := help

help:
	@echo "make setup   create the venv and install deps (uv)"
	@echo "make test    unit tests"
	@echo "make gate    run every phase gate in tests/gates/"
	@echo "make demo    run the thing end to end (needs a mic)"
	@echo "make turn    one instrumented turn from a recording — no mic needed"
	@echo "make arms    call every registered model arm once and print the log (VOX-006)"
	@echo "make board   verify the Odoo board MCP connection (auth + project pin)"

setup:
	@command -v uv >/dev/null || { echo "uv not installed: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }
	uv sync
	@test -f .env || { cp .env.example .env; echo "wrote .env from .env.example — fill it in"; }
	@echo "ok. next: make test"

test:
	uv run pytest tests -q --ignore=tests/gates

gate:
	@test -n "$$(ls tests/gates/*.py 2>/dev/null)" || { echo "no gates written yet — see tests/gates/README.md"; exit 1; }
	uv run pytest tests/gates -q

# One chained turn: mic -> silero-vad -> whisper-large-v3-turbo -> Llama-3.1-8B -> Kokoro-82M.
# Needs a working microphone and speakers. First run downloads the Kokoro weights (~350 MB).
demo:
	uv run python -m src.loop

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

clean:
	rm -rf .venv .pytest_cache **/__pycache__
