.PHONY: setup test gate demo board clean
.DEFAULT_GOAL := help

help:
	@echo "make setup   create the venv and install deps (uv)"
	@echo "make test    unit tests"
	@echo "make gate    run every phase gate in tests/gates/"
	@echo "make demo    run the thing end to end"
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

demo:
	@echo "not implemented yet. make demo must run the system end to end from a clean clone."
	@exit 1

# Creds come from ~/.config/ai-course-board.env (ODOO_USER + ODOO_KEY), never from the repo.
# Board coordinates come from .mcp.json env, so this checks the same config Claude Code uses.
board:
	ODOO_URL=https://odoo.fiftyfivetech.io ODOO_DB=odoo-db ODOO_PROJECT_ID=60 \
		python tools/board_mcp/odoo_board_mcp.py --selftest

clean:
	rm -rf .venv .pytest_cache **/__pycache__
