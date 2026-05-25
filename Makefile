.PHONY: install format lint test run

install:
	uv sync --all-extras

format:
	uv run ruff format src tests
	uv run ruff check --fix src tests

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests

test:
	uv run pytest

run:
	uv run uvicorn rag.main:app --reload --host 0.0.0.0 --port 8000
