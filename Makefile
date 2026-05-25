.PHONY: install format lint test eval run docker-up docker-down docker-build

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

eval:
	uv run python scripts/run_evaluation.py --dataset $(DATASET) --threshold 0.95

run:
	uv run uvicorn rag.main:app --reload --host 0.0.0.0 --port 8000

docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down
