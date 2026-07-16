.PHONY: install lint type test check ingest ask eval pg-up pg-down

install:
	uv pip install -e ".[dev]"

lint:
	ruff check src tests eval
	ruff format --check src tests eval

type:
	mypy

test:
	pytest

check: lint type test

ingest:
	agentic-rag ingest docs

ask:
	agentic-rag ask "$(Q)"

eval:
	python eval/run_eval.py

# Optional pgvector backend
pg-up:
	docker compose up -d

pg-down:
	docker compose down
