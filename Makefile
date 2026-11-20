PYTHON := .venv/Scripts/python

.PHONY: setup setup-env models migrate seed health ingest index bootstrap verify eval train-embed serve mcp test test-unit test-integration all db-up db-down

setup-env:
	uv sync --group dev

models:
	ollama pull qwen3:4b
	ollama pull qwen2.5:3b-instruct
	$(PYTHON) scripts/download_models.py

setup: setup-env
	$(PYTHON) scripts/detect_hardware.py > reports/raw/00_hardware.json
	docker compose up -d
	$(PYTHON) scripts/wait_for_db.py

migrate:
	$(PYTHON) -m nusalens.warehouse.migrations.runner

seed:
	$(PYTHON) -m nusalens.warehouse.seed

db-up:
	docker compose up -d

db-down:
	docker compose down

health:
	$(PYTHON) -m nusalens.cli health

ingest:
	$(PYTHON) -m nusalens.cli data ingest

index:
	$(PYTHON) -m nusalens.retrieval.chunker
	$(PYTHON) -m nusalens.retrieval.embed_chunks

bootstrap: setup models migrate ingest seed index health

eval:
	$(PYTHON) scripts/phase5_eval.py

train-embed:
	$(PYTHON) training/train_embed.py

serve:
	$(PYTHON) -m nusalens.cli serve

mcp:
	$(PYTHON) -m nusalens.mcp_server.main --transport stdio

test:
	$(PYTHON) -m pytest -m "not integration" -v

test-unit:
	$(PYTHON) -m pytest -m "not integration and not safe_integration" -v

test-integration:
	$(PYTHON) scripts/preflight_runtime.py
	$(PYTHON) -m pytest -m safe_integration -v
	$(PYTHON) -m pytest -m integration -v

verify:
	$(PYTHON) scripts/verify_evaluation.py

all: bootstrap verify test
