PYTHON := .venv/Scripts/python

.PHONY: setup health ingest index eval train-embed serve mcp test all db-up db-down

setup:
	uv sync
	$(PYTHON) scripts/detect_hardware.py > reports/raw/00_hardware.json
	docker compose up -d
	$(PYTHON) scripts/wait_for_db.py
	docker compose exec -T db psql -U nusalens -d nusalens -c "CREATE EXTENSION IF NOT EXISTS vector;"
	$(PYTHON) -m nusalens.cli health

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

eval:
	$(PYTHON) scripts/phase5_eval.py

train-embed:
	$(PYTHON) training/train_embed.py

serve:
	$(PYTHON) -m nusalens.cli serve

mcp:
	$(PYTHON) -m nusalens.mcp_server.main --transport stdio

test:
	$(PYTHON) -m pytest tests/ -v

all: setup ingest index eval train-embed test
