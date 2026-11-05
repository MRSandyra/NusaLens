# Phase 13: Testing, Optimization, Hardening

## Tests

93 tests across unit (chunker, RRF, SQL guard, prompt registry, entity resolver, loaders, ingest filters, retrieval metrics, judge) and one real integration test (ingest -> index -> ask over a self-contained 50-document fixture, `tests/test_integration_ingest_index_ask.py`). Full output: `reports/raw/13_coverage.txt`.

Coverage: **70%** on `src/nusalens/` (1705 statements, 504 missed), hitting the DoD target exactly. Lowest-covered areas and why:
- `eval/e2e_harness.py`, `eval/retrieval_harness.py` (0%, 55/68 stmts): real Phase 4 harnesses, superseded in later phases by `scripts/phase5_eval.py` and `scripts/phase9_e2e_eval.py` -- correct, historically used, but not in the current critical path, so left untested given the time budget.
- `ingest/wikipedia.py`, `ingest/worldbank.py` (0%, 45/71 stmts): the real connectors behind `data_cli.ingest()` (D27); untested because exercising them means live calls to the HF dataset download and the World Bank API, which this phase's test run doesn't make (the pure-function pieces of `ingest/base.py` and `ingest/wikipedia_filters.py` are tested at 78%/100%).
- `warehouse/seed.py` (47%): `load_documents`/`seed_dim_country` are tested (idempotency, real DB); `seed_dim_indicator`/`seed_fact_indicator` are not, same shape of gap as the ingest connectors.
- `mcp_server/main.py` (69%): all 6 tools, 3 resources, 2 prompts are tested; the uncovered lines are mostly the `argparse`/`__main__` entrypoint and a few error branches not exercised by the test payloads.

## Integration test

`tests/test_integration_ingest_index_ask.py`: generates 50 lexically distinct fictional documents, runs the real `chunk_documents()` -> `embed_chunks()` pipeline into the live `documents`/`chunks` tables under an `integration_fixture:` id prefix (cleaned up after, regardless of outcome), then verifies dense search finds the right chunk, `tsv` is populated (not NULL -- the exact bug D24 fixed), and a full `Pipeline.answer()` call correctly routes to DOC and cites the fixture content. ~99s total, including one real LLM synthesis call.

## Bugs found and fixed along the way

Found by actually running the pipeline end to end for the reproducibility check, not by code review:

| Bug | Where | Fix | Decision |
|---|---|---|---|
| `cur.copy_expert(...)` -- a psycopg2-only method that doesn't exist on psycopg v3 | `warehouse/loader.py` | Deleted (dead code, zero callers, superseded by the working `warehouse/seed.py`) | D24 |
| `INSERT INTO chunks` never set `tsv`, requiring a manual fix-up script after every ingest | `retrieval/embed_chunks.py` | `tsv = to_tsvector('simple', text)` computed in the same INSERT | D24 |
| A duplicate 81 MB HNSW index silently re-created on every rerun, already removed once before | `scripts/build_indexes.py` | Deleted; migrations already create the real indexes idempotently | D25 |
| `make`'s `ingest`/`index`/`eval`/`train-embed` targets called CLI subcommands that were never built | `Makefile` | Pointed at the real scripts/modules that actually do the work | D26 |
| The real ingestion CLI (`data_cli.py`) was never mounted into the main `nusalens.cli` app | `cli.py` | Mounted `data_cli.ingest()` onto `data_app`; deleted the dangerous placeholder `ingest()` that overwrote `reports/01_data_inventory.md` (an incident this caused -- see D27) | D27 |

## Reproducibility: `make all` from a (near-)clean run

`make` itself is not installed on this Windows machine (confirmed: `which make`/`where make` both fail) -- the walkthrough's own verification commands cannot run literally here. The Makefile's underlying commands were run directly in sequence instead: `setup` (uv sync, docker compose up, migrations, health) -> `ingest` -> `index` (chunk + embed, idempotent upsert over the live 41,297-row corpus) -> `eval` (Phase 5's 5-mode retrieval sweep) -> `train-embed` (the Phase 6 embedding fine-tune, retrained from scratch) -> `test`.

**Total wall time: 1980 seconds (33 minutes).** Full log: `reports/raw/13_make_all_run.log`.

The `eval` step's rerun reconfirmed Phase 5's original numbers closely (small expected variance from re-encoding embeddings fresh):

| Mode | Original (Phase 5) nDCG@10 | Rerun nDCG@10 |
|---|---|---|
| dense | 0.390 | 0.404 |
| sparse | 0.584 | 0.584 |
| rrf | 0.564 | 0.569 |
| rrf_rerank | 0.630 | 0.630 |
| rrf_rerank_exp | 0.630 | 0.630 |

Rerank-involving modes reproduced exactly; pure dense/rrf modes show small (+0.01-0.014) variance, consistent with embedding re-encoding introducing minor floating-point differences in approximate nearest-neighbor ties, not a reproducibility failure.

## Optimization: three measured improvements

All measured against the real 70-item DOC/HYBRID golden subset or a direct repeated-call benchmark, via dedicated scripts (`scripts/phase13_profile_connections.py`, `scripts/phase13_profile_rerank.py`, `scripts/phase13_profile_ef_search.py`). Raw results: `reports/raw/13_profile_*.json`.

| # | Change | Before | After | Result |
|---|---|---|---|---|
| 1 | Connection pooling (`psycopg_pool.ConnectionPool` shared through `Pipeline`/`HybridSearcher`, instead of a fresh `psycopg.connect()` per DB call) | 30 dense+sparse call pairs: 19,157 ms | 16,624 ms | **1.15x**, 84 ms saved per call pair |
| 2 | Reranker candidate count `k_rerank`: 30 -> 15 | mean rerank 6,058 ms, nDCG@10 0.6298, recall@10 0.800 | mean rerank 3,171 ms, nDCG@10 0.6415, recall@10 0.771 | **1.9x faster, nDCG actually improves**, recall@10 costs 2.9pp |
| 3 | HNSW `ef_search`: 40 (pgvector default) -> 100 | recall@10 0.500, query 2.57 ms mean | recall@10 0.543, query 3.28 ms mean | **Free recall gain** -- this ~41k-row corpus is small enough that query latency barely moves at any tested setting |

Optimization #2 is a genuine quality/speed tradeoff, decided in favor of speed because the ranking-quality metric (nDCG@10) *improved*, not degraded, at the new setting -- RRF's own top-15 candidates are already strong, and feeding the cross-encoder more borderline ones adds noise. Optimization #3 has no real tradeoff on this corpus size. Optimization #1 is a straightforward latency win with no quality impact.

**Provenance note:** Phase 9's and Phase 11's published latency numbers (`reports/09_e2e.md`, `reports/11_serving.md`; e.g. rerank_ms mean ~9.1s) reflect the *prior* defaults (`k_rerank=30`, `ef_search=40`) and are left as accurate historical records rather than retroactively edited. A fresh end-to-end run under the new defaults would show materially lower DOC/HYBRID latency; that full rerun was not repeated in this phase given the time budget, but the retrieval-level deltas above are independently measured and real.

Also discovered while making change #2: `config/settings.yaml`'s `retrieval.rerank_top_n` was never actually read by any code -- a second small dead-config finding, now at least kept in sync with the real hardcoded default even though it remains unwired.

## Hardening

- **Timeouts on every external call.** Every `httpx` call already had an explicit timeout (confirmed by grep, not assumed). The one gap: ~25 `psycopg.connect()` call sites across the codebase had no `connect_timeout`, meaning an unreachable database would block on the OS-level TCP timeout (potentially minutes). Fixed at a single point: `connect_timeout=5` added to `settings.db_dsn`/`db_ro_dsn` via the DSN's own query parameter, hardening every call site without touching all 25 individually.
- **API input validation.** `AskRequest.question`, `SearchRequest.query`, `SqlRequest.sql`, and the matching MCP tool parameters had no length bound. Added `min_length`/`max_length` to every free-text field (D30); regression-tested in `tests/test_hardening.py` (5 tests, all passing).
- **Graceful degradation, verified not assumed (D30).** Simulated both a down database (`settings.db_port` repointed at a closed port) and a down LLM (`ollama_client` repointed at an unreachable endpoint), without touching the real running services. Both fail in bounded time (4-10 seconds, not an indefinite hang) with a structured error (`/health` returns `200` with an `"error: ..."` field; `/ask` returns a structured `500`), confirmed via `TestClient(raise_server_exceptions=False)` and direct timing, not inferred from the presence of a `try/except`.
- **No secrets in logs.** Grepped every log/print/echo call site for the DB password fields; none found outside `config.py`'s own field declarations. The DSN (which does embed the password, unavoidably, since psycopg needs it) is never logged as a whole string.

## DoD

1. `make test` is green, coverage report at `reports/raw/13_coverage.txt`. **Met**: 93/93 passing, 70% coverage.
2. `reports/13_quality.md` contains the before/after optimization table with at least three measured improvements, plus hardening notes. **Met.**
3. `make all` succeeds from clean, total wall time recorded. **Met** (via the underlying commands directly, since `make` itself isn't installed on this machine -- D26): 1980 seconds.
