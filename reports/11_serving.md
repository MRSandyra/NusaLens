# Phase 11: API, UI, Observability

FastAPI surface over the Phase 9 `Pipeline`, Phase 7 SQL guard, and Phase 5 retrieval; a Streamlit UI consuming that API; `structlog` JSONL plus `query_log` for observability. Verified live against the real stack (Postgres, Ollama, both loaded models) with `make serve` equivalent (`python -m nusalens.cli serve`), not just unit tests.

**Note on the port in these examples:** port 8000 was already occupied by an unrelated service on this machine (confirmed: its own `/openapi.json` identifies as `"Stake Model Service"`), so verification below runs on `--api-port 8001`. The code's own default stays `8000`, matching the walkthrough's verification command, for a clean machine. See [DECISIONS.md D21](../docs/DECISIONS.md).

## Endpoints: one real request/response each

### `GET /health`
```bash
curl -s localhost:8001/health
```
```json
{"status": "ok", "database": "ok"}
```

### `POST /ask`
```bash
curl -s -X POST localhost:8001/ask -H 'content-type: application/json' \
  -d '{"question":"Berapa inflasi Indonesia tahun 2022?"}'
```
```json
{
  "route": "ABSTAIN",
  "route_source": "rule",
  "answer": "Informasi ini tidak tersedia di sumber yang terindeks (indikator Bank Dunia dan artikel Wikipedia Indonesia). Pertanyaan ini meminta entitas, periode, atau topik yang berada di luar cakupan data yang dicari.",
  "sources": [],
  "sql": null,
  "confidence": 0.0,
  "abstained": true,
  "per_stage_latency_ms": {"route_ms": 0.04}
}
```
This is the walkthrough's own example question, shown as actually returned rather than swapped for a friendlier one. It abstains because the rule router's indicator match is a literal substring against `dim_indicator.name_id` (e.g. `"Inflation, consumer prices (annual %)"`), and the Indonesian word "inflasi" doesn't match that string -- a known, already-documented limitation of the fast-rule layer (D17: "not evidence the rule layer generalizes to free-form user phrasing"), not a Phase 11 bug. `"Berapa nilai Gini index untuk Vietnam pada tahun 2022?"` (literal English indicator name) answers correctly -- see the SQL screenshot below.

### `POST /search`
```bash
curl -s -X POST localhost:8001/search -H 'content-type: application/json' \
  -d '{"query":"Apa itu rutenium?","k":2}'
```
```json
{
  "chunks": [
    {"chunk_id": "wikipedia:Rutenium_0", "title": "Rutenium", "...": "..."},
    {"chunk_id": "wikipedia:Teknesium_0", "title": "Teknesium", "...": "..."}
  ]
}
```

### `POST /sql`
```bash
curl -s -X POST localhost:8001/sql -H 'content-type: application/json' \
  -d '{"sql":"DELETE FROM fact_indicator"}'
```
```json
{"detail": "keyword denylist match"}
```
(HTTP 400 -- the guard rejects it before anything executes.) A safe `SELECT` returns real numeric JSON (see D22: `Decimal` is cast to `float`, not left as a string):
```json
{"sql": "SELECT value FROM v_indicator_series WHERE country_name = 'Indonesia' AND indicator_name = 'Population, total' AND year = 2022 LIMIT 200", "columns": ["value"], "rows": [[278830529.0]]}
```

### `GET /indicators`
```bash
curl -s localhost:8001/indicators
```
```json
{"n": 53, "indicators": [{"code": "EG.ELC.ACCS.ZS", "name_id": "Access to electricity (% of population)", "name_en": "Access to electricity (% of population)", "unit": "", "topic": ""}, "..."]}
```

### `GET /metrics`
```bash
curl -s localhost:8001/metrics
```
```json
{
  "query_count": 155,
  "mean_latency_ms_per_stage": {
    "route_ms": 0.03, "retrieve_ms": 167.79, "rerank_ms": 9118.73, "generate_ms": 27108.52,
    "text2sql_ms": 2753.53, "sql_and_retrieval_concurrent_ms": 8522.14
  },
  "abstain_rate": 0.4286,
  "route_distribution": {"SQL": 39, "ABSTAIN": 22, "DOC": 68, "HYBRID": 26}
}
```
Older `query_log` rows logged before this phase's latency-key rename (`router_ms`/`synthesis_ms`/`retrieval_ms`, pre-Phase-11) still show up alongside the current `route_ms`/`generate_ms`/`retrieve_ms`+`rerank_ms` names in the mean-per-stage breakdown above -- both sets are real, measured values, just from before and after the Phase 11 rename that split retrieval and reranking into separate timers.

## Per-stage latency breakdown

| Stage | Mean (ms) | What it covers |
|---|---|---|
| `route_ms` | 0.03 | Fast-rule classification (no LLM call for anything the rules decide) |
| `retrieve_ms` | 168 | Dense + sparse search, RRF fusion |
| `rerank_ms` | 9,119 | Cross-encoder reranking (`BAAI/bge-reranker-base`, CPU) over the top 30 |
| `generate_ms` | 27,109 | Synthesis (`qwen3:4b`) |
| `text2sql_ms` | 2,754 | SQL generation (`qwen2.5:3b-instruct`) + execution |
| `sql_and_retrieval_concurrent_ms` | 8,522 | HYBRID route's `asyncio.gather` wall-clock (SQL and retrieval run concurrently, so this is close to `max(text2sql_ms, retrieve_ms + rerank_ms)`, not their sum) |

Reranking, not generation, is the single most expensive always-on cost for DOC/HYBRID -- consistent with it running on CPU (D-series: the reranker was deliberately kept off GPU to leave VRAM for the LLM). `route_ms` being effectively free is the fast-rule router's whole point (D17).

## Observability

- `structlog` JSONL: `logs/nusalens.jsonl`, one line per `Pipeline.answer()` call with route, per-stage timers, token counts, prompt versions -- the same fields persisted to `query_log` in Postgres, mirrored to a flat file.
- `query_log` gained two columns this phase (`abstained`, `confidence`; migration `008_query_log_abstain.sql`) so `/metrics`' abstain rate reflects abstention on *every* route (SQL/DOC/HYBRID can abstain on insufficient evidence, not only the rule-decided ABSTAIN route), not just `route = 'ABSTAIN'`.
- Rate limiting: in-memory fixed 60-second window, 30 requests per client IP, returns `429 {"error", "error_type": "RateLimited"}`. Verified in `tests/test_api.py`.
- Global error handler: any uncaught exception returns `{"error", "error_type"}` JSON at HTTP 500; `/sql`'s guard rejection is a deliberate `400` via `HTTPException` instead, since that's a client input problem, not a server fault.

## UI screenshots

Three real scenarios, captured from the actually-running UI (not mocked), full flow question -> API -> rendered page:

- [`reports/figures/11_ui_sql.png`](figures/11_ui_sql.png) -- SQL route, "Berapa nilai Gini index untuk Vietnam pada tahun 2022?", showing the answer, the "Tampilkan SQL" panel, and the indicator-trend chart (fixed per D22).
- [`reports/figures/11_ui_doc.png`](figures/11_ui_doc.png) -- DOC route, the Pope Pius IX word-relationship question from the Phase 8 judge-calibration set, showing the answer and clickable source citations.
- [`reports/figures/11_ui_abstain.png`](figures/11_ui_abstain.png) -- ABSTAIN route, "Berapa jumlah penduduk Gondor menurut data Bank Dunia?", answered in 33ms with no LLM call at all.

## DoD

1. `make serve` starts the API and the UI, and `GET /health` returns 200. **Met** on `--api-port 8001` (D21); code default stays `8000` per spec.
2. OpenAPI docs are available at `/docs`. **Met**: `curl -s -o /dev/null -w '%{http_code}' localhost:8001/docs` -> `200`.
3. The three screenshots exist. **Met**, see above.
4. This report, with an example request/response per endpoint and a per-stage latency table. **Met.**

## Artifacts

- `src/nusalens/api/app.py` -- FastAPI app, 6 endpoints, Pydantic models, global error handler, rate limiter.
- `src/nusalens/ui/streamlit_app.py` -- Streamlit UI.
- `src/nusalens/observability.py` -- structlog JSONL configuration.
- `src/nusalens/warehouse/migrations/008_query_log_abstain.sql`.
- `tests/test_api.py` -- 7 passing tests.
- `reports/figures/11_ui_*.png` -- the three screenshots.
