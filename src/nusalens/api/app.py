# Every route returns a Pydantic response model; every exception (ours or FastAPI's own) is caught
# by the global handler below and returned as {"error", "error_type"} JSON, never a raw traceback.
import collections
import statistics
import time
from decimal import Decimal
from pathlib import Path

import psycopg
import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from nusalens.agents.sql_guard import execute_readonly, validate_sql
from nusalens.config import settings
from nusalens.pipeline import Pipeline
from nusalens.retrieval.hybrid_search import HybridSearcher
from nusalens.warehouse.migrations.runner import MIGRATIONS_DIR

app = FastAPI(title="NusaLens API", version="0.1.0")

_pipeline: Pipeline | None = None
_searcher: HybridSearcher | None = None


def get_pipeline() -> Pipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = Pipeline()
    return _pipeline


def get_searcher() -> HybridSearcher:
    global _searcher
    if _searcher is None:
        _searcher = HybridSearcher(settings.db_dsn, device="cpu")
    return _searcher


def readiness() -> dict[str, str]:
    """Return dependency states without constructing models or running a user query."""
    state = {"database": "unavailable", "migrations": "unavailable", "index": "unavailable", "ollama": "unavailable"}
    try:
        with psycopg.connect(settings.db_dsn, connect_timeout=5) as conn, conn.cursor() as cur:
            cur.execute("SELECT 1")
            state["database"] = "ok"
            cur.execute("SELECT to_regclass('public.schema_migrations')")
            if cur.fetchone()[0] is None:
                state["migrations"] = "missing"
            else:
                cur.execute("SELECT count(*) FROM schema_migrations")
                applied = cur.fetchone()[0]
                expected = len(list(Path(MIGRATIONS_DIR).glob("[0-9][0-9][0-9]_*.sql")))
                state["migrations"] = "ok" if applied >= expected else f"pending ({applied}/{expected})"
                if state["migrations"] == "ok":
                    cur.execute("SELECT count(*) FROM chunks")
                    n_chunks = cur.fetchone()[0]
                    state["index"] = "ok" if n_chunks else "empty"
    except Exception:
        # The detail belongs in server logs, not an unauthenticated readiness response.
        pass

    try:
        response = httpx.get(f"{settings.llm_endpoint}/api/tags", timeout=3.0)
        response.raise_for_status()
        available = {model["name"] for model in response.json().get("models", [])}
        required = {settings.generator_model, settings.text2sql_model, settings.router_model}
        missing = required - available
        state["ollama"] = "ok" if not missing else f"missing models: {', '.join(sorted(missing))}"
    except Exception:
        pass
    return state


def require_ready(*requirements: str) -> None:
    state = readiness()
    missing = {name: state[name] for name in requirements if state[name] != "ok"}
    if missing:
        detail = "; ".join(f"{name}={value}" for name, value in missing.items())
        raise HTTPException(status_code=503, detail=f"service unavailable: {detail}")


# ---------------------------------------------------------------------------
# In-memory rate limit: fixed 60s window, per client IP.
# ---------------------------------------------------------------------------

RATE_LIMIT_PER_MINUTE = 30
_request_log: dict[str, collections.deque] = collections.defaultdict(collections.deque)


class RateLimitMiddleware:
    """Small ASGI middleware; avoids BaseHTTPMiddleware's streaming/task-group edge cases."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        client = scope.get("client")
        client_ip = client[0] if client else "unknown"
        now = time.time()
        window = _request_log[client_ip]
        while window and now - window[0] > 60:
            window.popleft()
        if len(window) >= RATE_LIMIT_PER_MINUTE:
            response = JSONResponse(
                status_code=429,
                content={"error": f"rate limit exceeded: {RATE_LIMIT_PER_MINUTE} requests per 60s", "error_type": "RateLimited"},
            )
            await response(scope, receive, send)
            return
        window.append(now)
        await self.app(scope, receive, send)


app.add_middleware(RateLimitMiddleware)


@app.exception_handler(Exception)
async def global_error_handler(request: Request, exc: Exception) -> JSONResponse:
    # Do not leak DSNs, SQL, or internal topology through an HTTP error.
    return JSONResponse(status_code=500, content={"error": "internal server error", "error_type": type(exc).__name__})


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000, description="a question in Indonesian")


class SourceModel(BaseModel):
    title: str
    url: str | None = None


class AskResponse(BaseModel):
    route: str
    route_source: str
    answer: str
    sources: list[SourceModel] = Field(default_factory=list)
    sql: str | None = None
    confidence: float
    abstained: bool
    per_stage_latency_ms: dict = Field(default_factory=dict)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    k: int = Field(default=8, ge=1, le=20)


class ChunkModel(BaseModel):
    chunk_id: str
    text: str
    title: str | None = None
    url: str | None = None


class SearchResponse(BaseModel):
    chunks: list[ChunkModel]


class SqlRequest(BaseModel):
    sql: str = Field(min_length=1, max_length=2000, description="a single read-only SELECT over v_indicator_series")


class SqlResponse(BaseModel):
    sql: str
    columns: list[str]
    rows: list[list]


class IndicatorModel(BaseModel):
    code: str
    name_id: str
    name_en: str
    unit: str | None = None
    topic: str | None = None


class IndicatorsResponse(BaseModel):
    indicators: list[IndicatorModel]
    n: int


class HealthResponse(BaseModel):
    status: str
    database: str
    migrations: str
    index: str
    ollama: str


class MetricsResponse(BaseModel):
    query_count: int
    mean_latency_ms_per_stage: dict[str, float]
    abstain_rate: float
    route_distribution: dict[str, int]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    state = readiness()
    return HealthResponse(status="ok" if all(value == "ok" for value in state.values()) else "degraded", **state)


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    require_ready("database", "migrations", "index", "ollama")
    result = get_pipeline().answer(req.question)
    return AskResponse(
        route=result.route, route_source=result.route_source, answer=result.summary,
        sources=[SourceModel(title=s.title, url=s.url) for s in result.sources],
        sql=result.sql, confidence=result.confidence, abstained=result.abstained,
        per_stage_latency_ms=result.per_stage_latency_ms,
    )


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest) -> SearchResponse:
    require_ready("database", "migrations", "index")
    hits, _retrieve_ms, _rerank_ms = get_searcher().search_rrf_rerank_exp_timed(req.query, k=req.k)
    doc_ids = list(dict.fromkeys(h.doc_id for h, _ in hits))
    titles: dict[str, tuple] = {}
    if doc_ids:
        with psycopg.connect(settings.db_dsn) as conn, conn.cursor() as cur:
            cur.execute("SELECT doc_id, title, url FROM documents WHERE doc_id = ANY(%s)", (doc_ids,))
            titles = {r[0]: (r[1], r[2]) for r in cur.fetchall()}
    return SearchResponse(chunks=[
        ChunkModel(chunk_id=h.chunk_id, text=h.text, title=titles.get(h.doc_id, (None, None))[0],
                   url=titles.get(h.doc_id, (None, None))[1])
        for h, _ in hits
    ])


@app.post("/sql", response_model=SqlResponse)
def run_sql(req: SqlRequest) -> SqlResponse:
    require_ready("database", "migrations")
    result = validate_sql(req.sql, settings.db_dsn)
    if not result.ok:
        raise HTTPException(status_code=400, detail=result.reason)
    cols, rows = execute_readonly(result.sql, settings.db_ro_dsn, dsn_for_explain=settings.db_dsn)
    # Postgres NUMERIC comes back as Decimal, which Pydantic serializes to a JSON string rather
    # than a number; cast to float so clients (the UI's chart included) get real numeric JSON.
    out_rows = [[float(v) if isinstance(v, Decimal) else v for v in r] for r in rows]
    return SqlResponse(sql=result.sql, columns=cols, rows=out_rows)


@app.get("/indicators", response_model=IndicatorsResponse)
def indicators() -> IndicatorsResponse:
    require_ready("database", "migrations")
    with psycopg.connect(settings.db_dsn) as conn, conn.cursor() as cur:
        cur.execute("SELECT code, name_id, name_en, unit, topic FROM dim_indicator ORDER BY name_id")
        cols = [d[0] for d in cur.description]
        rows = [dict(zip(cols, row)) for row in cur.fetchall()]
    return IndicatorsResponse(indicators=[IndicatorModel(**r) for r in rows], n=len(rows))


@app.get("/metrics", response_model=MetricsResponse)
def metrics() -> MetricsResponse:
    require_ready("database", "migrations")
    with psycopg.connect(settings.db_dsn) as conn, conn.cursor() as cur:
        cur.execute("SELECT route, abstained, per_stage_latency_ms FROM query_log")
        rows = cur.fetchall()

    route_distribution: dict[str, int] = collections.Counter()
    abstain_flags = []
    stage_values: dict[str, list[float]] = collections.defaultdict(list)
    for route, abstained, latency in rows:
        route_distribution[route or "unknown"] += 1
        if abstained is not None:
            abstain_flags.append(bool(abstained))
        for stage, ms in (latency or {}).items():
            stage_values[stage].append(ms)

    mean_latency = {stage: round(statistics.mean(values), 2) for stage, values in stage_values.items()}
    abstain_rate = (sum(abstain_flags) / len(abstain_flags)) if abstain_flags else 0.0

    return MetricsResponse(
        query_count=len(rows),
        mean_latency_ms_per_stage=mean_latency,
        abstain_rate=round(abstain_rate, 4),
        route_distribution=dict(route_distribution),
    )
