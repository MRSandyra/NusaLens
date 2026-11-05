# Written against the installed mcp==2.3.0 API (mcp.server.mcpserver.MCPServer), verified directly
# by inspection before writing any of this, not from memory; see docs/DECISIONS.md D20.
import argparse
import json

import httpx
import psycopg
from mcp.server.mcpserver import MCPServer
from pydantic import BaseModel, Field

from nusalens.agents.sql_guard import execute_readonly, validate_sql
from nusalens.config import settings
from nusalens.pipeline import Pipeline
from nusalens.retrieval.hybrid_search import HybridSearcher
from nusalens.warehouse.schema_card import generate_schema_card

BMKG_LATEST_EARTHQUAKE_URL = "https://data.bmkg.go.id/DataMKG/TEWS/autogempa.json"

server = MCPServer(
    "nusalens",
    version="0.1.0",
    instructions=(
        "Indonesian public-data analytics assistant over World Bank indicators and indexed "
        "Indonesian Wikipedia articles. Use query_indicators or run_readonly_sql for numeric "
        "questions, search_documents for article-based questions, and answer_question for a "
        "single end-to-end answer that routes and synthesizes automatically."
    ),
)

# Heavy resources (embedding model, reranker, DB-backed name lookups) load lazily on first use,
# not at server startup, so list_tools/list_resources/list_prompts stay fast.
_pipeline: Pipeline | None = None
_searcher: HybridSearcher | None = None
_entity_names: dict | None = None


def _get_pipeline() -> Pipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = Pipeline()
    return _pipeline


def _get_searcher() -> HybridSearcher:
    global _searcher
    if _searcher is None:
        _searcher = HybridSearcher(settings.db_dsn, device="cpu")
    return _searcher


def _get_entity_names() -> dict:
    global _entity_names
    if _entity_names is None:
        with psycopg.connect(settings.db_dsn) as conn, conn.cursor() as cur:
            cur.execute("SELECT name_id FROM dim_country")
            countries = {r[0].lower(): r[0] for r in cur.fetchall()}
            cur.execute("SELECT name_id FROM dim_indicator")
            indicators = {r[0].lower(): r[0] for r in cur.fetchall()}
        _entity_names = {"countries": countries, "indicators": indicators}
    return _entity_names


def _error(exc: Exception, **extra) -> dict:
    # Every tool returns this shape on failure instead of letting the exception propagate as a
    # raw traceback over the protocol (Phase 10 checklist: "structured errors").
    return {"ok": False, "error": str(exc), "error_type": type(exc).__name__, **extra}


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

class SearchDocumentsParams(BaseModel):
    query: str = Field(min_length=1, max_length=1000, description="Indonesian search query")
    k: int = Field(default=8, ge=1, le=20, description="number of chunks to return")


@server.tool(description=(
    "Hybrid search (dense + sparse + RRF + cross-encoder rerank) over the indexed Indonesian "
    "Wikipedia articles. Returns chunks with their source document title and URL as citations. "
    "Use this for questions about facts, history, or definitions covered by an article -- not for "
    "numeric World Bank indicator values, which query_indicators or run_readonly_sql answer directly."
))
def search_documents(params: SearchDocumentsParams) -> dict:
    try:
        hits = _get_searcher().search_rrf_rerank_exp(params.query, k=params.k)
        doc_ids = list(dict.fromkeys(h.doc_id for h in hits))
        titles: dict[str, tuple] = {}
        if doc_ids:
            with psycopg.connect(settings.db_dsn) as conn, conn.cursor() as cur:
                cur.execute("SELECT doc_id, title, url FROM documents WHERE doc_id = ANY(%s)", (doc_ids,))
                titles = {r[0]: (r[1], r[2]) for r in cur.fetchall()}
        return {
            "ok": True,
            "chunks": [
                {
                    "chunk_id": h.chunk_id,
                    "text": h.text,
                    "title": titles.get(h.doc_id, (None, None))[0],
                    "url": titles.get(h.doc_id, (None, None))[1],
                }
                for h in hits
            ],
        }
    except Exception as exc:
        return _error(exc)


class QueryIndicatorsParams(BaseModel):
    country: str | None = Field(default=None, description="Indonesian country name, e.g. 'Indonesia', 'Vietnam'")
    indicator: str | None = Field(
        default=None,
        description="exact World Bank indicator name as returned by list_indicators, e.g. 'GDP per capita (current US$)'",
    )
    year_from: int | None = Field(default=None, description="inclusive start year")
    year_to: int | None = Field(default=None, description="inclusive end year")
    limit: int = Field(default=50, ge=1, le=500)


@server.tool(description=(
    "Structured query over World Bank indicator data by country, indicator name, and year range. "
    "Not free-form SQL -- call list_indicators first to find the exact indicator name string. Use "
    "run_readonly_sql instead for anything this structured shape cannot express (aggregates, "
    "ranking, joins across years)."
))
def query_indicators(params: QueryIndicatorsParams) -> dict:
    try:
        names = _get_entity_names()
        clauses, args = [], []
        if params.country:
            canon = names["countries"].get(params.country.lower())
            if canon is None:
                return _error(ValueError(f"unknown country: {params.country}"))
            clauses.append("country_name = %s")
            args.append(canon)
        if params.indicator:
            canon = names["indicators"].get(params.indicator.lower())
            if canon is None:
                return _error(ValueError(f"unknown indicator: {params.indicator}"))
            clauses.append("indicator_name = %s")
            args.append(canon)
        if params.year_from is not None:
            clauses.append("year >= %s")
            args.append(params.year_from)
        if params.year_to is not None:
            clauses.append("year <= %s")
            args.append(params.year_to)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = (
            f"SELECT country_name, indicator_name, year, value, unit FROM v_indicator_series {where} "
            "ORDER BY country_name, indicator_name, year LIMIT %s"
        )
        args.append(params.limit)
        with psycopg.connect(settings.db_dsn) as conn, conn.cursor() as cur:
            cur.execute(sql, args)
            cols = [d[0] for d in cur.description]
            rows = [dict(zip(cols, row)) for row in cur.fetchall()]
        return {"ok": True, "rows": rows, "n": len(rows)}
    except Exception as exc:
        return _error(exc)


class RunReadonlySqlParams(BaseModel):
    sql: str = Field(min_length=1, max_length=2000, description="a single read-only SELECT statement over v_indicator_series")


@server.tool(description=(
    "Execute a read-only SQL SELECT over v_indicator_series, behind the full Phase 7 guardrails: "
    "AST-parsed, keyword-denylisted, EXPLAIN-cost-checked, auto-LIMITed, and run under a "
    "statement_timeout read-only database role. Any write statement, multi-statement input, or "
    "over-budget query is rejected with a structured error and never executed."
))
def run_readonly_sql(params: RunReadonlySqlParams) -> dict:
    try:
        result = validate_sql(params.sql, settings.db_dsn)
        if not result.ok:
            return _error(ValueError(result.reason), sql=params.sql)
        cols, rows = execute_readonly(result.sql, settings.db_ro_dsn)
        return {"ok": True, "sql": result.sql, "columns": cols, "rows": [list(r) for r in rows]}
    except Exception as exc:
        return _error(exc)


@server.tool(description="Catalog of every available World Bank indicator: code, Indonesian/English name, unit, topic.")
def list_indicators() -> dict:
    try:
        with psycopg.connect(settings.db_dsn) as conn, conn.cursor() as cur:
            cur.execute("SELECT code, name_id, name_en, unit, topic FROM dim_indicator ORDER BY name_id")
            cols = [d[0] for d in cur.description]
            rows = [dict(zip(cols, row)) for row in cur.fetchall()]
        return {"ok": True, "indicators": rows, "n": len(rows)}
    except Exception as exc:
        return _error(exc)


@server.tool(description=(
    "Fetch the latest earthquake reported by BMKG (Indonesia's meteorology, climatology, and "
    "geophysics agency), live from data.bmkg.go.id. Demonstrates a dynamic, non-indexed data "
    "source rather than the static document/indicator index."
))
def get_latest_earthquake() -> dict:
    try:
        resp = httpx.get(BMKG_LATEST_EARTHQUAKE_URL, timeout=10.0)
        resp.raise_for_status()
        return {"ok": True, "earthquake": resp.json()["Infogempa"]["gempa"]}
    except Exception as exc:
        return _error(exc)


class AnswerQuestionParams(BaseModel):
    question: str = Field(min_length=1, max_length=1000, description="a question in Indonesian about World Bank indicators and/or indexed Wikipedia articles")


@server.tool(description=(
    "Answer a question end to end through the full NusaLens pipeline: routes it to SQL, document "
    "retrieval, or both, synthesizes a cited answer, and abstains rather than fabricating when the "
    "evidence is insufficient. Use this for a direct answer; use the other tools individually when "
    "the intermediate evidence (retrieved chunks, raw SQL rows) is what's needed instead."
))
def answer_question(params: AnswerQuestionParams) -> dict:
    try:
        result = _get_pipeline().answer(params.question)
        return {
            "ok": True,
            "route": result.route,
            "route_source": result.route_source,
            "answer": result.summary,
            "sources": [{"title": s.title, "url": s.url} for s in result.sources],
            "sql": result.sql,
            "confidence": result.confidence,
            "abstained": result.abstained,
        }
    except Exception as exc:
        return _error(exc)


# ---------------------------------------------------------------------------
# Resources (returned as pre-serialized JSON strings: schema/sample rows contain Decimal, date,
# and pgvector values that plain json.dumps cannot encode without default=str)
# ---------------------------------------------------------------------------

@server.resource("nusalens://schema", description="Live database schema: tables, columns, nullability, and 3 sample rows each.")
def schema_resource() -> str:
    return json.dumps(generate_schema_card(settings.db_dsn), default=str)


@server.resource("nusalens://sources", description="Data sources backing this index: counts, license, and provenance, plus live sources.")
def sources_resource() -> str:
    with psycopg.connect(settings.db_dsn) as conn, conn.cursor() as cur:
        cur.execute("SELECT source, license, count(*) FROM documents GROUP BY source, license")
        doc_sources = [{"source": r[0], "license": r[1], "n_documents": r[2]} for r in cur.fetchall()]
        cur.execute("SELECT source, count(*) FROM dim_indicator GROUP BY source")
        indicator_sources = [{"source": r[0], "n_indicators": r[1]} for r in cur.fetchall()]
    return json.dumps({
        "documents": doc_sources,
        "indicators": indicator_sources,
        "live": [{"name": "BMKG latest earthquake feed", "url": BMKG_LATEST_EARTHQUAKE_URL, "indexed": False}],
    }, default=str)


@server.resource("nusalens://stats", description="Row counts for every major table, read live from the database.")
def stats_resource() -> str:
    tables = ["documents", "chunks", "dim_country", "dim_indicator", "fact_indicator", "query_log"]
    out = {}
    with psycopg.connect(settings.db_dsn) as conn, conn.cursor() as cur:
        for t in tables:
            cur.execute(f"SELECT count(*) FROM {t}")
            out[t] = cur.fetchone()[0]
    return json.dumps(out, default=str)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

@server.prompt(description="Draft an analysis of one indicator's trend for one country over a year range.")
def analyze_indicator_trend(indicator: str, country: str, year_from: str = "2000", year_to: str = "2024") -> str:
    return (
        f"Gunakan tool query_indicators untuk mengambil data {indicator} untuk {country} dari tahun "
        f"{year_from} sampai {year_to}, lalu analisis trennya: naik, turun, atau stabil, dan sebutkan "
        f"tahun dengan perubahan paling besar."
    )


@server.prompt(description="Draft a comparison of one indicator across several countries for one year.")
def compare_countries(indicator: str, countries: str, year: str = "2024") -> str:
    return (
        f"Gunakan tool query_indicators untuk setiap negara dalam daftar ({countries}) pada "
        f"indikator {indicator} tahun {year}, lalu bandingkan nilainya dan urutkan dari tertinggi "
        f"ke terendah."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="NusaLens MCP server")
    parser.add_argument("--transport", default="stdio", choices=["stdio", "sse", "streamable-http"])
    args = parser.parse_args()
    server.run(transport=args.transport)


if __name__ == "__main__":
    main()
