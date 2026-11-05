# Routing: fast rules first (RuleRouter), LLM only when no rule fires (see agents/router.py).
# HYBRID mode runs SQL and retrieval concurrently via asyncio.to_thread, merges the evidence into
# one chunk list, and synthesizes once. The abstain gate fires before any LLM call for a
# rule-decided ABSTAIN, and after SQL/retrieval when the evidence itself turns out empty or
# low-confidence: never fabricates a number or citation.
import asyncio
import json
import time
from dataclasses import dataclass, field

from psycopg_pool import ConnectionPool

from nusalens.observability import log as obs_log
from nusalens.agents.router import RuleRouter, route as route_question
from nusalens.agents.synthesizer import synthesize
from nusalens.agents.text2sql import Text2SQLAgent
from nusalens.config import settings
from nusalens.llm.prompt_registry import active_version
from nusalens.retrieval.hybrid_search import HybridSearcher

ABSTAIN_MESSAGE = (
    "Informasi ini tidak tersedia di sumber yang terindeks (indikator Bank Dunia dan artikel "
    "Wikipedia Indonesia). Pertanyaan ini meminta entitas, periode, atau topik yang berada di "
    "luar cakupan data yang dicari."
)


@dataclass
class Source:
    title: str
    url: str | None


@dataclass
class AnswerResult:
    question: str
    route: str
    route_source: str
    summary: str
    sources: list[Source] = field(default_factory=list)
    sql: str | None = None
    confidence: float = 0.0
    abstained: bool = False
    per_stage_latency_ms: dict = field(default_factory=dict)
    token_counts: dict = field(default_factory=dict)
    prompt_versions: dict = field(default_factory=dict)
    retrieved_chunk_ids: list[str] = field(default_factory=list)
    sql_rows: list = field(default_factory=list)  # raw Text2SQL rows, for exact execution-accuracy checks

    def formatted(self) -> str:
        lines = [self.summary]
        if self.sources:
            lines.append("\nSumber:")
            for s in self.sources:
                lines.append(f"- {s.title}" + (f" ({s.url})" if s.url else ""))
        if self.sql:
            lines.append(f"\nSQL: {self.sql}")
        lines.append(f"\nKeyakinan: {self.confidence:.2f}")
        return "\n".join(lines)


def _strip_sql_suffix(answer: str) -> str:
    # Text2SQLAgent.verbalize() appends "\n\nSQL: ..." to its own answer text; the pipeline's
    # AnswerResult carries the SQL separately (result.sql) and formatted() re-appends it once,
    # so the raw text2sql answer is trimmed here to avoid printing the statement twice.
    return answer.split("\n\nSQL:")[0]


class Pipeline:
    def __init__(self, dsn: str | None = None, ro_dsn: str | None = None, device: str = "cpu"):
        self.dsn = dsn or settings.db_dsn
        self.ro_dsn = ro_dsn or settings.db_ro_dsn
        # Shared pool instead of a fresh psycopg.connect() per call: every DOC/HYBRID answer()
        # call previously opened 4+ separate connections (dense search, sparse search,
        # _sources_for, _log); measured before/after in reports/13_quality.md (Phase 13).
        self._pool = ConnectionPool(self.dsn, min_size=1, max_size=5, open=True)
        self.rule_router = RuleRouter(self.dsn)
        self.searcher = HybridSearcher(self.dsn, device=device, pool=self._pool)
        self.text2sql = Text2SQLAgent(dsn=self.dsn, ro_dsn=self.ro_dsn)

    def _sources_for(self, doc_ids: list[str]) -> list[Source]:
        unique_ids = list(dict.fromkeys(doc_ids))
        if not unique_ids:
            return []
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute("SELECT doc_id, title, url FROM documents WHERE doc_id = ANY(%s)", (unique_ids,))
            rows = {r[0]: (r[1], r[2]) for r in cur.fetchall()}
        return [Source(title=rows[d][0], url=rows[d][1]) for d in unique_ids if d in rows]

    def _retrieve(self, question: str, k: int = 8):
        return self.searcher.search_rrf_rerank_exp_timed(question, k=k)

    def _abstain_result(self, question: str, route: str, route_source: str, latency: dict,
                         tokens: dict, prompt_versions: dict, sql: str | None = None,
                         confidence: float = 0.0, chunk_ids: list[str] | None = None) -> "AnswerResult":
        result = AnswerResult(question=question, route=route, route_source=route_source,
                               summary=ABSTAIN_MESSAGE, abstained=True, confidence=confidence, sql=sql,
                               per_stage_latency_ms=latency, token_counts=tokens,
                               prompt_versions=prompt_versions, retrieved_chunk_ids=chunk_ids or [])
        self._log(result)
        return result

    def answer(self, question: str) -> AnswerResult:
        latency: dict = {}
        tokens = {"router": 0, "text2sql": 0, "synthesizer": 0}
        prompt_versions: dict = {}

        t0 = time.perf_counter()
        route_result = route_question(question, rule_router=self.rule_router)
        latency["route_ms"] = (time.perf_counter() - t0) * 1000
        tokens["router"] = route_result.prompt_tokens + route_result.completion_tokens
        if route_result.source == "llm":
            prompt_versions["router"] = active_version("router")

        # An unparsed LLM label defaults to ABSTAIN rather than guessing a route -- never fabricate.
        route_type = route_result.type or "ABSTAIN"

        if route_type == "ABSTAIN":
            return self._abstain_result(question, "ABSTAIN", route_result.source, latency, tokens, prompt_versions)

        if route_type == "SQL":
            t1 = time.perf_counter()
            sql_result = self.text2sql.ask(question)
            latency["text2sql_ms"] = (time.perf_counter() - t1) * 1000
            tokens["text2sql"] = sql_result.prompt_tokens + sql_result.completion_tokens
            prompt_versions["text2sql"] = str(self.text2sql.prompt.get("version"))

            if not sql_result.rows:
                return self._abstain_result(question, "SQL", route_result.source, latency, tokens,
                                             prompt_versions, sql=sql_result.sql, confidence=0.1)

            result = AnswerResult(question=question, route="SQL", route_source=route_result.source,
                                   summary=_strip_sql_suffix(sql_result.answer), sql=sql_result.sql,
                                   confidence=0.9, per_stage_latency_ms=latency, token_counts=tokens,
                                   prompt_versions=prompt_versions, sql_rows=sql_result.rows)
            self._log(result)
            return result

        if route_type == "DOC":
            scored_hits, retrieve_ms, rerank_ms = self._retrieve(question)
            latency["retrieve_ms"] = retrieve_ms
            latency["rerank_ms"] = rerank_ms
            chunk_ids = [h.chunk_id for h, _ in scored_hits]
            top_score = scored_hits[0][1] if scored_hits else 0.0

            if not scored_hits or top_score < settings.abstain_rerank_threshold:
                return self._abstain_result(question, "DOC", route_result.source, latency, tokens,
                                             prompt_versions, confidence=top_score, chunk_ids=chunk_ids)

            chunks = [{"chunk_id": h.chunk_id, "text": h.text} for h, _ in scored_hits]
            t2 = time.perf_counter()
            synth = synthesize(question, chunks)
            latency["generate_ms"] = (time.perf_counter() - t2) * 1000
            tokens["synthesizer"] = synth.prompt_tokens + synth.completion_tokens
            prompt_versions["synthesizer"] = active_version("synthesizer")

            sources = self._sources_for([h.doc_id for h, _ in scored_hits])
            result = AnswerResult(question=question, route="DOC", route_source=route_result.source,
                                   summary=synth.answer, sources=sources, confidence=top_score,
                                   abstained=synth.abstained, per_stage_latency_ms=latency,
                                   token_counts=tokens, prompt_versions=prompt_versions,
                                   retrieved_chunk_ids=chunk_ids)
            self._log(result)
            return result

        if route_type == "HYBRID":
            async def _run_both():
                return await asyncio.gather(
                    asyncio.to_thread(self.text2sql.ask, question),
                    asyncio.to_thread(self._retrieve, question),
                )

            t1 = time.perf_counter()
            sql_result, (scored_hits, retrieve_ms, rerank_ms) = asyncio.run(_run_both())
            latency["sql_and_retrieval_concurrent_ms"] = (time.perf_counter() - t1) * 1000
            latency["retrieve_ms"] = retrieve_ms
            latency["rerank_ms"] = rerank_ms
            tokens["text2sql"] = sql_result.prompt_tokens + sql_result.completion_tokens
            prompt_versions["text2sql"] = str(self.text2sql.prompt.get("version"))

            chunk_ids = [h.chunk_id for h, _ in scored_hits]
            top_score = scored_hits[0][1] if scored_hits else 0.0
            chunks = [{"chunk_id": h.chunk_id, "text": h.text} for h, _ in scored_hits]
            if sql_result.rows:
                chunks.append({"chunk_id": "sql_result", "text": sql_result.answer})

            if not chunks:
                return self._abstain_result(question, "HYBRID", route_result.source, latency, tokens,
                                             prompt_versions, sql=sql_result.sql, chunk_ids=chunk_ids)

            t2 = time.perf_counter()
            synth = synthesize(question, chunks)
            latency["generate_ms"] = (time.perf_counter() - t2) * 1000
            tokens["synthesizer"] = synth.prompt_tokens + synth.completion_tokens
            prompt_versions["synthesizer"] = active_version("synthesizer")

            sources = self._sources_for([h.doc_id for h, _ in scored_hits])
            confidence = (top_score + (0.9 if sql_result.rows else 0.1)) / 2
            result = AnswerResult(question=question, route="HYBRID", route_source=route_result.source,
                                   summary=synth.answer, sources=sources, sql=sql_result.sql,
                                   confidence=confidence, abstained=synth.abstained,
                                   per_stage_latency_ms=latency, token_counts=tokens,
                                   prompt_versions=prompt_versions, retrieved_chunk_ids=chunk_ids,
                                   sql_rows=sql_result.rows)
            self._log(result)
            return result

        return self._abstain_result(question, "ABSTAIN", route_result.source, latency, tokens, prompt_versions)

    def _log(self, result: AnswerResult) -> None:
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO query_log (question, route, sql, retrieved_chunk_ids, per_stage_latency_ms, "
                "token_counts, prompt_versions, abstained, confidence) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (result.question, result.route, result.sql, result.retrieved_chunk_ids or None,
                 json.dumps(result.per_stage_latency_ms), json.dumps(result.token_counts),
                 json.dumps(result.prompt_versions), result.abstained, result.confidence),
            )
            conn.commit()
        obs_log.info(
            "query",
            question=result.question, route=result.route, route_source=result.route_source,
            abstained=result.abstained, confidence=result.confidence,
            per_stage_latency_ms=result.per_stage_latency_ms, token_counts=result.token_counts,
            prompt_versions=result.prompt_versions,
        )
