import re
from dataclasses import dataclass, field

import psycopg
from sentence_transformers import SentenceTransformer

from nusalens.agents.entity_resolution import EntityResolver
from nusalens.agents.sql_guard import execute_readonly, validate_sql
from nusalens.config import settings
from nusalens.llm.ollama_client import generate_with_meta as ollama_generate_with_meta
from nusalens.llm.prompt_registry import load_prompt

_SQL_LINE = re.compile(r"SELECT\b.*", re.IGNORECASE | re.DOTALL)
MAX_ATTEMPTS = 2


@dataclass
class Text2SQLResult:
    question: str
    sql: str | None
    answer: str
    rows: list[tuple] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)
    attempts: int = 0
    error: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0


def _extract_sql(text: str) -> str:
    match = _SQL_LINE.search(text)
    if not match:
        return ""
    sql = match.group(0).strip()
    # Stop at the first blank line or markdown fence the model may add after the statement.
    sql = sql.split("\n\n")[0].split("```")[0].strip()
    first_line = sql.splitlines()[0] if sql.splitlines() else sql
    return first_line.strip().rstrip(";")


class Text2SQLAgent:
    def __init__(self, dsn: str = None, ro_dsn: str = None, model: str = None, endpoint: str = None, prompt_version: str | None = "v2"):
        self.dsn = dsn or settings.db_dsn
        self.ro_dsn = ro_dsn or settings.db_ro_dsn
        self.model = model or settings.text2sql_model
        self.endpoint = endpoint or settings.llm_endpoint
        self.prompt = load_prompt("text2sql", prompt_version)

        embedder = SentenceTransformer(settings.embedding_model, device="cpu")
        self.resolver = EntityResolver(self.dsn, embedder)
        self._country_names = set(self.resolver.country_names)
        self._indicator_names = set(self.resolver.indicator_names)

    def _generate(self, prompt_text: str) -> tuple[str, int, int]:
        return ollama_generate_with_meta(self.model, prompt_text, temperature=0.0, num_predict=200, endpoint=self.endpoint)

    def _snap_candidates(self, sql: str, candidates: list[str]) -> str:
        # The model is told to copy a candidate string exactly; if it still returns a near miss
        # or a translated alias (e.g. "New Zealand" for the view's Indonesian "Selandia Baru"),
        # correct it rather than letting a silent mismatch zero out the result.
        for literal in re.findall(r"'([^']*)'", sql):
            if literal in candidates:
                continue
            alias_hit = self.resolver.alias_to_canonical.get(literal.lower())
            if alias_hit is not None:
                sql = sql.replace(f"'{literal}'", f"'{alias_hit}'")
                continue
            best = max(candidates, key=lambda c: _similar(c, literal), default=None)
            if best is not None and _similar(best, literal) > 0.6:
                sql = sql.replace(f"'{literal}'", f"'{best}'")
        return sql

    def ask(self, question: str) -> Text2SQLResult:
        candidates = self.resolver.resolve(question)
        base_prompt = self.prompt["template"].format(
            question=question,
            country_candidates=", ".join(candidates.countries) or "(tidak ada)",
            indicator_candidates=", ".join(candidates.indicators) or "(tidak ada)",
        )

        sql = ""
        last_error = ""
        prompt_tokens = 0
        completion_tokens = 0
        for attempt in range(1, MAX_ATTEMPTS + 1):
            if attempt == 1:
                raw, p_tok, c_tok = self._generate(base_prompt)
            else:
                repair_prompt = self.prompt["repair_template"].format(previous_sql=sql, error=last_error)
                raw, p_tok, c_tok = self._generate(repair_prompt)
            prompt_tokens += p_tok
            completion_tokens += c_tok

            sql = _extract_sql(raw)
            sql = self._snap_candidates(sql, candidates.countries + candidates.indicators)

            result = validate_sql(sql, self.dsn)
            if not result.ok:
                last_error = result.reason
                continue

            try:
                cols, rows = execute_readonly(result.sql, self.ro_dsn)
            except Exception as exc:
                last_error = str(exc)
                continue

            return Text2SQLResult(
                question=question,
                sql=result.sql,
                answer=verbalize(question, cols, rows, result.sql),
                rows=rows,
                columns=cols,
                attempts=attempt,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )

        return Text2SQLResult(
            question=question,
            sql=sql or None,
            answer=f"Tidak dapat menjawab pertanyaan ini secara aman: {last_error}",
            attempts=MAX_ATTEMPTS,
            error=last_error,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )


def _similar(a: str, b: str) -> float:
    import difflib
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


def verbalize(question: str, columns: list[str], rows: list[tuple], sql: str) -> str:
    if not rows:
        return f"Tidak ditemukan data untuk pertanyaan ini.\n\nSQL: {sql}"

    if len(rows) == 1 and len(columns) == 1:
        return f"Jawaban: {rows[0][0]}.\n\nSQL: {sql}"

    lines = [", ".join(f"{c}={v}" for c, v in zip(columns, row)) for row in rows[:10]]
    body = "\n".join(f"- {l}" for l in lines)
    return f"Hasil:\n{body}\n\nSQL: {sql}"
