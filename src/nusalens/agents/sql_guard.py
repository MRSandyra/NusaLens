import re
from dataclasses import dataclass

import psycopg
import sqlglot
import yaml
from sqlglot import exp

from nusalens.config import PROJECT_ROOT, settings

_DENYLIST = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|GRANT|REVOKE|TRUNCATE|CREATE|COPY)\b|\bpg_",
    re.IGNORECASE,
)


def _load_sql_guard_config() -> dict:
    path = PROJECT_ROOT / "config" / "settings.yaml"
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return cfg.get("sql_guard", {})


@dataclass
class ValidationResult:
    ok: bool
    sql: str
    reason: str = ""
    explain_cost: float | None = None


def validate_sql(raw_sql: str, dsn_for_explain: str) -> ValidationResult:
    cfg = _load_sql_guard_config()
    default_limit = int(cfg.get("default_limit", 200))
    max_explain_cost = float(cfg.get("max_explain_cost", 1_000_000))

    sql = raw_sql.strip().rstrip(";").strip()
    if not sql:
        return ValidationResult(False, sql, "empty SQL")

    if _DENYLIST.search(sql):
        return ValidationResult(False, sql, "keyword denylist match")

    try:
        statements = sqlglot.parse(sql, read="postgres")
    except Exception as exc:
        return ValidationResult(False, sql, f"sqlglot parse error: {exc}")

    statements = [s for s in statements if s is not None]
    if len(statements) != 1:
        return ValidationResult(False, sql, f"expected exactly one statement, got {len(statements)}")

    parsed = statements[0]
    if not isinstance(parsed, exp.Select):
        return ValidationResult(False, sql, f"expected a SELECT, got {type(parsed).__name__}")

    if parsed.args.get("limit") is None:
        parsed = parsed.limit(default_limit)
    bounded_sql = parsed.sql(dialect="postgres")

    try:
        with psycopg.connect(dsn_for_explain) as conn, conn.cursor() as cur:
            cur.execute(f"EXPLAIN (FORMAT JSON) {bounded_sql}")
            plan = cur.fetchone()[0]
    except Exception as exc:
        return ValidationResult(False, bounded_sql, f"EXPLAIN failed: {exc}")

    total_cost = plan[0]["Plan"]["Total Cost"]
    if total_cost > max_explain_cost:
        return ValidationResult(False, bounded_sql, f"estimated cost {total_cost} exceeds {max_explain_cost}", total_cost)

    return ValidationResult(True, bounded_sql, "ok", total_cost)


def execute_readonly(sql: str, dsn_ro: str) -> tuple[list[str], list[tuple]]:
    with psycopg.connect(dsn_ro) as conn, conn.cursor() as cur:
        cur.execute(sql)
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
    return cols, rows
