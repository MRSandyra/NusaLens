from nusalens.agents.sql_guard import validate_sql
from nusalens.config import settings

DESTRUCTIVE_PAYLOADS = [
    "DROP TABLE documents",
    "DELETE FROM fact_indicator WHERE iso3 = 'IDN'",
    "UPDATE fact_indicator SET value = 999999 WHERE iso3 = 'IDN' AND year = 2019",
    "SELECT * FROM pg_shadow",
    "CREATE TABLE backup_documents AS SELECT * FROM documents",
    "GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO nusalens_ro",
    "SELECT 1; DROP TABLE chunks",
    "COPY documents TO '/tmp/dump.csv'",
]


def test_destructive_payloads_are_rejected():
    for payload in DESTRUCTIVE_PAYLOADS:
        result = validate_sql(payload, settings.db_dsn)
        assert not result.ok, f"guard let through a destructive payload: {payload}"


def test_comment_smuggled_second_statement_is_rejected():
    result = validate_sql(
        "SELECT * FROM v_indicator_series WHERE 1=1; -- drop everything after this", settings.db_dsn
    )
    assert not result.ok


def test_safe_select_is_accepted_and_gets_a_limit():
    result = validate_sql(
        "SELECT value FROM v_indicator_series WHERE country_name = 'Indonesia' "
        "AND indicator_name = 'GDP per capita (current US$)' AND year = 2019",
        settings.db_dsn,
    )
    assert result.ok
    assert "LIMIT" in result.sql.upper()


def test_existing_limit_is_not_duplicated():
    result = validate_sql(
        "SELECT value FROM v_indicator_series WHERE country_name = 'Indonesia' LIMIT 5",
        settings.db_dsn,
    )
    assert result.ok
    assert result.sql.upper().count("LIMIT") == 1


if __name__ == "__main__":
    test_destructive_payloads_are_rejected()
    test_comment_smuggled_second_statement_is_rejected()
    test_safe_select_is_accepted_and_gets_a_limit()
    test_existing_limit_is_not_duplicated()
    print("all guardrail checks passed")
