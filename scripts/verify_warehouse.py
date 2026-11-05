import os
import psycopg

RW_DSN = os.getenv("DATABASE_URL", "postgresql://nusalens:nusalens@localhost:55433/nusalens")
RO_DSN = "postgresql://nusalens_ro:readonly_pass@localhost:55433/nusalens"


def counts():
    with psycopg.connect(RW_DSN) as conn, conn.cursor() as cur:
        for t in ("dim_country", "dim_indicator", "fact_indicator", "documents", "chunks"):
            cur.execute(f"SELECT count(*) FROM {t}")
            print(f"{t}: {cur.fetchone()[0]}")


def sample_query():
    sql = """
        SELECT country_name, indicator_name, year, value
        FROM v_indicator_series
        WHERE indicator_name LIKE 'GDP per capita%' AND year = 2023
        ORDER BY value DESC
    """
    with psycopg.connect(RW_DSN) as conn, conn.cursor() as cur:
        cur.execute(sql)
        for row in cur.fetchall():
            print(row)


def readonly_enforcement():
    with psycopg.connect(RO_DSN) as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM fact_indicator")
        print("ro SELECT ok, rows:", cur.fetchone()[0])
    try:
        with psycopg.connect(RO_DSN) as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM fact_indicator WHERE year = 2024")
        print("ro DELETE: NOT BLOCKED (failure)")
    except Exception as e:
        print("ro DELETE blocked:", type(e).__name__, str(e).strip().splitlines()[0])
    try:
        with psycopg.connect(RO_DSN) as conn, conn.cursor() as cur:
            cur.execute("CREATE TABLE evil (x int)")
        print("ro CREATE: NOT BLOCKED (failure)")
    except Exception as e:
        print("ro CREATE blocked:", type(e).__name__, str(e).strip().splitlines()[0])


if __name__ == "__main__":
    print("-- row counts --")
    counts()
    print("-- GDP per capita 2023 across countries --")
    sample_query()
    print("-- read-only role enforcement --")
    readonly_enforcement()
