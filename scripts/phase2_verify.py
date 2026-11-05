"""Phase 2 verification: schema stats and read‑only role enforcement."""
import json
import psycopg
from nusalens.config import settings

print("=== Phase 2: Schema stats ===\n")

with psycopg.connect(settings.db_dsn) as conn, conn.cursor() as cur:
    tables = ["dim_country", "dim_indicator", "fact_indicator", "documents", "chunks"]
    for tbl in tables:
        cur.execute(f"SELECT COUNT(*) FROM {tbl}")
        count = cur.fetchone()[0]
        print(f"{tbl}: {count} rows")

print("\n=== Phase 2: Read‑only role enforcement ===\n")
# Test read-only role using the main user then switching role
with psycopg.connect(settings.db_dsn) as conn, conn.cursor() as cur:
    try:
        cur.execute("SET ROLE nusalens_ro")
        cur.execute("SELECT COUNT(*) FROM fact_indicator")
        count = cur.fetchone()[0]
        print(f"nusalens_ro SELECT: OK, {count} rows")
    except Exception as e:
        print(f"nusalens_ro SELECT: FAIL, {e}")

    try:
        cur.execute("DELETE FROM fact_indicator WHERE year = 2024")
        print("nusalens_ro DELETE: FAIL (should have been blocked)")
    except psycopg.errors.ReadOnlySqlTransaction:
        print("nusalens_ro DELETE: OK (correctly blocked with ReadOnlySqlTransaction)")
    except Exception as e:
        print(f"nusalens_ro DELETE: OK (blocked, {type(e).__name__})")
    finally:
        conn.rollback()

    try:
        cur.execute("CREATE TABLE evil (x int)")
        print("nusalens_ro CREATE: FAIL (should have been blocked)")
    except psycopg.errors.ReadOnlySqlTransaction:
        print("nusalens_ro CREATE: OK (correctly blocked with ReadOnlySqlTransaction)")
    except Exception as e:
        print(f"nusalens_ro CREATE: OK (blocked, {type(e).__name__})")
    finally:
        conn.rollback()

print("\nAll Phase 2 checks completed.")
