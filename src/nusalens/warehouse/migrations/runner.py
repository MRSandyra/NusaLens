import os
from pathlib import Path
import psycopg
from psycopg.sql import SQL, Identifier, Literal

MIGRATIONS_DIR = Path(__file__).resolve().parent


def _applied_versions(conn: psycopg.Connection) -> set[int]:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT version FROM schema_migrations
        """)
        return {row[0] for row in cur.fetchall()}


def run_migrations(dsn: str) -> None:
    with psycopg.connect(dsn) as conn:
        # Ensure the migrations tracking table exists first
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version INT PRIMARY KEY,
                    applied_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
        conn.commit()
        applied = _applied_versions(conn)
        files = sorted(MIGRATIONS_DIR.glob("[0-9][0-9][0-9]_*.sql"))
        for path in files:
            version = int(path.name[:3])
            if version in applied:
                continue
            sql = path.read_text(encoding="utf-8")
            with conn.cursor() as cur:
                cur.execute(sql)
                cur.execute(
                    "INSERT INTO schema_migrations (version) VALUES (%s) ON CONFLICT DO NOTHING",
                    (version,),
                )
            conn.commit()
            print(f"Applied migration {version}: {path.name}")
        print("Migrations up to date")


if __name__ == "__main__":
    dsn = os.getenv("DATABASE_URL", "postgresql://nusalens:nusalens@localhost:55433/nusalens")
    run_migrations(dsn)
