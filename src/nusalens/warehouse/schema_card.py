import psycopg
from typing import Any


def _sample_rows(cur, table: str, n: int = 3) -> list[dict]:
    cur.execute(f"SELECT * FROM {table} LIMIT {n}")
    cols = [desc[0] for desc in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def generate_schema_card(dsn: str) -> dict[str, Any]:
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
                ORDER BY table_name
            """)
            tables = [row[0] for row in cur.fetchall()]
            result = {}
            for t in tables:
                cur.execute(f"""
                    SELECT column_name, data_type, is_nullable
                    FROM information_schema.columns
                    WHERE table_schema = 'public' AND table_name = %s
                    ORDER BY ordinal_position
                """, (t,))
                columns = [{"name": c[0], "type": c[1], "nullable": c[2]} for c in cur.fetchall()]
                sample = _sample_rows(cur, t, 3)
                result[t] = {"columns": columns, "sample_rows": sample}
    return result


if __name__ == "__main__":
    import json, os
    dsn = os.getenv("DATABASE_URL", "postgresql://nusalens:nusalens@localhost:55433/nusalens")
    card = generate_schema_card(dsn)
    print(json.dumps(card, indent=2, default=str))
