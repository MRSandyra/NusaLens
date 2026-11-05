import psycopg

DSN = "postgresql://nusalens:nusalens@localhost:55433/nusalens"


def main() -> None:
    with psycopg.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM chunks")
        total = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM chunks WHERE embedding IS NULL")
        nulls = cur.fetchone()[0]
        cur.execute("SELECT vector_dims(embedding) FROM chunks LIMIT 1")
        dims = cur.fetchone()[0]
        cur.execute("SELECT count(DISTINCT vector_dims(embedding)) FROM chunks")
        distinct_dims = cur.fetchone()[0]

        # Norm computed server-side: pgvector values arrive as text in psycopg
        # without the type adapter registered, so client-side math is unreliable.
        cur.execute(
            "SELECT min(vector_norm(embedding)), max(vector_norm(embedding)), "
            "avg(vector_norm(embedding)), stddev(vector_norm(embedding)) FROM chunks"
        )
        nmin, nmax, navg, nstd = cur.fetchone()
        cur.execute(
            "SELECT count(*) FROM chunks WHERE abs(vector_norm(embedding) - 1.0) < 1e-3"
        )
        near_unit = cur.fetchone()[0]

        print(f"total chunks: {total}")
        print(f"null embeddings: {nulls}")
        print(f"non-null embeddings: {total - nulls}")
        print(f"embedding dims: {dims} (distinct dim values: {distinct_dims})")
        print(f"norm min: {nmin:.8f}")
        print(f"norm max: {nmax:.8f}")
        print(f"norm avg: {navg:.8f}")
        print(f"norm stddev: {nstd:.10f}")
        print(f"norms within 1e-3 of 1.0: {near_unit} / {total}")


if __name__ == "__main__":
    main()
