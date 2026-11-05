import time

import psycopg

DSN = "postgresql://nusalens:nusalens@localhost:55433/nusalens"


def main() -> None:
    with psycopg.connect(DSN) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            # Migration 002 already ships an equivalent HNSW index; the second
            # one built by an earlier run only duplicates 81MB of storage.
            cur.execute("DROP INDEX IF EXISTS idx_chunks_embedding_hnsw")
            print("dropped duplicate idx_chunks_embedding_hnsw")

            # idx_chunks_tsv indexes the tsv column, which the embedding loader
            # never fills, so the GIN index was covering an empty column.
            start = time.time()
            cur.execute(
                "UPDATE chunks SET tsv = to_tsvector('simple', text) WHERE tsv IS NULL"
            )
            print(f"populated tsv for {cur.rowcount} rows in {time.time() - start:.1f}s")

            cur.execute("SELECT count(*) FROM chunks WHERE tsv IS NULL")
            print(f"tsv still null: {cur.fetchone()[0]}")

            start = time.time()
            cur.execute("REINDEX INDEX idx_chunks_tsv")
            print(f"reindexed idx_chunks_tsv in {time.time() - start:.1f}s")

            cur.execute(
                "SELECT indexname, pg_size_pretty(pg_relation_size(indexname::regclass)) "
                "FROM pg_indexes WHERE tablename = 'chunks' ORDER BY indexname"
            )
            for name, size in cur.fetchall():
                print(f"index {name}: {size}")


if __name__ == "__main__":
    main()
