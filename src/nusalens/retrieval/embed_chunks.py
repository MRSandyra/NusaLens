import time
from pathlib import Path

import pyarrow.parquet as pq
from sentence_transformers import SentenceTransformer
import psycopg

MODEL_NAME = "intfloat/multilingual-e5-small"


def _format_vector(vec: list[float]) -> str:
    return "[" + ",".join(str(v) for v in vec) + "]"


def embed_chunks(parquet_path: Path, dsn: str, batch_size: int = 64) -> None:
    model = SentenceTransformer(MODEL_NAME, device="cuda")
    table = pq.read_table(parquet_path)
    df = table.to_pandas()
    texts = df["text"].tolist()
    start = time.time()
    all_emb = model.encode(texts, batch_size=batch_size, normalize_embeddings=True)
    elapsed = time.time() - start
    print(f"Encoded {len(texts)} chunks in {elapsed:.1f}s ({len(texts)/elapsed:.2f} chunks/sec)")
    # Bulk upsert using executemany with vector formatted as string
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            rows = []
            for i, row in df.iterrows():
                emb = _format_vector(all_emb[i].tolist())
                rows.append((row["chunk_id"], row["doc_id"], row["ord"], row["heading_path"], row["text"], row["token_count"], emb))
            # tsv is populated here too (to_tsvector on the same text), not left NULL for a
            # separate manual fix-up pass -- see scripts/fix_indexes.py, which this closes the gap
            # that script was written to patch (Phase 13 hardening, docs/DECISIONS.md D24).
            sql = """
                INSERT INTO chunks (chunk_id, doc_id, ord, heading_path, text, token_count, embedding, tsv)
                VALUES (%s, %s, %s, %s, %s, %s, %s::vector, to_tsvector('simple', %s))
                ON CONFLICT (chunk_id) DO UPDATE SET
                    doc_id = EXCLUDED.doc_id,
                    ord = EXCLUDED.ord,
                    heading_path = EXCLUDED.heading_path,
                    text = EXCLUDED.text,
                    token_count = EXCLUDED.token_count,
                    embedding = EXCLUDED.embedding,
                    tsv = EXCLUDED.tsv
            """
            rows = [(r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[4]) for r in rows]
            cur.executemany(sql, rows)
        conn.commit()
    print(f"Loaded {len(rows)} chunk embeddings into DB")

if __name__ == "__main__":
    dsn = "postgresql://nusalens:nusalens@localhost:55433/nusalens"
    embed_chunks(Path("data/gold/chunks.parquet"), dsn)
