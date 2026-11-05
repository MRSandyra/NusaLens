"""Phase 6: re-embed the full corpus with the fine-tuned model into the embedding_ft slot."""
import time
from pathlib import Path

import psycopg
import pyarrow.parquet as pq
from sentence_transformers import SentenceTransformer

from nusalens.config import PROJECT_ROOT, settings

MODEL_PATH = PROJECT_ROOT / "models" / "e5-nusalens-v1"


def _format_vector(vec: list[float]) -> str:
    return "[" + ",".join(str(v) for v in vec) + "]"


def main() -> None:
    model = SentenceTransformer(str(MODEL_PATH), device="cuda")
    df = pq.read_table(PROJECT_ROOT / "data" / "gold" / "chunks.parquet").to_pandas()
    texts = [f"passage: {t}" for t in df["text"].tolist()]

    start = time.time()
    all_emb = model.encode(texts, batch_size=64, normalize_embeddings=True)
    elapsed = time.time() - start
    print(f"Encoded {len(texts)} chunks with the fine-tuned model in {elapsed:.1f}s ({len(texts)/elapsed:.2f} chunks/sec)")

    rows = [(_format_vector(all_emb[i].tolist()), row["chunk_id"]) for i, row in df.iterrows()]
    with psycopg.connect(settings.db_dsn) as conn, conn.cursor() as cur:
        cur.executemany("UPDATE chunks SET embedding_ft = %s::vector WHERE chunk_id = %s", rows)
        conn.commit()
    print(f"updated embedding_ft for {len(rows)} chunks")


if __name__ == "__main__":
    main()
