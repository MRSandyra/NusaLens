from dataclasses import dataclass

import psycopg
from sentence_transformers import SentenceTransformer

MODEL_NAME = "intfloat/multilingual-e5-small"


@dataclass
class DenseHit:
    chunk_id: str
    doc_id: str
    text: str
    score: float


class DenseSearcher:
    def __init__(self, dsn: str, device: str = "cpu", model_name: str = MODEL_NAME,
                 embedding_column: str = "embedding", pool=None, ef_search: int = 100):
        self.dsn = dsn
        self.model = SentenceTransformer(model_name, device=device)
        self.embedding_column = embedding_column
        self.pool = pool  # optional psycopg_pool.ConnectionPool (Phase 13 hardening/optimization)
        # pgvector's own default is 40; measured (reports/raw/13_profile_ef_search.json) on this
        # ~41k-row corpus that 100 raises recall@10 0.50->0.54 at a query-time cost increase too
        # small to matter (2.6ms->3.3ms mean) -- a recall win with no real latency tradeoff here.
        self.ef_search = ef_search

    def _connect(self):
        return self.pool.connection() if self.pool is not None else psycopg.connect(self.dsn)

    def encode_query(self, query: str) -> str:
        # E5 requires the asymmetric "query: " prefix; omitting it costs recall.
        vec = self.model.encode(f"query: {query}", normalize_embeddings=True)
        return "[" + ",".join(str(v) for v in vec.tolist()) + "]"

    def search(self, query: str, k: int = 10) -> list[DenseHit]:
        qvec = self.encode_query(query)
        sql = f"""
            SELECT chunk_id, doc_id, text, 1 - ({self.embedding_column} <=> %s::vector) AS score
            FROM chunks
            WHERE {self.embedding_column} IS NOT NULL
            ORDER BY {self.embedding_column} <=> %s::vector
            LIMIT %s
        """
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(f"SET hnsw.ef_search = {self.ef_search}")
            cur.execute(sql, (qvec, qvec, k))
            return [DenseHit(r[0], r[1], r[2], float(r[3])) for r in cur.fetchall()]
