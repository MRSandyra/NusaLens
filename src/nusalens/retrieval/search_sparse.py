import re
from dataclasses import dataclass

import psycopg

_TOKEN = re.compile(r"[0-9a-zA-ZÀ-ÿ]+", re.UNICODE)

_STOP = {
    "dalam", "artikel", "bagaimana", "apa", "mengapa", "kapan", "siapa", "mana",
    "yang", "dan", "atau", "dengan", "untuk", "pada", "dari", "itu",
    "ini", "adalah", "akan", "telah", "sudah", "dijelaskan", "kaitannya", "kaitan",
    "terkait", "berapa", "sebutkan", "jelaskan", "tersebut", "juga", "oleh",
    "sebagai", "tentang", "menurut", "hubungan", "antara",
}


def query_terms(query: str, limit: int = 8) -> list[str]:
    """Content terms from a natural language question, stopwords removed."""
    seen: list[str] = []
    for tok in _TOKEN.findall(query.lower()):
        if len(tok) < 3 or tok in _STOP or tok in seen:
            continue
        seen.append(tok)
        if len(seen) >= limit:
            break
    return seen


@dataclass
class SparseHit:
    chunk_id: str
    doc_id: str
    text: str
    score: float


class SparseSearcher:
    def __init__(self, dsn: str, pool=None):
        self.dsn = dsn
        self.pool = pool  # optional psycopg_pool.ConnectionPool (Phase 13 hardening/optimization)

    def _connect(self):
        return self.pool.connection() if self.pool is not None else psycopg.connect(self.dsn)

    def search(self, query: str, k: int = 10) -> list[SparseHit]:
        terms = query_terms(query)
        if not terms:
            return []
        tsquery = " | ".join(terms)
        sql = """
            SELECT chunk_id, doc_id, text,
                   ts_rank(tsv, to_tsquery('simple', %s)) AS score
            FROM chunks
            WHERE tsv @@ to_tsquery('simple', %s)
            ORDER BY score DESC
            LIMIT %s
        """
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(sql, (tsquery, tsquery, k))
            return [SparseHit(r[0], r[1], r[2], float(r[3])) for r in cur.fetchall()]
