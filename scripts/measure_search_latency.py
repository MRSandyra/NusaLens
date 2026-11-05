import json
import statistics
import time
from pathlib import Path

import psycopg

from nusalens.retrieval.search_dense import DenseSearcher

DSN = "postgresql://nusalens:nusalens@localhost:55433/nusalens"
OUT = Path("reports/raw/03_search_latency.json")


def build_queries(n: int = 100) -> list[str]:
    # Queries drawn from stored titles so every probe hits real corpus vocabulary.
    with psycopg.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT title FROM documents WHERE length(title) BETWEEN 8 AND 60 "
            "ORDER BY doc_id LIMIT %s",
            (n,),
        )
        return [r[0] for r in cur.fetchall()]


def main() -> None:
    queries = build_queries()
    searcher = DenseSearcher(DSN)
    encode_ms: list[float] = []
    total_ms: list[float] = []
    hit_counts: list[int] = []

    for q in queries:
        t0 = time.perf_counter()
        qvec = searcher.encode_query(q)
        t1 = time.perf_counter()
        with psycopg.connect(DSN) as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT chunk_id, 1 - (embedding <=> %s::vector) FROM chunks "
                "ORDER BY embedding <=> %s::vector LIMIT 10",
                (qvec, qvec),
            )
            rows = cur.fetchall()
        t2 = time.perf_counter()
        encode_ms.append((t1 - t0) * 1000)
        total_ms.append((t2 - t0) * 1000)
        hit_counts.append(len(rows))

    def pct(values: list[float], p: float) -> float:
        ordered = sorted(values)
        idx = min(int(p * len(ordered)), len(ordered) - 1)
        return ordered[idx]

    result = {
        "queries": len(queries),
        "k": 10,
        "encode_ms": {
            "p50": round(statistics.median(encode_ms), 2),
            "p95": round(pct(encode_ms, 0.95), 2),
        },
        "total_ms": {
            "p50": round(statistics.median(total_ms), 2),
            "p95": round(pct(total_ms, 0.95), 2),
            "min": round(min(total_ms), 2),
            "max": round(max(total_ms), 2),
            "mean": round(statistics.mean(total_ms), 2),
        },
        "hits_per_query_min": min(hit_counts),
        "queries_returning_k": sum(1 for c in hit_counts if c == 10),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
