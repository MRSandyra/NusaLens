"""Phase 13: profile hotspot #2 -- a fresh psycopg.connect() per DB call instead of a shared pool.
Isolates the connection+query cost (not the cross-encoder rerank, which would dwarf it) by timing
DenseSearcher.search() + SparseSearcher.search() over N repeated calls, pooled vs unpooled.
"""
import time

from psycopg_pool import ConnectionPool

from nusalens.config import settings
from nusalens.retrieval.search_dense import DenseSearcher
from nusalens.retrieval.search_sparse import SparseSearcher

N = 30
QUERY = "Berapa nilai Gini index untuk Vietnam pada tahun 2022?"


def timed_run(pool) -> float:
    dense = DenseSearcher(settings.db_dsn, device="cpu", pool=pool)
    sparse = SparseSearcher(settings.db_dsn, pool=pool)
    t0 = time.perf_counter()
    for _ in range(N):
        dense.search(QUERY, k=10)
        sparse.search(QUERY, k=10)
    return (time.perf_counter() - t0) * 1000


def main() -> None:
    unpooled_ms = timed_run(pool=None)
    print(f"unpooled: {N} dense+sparse calls in {unpooled_ms:.1f}ms ({unpooled_ms / N:.2f}ms/call pair)")

    pool = ConnectionPool(settings.db_dsn, min_size=1, max_size=5, open=True)
    pool.wait()  # ensure the pool is warm before timing, same as it is in a live Pipeline
    pooled_ms = timed_run(pool)
    print(f"pooled:   {N} dense+sparse calls in {pooled_ms:.1f}ms ({pooled_ms / N:.2f}ms/call pair)")
    print(f"speedup: {unpooled_ms / pooled_ms:.2f}x, saved {unpooled_ms - pooled_ms:.1f}ms over {N} call pairs "
          f"({(unpooled_ms - pooled_ms) / N:.2f}ms/call pair)")


if __name__ == "__main__":
    main()
