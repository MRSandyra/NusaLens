# Phase 3: Chunking, embedding, and retrieval indexes

15,000 Wikipedia documents are split into 41,297 chunks, embedded with a local multilingual model, and indexed in Postgres for both dense (HNSW/cosine) and lexical (GIN/tsvector) search.

## Chunking

| Metric | Value |
|---|---|
| Input documents | 15,000 |
| Output chunks | 41,297 |
| Target token size | 450 |
| Overlap | 80 tokens |
| Chunking time | 4.7s |

Evidence: `reports/raw/03_chunk.log` (chunker run output), `scripts/probe_chunks.py` result recorded at the time (`chunks rows 41297`).

## Embedding

| Metric | Value |
|---|---|
| Model | intfloat/multilingual-e5-small |
| Embedding dimension | 384 |
| Device | CUDA (RTX 3060 Laptop, 6GB VRAM) |
| Chunks encoded | 41,297 |
| Encoding time | 397.1s |
| Throughput | 104.0 chunks/sec |

Evidence: `reports/raw/03_embed_reload.log`.

An earlier load stored raw Python list reprs as text because the pgvector Python adapter was not registered with psycopg3, so `embedding` values round-tripped as strings rather than vectors. Fixed by formatting embeddings as pgvector literal strings (`"[v1,v2,...]"`) and casting with `%s::vector` on insert; reloaded from scratch after truncating the table.

## Embedding integrity

Measured directly against the loaded `chunks` table with server-side `vector_norm()`, avoiding client-side parsing of the vector type:

| Check | Result |
|---|---|
| Total chunks | 41,297 |
| Null embeddings | 0 |
| Embedding dimension (distinct values seen) | 384 (1 distinct value, i.e. uniform) |
| Norm min | 0.99999993 |
| Norm max | 1.00000012 |
| Norm mean | 1.00000003 |
| Norm stddev | 0.0000000291 |
| Chunks with norm within 1e-3 of 1.0 | 41,297 / 41,297 |

Evidence: `reports/raw/03_embedding_verify.txt`. `normalize_embeddings=True` at encode time is confirmed correct on the stored data, not just at the point of encoding.

## Indexes

| Index | Type | Size | Build time |
|---|---|---|---|
| idx_chunks_embedding | HNSW (m=16, ef_construction=64), cosine ops | 81 MB | 18.5s |
| idx_chunks_tsv | GIN on `tsv` tsvector column | 44 MB | 11.3s (reindex after backfill) |
| chunks_pkey | btree | 3,520 kB | n/a |

The HNSW index already existed from migration `002_vector.sql`; a duplicate index was accidentally built under a different name during this phase and was dropped (`reports/raw/03_index_fix.txt`) to avoid maintaining two identical 81 MB structures.

The GIN index's target column, `tsv`, was created by migration `002_vector.sql` but never populated by the embedding loader, so the first `CREATE INDEX IF NOT EXISTS` on it silently indexed an empty column (0.0s build time, the tell). Fixed: backfilled `tsv = to_tsvector('simple', text)` for all 41,297 rows (83.0s), then reindexed. `tsv IS NULL` count is 0 after the backfill.

HNSW build required disabling parallel maintenance workers and reducing `maintenance_work_mem` to 256MB: the container's default `/dev/shm` size is smaller than the shared memory segment Postgres requests for a parallel HNSW build, producing `DiskFull` on the default settings.

## Dense search latency

`search_dense.py` implements query encoding (with the required `"query: "` E5 prefix) plus a cosine-distance HNSW lookup. Measured over 100 queries drawn from real `documents.title` values already in the corpus (not synthetic strings), k=10:

| Metric | Value |
|---|---|
| Queries | 100 |
| k | 10 |
| Encode time p50 | 17.26 ms |
| Encode time p95 | 23.54 ms |
| End-to-end (encode + DB round trip) p50 | 37.04 ms |
| End-to-end p95 | 55.16 ms |
| End-to-end min / max / mean | 32.66 / 713.72 / 46.50 ms |
| Queries returning full k=10 hits | 100 / 100 |

Evidence: `reports/raw/03_search_latency.json`. The single 713.72ms outlier is the first query in the run (cold connection/model state); p50 and p95 are computed over the full 100-query distribution and are not sensitive to that one outlier.

## Summary against Definition of Done

| Requirement | Status |
|---|---|
| Chunks table populated from all documents | met, 41,297 chunks from 15,000 documents |
| Embeddings non-null, correctly normalized | met, 0 nulls, all norms within 1e-3 of 1.0 |
| HNSW index built on embedding column | met, 81 MB, 18.5s build |
| GIN index built on lexical column | met, 44 MB, backfilled tsv then indexed |
| Dense search latency measured over a real query set | met, p50 37.04ms / p95 55.16ms over 100 queries |
