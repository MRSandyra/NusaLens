ALTER TABLE chunks ADD COLUMN IF NOT EXISTS embedding_ft vector(384);

CREATE INDEX IF NOT EXISTS idx_chunks_embedding_ft ON chunks USING hnsw (embedding_ft vector_cosine_ops) WITH (m=16, ef_construction=64);
