CREATE TABLE IF NOT EXISTS query_log (
    log_id SERIAL PRIMARY KEY,
    question TEXT NOT NULL,
    route TEXT,
    sql TEXT,
    retrieved_chunk_ids TEXT[],
    per_stage_latency_ms JSONB,
    token_counts JSONB,
    prompt_versions JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS eval_run (
    run_id TEXT PRIMARY KEY,
    tag TEXT NOT NULL,
    started_at TIMESTAMPTZ DEFAULT NOW(),
    finished_at TIMESTAMPTZ,
    metrics JSONB
);

CREATE TABLE IF NOT EXISTS schema_migrations (
    version INT PRIMARY KEY,
    applied_at TIMESTAMPTZ DEFAULT NOW()
);
