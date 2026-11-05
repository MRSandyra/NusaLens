CREATE INDEX IF NOT EXISTS idx_documents_tsv ON documents USING gin(to_tsvector('indonesian', text));
