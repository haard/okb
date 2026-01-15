-- Initial schema - documents, chunks, indexes
-- depends:

CREATE EXTENSION IF NOT EXISTS vector;

-- Main documents table
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_path TEXT NOT NULL,
    source_type TEXT NOT NULL,
    title TEXT,
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    content_hash TEXT NOT NULL,
    UNIQUE(content_hash)
);

-- Chunks for semantic search
CREATE TABLE IF NOT EXISTS chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding_text TEXT NOT NULL,
    embedding vector(768),
    token_count INTEGER,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- HNSW index for similarity search
CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Full-text search indexes
CREATE INDEX IF NOT EXISTS documents_content_fts ON documents
USING gin(to_tsvector('english', content));

CREATE INDEX IF NOT EXISTS chunks_content_fts ON chunks
USING gin(to_tsvector('english', content));

-- Lookup indexes
CREATE INDEX IF NOT EXISTS documents_source_path_idx ON documents(source_path);
CREATE INDEX IF NOT EXISTS documents_source_type_idx ON documents(source_type);
CREATE INDEX IF NOT EXISTS documents_metadata_idx ON documents USING gin(metadata);

-- Function to update timestamp
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger (drop first to make idempotent)
DROP TRIGGER IF EXISTS documents_updated_at ON documents;
CREATE TRIGGER documents_updated_at
    BEFORE UPDATE ON documents
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

-- Helper view for search results
CREATE OR REPLACE VIEW search_results AS
SELECT
    c.id as chunk_id,
    c.content,
    c.chunk_index,
    c.token_count,
    c.embedding,
    d.id as document_id,
    d.source_path,
    d.source_type,
    d.title,
    d.metadata
FROM chunks c
JOIN documents d ON c.document_id = d.id;

-- Stats view
CREATE OR REPLACE VIEW index_stats AS
SELECT
    source_type,
    COUNT(DISTINCT d.id) as document_count,
    COUNT(c.id) as chunk_count,
    SUM(c.token_count) as total_tokens
FROM documents d
LEFT JOIN chunks c ON c.document_id = d.id
GROUP BY source_type;
