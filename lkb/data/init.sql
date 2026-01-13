-- Knowledge Base Schema for pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- Main documents table
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_path TEXT NOT NULL,
    source_type TEXT NOT NULL,  -- 'markdown', 'code', 'pdf', 'note'
    title TEXT,
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    content_hash TEXT NOT NULL,  -- For deduplication/change detection
    UNIQUE(content_hash)
);

-- Chunks for semantic search
CREATE TABLE chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,           -- Original chunk text (for display)
    embedding_text TEXT NOT NULL,    -- Contextualized text (what was embedded)
    embedding vector(768),           -- nomic-embed-text-v1.5 dimension
    token_count INTEGER,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Optimized index for similarity search
-- Using HNSW for better query performance (slightly slower build than IVFFlat)
CREATE INDEX chunks_embedding_idx ON chunks 
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Full-text search as fallback/hybrid
CREATE INDEX documents_content_fts ON documents 
USING gin(to_tsvector('english', content));

CREATE INDEX chunks_content_fts ON chunks 
USING gin(to_tsvector('english', content));

-- Source path index for updates
CREATE INDEX documents_source_path_idx ON documents(source_path);

-- Source type index for filtering
CREATE INDEX documents_source_type_idx ON documents(source_type);

-- Metadata GIN index for JSONB queries
CREATE INDEX documents_metadata_idx ON documents USING gin(metadata);

-- Function to update timestamp
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER documents_updated_at
    BEFORE UPDATE ON documents
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

-- Helper view for search results
CREATE VIEW search_results AS
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
CREATE VIEW index_stats AS
SELECT 
    source_type,
    COUNT(DISTINCT d.id) as document_count,
    COUNT(c.id) as chunk_count,
    SUM(c.token_count) as total_tokens
FROM documents d
LEFT JOIN chunks c ON c.document_id = d.id
GROUP BY source_type;
