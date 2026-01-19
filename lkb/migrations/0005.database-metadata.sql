-- Database metadata table for storing LLM-enhanced descriptions
-- depends: 0004.tokens

CREATE TABLE IF NOT EXISTS database_metadata (
    id SERIAL PRIMARY KEY,
    key TEXT NOT NULL UNIQUE,  -- 'description', 'topics', etc.
    value JSONB NOT NULL,
    source TEXT NOT NULL DEFAULT 'llm',  -- 'config' or 'llm'
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Index for quick key lookups
CREATE INDEX IF NOT EXISTS idx_database_metadata_key ON database_metadata(key);

-- Insert default entries that can be updated by LLM
INSERT INTO database_metadata (key, value, source) VALUES
    ('llm_description', 'null'::jsonb, 'llm'),
    ('llm_topics', '[]'::jsonb, 'llm')
ON CONFLICT (key) DO NOTHING;
