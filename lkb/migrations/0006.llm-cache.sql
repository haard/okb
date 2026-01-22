-- LLM response cache for avoiding redundant API calls
-- depends: 0005.database-metadata

CREATE TABLE IF NOT EXISTS llm_cache (
    content_hash TEXT PRIMARY KEY,  -- SHA256 of (prompt + system + model)
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    response TEXT NOT NULL,  -- JSON: {content, input_tokens, output_tokens}
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for cache cleanup queries
CREATE INDEX IF NOT EXISTS llm_cache_created_at_idx ON llm_cache(created_at);
