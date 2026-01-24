-- API tokens for HTTP authentication
-- depends: 0003.structured-fields

CREATE TABLE IF NOT EXISTS tokens (
    token_hash TEXT PRIMARY KEY,  -- SHA256(full_token), hex encoded
    permissions TEXT NOT NULL CHECK (permissions IN ('ro', 'rw')),
    description TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_used_at TIMESTAMPTZ
);

-- Index for listing tokens
CREATE INDEX IF NOT EXISTS tokens_created_at_idx ON tokens(created_at DESC);
