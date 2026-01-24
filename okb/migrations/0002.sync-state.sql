-- Sync state for API sources (plugin system)
-- depends: 0001.initial-schema

CREATE TABLE IF NOT EXISTS sync_state (
    source_name TEXT NOT NULL,
    database_name TEXT NOT NULL DEFAULT 'default',
    last_sync TIMESTAMPTZ,
    cursor TEXT,
    extra JSONB DEFAULT '{}',
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (source_name, database_name)
);

-- Index for listing sync states
CREATE INDEX IF NOT EXISTS sync_state_updated_idx ON sync_state(updated_at DESC);

-- Trigger for updated_at
DROP TRIGGER IF EXISTS sync_state_updated_at ON sync_state;
CREATE TRIGGER sync_state_updated_at
    BEFORE UPDATE ON sync_state
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();
