-- Replace enrichment with synthesis
-- depends: 0010.token-id

CREATE TABLE IF NOT EXISTS pending_synthesis (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    project TEXT,
    model TEXT NOT NULL,
    origin TEXT NOT NULL DEFAULT 'cli',
    status TEXT NOT NULL DEFAULT 'pending',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    reviewed_at TIMESTAMPTZ
);
CREATE INDEX idx_pending_synthesis_status ON pending_synthesis(status);

-- Drop enrichment infrastructure
ALTER TABLE documents DROP COLUMN IF EXISTS enriched_at;
ALTER TABLE documents DROP COLUMN IF EXISTS enrichment_version;
DROP INDEX IF EXISTS idx_documents_needs_enrichment;
DROP TABLE IF EXISTS cross_doc_entity_candidates CASCADE;
DROP TABLE IF EXISTS pending_cluster_merges CASCADE;
DROP TABLE IF EXISTS topic_cluster_members CASCADE;
DROP TABLE IF EXISTS topic_clusters CASCADE;
DROP TABLE IF EXISTS entity_relationships CASCADE;
DROP TABLE IF EXISTS pending_entity_merges CASCADE;
DROP TABLE IF EXISTS entity_aliases CASCADE;
DROP TABLE IF EXISTS entity_refs CASCADE;
DROP TABLE IF EXISTS pending_entities CASCADE;
DROP TABLE IF EXISTS consolidation_runs CASCADE;
