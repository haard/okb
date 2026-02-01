-- Entity consolidation: deduplication, cross-doc detection, clustering, relationships
-- depends: 0008.enrichment

-- Canonical mappings: alias text -> entity document
-- Used for deduplication and alias resolution
CREATE TABLE IF NOT EXISTS entity_aliases (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    alias_text TEXT NOT NULL,
    entity_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    confidence REAL,  -- How confident we are this alias belongs to entity
    source TEXT DEFAULT 'manual',  -- 'manual', 'merge', 'extraction'
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(alias_text, entity_id)
);

CREATE INDEX IF NOT EXISTS idx_entity_aliases_text ON entity_aliases(LOWER(alias_text));
CREATE INDEX IF NOT EXISTS idx_entity_aliases_entity ON entity_aliases(entity_id);

-- Proposed entity merges awaiting user confirmation
CREATE TABLE IF NOT EXISTS pending_entity_merges (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    canonical_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    duplicate_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    confidence REAL NOT NULL,  -- How confident we are these are the same
    reason TEXT,  -- Why we think they're the same ("embedding_similarity", "alias_match", "llm")
    detected_at TIMESTAMPTZ DEFAULT NOW(),
    status TEXT DEFAULT 'pending',  -- 'pending', 'approved', 'rejected'
    reviewed_at TIMESTAMPTZ,
    UNIQUE(canonical_id, duplicate_id)
);

CREATE INDEX IF NOT EXISTS idx_pending_merges_status ON pending_entity_merges(status);
CREATE INDEX IF NOT EXISTS idx_pending_merges_confidence ON pending_entity_merges(confidence DESC);

-- Entity-to-entity relationships
CREATE TABLE IF NOT EXISTS entity_relationships (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_entity_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    target_entity_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    relationship_type TEXT NOT NULL,  -- 'works_for', 'uses', 'belongs_to', 'related_to'
    confidence REAL,
    source TEXT DEFAULT 'extraction',  -- 'extraction', 'manual'
    context TEXT,  -- Supporting context for the relationship
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(source_entity_id, target_entity_id, relationship_type)
);

CREATE INDEX IF NOT EXISTS idx_entity_rel_source ON entity_relationships(source_entity_id);
CREATE INDEX IF NOT EXISTS idx_entity_rel_target ON entity_relationships(target_entity_id);
CREATE INDEX IF NOT EXISTS idx_entity_rel_type ON entity_relationships(relationship_type);

-- Topic clusters group related entities and documents
CREATE TABLE IF NOT EXISTS topic_clusters (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    description TEXT,
    centroid vector(768),  -- Cluster centroid embedding
    member_count INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_topic_clusters_centroid ON topic_clusters
    USING hnsw (centroid vector_cosine_ops);

-- Cluster membership: entities and documents can belong to clusters
CREATE TABLE IF NOT EXISTS topic_cluster_members (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    cluster_id UUID NOT NULL REFERENCES topic_clusters(id) ON DELETE CASCADE,
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    distance REAL,  -- Distance from cluster centroid
    is_entity BOOLEAN DEFAULT FALSE,  -- True if document is an entity
    added_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(cluster_id, document_id)
);

CREATE INDEX IF NOT EXISTS idx_cluster_members_cluster ON topic_cluster_members(cluster_id);
CREATE INDEX IF NOT EXISTS idx_cluster_members_document ON topic_cluster_members(document_id);

-- Proposed cluster merges awaiting confirmation
CREATE TABLE IF NOT EXISTS pending_cluster_merges (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    primary_cluster_id UUID NOT NULL REFERENCES topic_clusters(id) ON DELETE CASCADE,
    secondary_cluster_id UUID NOT NULL REFERENCES topic_clusters(id) ON DELETE CASCADE,
    similarity REAL NOT NULL,  -- How similar the clusters are
    status TEXT DEFAULT 'pending',  -- 'pending', 'approved', 'rejected'
    detected_at TIMESTAMPTZ DEFAULT NOW(),
    reviewed_at TIMESTAMPTZ,
    UNIQUE(primary_cluster_id, secondary_cluster_id)
);

CREATE INDEX IF NOT EXISTS idx_pending_cluster_merges_status ON pending_cluster_merges(status);

-- Cross-document entity candidates: detected mentions not yet extracted as entities
CREATE TABLE IF NOT EXISTS cross_doc_entity_candidates (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    text TEXT NOT NULL,  -- The mention text (normalized)
    document_ids UUID[] NOT NULL,  -- Array of document IDs containing this mention
    document_count INTEGER NOT NULL,  -- Number of documents (for quick filtering)
    sample_contexts JSONB DEFAULT '[]',  -- Sample text contexts where it appears
    suggested_type TEXT,  -- Suggested entity type
    confidence REAL,
    status TEXT DEFAULT 'pending',  -- 'pending', 'approved', 'rejected', 'exists'
    created_at TIMESTAMPTZ DEFAULT NOW(),
    reviewed_at TIMESTAMPTZ,
    UNIQUE(text)
);

CREATE INDEX IF NOT EXISTS idx_cross_doc_status ON cross_doc_entity_candidates(status);
CREATE INDEX IF NOT EXISTS idx_cross_doc_count ON cross_doc_entity_candidates(document_count DESC);

-- Track consolidation runs
CREATE TABLE IF NOT EXISTS consolidation_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_type TEXT NOT NULL,  -- 'dedup', 'cross_doc', 'cluster', 'relationship', 'full'
    started_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    stats JSONB DEFAULT '{}',  -- Run statistics
    error TEXT  -- Error message if failed
);
