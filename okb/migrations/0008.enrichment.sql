-- LLM enrichment for document annotation (TODOs and entities)
-- depends: 0006.llm-cache

-- Track enrichment state on documents
ALTER TABLE documents ADD COLUMN IF NOT EXISTS enriched_at TIMESTAMPTZ;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS enrichment_version INTEGER;

-- Index for "needs enrichment" queries
CREATE INDEX IF NOT EXISTS idx_documents_needs_enrichment
    ON documents(enriched_at) WHERE enriched_at IS NULL;

-- Pending entity suggestions (before approval)
CREATE TABLE IF NOT EXISTS pending_entities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    entity_name TEXT NOT NULL,
    entity_type TEXT NOT NULL,  -- person, project, technology, concept, organization
    aliases JSONB DEFAULT '[]',
    description TEXT,
    mentions JSONB DEFAULT '[]',  -- Context snippets from source document
    confidence REAL,
    status TEXT DEFAULT 'pending',  -- pending, approved, rejected
    created_at TIMESTAMPTZ DEFAULT NOW(),
    reviewed_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_pending_entities_status ON pending_entities(status);
CREATE INDEX IF NOT EXISTS idx_pending_entities_source ON pending_entities(source_document_id);
CREATE INDEX IF NOT EXISTS idx_pending_entities_type ON pending_entities(entity_type);

-- Entity references (links entity documents to source documents)
-- When an entity is approved, it becomes a document with source_path like okb://entity/person/john-smith
-- This table tracks which documents mention each entity
CREATE TABLE IF NOT EXISTS entity_refs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    entity_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    mention_text TEXT NOT NULL,
    context TEXT,  -- Surrounding text for context
    confidence REAL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(entity_id, document_id, mention_text)
);

CREATE INDEX IF NOT EXISTS idx_entity_refs_entity ON entity_refs(entity_id);
CREATE INDEX IF NOT EXISTS idx_entity_refs_document ON entity_refs(document_id);
