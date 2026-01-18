-- Structured fields for actionable items (tasks, events, emails)
-- depends: 0002.sync-state

-- Add structured fields to documents table for temporal/status queries
ALTER TABLE documents ADD COLUMN IF NOT EXISTS due_date TIMESTAMPTZ;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS event_start TIMESTAMPTZ;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS event_end TIMESTAMPTZ;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS status TEXT;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS priority INTEGER;

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS documents_due_date_idx ON documents(due_date) WHERE due_date IS NOT NULL;
CREATE INDEX IF NOT EXISTS documents_event_start_idx ON documents(event_start) WHERE event_start IS NOT NULL;
CREATE INDEX IF NOT EXISTS documents_status_idx ON documents(status) WHERE status IS NOT NULL;

-- Composite index for "incomplete tasks due soon" queries
CREATE INDEX IF NOT EXISTS documents_actionable_idx ON documents(due_date, status)
    WHERE due_date IS NOT NULL AND status IS NOT NULL;

-- Composite index for "events in date range" queries
CREATE INDEX IF NOT EXISTS documents_event_range_idx ON documents(event_start, event_end)
    WHERE event_start IS NOT NULL;
