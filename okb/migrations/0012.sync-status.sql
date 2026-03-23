-- Add status tracking to sync_state for non-blocking sync
-- depends: 0011.synthesis

ALTER TABLE sync_state ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'idle';
ALTER TABLE sync_state ADD COLUMN IF NOT EXISTS started_at TIMESTAMPTZ;
ALTER TABLE sync_state ADD COLUMN IF NOT EXISTS error TEXT;
