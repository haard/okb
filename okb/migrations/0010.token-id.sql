-- Add ID column to tokens for easier revocation
-- depends: 0009.entity-consolidation

ALTER TABLE tokens ADD COLUMN IF NOT EXISTS id SERIAL;

-- Create index for ID lookups
CREATE INDEX IF NOT EXISTS tokens_id_idx ON tokens(id);
