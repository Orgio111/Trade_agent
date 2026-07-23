-- Restart-safe incremental feature state.

CREATE TABLE IF NOT EXISTS feature_checkpoints (
    symbol                  VARCHAR(64) PRIMARY KEY,
    state_version           BIGINT NOT NULL CHECK (state_version > 0),
    last_market_event_id    UUID NOT NULL,
    checkpoint              JSONB NOT NULL,
    checkpoint_checksum     CHAR(64) NOT NULL
                                CHECK (checkpoint_checksum ~ '^[0-9a-f]{64}$'),
    feature_snapshot        JSONB NOT NULL,
    feature_checksum        CHAR(64) NOT NULL
                                CHECK (feature_checksum ~ '^[0-9a-f]{64}$'),
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (symbol, last_market_event_id)
);

CREATE INDEX IF NOT EXISTS idx_feature_checkpoints_updated
    ON feature_checkpoints (updated_at DESC);
