CREATE TABLE IF NOT EXISTS worker_leases (
    worker_id VARCHAR(160) PRIMARY KEY,
    service_name VARCHAR(64) NOT NULL,
    instance_id VARCHAR(128) NOT NULL,
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_heartbeat TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    lease_expires_at TIMESTAMPTZ NOT NULL,
    status VARCHAR(32) NOT NULL,
    last_error VARCHAR(512),
    build_version VARCHAR(128) NOT NULL,
    consumer_name VARCHAR(128) NOT NULL,
    consumer_lag BIGINT CHECK (consumer_lag IS NULL OR consumer_lag >= 0)
);

CREATE INDEX IF NOT EXISTS idx_worker_leases_expiry
    ON worker_leases (lease_expires_at);