-- Restart-safe transactional outbox dispatch and bounded dead-letter state.

ALTER TABLE event_inbox
    ALTER COLUMN stream_sequence DROP NOT NULL;

ALTER TABLE event_outbox
    DROP CONSTRAINT IF EXISTS event_outbox_status_check;
ALTER TABLE event_outbox
    ADD CONSTRAINT event_outbox_status_check
    CHECK (status IN ('pending', 'publishing', 'published', 'dead_letter'));
ALTER TABLE event_outbox
    ADD COLUMN IF NOT EXISTS next_attempt_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ADD COLUMN IF NOT EXISTS lease_owner VARCHAR(160),
    ADD COLUMN IF NOT EXISTS lease_expires_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS source_subject VARCHAR(256),
    ADD COLUMN IF NOT EXISTS source_event_id VARCHAR(128);

CREATE TABLE IF NOT EXISTS dead_letter_events (
    id                      BIGSERIAL PRIMARY KEY,
    original_subject        VARCHAR(256) NOT NULL,
    original_event_id       VARCHAR(128) NOT NULL,
    payload_hash            CHAR(64) NOT NULL
                                CHECK (payload_hash ~ '^[0-9a-f]{64}$'),
    failure_class           VARCHAR(128) NOT NULL,
    failure_message         VARCHAR(512) NOT NULL,
    consumer                VARCHAR(128) NOT NULL,
    first_failure_at        TIMESTAMPTZ NOT NULL,
    last_failure_at         TIMESTAMPTZ NOT NULL,
    retry_count             INTEGER NOT NULL CHECK (retry_count > 0),
    stack_trace_fingerprint CHAR(64) NOT NULL
                                CHECK (stack_trace_fingerprint ~ '^[0-9a-f]{64}$'),
    schema_version          VARCHAR(16) NOT NULL DEFAULT '1.0',
    replayed_at             TIMESTAMPTZ,
    replayed_by             VARCHAR(128),
    replay_reason           VARCHAR(512),
    UNIQUE (consumer, original_event_id)
);

DROP INDEX IF EXISTS idx_event_outbox_pending;
CREATE INDEX IF NOT EXISTS idx_event_outbox_dispatchable
    ON event_outbox (next_attempt_at, created_at)
    WHERE status IN ('pending', 'publishing');
CREATE INDEX IF NOT EXISTS idx_event_outbox_lease_expiry
    ON event_outbox (lease_expires_at)
    WHERE status = 'publishing';
