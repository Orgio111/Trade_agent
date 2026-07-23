-- Explicit operator authorization evidence for bounded DLQ replay.

CREATE TABLE IF NOT EXISTS replay_audit_log (
    id                  BIGSERIAL PRIMARY KEY,
    original_event_id   VARCHAR(128) NOT NULL,
    operator_identity   VARCHAR(128) NOT NULL,
    reason              VARCHAR(512) NOT NULL CHECK (length(reason) >= 10),
    dry_run             BOOLEAN NOT NULL,
    payload_hash        CHAR(64) NOT NULL
                            CHECK (payload_hash ~ '^[0-9a-f]{64}$'),
    consumer            VARCHAR(128) NOT NULL,
    authorized_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (original_event_id)
);

CREATE OR REPLACE FUNCTION quantex_reject_replay_audit_mutation()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    RAISE EXCEPTION 'replay_audit_log is append-only'
        USING ERRCODE = '55000';
END;
$$;

DROP TRIGGER IF EXISTS trg_replay_audit_immutable ON replay_audit_log;
CREATE TRIGGER trg_replay_audit_immutable
BEFORE UPDATE OR DELETE ON replay_audit_log
FOR EACH ROW EXECUTE FUNCTION quantex_reject_replay_audit_mutation();
