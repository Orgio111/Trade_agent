-- QUANTEX runtime durability scaffolding.
-- Additive only: transport inbox/outbox state and authoritative projections.

CREATE TABLE IF NOT EXISTS event_inbox (
    stream_name         VARCHAR(128) NOT NULL,
    durable_name        VARCHAR(64) NOT NULL,
    event_id            VARCHAR(128) NOT NULL,
    subject             VARCHAR(256) NOT NULL,
    stream_sequence     BIGINT NOT NULL CHECK (stream_sequence > 0),
    delivery_count      INTEGER NOT NULL DEFAULT 1 CHECK (delivery_count > 0),
    payload             JSONB NOT NULL,
    payload_checksum    CHAR(64) NOT NULL
                            CHECK (payload_checksum ~ '^[0-9a-f]{64}$'),
    received_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    processed_at        TIMESTAMPTZ,
    last_error_code     VARCHAR(128),
    PRIMARY KEY (stream_name, durable_name, event_id),
    UNIQUE (stream_name, durable_name, stream_sequence)
);

CREATE TABLE IF NOT EXISTS event_outbox (
    event_id            VARCHAR(128) PRIMARY KEY,
    aggregate_type      VARCHAR(64) NOT NULL,
    aggregate_id        VARCHAR(128) NOT NULL,
    subject             VARCHAR(256) NOT NULL,
    payload             JSONB NOT NULL,
    payload_checksum    CHAR(64) NOT NULL
                            CHECK (payload_checksum ~ '^[0-9a-f]{64}$'),
    headers             JSONB NOT NULL DEFAULT '{}'::jsonb,
    status              VARCHAR(16) NOT NULL DEFAULT 'pending'
                            CHECK (status IN ('pending', 'published', 'dead_letter')),
    publish_attempts    INTEGER NOT NULL DEFAULT 0
                            CHECK (publish_attempts BETWEEN 0 AND 20),
    available_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    published_at        TIMESTAMPTZ,
    last_error_code     VARCHAR(128)
);

CREATE TABLE IF NOT EXISTS consumer_offsets (
    stream_name         VARCHAR(128) NOT NULL,
    durable_name        VARCHAR(64) NOT NULL,
    last_stream_sequence BIGINT NOT NULL DEFAULT 0
                            CHECK (last_stream_sequence >= 0),
    last_event_id       VARCHAR(128),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (stream_name, durable_name)
);

-- Bind the model-produced candidate envelope to the immutable signal row.
ALTER TABLE signals
    ADD COLUMN IF NOT EXISTS candidate_event_id VARCHAR(64);

-- Immutable portfolio state consumed by deterministic risk evaluation.
CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    account_id          VARCHAR(64) NOT NULL,
    state_id            VARCHAR(64) NOT NULL,
    schema_version      VARCHAR(16) NOT NULL DEFAULT '1.0',
    source_sequence     BIGINT NOT NULL CHECK (source_sequence >= 0),
    equity              NUMERIC(38, 18) NOT NULL CHECK (equity > 0),
    cash                NUMERIC(38, 18) NOT NULL CHECK (cash >= 0),
    daily_pnl           NUMERIC(38, 18) NOT NULL,
    weekly_pnl          NUMERIC(38, 18) NOT NULL,
    consecutive_losses  INTEGER NOT NULL CHECK (consecutive_losses >= 0),
    open_positions      INTEGER NOT NULL CHECK (open_positions >= 0),
    gross_exposure      NUMERIC(38, 18) NOT NULL CHECK (gross_exposure >= 0),
    kill_switch_active  BOOLEAN NOT NULL DEFAULT TRUE,
    reconciled_at       TIMESTAMPTZ NOT NULL,
    payload             JSONB NOT NULL,
    payload_checksum    CHAR(64) NOT NULL
                            CHECK (payload_checksum ~ '^[0-9a-f]{64}$'),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (account_id, state_id),
    UNIQUE (account_id, source_sequence)
);

-- Versioned venue rules.  There is intentionally no active/default seed.
CREATE TABLE IF NOT EXISTS instrument_constraints (
    venue               VARCHAR(32) NOT NULL,
    market_type         VARCHAR(32) NOT NULL,
    instrument_id       VARCHAR(64) NOT NULL,
    version             VARCHAR(64) NOT NULL,
    tick_size           NUMERIC(38, 18) NOT NULL CHECK (tick_size > 0),
    step_size           NUMERIC(38, 18) NOT NULL CHECK (step_size > 0),
    min_quantity        NUMERIC(38, 18) NOT NULL CHECK (min_quantity > 0),
    min_notional        NUMERIC(38, 18) NOT NULL CHECK (min_notional > 0),
    effective_at        TIMESTAMPTZ NOT NULL,
    expires_at          TIMESTAMPTZ,
    payload             JSONB NOT NULL,
    payload_checksum    CHAR(64) NOT NULL
                            CHECK (payload_checksum ~ '^[0-9a-f]{64}$'),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (venue, market_type, instrument_id, version),
    CHECK (expires_at IS NULL OR expires_at > effective_at)
);

-- Derived position and lot scaffolding.  These rows are projections, never
-- broker authority, and must point back to the authoritative fill ledger.
CREATE TABLE IF NOT EXISTS position_projections (
    account_id          VARCHAR(64) NOT NULL,
    venue               VARCHAR(32) NOT NULL,
    market_type         VARCHAR(32) NOT NULL,
    instrument_id       VARCHAR(64) NOT NULL,
    side                VARCHAR(8) NOT NULL CHECK (side IN ('buy', 'sell')),
    quantity            NUMERIC(38, 18) NOT NULL CHECK (quantity >= 0),
    average_entry_price NUMERIC(38, 18),
    realized_pnl        NUMERIC(38, 18) NOT NULL DEFAULT 0,
    source_sequence     BIGINT NOT NULL CHECK (source_sequence >= 0),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (account_id, venue, market_type, instrument_id, side),
    CHECK (
        (quantity = 0 AND average_entry_price IS NULL)
        OR (quantity > 0 AND average_entry_price > 0)
    )
);

CREATE TABLE IF NOT EXISTS position_lots (
    lot_id              VARCHAR(128) PRIMARY KEY,
    account_id          VARCHAR(64) NOT NULL,
    venue               VARCHAR(32) NOT NULL,
    market_type         VARCHAR(32) NOT NULL,
    instrument_id       VARCHAR(64) NOT NULL,
    side                VARCHAR(8) NOT NULL CHECK (side IN ('buy', 'sell')),
    opening_fill_id     VARCHAR(128) NOT NULL REFERENCES fills(id),
    opened_quantity     NUMERIC(38, 18) NOT NULL CHECK (opened_quantity > 0),
    remaining_quantity  NUMERIC(38, 18) NOT NULL CHECK (remaining_quantity >= 0),
    entry_price         NUMERIC(38, 18) NOT NULL CHECK (entry_price > 0),
    opened_at           TIMESTAMPTZ NOT NULL,
    closed_at           TIMESTAMPTZ,
    source_sequence     BIGINT NOT NULL CHECK (source_sequence >= 0),
    CHECK (remaining_quantity <= opened_quantity),
    CHECK (closed_at IS NULL OR remaining_quantity = 0)
);

CREATE OR REPLACE FUNCTION quantex_reject_immutable_mutation()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    RAISE EXCEPTION '% is append-only', TG_TABLE_NAME
        USING ERRCODE = '55000';
END;
$$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_trigger
        WHERE tgname = 'trg_risk_decisions_immutable'
          AND tgrelid = 'risk_decisions'::regclass
    ) THEN
        CREATE TRIGGER trg_risk_decisions_immutable
        BEFORE UPDATE OR DELETE ON risk_decisions
        FOR EACH ROW EXECUTE FUNCTION quantex_reject_immutable_mutation();
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_trigger
        WHERE tgname = 'trg_signals_immutable'
          AND tgrelid = 'signals'::regclass
    ) THEN
        CREATE TRIGGER trg_signals_immutable
        BEFORE UPDATE OR DELETE ON signals
        FOR EACH ROW EXECUTE FUNCTION quantex_reject_immutable_mutation();
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_trigger
        WHERE tgname = 'trg_portfolio_snapshots_immutable'
          AND tgrelid = 'portfolio_snapshots'::regclass
    ) THEN
        CREATE TRIGGER trg_portfolio_snapshots_immutable
        BEFORE UPDATE OR DELETE ON portfolio_snapshots
        FOR EACH ROW EXECUTE FUNCTION quantex_reject_immutable_mutation();
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_trigger
        WHERE tgname = 'trg_instrument_constraints_immutable'
          AND tgrelid = 'instrument_constraints'::regclass
    ) THEN
        CREATE TRIGGER trg_instrument_constraints_immutable
        BEFORE UPDATE OR DELETE ON instrument_constraints
        FOR EACH ROW EXECUTE FUNCTION quantex_reject_immutable_mutation();
    END IF;
END;
$$;

CREATE INDEX IF NOT EXISTS idx_event_inbox_unprocessed
    ON event_inbox (stream_name, durable_name, stream_sequence)
    WHERE processed_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_event_outbox_pending
    ON event_outbox (available_at, created_at)
    WHERE status = 'pending';
CREATE INDEX IF NOT EXISTS idx_portfolio_snapshots_latest
    ON portfolio_snapshots (account_id, source_sequence DESC);
-- One immutable candidate signal receives exactly one deterministic verdict.
-- JetStream redelivery must reuse that verdict even after risk inputs advance.
CREATE UNIQUE INDEX IF NOT EXISTS uq_risk_decision_signal_once
    ON risk_decisions (signal_id);
CREATE UNIQUE INDEX IF NOT EXISTS uq_signal_candidate_event
    ON signals (candidate_event_id)
    WHERE candidate_event_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_constraints_effective
    ON instrument_constraints (
        venue, market_type, instrument_id, effective_at DESC
    );
CREATE INDEX IF NOT EXISTS idx_position_lots_open
    ON position_lots (account_id, venue, market_type, instrument_id, opened_at)
    WHERE remaining_quantity > 0;
