-- QUANTEX deterministic execution ledger and promotion controls.
-- Phase 0-4 foundation: append-oriented decisions, intents, orders, and fills.

CREATE TABLE IF NOT EXISTS ingest_runs (
    id                  VARCHAR(128) PRIMARY KEY,
    venue               VARCHAR(32) NOT NULL,
    source_mode         VARCHAR(16) NOT NULL
                            CHECK (source_mode IN ('historical', 'replay', 'paper_live', 'live')),
    started_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at        TIMESTAMPTZ,
    raw_manifest_uri    TEXT,
    manifest_checksum   VARCHAR(128),
    status              VARCHAR(16) NOT NULL DEFAULT 'running'
                            CHECK (status IN ('running', 'complete', 'failed', 'quarantined')),
    metadata            JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS data_quality_events (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    ingest_run_id       VARCHAR(128) REFERENCES ingest_runs(id),
    event_id            VARCHAR(64),
    venue               VARCHAR(32) NOT NULL,
    instrument_id       VARCHAR(64) NOT NULL,
    event_type          VARCHAR(64) NOT NULL,
    exchange_ts         TIMESTAMPTZ,
    received_ts         TIMESTAMPTZ NOT NULL,
    accepted            BOOLEAN NOT NULL,
    reason_codes        JSONB NOT NULL DEFAULT '[]'::jsonb,
    details             JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS strategy_versions (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    strategy_name       VARCHAR(128) NOT NULL,
    version             VARCHAR(64) NOT NULL,
    code_hash           VARCHAR(128) NOT NULL,
    config              JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (strategy_name, version, code_hash)
);

CREATE TABLE IF NOT EXISTS signals (
    id                  VARCHAR(64) PRIMARY KEY,
    trace_id            VARCHAR(128) NOT NULL,
    strategy_version_id UUID REFERENCES strategy_versions(id),
    instrument_id       VARCHAR(64) NOT NULL,
    side                VARCHAR(8) NOT NULL CHECK (side IN ('buy', 'sell')),
    source_mode         VARCHAR(16) NOT NULL
                            CHECK (source_mode IN ('historical', 'replay', 'paper_live', 'live')),
    reference_price     NUMERIC(38, 18) NOT NULL CHECK (reference_price > 0),
    stop_price          NUMERIC(38, 18) NOT NULL CHECK (stop_price > 0),
    take_profit_price   NUMERIC(38, 18),
    confidence          NUMERIC(20, 19) CHECK (confidence BETWEEN 0 AND 1),
    feature_snapshot_id VARCHAR(64),
    market_event_id     VARCHAR(64) NOT NULL,
    payload             JSONB NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS risk_policies (
    version             VARCHAR(64) PRIMARY KEY,
    paper_only          BOOLEAN NOT NULL DEFAULT TRUE,
    policy              JSONB NOT NULL,
    policy_hash         VARCHAR(128) NOT NULL UNIQUE,
    active              BOOLEAN NOT NULL DEFAULT FALSE,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_active_risk_policy
    ON risk_policies (active)
    WHERE active;

CREATE TABLE IF NOT EXISTS risk_decisions (
    id                  VARCHAR(64) PRIMARY KEY,
    signal_id           VARCHAR(64) NOT NULL REFERENCES signals(id),
    trace_id            VARCHAR(128) NOT NULL,
    account_id          VARCHAR(64) NOT NULL,
    policy_version      VARCHAR(64) NOT NULL,
    state_snapshot_id   VARCHAR(64) NOT NULL,
    approved            BOOLEAN NOT NULL,
    reason_codes        JSONB NOT NULL,
    candidate_hash      VARCHAR(64) NOT NULL,
    approved_quantity   NUMERIC(38, 18) NOT NULL DEFAULT 0
                            CHECK (approved_quantity >= 0),
    approved_risk_pct   NUMERIC(20, 19) NOT NULL DEFAULT 0
                            CHECK (approved_risk_pct BETWEEN 0 AND 1),
    state_snapshot      JSONB NOT NULL,
    decision_payload    JSONB NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (signal_id, policy_version, state_snapshot_id)
);

CREATE TABLE IF NOT EXISTS order_intents (
    id                  VARCHAR(64) PRIMARY KEY,
    client_order_id     VARCHAR(64) NOT NULL UNIQUE,
    risk_decision_id    VARCHAR(64) NOT NULL UNIQUE REFERENCES risk_decisions(id),
    trace_id            VARCHAR(128) NOT NULL,
    account_id          VARCHAR(64) NOT NULL,
    venue               VARCHAR(32) NOT NULL,
    market_type         VARCHAR(32) NOT NULL,
    instrument_id       VARCHAR(64) NOT NULL,
    side                VARCHAR(8) NOT NULL CHECK (side IN ('buy', 'sell')),
    order_type          VARCHAR(16) NOT NULL CHECK (order_type IN ('market', 'limit')),
    quantity            NUMERIC(38, 18) NOT NULL CHECK (quantity > 0),
    limit_price         NUMERIC(38, 18),
    source_mode         VARCHAR(16) NOT NULL
                            CHECK (source_mode IN ('replay', 'paper_live')),
    status              VARCHAR(24) NOT NULL DEFAULT 'persisted'
                            CHECK (status IN ('persisted', 'submitted', 'ambiguous', 'terminal')),
    payload             JSONB NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS broker_orders (
    id                  VARCHAR(128) PRIMARY KEY,
    order_intent_id     VARCHAR(64) NOT NULL REFERENCES order_intents(id),
    client_order_id     VARCHAR(64) NOT NULL UNIQUE,
    venue_order_id      VARCHAR(128),
    status              VARCHAR(24) NOT NULL
                            CHECK (status IN (
                                'pending_submit', 'ambiguous', 'open',
                                'partially_filled', 'filled', 'canceled', 'rejected'
                            )),
    requested_quantity  NUMERIC(38, 18) NOT NULL CHECK (requested_quantity > 0),
    filled_quantity     NUMERIC(38, 18) NOT NULL DEFAULT 0 CHECK (filled_quantity >= 0),
    average_price       NUMERIC(38, 18),
    total_fee           NUMERIC(38, 18) NOT NULL DEFAULT 0 CHECK (total_fee >= 0),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (order_intent_id),
    UNIQUE (venue_order_id)
);

CREATE TABLE IF NOT EXISTS order_events (
    sequence_id         BIGSERIAL PRIMARY KEY,
    order_id            VARCHAR(128) NOT NULL REFERENCES broker_orders(id),
    event_id            VARCHAR(128) NOT NULL UNIQUE,
    event_type          VARCHAR(32) NOT NULL,
    previous_status     VARCHAR(24),
    new_status          VARCHAR(24) NOT NULL,
    exchange_ts         TIMESTAMPTZ,
    received_ts         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    payload             JSONB NOT NULL,
    payload_checksum    VARCHAR(128) NOT NULL
);

CREATE TABLE IF NOT EXISTS fills (
    id                  VARCHAR(128) PRIMARY KEY,
    order_id            VARCHAR(128) NOT NULL REFERENCES broker_orders(id),
    venue_fill_id       VARCHAR(128) NOT NULL,
    quantity            NUMERIC(38, 18) NOT NULL CHECK (quantity > 0),
    price               NUMERIC(38, 18) NOT NULL CHECK (price > 0),
    fee                 NUMERIC(38, 18) NOT NULL DEFAULT 0 CHECK (fee >= 0),
    fee_asset           VARCHAR(32),
    liquidity_role      VARCHAR(8) CHECK (liquidity_role IN ('maker', 'taker', 'paper')),
    exchange_ts         TIMESTAMPTZ,
    received_ts         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    payload             JSONB NOT NULL DEFAULT '{}'::jsonb,
    UNIQUE (order_id, venue_fill_id)
);

CREATE TABLE IF NOT EXISTS cash_ledger (
    sequence_id         BIGSERIAL PRIMARY KEY,
    account_id          VARCHAR(64) NOT NULL,
    asset               VARCHAR(32) NOT NULL,
    entry_type          VARCHAR(32) NOT NULL
                            CHECK (entry_type IN ('deposit', 'withdrawal', 'trade', 'fee', 'funding', 'adjustment')),
    amount              NUMERIC(38, 18) NOT NULL,
    order_id            VARCHAR(128) REFERENCES broker_orders(id),
    fill_id             VARCHAR(128) REFERENCES fills(id),
    idempotency_key     VARCHAR(128) NOT NULL UNIQUE,
    occurred_at         TIMESTAMPTZ NOT NULL,
    metadata            JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS reconciliation_runs (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    account_id          VARCHAR(64) NOT NULL,
    venue               VARCHAR(32) NOT NULL,
    source_mode         VARCHAR(16) NOT NULL
                            CHECK (source_mode IN ('replay', 'paper_live', 'live')),
    started_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at        TIMESTAMPTZ,
    status              VARCHAR(16) NOT NULL
                            CHECK (status IN ('running', 'matched', 'discrepancy', 'failed')),
    discrepancy_count   INTEGER NOT NULL DEFAULT 0 CHECK (discrepancy_count >= 0),
    report              JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS kill_switch_state (
    account_id          VARCHAR(64) PRIMARY KEY,
    active              BOOLEAN NOT NULL DEFAULT TRUE,
    reason              TEXT NOT NULL DEFAULT 'fail_closed_until_explicitly_initialized',
    changed_by          VARCHAR(128) NOT NULL DEFAULT 'system',
    changed_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    version             BIGINT NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS promotion_gates (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    strategy_version_id UUID NOT NULL REFERENCES strategy_versions(id),
    target_mode         VARCHAR(16) NOT NULL
                            CHECK (target_mode IN ('paper_live', 'shadow', 'canary', 'live')),
    status              VARCHAR(16) NOT NULL DEFAULT 'pending'
                            CHECK (status IN ('pending', 'approved', 'rejected', 'expired', 'revoked')),
    evidence            JSONB NOT NULL,
    approved_by         VARCHAR(128),
    approved_at         TIMESTAMPTZ,
    expires_at          TIMESTAMPTZ,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_quality_instrument_time
    ON data_quality_events (instrument_id, received_ts DESC);
CREATE INDEX IF NOT EXISTS idx_signals_trace
    ON signals (trace_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_risk_account_time
    ON risk_decisions (account_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_intents_account_time
    ON order_intents (account_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_order_events_order_seq
    ON order_events (order_id, sequence_id);
CREATE INDEX IF NOT EXISTS idx_fills_order
    ON fills (order_id, received_ts);
CREATE INDEX IF NOT EXISTS idx_cash_account_time
    ON cash_ledger (account_id, occurred_at);
