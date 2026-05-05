-- Initial schema for the Trading Strategy Memory database
-- Run once during first deployment (handled automatically by MemoryAgent)

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";  -- For fast text search on rationale

-- ─── Core trade log ──────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS trades (
    order_id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id        UUID NOT NULL,
    symbol            TEXT NOT NULL,
    side              TEXT NOT NULL CHECK (side IN ('BUY', 'SELL')),
    entry_price       DOUBLE PRECISION NOT NULL,
    exit_price        DOUBLE PRECISION,
    quantity          DOUBLE PRECISION NOT NULL,
    pnl               DOUBLE PRECISION,
    pnl_pct           DOUBLE PRECISION,
    win               BOOLEAN,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    closed_at         TIMESTAMPTZ,
    council_json      JSONB,           -- Full CouncilDecision snapshot
    risk_json         JSONB,           -- Full RiskReport snapshot
    failure_analysis  TEXT,
    updated_prompts   JSONB
);

-- ─── Agent prompt evolution ───────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS agent_prompts (
    agent_name        TEXT PRIMARY KEY,
    base_prompt       TEXT NOT NULL DEFAULT '',
    refinements       TEXT[] DEFAULT '{}',
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ─── Portfolio snapshots ──────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    id            BIGSERIAL PRIMARY KEY,
    snapped_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    equity        DOUBLE PRECISION NOT NULL,
    cash          DOUBLE PRECISION NOT NULL,
    daily_pnl     DOUBLE PRECISION,
    drawdown_pct  DOUBLE PRECISION,
    positions     JSONB DEFAULT '{}'
);

-- ─── Model drift log ──────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS drift_events (
    id           BIGSERIAL PRIMARY KEY,
    detected_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    model_name   TEXT NOT NULL,
    feature_name TEXT NOT NULL,
    psi_score    DOUBLE PRECISION NOT NULL,
    retrain_triggered BOOLEAN DEFAULT FALSE
);

-- ─── Indexes ──────────────────────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_trades_symbol_closed ON trades(symbol, closed_at DESC);
CREATE INDEX IF NOT EXISTS idx_trades_session ON trades(session_id);
CREATE INDEX IF NOT EXISTS idx_trades_win ON trades(win);
CREATE INDEX IF NOT EXISTS idx_snapshots_time ON portfolio_snapshots(snapped_at DESC);
CREATE INDEX IF NOT EXISTS idx_drift_model ON drift_events(model_name, detected_at DESC);
CREATE INDEX IF NOT EXISTS idx_trades_rationale ON trades USING gin(failure_analysis gin_trgm_ops);
