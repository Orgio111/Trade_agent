-- ═══════════════════════════════════════════════════════════
-- QUANTEX — Initial Database Schema
-- Phase 1: Core tables for trade logging and agent memory
-- ═══════════════════════════════════════════════════════════

-- Enable pgvector for embedding support (Phase 2+)
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ── Trades ───────────────────────────────────────────────
CREATE TABLE trades (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol          VARCHAR(20) NOT NULL,
    side            VARCHAR(4) NOT NULL CHECK (side IN ('buy', 'sell')),
    entry_price     DECIMAL(20, 8),
    exit_price      DECIMAL(20, 8),
    quantity        DECIMAL(20, 8),
    leverage        INTEGER DEFAULT 1,
    pnl             DECIMAL(20, 8),
    fees            DECIMAL(20, 8),
    agent_id        VARCHAR(50),
    strategy_name   VARCHAR(100),
    confidence_score DECIMAL(5, 4),
    entry_time      TIMESTAMPTZ,
    exit_time       TIMESTAMPTZ,
    status          VARCHAR(20) DEFAULT 'open' CHECK (status IN ('open', 'closed', 'cancelled')),
    reason          TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_trades_symbol ON trades(symbol);
CREATE INDEX idx_trades_agent ON trades(agent_id);
CREATE INDEX idx_trades_created ON trades(created_at DESC);
CREATE INDEX idx_trades_status ON trades(status);

-- ── Positions ────────────────────────────────────────────
CREATE TABLE positions (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    trade_id        UUID REFERENCES trades(id) ON DELETE CASCADE,
    symbol          VARCHAR(20) NOT NULL,
    side            VARCHAR(4) NOT NULL CHECK (side IN ('buy', 'sell')),
    entry_price     DECIMAL(20, 8),
    current_price   DECIMAL(20, 8),
    quantity        DECIMAL(20, 8),
    leverage        INTEGER DEFAULT 1,
    unrealized_pnl  DECIMAL(20, 8),
    realized_pnl    DECIMAL(20, 8) DEFAULT 0,
    stop_loss       DECIMAL(20, 8),
    take_profits    JSONB,  -- [{price, qty_pct, filled, trail, trail_dist}]
    status          VARCHAR(20) DEFAULT 'open' CHECK (status IN ('open', 'closed', 'liquidated')),
    opened_at       TIMESTAMPTZ DEFAULT NOW(),
    closed_at       TIMESTAMPTZ
);

CREATE INDEX idx_positions_symbol ON positions(symbol);
CREATE INDEX idx_positions_status ON positions(status);

-- ── Agent Memory ─────────────────────────────────────────
CREATE TABLE agent_memory (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id        VARCHAR(50) NOT NULL,
    memory_type     VARCHAR(30) NOT NULL CHECK (memory_type IN ('trade', 'pattern', 'regime', 'signal')),
    content         TEXT,
    embedding       vector(1536),    -- pgvector for semantic search
    importance      DECIMAL(5, 4) DEFAULT 0.5,
    metadata        JSONB,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    expires_at      TIMESTAMPTZ
);

CREATE INDEX idx_agent_memory_type ON agent_memory(agent_id, memory_type);
CREATE INDEX idx_agent_memory_created ON agent_memory(created_at DESC);

-- ── Strategy Performance ─────────────────────────────────
CREATE TABLE strategy_performance (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    strategy_name   VARCHAR(100) NOT NULL,
    version         INTEGER DEFAULT 1,
    win_rate        DECIMAL(5, 4),
    sharpe_ratio    DECIMAL(8, 4),
    max_drawdown    DECIMAL(8, 4),
    profit_factor   DECIMAL(8, 4),
    total_trades    INTEGER DEFAULT 0,
    total_pnl       DECIMAL(20, 8),
    active          BOOLEAN DEFAULT false,
    params          JSONB,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_strategy_performance ON strategy_performance(strategy_name, version);

-- ── Market Regimes ───────────────────────────────────────
CREATE TABLE market_regimes (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    regime_type     VARCHAR(30) NOT NULL CHECK (regime_type IN ('bull', 'bear', 'ranging', 'volatile', 'weak_trend')),
    symbol          VARCHAR(20),
    start_time      TIMESTAMPTZ,
    end_time        TIMESTAMPTZ,
    confidence      DECIMAL(5, 4),
    indicators      JSONB,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_regimes_symbol ON market_regimes(symbol);
CREATE INDEX idx_regimes_time ON market_regimes(start_time DESC);

-- ── Daily PnL ────────────────────────────────────────────
CREATE TABLE daily_pnl (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    date            DATE NOT NULL,
    total_pnl       DECIMAL(20, 8),
    total_trades    INTEGER DEFAULT 0,
    win_count       INTEGER DEFAULT 0,
    loss_count      INTEGER DEFAULT 0,
    fees_total      DECIMAL(20, 8),
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(date)
);

-- ═══════════════════════════════════════════════════════════
-- Seed data: Initial strategy configuration
-- ═══════════════════════════════════════════════════════════

INSERT INTO strategy_performance (strategy_name, version, active, params)
VALUES ('ema_crossover_rsi', 1, true, '{
    "ema_fast": 9,
    "ema_slow": 21,
    "rsi_period": 14,
    "rsi_overbought": 70,
    "rsi_oversold": 30,
    "atr_multiplier_sl": 1.5,
    "atr_multiplier_tp1": 2.0,
    "atr_multiplier_tp2": 3.5,
    "timeframe": "1m"
}'::jsonb);
