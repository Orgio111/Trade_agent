"""
QUANTEX Database Layer — PostgreSQL persistence via asyncpg.

Repository pattern for:
- Trade logging and retrieval
- Position tracking
- Strategy performance history
- Daily PnL snapshots
- Agent memory storage
"""
import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, date
from typing import Optional

import asyncpg


@dataclass
class TradeRecord:
    symbol: str
    side: str
    entry_price: float
    exit_price: Optional[float] = None
    quantity: float = 0.0
    leverage: int = 1
    pnl: Optional[float] = None
    fees: float = 0.0
    agent_id: Optional[str] = None
    strategy_name: Optional[str] = None
    confidence_score: Optional[float] = None
    entry_time: Optional[datetime] = None
    exit_time: Optional[datetime] = None
    status: str = "open"
    reason: Optional[str] = None


@dataclass
class PositionRecord:
    trade_id: str
    symbol: str
    side: str
    entry_price: float
    current_price: float
    quantity: float
    leverage: int = 1
    stop_loss: Optional[float] = None
    take_profits: Optional[str] = None  # JSON string
    status: str = "open"


@dataclass
class StrategyPerfRecord:
    strategy_name: str
    version: int = 1
    win_rate: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    profit_factor: Optional[float] = None
    total_trades: int = 0
    total_pnl: Optional[float] = None
    active: bool = True
    params: Optional[str] = None  # JSON string


class Database:
    """
    PostgreSQL database interface using asyncpg.
    Handles connection pooling, migrations, and all CRUD operations.
    """

    def __init__(self):
        self._pool: Optional[asyncpg.Pool] = None
        self._dsn = os.getenv(
            "DATABASE_URL",
            "postgresql://quantex:secret@localhost:5432/quantex",
        )

    async def connect(self):
        """Initialize connection pool."""
        if self._pool is None:
            self._pool = await asyncpg.create_pool(
                dsn=self._dsn,
                min_size=2,
                max_size=10,
                command_timeout=30,
            )
            await self._run_migrations()
        return self

    async def disconnect(self):
        """Close connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None

    async def _run_migrations(self):
        """Run database migrations if needed."""
        async with self._pool.acquire() as conn:
            # Create tables if they don't exist
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    symbol          VARCHAR(20) NOT NULL,
                    side            VARCHAR(4) NOT NULL,
                    entry_price     DECIMAL(20, 8),
                    exit_price      DECIMAL(20, 8),
                    quantity        DECIMAL(20, 8),
                    leverage        INTEGER DEFAULT 1,
                    pnl             DECIMAL(20, 8),
                    fees            DECIMAL(20, 8) DEFAULT 0,
                    agent_id        VARCHAR(50),
                    strategy_name   VARCHAR(100),
                    confidence_score DECIMAL(5, 4),
                    entry_time      TIMESTAMPTZ,
                    exit_time       TIMESTAMPTZ,
                    status          VARCHAR(20) DEFAULT 'open',
                    reason          TEXT,
                    created_at      TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS positions (
                    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    trade_id        UUID REFERENCES trades(id) ON DELETE CASCADE,
                    symbol          VARCHAR(20) NOT NULL,
                    side            VARCHAR(4) NOT NULL,
                    entry_price     DECIMAL(20, 8),
                    current_price   DECIMAL(20, 8),
                    quantity        DECIMAL(20, 8),
                    leverage        INTEGER DEFAULT 1,
                    unrealized_pnl  DECIMAL(20, 8),
                    realized_pnl    DECIMAL(20, 8) DEFAULT 0,
                    stop_loss       DECIMAL(20, 8),
                    take_profits    JSONB,
                    status          VARCHAR(20) DEFAULT 'open',
                    opened_at       TIMESTAMPTZ DEFAULT NOW(),
                    closed_at       TIMESTAMPTZ
                )
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS strategy_performance (
                    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
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
                )
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS daily_pnl (
                    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    date            DATE NOT NULL UNIQUE,
                    total_pnl       DECIMAL(20, 8),
                    total_trades    INTEGER DEFAULT 0,
                    win_count       INTEGER DEFAULT 0,
                    loss_count      INTEGER DEFAULT 0,
                    fees_total      DECIMAL(20, 8) DEFAULT 0,
                    created_at      TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            # Agent Memory table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS agent_memory (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    agent_id VARCHAR(50) NOT NULL,
                    memory_type VARCHAR(30) NOT NULL CHECK (memory_type IN ('trade', 'pattern', 'regime', 'signal')),
                    content TEXT,
                    importance DECIMAL(5, 4) DEFAULT 0.5,
                    metadata JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    expires_at TIMESTAMPTZ
                )
            """)

            # Market Regimes table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS market_regimes (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    regime_type VARCHAR(30) NOT NULL CHECK (regime_type IN ('bull', 'bear', 'ranging', 'volatile', 'weak_trend')),
                    symbol VARCHAR(20),
                    start_time TIMESTAMPTZ,
                    end_time TIMESTAMPTZ,
                    confidence DECIMAL(5, 4),
                    indicators JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)

            # Create indexes
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_created ON trades(created_at DESC)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_daily_pnl_date ON daily_pnl(date DESC)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_agent_memory ON agent_memory(agent_id, memory_type)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_regimes_time ON market_regimes(start_time DESC)")


    # ── Trades ───────────────────────────────────────

    async def save_trade(self, trade: TradeRecord) -> str:
        """Save a trade record and return its ID."""
        async with self._pool.acquire() as conn:
            trade_id = str(uuid.uuid4())
            await conn.execute("""
                INSERT INTO trades (id, symbol, side, entry_price, exit_price, quantity,
                    leverage, pnl, fees, agent_id, strategy_name, confidence_score,
                    entry_time, exit_time, status, reason)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
            """,
                trade_id, trade.symbol, trade.side, trade.entry_price, trade.exit_price,
                trade.quantity, trade.leverage, trade.pnl, trade.fees, trade.agent_id,
                trade.strategy_name, trade.confidence_score,
                trade.entry_time or datetime.utcnow(), trade.exit_time,
                trade.status, trade.reason,
            )
            return trade_id

    async def close_trade(self, trade_id: str, exit_price: float, pnl: float, fees: float = 0.0):
        """Close a trade with exit price and PnL."""
        async with self._pool.acquire() as conn:
            await conn.execute("""
                UPDATE trades SET exit_price=$1, pnl=$2, fees=$3, status='closed',
                    exit_time=NOW()
                WHERE id=$4
            """, exit_price, pnl, fees, trade_id)

    async def get_recent_trades(self, limit: int = 50) -> list[dict]:
        """Get recent trades."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT * FROM trades ORDER BY created_at DESC LIMIT $1
            """, limit)
            return [dict(r) for r in rows]

    async def get_trade_stats(self, days: int = 30) -> dict:
        """Get trade statistics for the given period."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT 
                    COUNT(*) as total_trades,
                    COUNT(*) FILTER (WHERE pnl > 0) as winning_trades,
                    COUNT(*) FILTER (WHERE pnl <= 0) as losing_trades,
                    COALESCE(SUM(pnl), 0) as total_pnl,
                    COALESCE(SUM(fees), 0) as total_fees,
                    COALESCE(AVG(pnl) FILTER (WHERE pnl > 0), 0) as avg_win,
                    COALESCE(AVG(pnl) FILTER (WHERE pnl <= 0), 0) as avg_loss
                FROM trades 
                WHERE created_at >= NOW() - INTERVAL '1 day' * $1
                    AND status = 'closed'
            """, days)
            row = rows[0]
            total = row["total_trades"]
            wins = row["winning_trades"]
            return {
                "total_trades": total,
                "winning_trades": wins,
                "losing_trades": row["losing_trades"],
                "win_rate": round(wins / total, 4) if total > 0 else 0.0,
                "total_pnl": round(float(row["total_pnl"]), 4),
                "total_fees": round(float(row["total_fees"]), 4),
                "avg_win": round(float(row["avg_win"]), 4),
                "avg_loss": round(float(row["avg_loss"]), 4),
                "period_days": days,
            }

    # ── Daily PnL ────────────────────────────────────

    async def update_daily_pnl(self, pnl: float, fees: float = 0.0, won: bool = False):
        """Update daily PnL aggregation (upsert)."""
        today = date.today()
        async with self._pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO daily_pnl (date, total_pnl, total_trades, win_count, loss_count, fees_total)
                VALUES ($1, $2, 1, $3, $4, $5)
                ON CONFLICT (date) DO UPDATE SET
                    total_pnl = daily_pnl.total_pnl + $2,
                    total_trades = daily_pnl.total_trades + 1,
                    win_count = daily_pnl.win_count + $3,
                    loss_count = daily_pnl.loss_count + $4,
                    fees_total = daily_pnl.fees_total + $5
            """, today, pnl, 1 if won else 0, 0 if won else 1, fees)

    async def get_daily_pnl(self, days: int = 30) -> list[dict]:
        """Get daily PnL history."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT * FROM daily_pnl 
                WHERE date >= CURRENT_DATE - ($1 || ' days')::INTERVAL
                ORDER BY date DESC
            """, str(days))
            return [dict(r) for r in rows]

    # ── Agent Memory ─────────────────────────────────

    async def save_agent_memory(self, agent_id: str, memory_type: str,
                                 content: str, importance: float = 0.5,
                                 metadata: Optional[dict] = None) -> str:
        """Store agent memory."""
        async with self._pool.acquire() as conn:
            mem_id = str(uuid.uuid4())
            await conn.execute("""
                INSERT INTO agent_memory (id, agent_id, memory_type, content, importance, metadata)
                VALUES ($1, $2, $3, $4, $5, $6::jsonb)
            """, mem_id, agent_id, memory_type, content, importance,
                json.dumps(metadata) if metadata else None)
            return mem_id

    async def get_agent_memories(self, agent_id: str,
                                  memory_type: Optional[str] = None,
                                  limit: int = 50) -> list[dict]:
        """Retrieve agent memories."""
        async with self._pool.acquire() as conn:
            if memory_type:
                rows = await conn.fetch(
                    """SELECT * FROM agent_memory
                       WHERE agent_id=$1 AND memory_type=$2
                       ORDER BY created_at DESC LIMIT $3""",
                    agent_id, memory_type, limit)
            else:
                rows = await conn.fetch(
                    """SELECT * FROM agent_memory
                       WHERE agent_id=$1
                       ORDER BY created_at DESC LIMIT $2""",
                    agent_id, limit)
            return [dict(r) for r in rows]

    # ── Market Regimes ───────────────────────────────

    async def save_regime(self, regime_type: str, symbol: str,
                           confidence: float, indicators: Optional[dict] = None) -> str:
        """Record a market regime transition."""
        async with self._pool.acquire() as conn:
            regime_id = str(uuid.uuid4())
            # Close previous regime
            await conn.execute("""
                UPDATE market_regimes SET end_time=NOW()
                WHERE symbol=$1 AND end_time IS NULL
            """, symbol)
            # Insert new regime
            await conn.execute("""
                INSERT INTO market_regimes (id, regime_type, symbol, start_time, confidence, indicators)
                VALUES ($1, $2, $3, NOW(), $4, $5::jsonb)
            """, regime_id, regime_type, symbol, confidence,
                json.dumps(indicators) if indicators else None)
            return regime_id

    async def get_current_regime(self, symbol: str = "BTCUSDT") -> Optional[dict]:
        """Get current active regime for a symbol."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """SELECT * FROM market_regimes
                   WHERE symbol=$1 AND end_time IS NULL
                   ORDER BY start_time DESC LIMIT 1""",
                symbol)
            return dict(row) if row else None

    # ── Positions ────────────────────────────────────

    async def save_position(self, pos: PositionRecord) -> str:
        """Save a position record."""
        async with self._pool.acquire() as conn:
            pos_id = str(uuid.uuid4())
            await conn.execute("""
                INSERT INTO positions (id, trade_id, symbol, side, entry_price,
                    current_price, quantity, leverage, stop_loss, take_profits, status)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10::jsonb, $11)
            """,
                pos_id, pos.trade_id, pos.symbol, pos.side, pos.entry_price,
                pos.current_price, pos.quantity, pos.leverage, pos.stop_loss,
                pos.take_profits, pos.status,
            )
            return pos_id

    # ── Strategy Performance ─────────────────────────

    async def update_strategy_perf(self, perf: StrategyPerfRecord):
        """Update strategy performance metrics (upsert on strategy_name)."""
        async with self._pool.acquire() as conn:
            # Check if record exists
            existing = await conn.fetchrow(
                "SELECT id FROM strategy_performance WHERE strategy_name=$1 AND version=$2",
                perf.strategy_name, perf.version,
            )
            if existing:
                await conn.execute("""
                    UPDATE strategy_performance SET
                        win_rate=$1, sharpe_ratio=$2, max_drawdown=$3,
                        profit_factor=$4, total_trades=$5, total_pnl=$6,
                        active=$7, params=$8::jsonb, updated_at=NOW()
                    WHERE strategy_name=$9 AND version=$10
                """,
                    perf.win_rate, perf.sharpe_ratio, perf.max_drawdown,
                    perf.profit_factor, perf.total_trades, perf.total_pnl,
                    perf.active, perf.params, perf.strategy_name, perf.version,
                )
            else:
                await conn.execute("""
                    INSERT INTO strategy_performance (strategy_name, version, win_rate,
                        sharpe_ratio, max_drawdown, profit_factor, total_trades,
                        total_pnl, active, params)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10::jsonb)
                """,
                    perf.strategy_name, perf.version, perf.win_rate,
                    perf.sharpe_ratio, perf.max_drawdown, perf.profit_factor,
                    perf.total_trades, perf.total_pnl, perf.active, perf.params,
                )

    async def get_active_strategies(self) -> list[dict]:
        """Get all active strategy performances."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM strategy_performance WHERE active = true ORDER BY updated_at DESC"
            )
            return [dict(r) for r in rows]

    # ── Health ───────────────────────────────────────

    async def health(self) -> dict:
        """Check database connectivity."""
        try:
            async with self._pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
                return {"status": "ok", "connected": True}
        except Exception as e:
            return {"status": "error", "connected": False, "error": str(e)}
