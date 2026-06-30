"""RL Memory System — Redis (short-term) + Postgres (long-term) + Qdrant (vector)."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

# Optional imports - handle gracefully if not installed
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False
    psycopg2 = None

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qdrant_models
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    QdrantClient = None
    qdrant_models = None

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    SentenceTransformer = None


# ── Data Classes ──────────────────────────────────────────────


@dataclass
class TradeOutcome:
    """Complete trade result for memory storage."""

    trade_id: str
    timestamp: datetime
    symbol: str
    action: str  # BUY/SELL/HOLD
    entry_price: float
    exit_price: float | None
    quantity: float
    pnl_pct: float
    pnl_abs: float
    confidence: float
    regime: str | None
    indicators: dict[str, float]
    agent_reasoning: list[str]
    risk_decision: dict[str, Any]
    execution_latency_ms: float
    status: str  # FILLED/REJECTED/ERROR
    max_drawdown_pct: float = 0.0  # Max adverse excursion during trade

    def to_dict(self) -> dict:
        return {
            "trade_id": self.trade_id,
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "action": self.action,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "quantity": self.quantity,
            "pnl_pct": self.pnl_pct,
            "pnl_abs": self.pnl_abs,
            "confidence": self.confidence,
            "regime": self.regime,
            "indicators": self.indicators,
            "agent_reasoning": self.agent_reasoning,
            "risk_decision": self.risk_decision,
            "execution_latency_ms": self.execution_latency_ms,
            "status": self.status,
        }


@dataclass
class MarketState:
    """Market state snapshot for embedding."""

    timestamp: datetime
    symbol: str
    price: float
    regime: str
    indicators: dict[str, float]
    recent_pnl: float
    open_positions: int
    confidence: float

    def to_text(self) -> str:
        """Convert to text for embedding."""
        ind_str = ", ".join(f"{k}:{v:.4f}" for k, v in self.indicators.items())
        return (
            f"Symbol: {self.symbol}, Price: {self.price:.2f}, Regime: {self.regime}, "
            f"Indicators: [{ind_str}], Recent PnL: {self.recent_pnl:.4f}, "
            f"Open Positions: {self.open_positions}, Confidence: {self.confidence:.2f}"
        )

    def to_vector(self, embedder: SentenceTransformer) -> np.ndarray:
        """Generate embedding vector."""
        return embedder.encode(self.to_text())


# ── Memory Store Classes ──────────────────────────────────────


class RedisStore:
    """Short-term memory: recent trades, current session state."""

    def __init__(self, url: str | None = None):
        if not REDIS_AVAILABLE:
            raise RuntimeError("redis package not installed: pip install redis")
        self.url = url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.client = redis.from_url(self.url, decode_responses=True)
        # Don't test connection on init - let it fail lazily on first use
        self._connected = False

    def _ensure_connection(self):
        """Test connection lazily."""
        if not self._connected:
            try:
                self.client.ping()
                self._connected = True
            except Exception as e:
                raise RuntimeError(f"Redis connection failed: {e}")

    def save_trade(self, trade: TradeOutcome, ttl_seconds: int = 86400) -> None:
        """Save trade to Redis with TTL (default 24h)."""
        self._ensure_connection()
        key = f"trade:{trade.trade_id}"
        self.client.setex(key, ttl_seconds, json.dumps(trade.to_dict()))

    def get_trade(self, trade_id: str) -> TradeOutcome | None:
        self._ensure_connection()
        data = self.client.get(f"trade:{trade_id}")
        if data:
            d = json.loads(data)
            d["timestamp"] = datetime.fromisoformat(d["timestamp"])
            return TradeOutcome(**d)
        return None

    def get_recent_trades(self, limit: int = 100) -> list[TradeOutcome]:
        """Get most recent trades from Redis sorted set."""
        self._ensure_connection()
        # Use sorted set with timestamp as score
        keys = self.client.zrevrange("trades:recent", 0, limit - 1)
        trades = []
        for key in keys:
            data = self.client.get(key)
            if data:
                d = json.loads(data)
                d["timestamp"] = datetime.fromisoformat(d["timestamp"])
                trades.append(TradeOutcome(**d))
        return trades

    def add_to_recent(self, trade: TradeOutcome) -> None:
        """Add trade to recent sorted set."""
        self._ensure_connection()
        self.client.zadd("trades:recent", {f"trade:{trade.trade_id}": trade.timestamp.timestamp()})
        # Keep only last 1000
        self.client.zremrangebyrank("trades:recent", 0, -1001)

    def save_session_state(self, key: str, value: dict, ttl: int = 3600) -> None:
        self._ensure_connection()
        self.client.setex(f"session:{key}", ttl, json.dumps(value))

    def get_session_state(self, key: str) -> dict | None:
        self._ensure_connection()
        data = self.client.get(f"session:{key}")
        return json.loads(data) if data else None


class PostgresStore:
    """Long-term memory: persistent trade history, analytics."""

    def __init__(self, dsn: str | None = None):
        if not POSTGRES_AVAILABLE:
            raise RuntimeError("psycopg2 not installed: pip install psycopg2-binary")
        self.dsn = dsn or os.getenv("POSTGRES_DSN", "postgresql://postgres:***@localhost:5432/trading")
        self._connected = False
        self._init_schema()

    def _get_conn(self):
        return psycopg2.connect(self.dsn, cursor_factory=RealDictCursor)

    def _ensure_connection(self):
        if not self._connected:
            try:
                with self._get_conn() as conn:
                    conn.execute("SELECT 1")
                self._connected = True
            except Exception as e:
                raise RuntimeError(f"Postgres connection failed: {e}")

    def _init_schema(self):
        try:
            with self._get_conn() as conn, conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS trades (
                        trade_id VARCHAR(64) PRIMARY KEY,
                        timestamp TIMESTAMPTZ NOT NULL,
                        symbol VARCHAR(20) NOT NULL,
                        action VARCHAR(10) NOT NULL,
                        entry_price DECIMAL(20,8),
                        exit_price DECIMAL(20,8),
                        quantity DECIMAL(20,8),
                        pnl_pct DECIMAL(10,6),
                        pnl_abs DECIMAL(20,8),
                        confidence DECIMAL(5,4),
                        regime VARCHAR(30),
                        indicators JSONB,
                        agent_reasoning JSONB,
                        risk_decision JSONB,
                        execution_latency_ms DECIMAL(10,2),
                        status VARCHAR(20)
                    );
                    CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp DESC);
                    CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
                    CREATE INDEX IF NOT EXISTS idx_trades_pnl ON trades(pnl_pct);
                """)
                conn.commit()
        except Exception as e:
            # Schema init failed - DB might not be running, skip gracefully
            print(f"⚠️  Postgres schema init skipped: {e}")

    def save_trade(self, trade: TradeOutcome) -> None:
        self._ensure_connection()
        with self._get_conn() as conn, conn.cursor() as cur:
            cur.execute("""
                INSERT INTO trades (trade_id, timestamp, symbol, action, entry_price, exit_price,
                    quantity, pnl_pct, pnl_abs, confidence, regime, indicators, agent_reasoning,
                    risk_decision, execution_latency_ms, status)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (trade_id) DO UPDATE SET
                    exit_price = EXCLUDED.exit_price,
                    pnl_pct = EXCLUDED.pnl_pct,
                    pnl_abs = EXCLUDED.pnl_abs,
                    status = EXCLUDED.status
            """, (
                trade.trade_id, trade.timestamp, trade.symbol, trade.action,
                trade.entry_price, trade.exit_price, trade.quantity,
                trade.pnl_pct, trade.pnl_abs, trade.confidence, trade.regime,
                json.dumps(trade.indicators), json.dumps(trade.agent_reasoning),
                json.dumps(trade.risk_decision), trade.execution_latency_ms, trade.status
            ))
            conn.commit()

    def get_trades(
        self,
        symbol: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int = 1000
    ) -> list[TradeOutcome]:
        self._ensure_connection()
        with self._get_conn() as conn, conn.cursor() as cur:
            query = "SELECT * FROM trades WHERE 1=1"
            params = []
            if symbol:
                query += " AND symbol = %s"
                params.append(symbol)
            if start:
                query += " AND timestamp >= %s"
                params.append(start)
            if end:
                query += " AND timestamp <= %s"
                params.append(end)
            query += " ORDER BY timestamp DESC LIMIT %s"
            params.append(limit)
            cur.execute(query, params)
            rows = cur.fetchall()
            return [TradeOutcome(**{**r, "indicators": r["indicators"], "agent_reasoning": r["agent_reasoning"], "risk_decision": r["risk_decision"]}) for r in rows]

    def get_performance_stats(self, days: int = 30) -> dict:
        self._ensure_connection()
        with self._get_conn() as conn, conn.cursor() as cur:
            cur.execute("""
                SELECT
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN pnl_pct > 0 THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN pnl_pct <= 0 THEN 1 ELSE 0 END) as losses,
                    AVG(pnl_pct) as avg_pnl_pct,
                    STDDEV(pnl_pct) as std_pnl_pct,
                    MIN(pnl_pct) as min_pnl_pct,
                    MAX(pnl_pct) as max_pnl_pct,
                    SUM(pnl_abs) as total_pnl_abs
                FROM trades
                WHERE timestamp >= NOW() - INTERVAL '%s days'
            """, (days,))
            row = cur.fetchone()
            if row and row["total_trades"]:
                win_rate = row["wins"] / row["total_trades"]
                sharpe = (row["avg_pnl_pct"] / row["std_pnl_pct"] * np.sqrt(252)) if row["std_pnl_pct"] else 0
                return {
                    "total_trades": row["total_trades"],
                    "wins": row["wins"],
                    "losses": row["losses"],
                    "win_rate": win_rate,
                    "avg_pnl_pct": float(row["avg_pnl_pct"] or 0),
                    "std_pnl_pct": float(row["std_pnl_pct"] or 0),
                    "sharpe": sharpe,
                    "max_drawdown_pct": float(row["min_pnl_pct"] or 0),
                    "total_pnl_abs": float(row["total_pnl_abs"] or 0),
                }
            return {"total_trades": 0}


class QdrantStore:
    """Vector memory: embeddings for semantic retrieval."""

    def __init__(
        self,
        url: str | None = None,
        api_key: str | None = None,
        collection: str = "trade_memory",
        embedder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        vector_dim: int = 384,
    ):
        if not QDRANT_AVAILABLE:
            raise RuntimeError("qdrant-client not installed: pip install qdrant-client")
        if not EMBEDDINGS_AVAILABLE:
            raise RuntimeError("sentence-transformers not installed: pip install sentence-transformers")

        self.url = url or os.getenv("QDRANT_URL", "http://localhost:6333")
        self.api_key = api_key or os.getenv("QDRANT_API_KEY")
        self.collection = collection
        self.embedder = SentenceTransformer(embedder_name)
        self.vector_dim = vector_dim

        self.client = QdrantClient(url=self.url, api_key=self.api_key)
        self._connected = False
        self._init_collection()

    def _ensure_connection(self):
        if not self._connected:
            try:
                self.client.get_collections()
                self._connected = True
            except Exception as e:
                raise RuntimeError(f"Qdrant connection failed: {e}")

    def _init_collection(self):
        collections = self.client.get_collections().collections
        if not any(c.name == self.collection for c in collections):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=qdrant_models.VectorParams(size=self.vector_dim, distance=qdrant_models.Distance.COSINE),
            )

    def embed_state(self, state: MarketState) -> np.ndarray:
        return self.embedder.encode(state.to_text())

    def embed_text(self, text: str) -> np.ndarray:
        return self.embedder.encode(text)

    def save_trade_vector(self, trade: TradeOutcome, state: MarketState) -> None:
        self._ensure_connection()
        vector = self.embed_state(state)
        payload = trade.to_dict()
        payload["market_state_text"] = state.to_text()
        self.client.upsert(
            collection_name=self.collection,
            points=[
                qdrant_models.PointStruct(
                    id=hash(trade.trade_id) % (2**31),
                    vector=vector.tolist(),
                    payload=payload,
                )
            ],
        )

    def search_similar(
        self,
        query_state: MarketState,
        limit: int = 10,
        score_threshold: float = 0.7,
    ) -> list[dict]:
        self._ensure_connection()
        vector = self.embed_state(query_state)
        results = self.client.search(
            collection_name=self.collection,
            query_vector=vector.tolist(),
            limit=limit,
            score_threshold=score_threshold,
        )
        return [{"score": r.score, "payload": r.payload} for r in results]

    def search_by_text(self, query: str, limit: int = 10, score_threshold: float = 0.7) -> list[dict]:
        self._ensure_connection()
        vector = self.embed_text(query)
        results = self.client.search(
            collection_name=self.collection,
            query_vector=vector.tolist(),
            limit=limit,
            score_threshold=score_threshold,
        )
        return [{"score": r.score, "payload": r.payload} for r in results]


# ── Unified Memory Manager ────────────────────────────────────


class RLMemoryManager:
    """Unified interface for all three memory stores."""

    def __init__(
        self,
        redis_url: str | None = None,
        postgres_dsn: str | None = None,
        qdrant_url: str | None = None,
        qdrant_api_key: str | None = None,
        collection: str = "trade_memory",
    ):
        self.redis = None
        self.postgres = None
        self.qdrant = None
        self._available = {"redis": False, "postgres": False, "qdrant": False}

        # Initialize stores lazily with error handling
        if REDIS_AVAILABLE:
            try:
                self.redis = RedisStore(redis_url)
                self._available["redis"] = True
            except Exception as e:
                print(f"⚠️  Redis init skipped: {e}")

        if POSTGRES_AVAILABLE:
            try:
                self.postgres = PostgresStore(postgres_dsn)
                self._available["postgres"] = True
            except Exception as e:
                print(f"⚠️  Postgres init skipped: {e}")

        if QDRANT_AVAILABLE and EMBEDDINGS_AVAILABLE:
            try:
                self.qdrant = QdrantStore(qdrant_url, qdrant_api_key, collection)
                self._available["qdrant"] = True
            except Exception as e:
                print(f"⚠️  Qdrant init skipped: {e}")

    def is_available(self, store: str) -> bool:
        return self._available.get(store, False)

    def save_trade(self, trade: TradeOutcome, state: MarketState | None = None) -> None:
        """Persist trade to all available stores."""
        if self.redis:
            self.redis.save_trade(trade)
            self.redis.add_to_recent(trade)
        if self.postgres:
            self.postgres.save_trade(trade)
        if self.qdrant and state:
            self.qdrant.save_trade_vector(trade, state)

    def get_recent_trades(self, limit: int = 100) -> list[TradeOutcome]:
        if self.redis:
            return self.redis.get_recent_trades(limit)
        if self.postgres:
            return self.postgres.get_trades(limit=limit)
        return []

    def get_performance_stats(self, days: int = 30) -> dict:
        if self.postgres:
            return self.postgres.get_performance_stats(days)
        return {"total_trades": 0}

    def find_similar_states(self, state: MarketState, limit: int = 10) -> list[dict]:
        if self.qdrant:
            return self.qdrant.search_similar(state, limit)
        return []

    def search_by_query(self, query: str, limit: int = 10) -> list[dict]:
        if self.qdrant:
            return self.qdrant.search_by_text(query, limit)
        return []


# ── Factory ───────────────────────────────────────────────────


def create_memory_manager(config: dict | None = None) -> RLMemoryManager:
    """Create memory manager from config dict."""
    config = config or {}
    return RLMemoryManager(
        redis_url=config.get("redis_url"),
        postgres_dsn=config.get("postgres_dsn"),
        qdrant_url=config.get("qdrant_url"),
        qdrant_api_key=config.get("qdrant_api_key"),
        collection=config.get("collection", "trade_memory"),
    )


# ── Demo / Test ───────────────────────────────────────────────


if __name__ == "__main__":
    import uuid

    print("Testing RL Memory System...")

    # Create sample trade
    trade = TradeOutcome(
        trade_id=str(uuid.uuid4())[:8],
        timestamp=datetime.now(),
        symbol="BTCUSDT",
        action="BUY",
        entry_price=61250.0,
        exit_price=61500.0,
        quantity=0.005,
        pnl_pct=0.0041,
        pnl_abs=1.25,
        confidence=0.85,
        regime="trending_up",
        indicators={"rsi": 62.3, "atr": 580.0, "ema9": 61150.0},
        agent_reasoning=["RSI oversold", "EMA crossover"],
        risk_decision={"allow": True, "max_size": 0.05, "reason": "All checks passed"},
        execution_latency_ms=6.2,
        status="FILLED",
    )

    state = MarketState(
        timestamp=datetime.now(),
        symbol="BTCUSDT",
        price=61250.0,
        regime="trending_up",
        indicators={"rsi": 62.3, "atr": 580.0, "ema9": 61150.0},
        recent_pnl=0.0041,
        open_positions=1,
        confidence=0.85,
    )

    # Test with available stores
    try:
        mgr = create_memory_manager()
        print(f"Available stores: {mgr._available}")

        if mgr.is_available("redis"):
            mgr.save_trade(trade, state)
            recent = mgr.get_recent_trades(5)
            print(f"Redis: saved trade, recent count = {len(recent)}")

        if mgr.is_available("postgres"):
            mgr.save_trade(trade, state)
            stats = mgr.get_performance_stats(30)
            print(f"Postgres: stats = {stats}")

        if mgr.is_available("qdrant"):
            mgr.save_trade(trade, state)
            similar = mgr.find_similar_states(state, limit=3)
            print(f"Qdrant: saved vector, similar = {len(similar)}")

        print("✅ Memory system test passed")
    except Exception as e:
        print(f"⚠️  Test skipped (stores not running): {e}")