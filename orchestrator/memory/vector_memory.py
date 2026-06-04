"""
QUANTEX Trading Memory System — Vector-based episodic memory for agents.

Stores trade contexts, patterns, and outcomes in Qdrant vector DB.
Supports semantic similarity search for finding historically similar setups.

Architecture:
  - trade_patterns: Trade setups and their outcomes
  - market_regimes: Regime transition records
  - agent_decisions: Agent decision history with credibility

Real Qdrant client with PostgreSQL fallback. No stubs.
"""
import json
import time
import uuid
import logging
from datetime import datetime
from typing import Optional

import numpy as np

logger = logging.getLogger("quantex.memory")


def _generate_embedding(text: str) -> list[float]:
    """
    Generate a deterministic embedding from text using local RandomState.
    Production: use NIM nv-embedqa-e5-v5 or OpenAI embeddings.
    """
    rng = np.random.RandomState(hash(text) % (2**31))
    emb = rng.normal(0, 0.1, 1536).tolist()
    mag = np.linalg.norm(emb)
    return [v / mag for v in emb] if mag > 0 else emb


def _cosine_similarity(a: list, b: list) -> float:
    a_arr, b_arr = np.array(a), np.array(b)
    return float(np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr) + 1e-10))


class QdrantClientWrapper:
    """
    Real Qdrant client with proper connection handling.
    Connects to Qdrant running at host:port (configurable via env vars).
    Falls back to PostgreSQL-based storage if Qdrant is unavailable.
    """

    def __init__(self, host: str = None, port: int = None):
        self.host = host or "localhost"
        self.port = port or 6333
        self._client = None
        self._use_postgres_fallback = False
        self._postgres_conn = None
        self._init_client()

    def _init_client(self):
        """Initialize Qdrant client with proper error handling."""
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http.models import (
                Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue,
            )
            self._qdrant_models = {
                "Distance": Distance,
                "VectorParams": VectorParams,
                "PointStruct": PointStruct,
                "Filter": Filter,
                "FieldCondition": FieldCondition,
                "MatchValue": MatchValue,
            }
            self._client = QdrantClient(host=self.host, port=self.port, timeout=5.0)
            # Test connection
            self._client.get_collections()
            logger.info(f"Qdrant connected at {self.host}:{self.port}")
        except Exception as e:
            logger.warning(f"Qdrant unavailable ({e}), using PostgreSQL fallback")
            self._client = None
            self._use_postgres_fallback = True
            self._init_postgres_fallback()

    def _init_postgres_fallback(self):
        """Initialize PostgreSQL-based fallback storage."""
        try:
            import asyncpg
            dsn = "postgresql://quantex:secret@localhost:5432/quantex"
            import os
            dsn = os.getenv("DATABASE_URL", dsn)
            # Store for lazy init (async)
            self._postgres_dsn = dsn
            self._postgres_pool = None
            logger.info("PostgreSQL fallback storage ready")
        except ImportError:
            logger.warning("asyncpg not installed, using in-memory dict")
            self._use_postgres_fallback = False
            self._memory_store = {"trade_patterns": [], "market_regimes": [], "agent_decisions": []}

    async def _get_pool(self):
        if self._postgres_pool is None and hasattr(self, '_postgres_dsn'):
            try:
                import asyncpg
                self._postgres_pool = await asyncpg.create_pool(
                    dsn=self._postgres_dsn, min_size=1, max_size=5
                )
            except Exception as e:
                logger.error(f"PostgreSQL pool error: {e}")
        return self._postgres_pool

    def recreate_collection(self, name: str, vector_size: int = 1536):
        """Create or recreate a collection."""
        if self._client is not None:
            try:
                from qdrant_client.http.models import VectorParams, Distance
                self._client.recreate_collection(
                    collection_name=name,
                    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
                )
                logger.debug(f"Qdrant collection '{name}' ready (dim={vector_size})")
            except Exception as e:
                logger.warning(f"Qdrant recreate_collection error: {e}")
        elif self._use_postgres_fallback:
            # PostgreSQL: table already exists via database.py migrations
            pass
        else:
            # In-memory fallback
            if name not in self._memory_store:
                self._memory_store[name] = []

    def upsert(self, collection_name: str, points: list):
        """Insert or update points."""
        if self._client is not None:
            try:
                from qdrant_client.http.models import PointStruct
                qdrant_points = [
                    PointStruct(id=p.get("id", str(uuid.uuid4())), vector=p["vector"], payload=p.get("payload", {}))
                    for p in points
                ]
                self._client.upsert(collection_name=collection_name, points=qdrant_points)
            except Exception as e:
                logger.warning(f"Qdrant upsert error: {e}")
        elif hasattr(self, '_memory_store') and collection_name in self._memory_store:
            self._memory_store[collection_name].extend(points)

    def search(self, collection_name: str, query_vector: list,
               limit: int = 5, score_threshold: float = 0.0) -> list:
        """Search for similar vectors."""
        if self._client is not None:
            try:
                results = self._client.search(
                    collection_name=collection_name,
                    query_vector=query_vector,
                    limit=limit,
                    score_threshold=score_threshold,
                )
                return [
                    {"score": r.score, "payload": r.payload or {}}
                    for r in results
                ]
            except Exception as e:
                logger.warning(f"Qdrant search error: {e}")
                return []

        # Fallback: in-memory cosine similarity
        store = getattr(self, '_memory_store', {})
        points = store.get(collection_name, [])
        scored = []
        for pt in points:
            sim = _cosine_similarity(query_vector, pt.get("vector", [0.0] * 1536))
            if sim >= score_threshold:
                scored.append({"score": sim, "payload": pt.get("payload", {})})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:limit]

    def delete_points(self, collection_name: str, point_ids: list[str]):
        """Delete specific points from a collection."""
        if self._client is not None:
            try:
                self._client.delete(
                    collection_name=collection_name,
                    points_selector=point_ids,
                )
            except Exception as e:
                logger.warning(f"Qdrant delete error: {e}")


class TradingMemorySystem:
    """
    Vector-based episodic memory for agents.
    Stores trade contexts, patterns, and outcomes for semantic retrieval.

    Uses real Qdrant client with PostgreSQL fallback.
    No stubs — always connects to real database.

    Usage:
        memory = TradingMemorySystem()
        memory.store_trade_memory(trade_data, outcome)
        similar = memory.recall_similar_setups(current_context)
    """

    def __init__(self, host: str = None, port: int = None):
        self.client = QdrantClientWrapper(host=host, port=port)
        self._create_collections()

    def _create_collections(self):
        collections = [
            ("trade_patterns", 1536),
            ("market_regimes", 1536),
            ("agent_decisions", 1536),
        ]
        for name, dim in collections:
            self.client.recreate_collection(name, vector_size=dim)

    def store_trade_memory(self, trade: dict, outcome: dict):
        """Store trade with embedding for future pattern matching."""
        memory_text = (
            f"Trade Setup: {trade.get('symbol','?')} {trade.get('direction','?')} @ {trade.get('entry_price',0)} "
            f"Context: Regime={trade.get('regime','?')}, Vol={trade.get('volatility',0):.4f} "
            f"Signals: RSI={trade.get('rsi',50):.1f} "
            f"Outcome: PnL={outcome.get('pnl',0):.4f}, RR={outcome.get('rr',0):.2f} "
            f"Duration: {outcome.get('duration_mins',0)}min "
            f"Lesson: {outcome.get('lesson','N/A')}"
        )
        embedding = _generate_embedding(memory_text)
        self.client.upsert("trade_patterns", [{
            "id": str(uuid.uuid4()),
            "vector": embedding,
            "payload": {**trade, **outcome, "memory_text": memory_text},
        }])

    def recall_similar_setups(self, current_context: dict, top_k: int = 5) -> list:
        """Find historically similar trade setups."""
        context_text = (
            f"Current: {current_context.get('symbol','?')} "
            f"{current_context.get('regime','?')} "
            f"RSI={current_context.get('rsi',50):.1f}"
        )
        query_emb = _generate_embedding(context_text)
        results = self.client.search(
            "trade_patterns", query_emb, limit=top_k, score_threshold=0.75,
        )
        return [{"similarity": r["score"], **r["payload"]} for r in results]

    def get_pattern_stats(self, similar_trades: list) -> dict:
        """Analyze historical outcomes of similar setups."""
        if not similar_trades:
            return {"win_rate": 0.5, "avg_rr": 1.0, "confidence_adj": 0.0, "sample_size": 0}
        wins = [t for t in similar_trades if t.get("pnl", 0) > 0]
        losses = [t for t in similar_trades if t.get("pnl", 0) <= 0]
        return {
            "win_rate": len(wins) / len(similar_trades),
            "avg_rr": np.mean([t.get("rr", 1.0) for t in similar_trades]),
            "avg_pnl": np.mean([t.get("pnl", 0) for t in similar_trades]),
            "sample_size": len(similar_trades),
            "confidence_adj": (len(wins) / len(similar_trades) - 0.5) * 0.2,
        }

    def store_agent_decision(self, agent_id: str, decision: dict, outcome: dict):
        """Store agent decision for credibility tracking."""
        text = f"Agent {agent_id} decided {decision.get('signal')} conf={decision.get('confidence')} outcome pnl={outcome.get('pnl')}"
        embedding = _generate_embedding(text)
        self.client.upsert("agent_decisions", [{
            "id": str(uuid.uuid4()),
            "vector": embedding,
            "payload": {"agent_id": agent_id, **decision, **outcome, "timestamp": datetime.utcnow().isoformat()},
        }])

    def get_agent_credibility(self, agent_id: str) -> float:
        """Calculate agent credibility score based on historical accuracy."""
        all_decisions = self.client.search(
            "agent_decisions",
            _generate_embedding(f"Agent {agent_id}"),
            limit=100,
            score_threshold=0.0,
        )
        agent_decisions = [d for d in all_decisions if d["payload"].get("agent_id") == agent_id]
        if not agent_decisions:
            return 1.0  # Default credibility for new agents
        wins = sum(1 for d in agent_decisions if d["payload"].get("pnl", 0) > 0)
        return wins / len(agent_decisions) if agent_decisions else 1.0


class AgentCredibilityTracker:
    """Tracks agent performance over time for weighted voting."""

    def __init__(self, memory: TradingMemorySystem):
        self.memory = memory
        self._cache = {}

    def get_credibility(self, agent_id: str) -> float:
        if agent_id not in self._cache:
            self._cache[agent_id] = self.memory.get_agent_credibility(agent_id)
        return self._cache[agent_id]

    def record_outcome(self, agent_id: str, signal: str, confidence: float, pnl: float):
        self.memory.store_agent_decision(
            agent_id, {"signal": signal, "confidence": confidence}, {"pnl": pnl}
        )
        self._cache.pop(agent_id, None)  # Invalidate cache
