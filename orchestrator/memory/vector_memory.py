"""
QUANTEX Trading Memory System — Vector-based episodic memory for agents.

Stores trade contexts, patterns, and outcomes in Qdrant vector DB.
Supports semantic similarity search for finding historically similar setups.

Architecture:
  - trade_patterns: Trade setups and their outcomes
  - market_regimes: Regime transition records
  - agent_decisions: Agent decision history with credibility
"""
import json
import time
import uuid
from datetime import datetime
from typing import Optional

import numpy as np


class QdrantClientStub:
    """
    Lightweight in-memory stub for Qdrant vector DB.
    Production: swap with qdrant_client.QdrantClient.
    """

    def __init__(self):
        self.collections = {}

    def recreate_collection(self, name: str, vector_size: int = 1536):
        self.collections[name] = {
            "config": {"vector_size": vector_size},
            "points": [],
        }

    def upsert(self, collection_name: str, points: list):
        col = self.collections.get(collection_name)
        if col:
            for pt in points:
                col["points"].append(pt)

    def search(self, collection_name: str, query_vector: list,
               limit: int = 5, score_threshold: float = 0.0) -> list:
        col = self.collections.get(collection_name)
        if not col:
            return []
        scored = []
        for pt in col["points"]:
            sim = self._cosine_sim(query_vector, pt["vector"])
            if sim >= score_threshold:
                scored.append({"score": sim, "payload": pt.get("payload", {})})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:limit]

    @staticmethod
    def _cosine_sim(a: list, b: list) -> float:
        a, b = np.array(a), np.array(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


class TradingMemorySystem:
    """
    Vector-based episodic memory for agents.
    Stores trade contexts, patterns, and outcomes for semantic retrieval.

    Uses Qdrant-compatible interface (in-memory stub by default,
    swap to real Qdrant in production via qdrant_client).
    """

    def __init__(self, host: str = "localhost", port: int = 6333, use_stub: bool = True):
        self.use_stub = use_stub
        if use_stub:
            self.client = QdrantClientStub()
        else:
            try:
                from qdrant_client import QdrantClient
                self.client = QdrantClient(host=host, port=port)
            except ImportError:
                self.client = QdrantClientStub()
                self.use_stub = True

        self._create_collections()

    def _create_collections(self):
        collections = [
            ("trade_patterns", 1536),
            ("market_regimes", 1536),
            ("agent_decisions", 1536),
        ]
        for name, dim in collections:
            self.client.recreate_collection(name, vector_size=dim)

    def _make_embedding(self, text: str) -> list[float]:
        """Generate a deterministic embedding from text using local RandomState.
        Production: use NIM nv-embedqa-e5-v5 or OpenAI embeddings.
        """
        rng = np.random.RandomState(hash(text) % (2**31))
        emb = rng.normal(0, 0.1, 1536).tolist()
        mag = np.linalg.norm(emb)
        return [v / mag for v in emb] if mag > 0 else emb

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
        embedding = self._make_embedding(memory_text)
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
        query_emb = self._make_embedding(context_text)
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
        embedding = self._make_embedding(text)
        self.client.upsert("agent_decisions", [{
            "id": str(uuid.uuid4()),
            "vector": embedding,
            "payload": {"agent_id": agent_id, **decision, **outcome, "timestamp": datetime.utcnow().isoformat()},
        }])

    def get_agent_credibility(self, agent_id: str) -> float:
        """Calculate agent credibility score based on historical accuracy."""
        all_decisions = self.client.search(
            "agent_decisions",
            self._make_embedding(f"Agent {agent_id}"),
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
