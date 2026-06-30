"""
QUANTEX RAG Agent — Retrieval Augmented Generation for trade pattern memory.

Queries Qdrant vector store for historically similar trade setups,
provides context to the strategy agent for informed decisions.

Architecture:
  Current Market State → Embedding → Qdrant Search → Context Assembly → LLM

Usage:
    agent = RAGAgent(qdrant_url="http://localhost:6333")
    context = await agent.retrieve(
        symbol="BTCUSDT", regime="trending", rsi=65.0, ...
    )
    # context = {"documents": [...], "pattern_stats": {...}, "query_time_ms": 42}
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx
import numpy as np

logger = logging.getLogger("quantex.rag_agent")

DEFAULT_QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
DEFAULT_NIM_URL = os.getenv("NVIDIA_NIM_URL", "https://integrate.api.nvidia.com/v1")
DEFAULT_NIM_KEY = os.getenv("NVIDIA_API_KEY", "")
EMBEDDING_MODEL = "nvidia/nv-embedqa-e5-v5"
EMBEDDING_DIM = 1536


@dataclass
class RAGContext:
    """Structured output from RAG retrieval."""
    documents: list[dict] = field(default_factory=list)
    similar_trades: list[dict] = field(default_factory=list)
    pattern_stats: dict[str, Any] = field(default_factory=dict)
    query_time_ms: float = 0.0
    source: str = "qdrant"  # "qdrant", "local", "fallback"
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "documents": self.documents,
            "similar_trades": self.similar_trades,
            "pattern_stats": self.pattern_stats,
            "query_time_ms": self.query_time_ms,
            "source": self.source,
            "error": self.error,
        }


class RAGAgent:
    """
    RAG Agent for trade pattern retrieval.

    Uses Qdrant for vector storage and NIM for embeddings.
    Falls back to in-memory cosine similarity if Qdrant is unavailable.
    """

    def __init__(
        self,
        qdrant_url: str = DEFAULT_QDRANT_URL,
        nim_url: str = DEFAULT_NIM_URL,
        nim_key: str = DEFAULT_NIM_KEY,
        collection: str = "trade_patterns",
        top_k: int = 5,
    ):
        self.qdrant_url = qdrant_url.rstrip("/")
        self.nim_url = nim_url.rstrip("/")
        self.nim_key = nim_key
        self.collection = collection
        self.top_k = top_k
        self._connected = False
        self._local_store: list[dict] = []  # In-memory fallback

    async def retrieve(
        self,
        symbol: str = "BTCUSDT",
        regime: str = "unknown",
        rsi: float = 50.0,
        macd: float = 0.0,
        volume_ratio: float = 1.0,
        price: float = 0.0,
        vlm_trend: str = "",
        extra_context: str = "",
        top_k: int | None = None,
    ) -> RAGContext:
        """
        Retrieve similar trade patterns from vector memory.

        Args:
            symbol: Trading pair
            regime: Market regime
            rsi: Current RSI
            macd: Current MACD
            volume_ratio: Volume ratio vs SMA
            price: Current price
            vlm_trend: VLM analysis trend output
            extra_context: Additional text context
            top_k: Override default top_k

        Returns:
            RAGContext with documents, stats, and timing
        """
        start = time.perf_counter()
        k = top_k or self.top_k

        # Build query text
        query_text = self._build_query_text(
            symbol, regime, rsi, macd, volume_ratio, price, vlm_trend, extra_context,
        )

        # Generate embedding
        try:
            embedding = await self._embed(query_text)
        except Exception as e:
            logger.warning(f"Embedding failed ({e}), using hash-based fallback")
            embedding = self._hash_embedding(query_text)

        # Search Qdrant
        try:
            results = await self._search_qdrant(embedding, k)
            source = "qdrant"
        except Exception as e:
            logger.warning(f"Qdrant search failed ({e}), using local fallback")
            results = self._search_local(embedding, k)
            source = "local"

        # Compute pattern statistics
        pattern_stats = self._compute_stats(results)

        query_time_ms = (time.perf_counter() - start) * 1000

        return RAGContext(
            documents=[r.get("payload", {}) for r in results],
            similar_trades=[
                {**r.get("payload", {}), "similarity": r.get("score", 0)}
                for r in results
            ],
            pattern_stats=pattern_stats,
            query_time_ms=round(query_time_ms, 2),
            source=source,
        )

    async def store_trade(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        regime: str,
        rsi: float,
        pnl: float = 0.0,
        outcome: str = "pending",
        lesson: str = "",
    ) -> bool:
        """
        Store a trade pattern for future retrieval.

        Returns True if stored successfully.
        """
        text = (
            f"Trade: {symbol} {direction} @ {entry_price:.2f} "
            f"Regime={regime} RSI={rsi:.1f} "
            f"PnL={pnl:.4f} Outcome={outcome} Lesson={lesson}"
        )

        try:
            embedding = await self._embed(text)
        except Exception:
            embedding = self._hash_embedding(text)

        point = {
            "id": str(uuid.uuid4()),
            "vector": embedding,
            "payload": {
                "symbol": symbol,
                "direction": direction,
                "entry_price": entry_price,
                "regime": regime,
                "rsi": rsi,
                "pnl": pnl,
                "outcome": outcome,
                "lesson": lesson,
                "text": text,
                "timestamp": time.time(),
            },
        }

        # Try Qdrant upsert
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.put(
                    f"{self.qdrant_url}/collections/{self.collection}/points",
                    json={"points": [point]},
                )
                if resp.status_code in (200, 201):
                    return True
        except Exception:
            pass

        # Fallback: store locally
        self._local_store.append(point)
        if len(self._local_store) > 1000:
            self._local_store = self._local_store[-500:]
        return True

    def _build_query_text(
        self, symbol, regime, rsi, macd, volume_ratio, price, vlm_trend, extra,
    ) -> str:
        parts = [
            f"Symbol: {symbol}",
            f"Regime: {regime}",
            f"RSI: {rsi:.1f}",
            f"MACD: {macd:.4f}",
            f"Volume ratio: {volume_ratio:.2f}",
            f"Price: {price:.2f}",
        ]
        if vlm_trend:
            parts.append(f"VLM trend: {vlm_trend}")
        if extra:
            parts.append(extra)
        return " | ".join(parts)

    async def _embed(self, text: str) -> list[float]:
        """Generate embedding via NVIDIA NIM."""
        if not self.nim_key:
            return self._hash_embedding(text)

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{self.nim_url}/embeddings",
                headers={
                    "Authorization": f"Bearer {self.nim_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": EMBEDDING_MODEL,
                    "input": text,
                    "input_type": "query",
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return data["data"][0]["embedding"]

    def _hash_embedding(self, text: str) -> list[float]:
        """Deterministic hash-based embedding (no API needed)."""
        rng = np.random.RandomState(abs(hash(text)) % (2**31))
        emb = rng.normal(0, 0.1, EMBEDDING_DIM).tolist()
        mag = np.linalg.norm(emb)
        return [v / mag for v in emb] if mag > 0 else emb

    async def _search_qdrant(self, query_vector: list[float], k: int) -> list[dict]:
        """Search Qdrant collection."""
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(
                f"{self.qdrant_url}/collections/{self.collection}/points/search",
                json={
                    "vector": query_vector,
                    "limit": k,
                    "with_payload": True,
                    "score_threshold": 0.3,
                },
            )
            resp.raise_for_status()
            return resp.json().get("result", [])

    def _search_local(self, query_vector: list[float], k: int) -> list[dict]:
        """Fallback: cosine similarity on local store."""
        if not self._local_store:
            return []

        q = np.array(query_vector)
        scored = []
        for pt in self._local_store:
            v = np.array(pt.get("vector", [0.0] * EMBEDDING_DIM))
            sim = float(np.dot(q, v) / (np.linalg.norm(q) * np.linalg.norm(v) + 1e-10))
            scored.append({"score": sim, "payload": pt.get("payload", {})})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:k]

    def _compute_stats(self, results: list[dict]) -> dict:
        """Compute aggregate statistics from retrieved patterns."""
        if not results:
            return {"win_rate": 0.5, "avg_pnl": 0.0, "sample_size": 0}

        pnls = [r.get("payload", {}).get("pnl", 0) for r in results]
        wins = [p for p in pnls if p > 0]

        return {
            "win_rate": len(wins) / len(pnls) if pnls else 0.5,
            "avg_pnl": float(np.mean(pnls)) if pnls else 0.0,
            "max_pnl": max(pnls) if pnls else 0.0,
            "min_pnl": min(pnls) if pnls else 0.0,
            "sample_size": len(results),
            "confidence_adjustment": (len(wins) / len(pnls) - 0.5) * 0.2 if pnls else 0.0,
        }

    async def health_check(self) -> bool:
        """Check Qdrant availability."""
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(f"{self.qdrant_url}/collections")
                return resp.status_code == 200
        except Exception:
            return False
