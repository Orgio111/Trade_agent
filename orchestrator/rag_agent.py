"""
QUANTEX RAG Agent — Retrieval Augmented Generation for trade pattern memory.

Queries Qdrant vector store for historically similar trade setups,
provides context to the strategy agent for informed decisions.

ALL external APIs are optional — works fully with local-only inference.
Embedding fallback chain: NIM cloud → sentence-transformers (local) → hash-based.
Vector store: Qdrant (optional) → in-memory fallback (always available).

Usage:
    agent = RAGAgent()  # No API keys needed
    context = await agent.retrieve(
        symbol="BTCUSDT", regime="trending", rsi=65.0, ...
    )
"""
from __future__ import annotations

import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger("quantex.rag_agent")

DEFAULT_QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
DEFAULT_NIM_URL = os.getenv("NVIDIA_NIM_URL", "https://integrate.api.nvidia.com/v1")
DEFAULT_NIM_KEY = os.getenv("NVIDIA_API_KEY", "")
EMBEDDING_MODEL = "nvidia/nv-embedqa-e5-v5"
EMBEDDING_MODEL = "nvidia/nv-embedqa-e5-v5"
EMBEDDING_DIM = 1536  # NIM model dimension (cloud only)

# Local embedding model (sentence-transformers) — runs on CPU, ~80MB download
LOCAL_EMBEDDING_MODEL = os.getenv(
    "RAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
LOCAL_EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 outputs 384-dim vectors

# Canonical embedding dimension — all backends must produce vectors of this size.
# When local embedder is available, use its dimension (384). Otherwise fallback to 1536.
DEFAULT_EMBEDDING_DIM = int(os.getenv("RAG_EMBEDDING_DIM", str(LOCAL_EMBEDDING_DIM)))


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


class LocalEmbedder:
    """
    Local sentence-transformers embedding backend.

    Loads a small model (~80MB) on first use and caches it.
    Produces real semantic embeddings that capture meaning,
    unlike the hash-based fallback which produces random vectors.

    Supported models:
      - sentence-transformers/all-MiniLM-L6-v2 (384-dim, 80MB, fast)
      - sentence-transformers/all-mpnet-base-v2 (768-dim, 400MB, better quality)
    """

    def __init__(self, model_name: str = LOCAL_EMBEDDING_MODEL):
        self.model_name = model_name
        self._model = None
        self._dim = LOCAL_EMBEDDING_DIM

    def _load_model(self):
        """Lazy-load sentence-transformers model."""
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading local embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            self._dim = self._model.get_sentence_embedding_dimension()
            logger.info(
                f"Local embedding model loaded: {self.model_name} "
                f"(dim={self._dim})"
            )
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load embedding model: {e}")

    def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        self._load_model()
        embedding = self._model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        self._load_model()
        embeddings = self._model.encode(texts, normalize_embeddings=True)
        return [e.tolist() for e in embeddings]

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def is_available(self) -> bool:
        """Check if sentence-transformers is importable."""
        try:
            import sentence_transformers
            return True
        except ImportError:
            return False


class RAGAgent:
    """
    RAG Agent for trade pattern retrieval.

    Embedding fallback chain:
      1. NIM cloud (optional, best quality, needs API key)
      2. sentence-transformers local (optional, good quality, CPU, ~80MB)
      3. Hash-based (always available, low quality, random vectors)

    Vector store fallback chain:
      1. Qdrant (optional, best performance)
      2. In-memory cosine similarity (always available)
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
        self._embedding_backend = "hash"  # Track which backend is active

        # Local embedding backend (lazy-init)
        self._local_embedder: LocalEmbedder | None = None

        # Lazy-import httpx only when needed
        self._http = None

    def _get_local_embedder(self) -> LocalEmbedder | None:
        """Get or create local embedder (lazy)."""
        if self._local_embedder is None:
            embedder = LocalEmbedder()
            if embedder.is_available:
                self._local_embedder = embedder
                logger.info("Local sentence-transformers embedding available")
            else:
                logger.info(
                    "sentence-transformers not installed — "
                    "install for better embeddings: pip install sentence-transformers"
                )
        return self._local_embedder

    async def _get_http(self):
        """Lazy-import httpx."""
        if self._http is None:
            import httpx
            self._http = httpx.AsyncClient(timeout=10.0)
        return self._http

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

        Works without any API keys — uses local embeddings and local store.
        """
        start = time.perf_counter()
        k = top_k or self.top_k

        # Build query text
        query_text = self._build_query_text(
            symbol, regime, rsi, macd, volume_ratio, price, vlm_trend, extra_context,
        )

        # Generate embedding (NIM → local sentence-transformers → hash)
        embedding = await self._embed(query_text)

        # Search Qdrant → local fallback
        try:
            results = await self._search_qdrant(embedding, k)
            source = "qdrant"
        except Exception as e:
            logger.debug(f"Qdrant search failed ({e}), using local fallback")
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
        """Store a trade pattern for future retrieval."""
        text = (
            f"Trade: {symbol} {direction} @ {entry_price:.2f} "
            f"Regime={regime} RSI={rsi:.1f} "
            f"PnL={pnl:.4f} Outcome={outcome} Lesson={lesson}"
        )

        embedding = await self._embed(text)

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
            http = await self._get_http()
            resp = await http.put(
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

    def _get_canonical_dim(self) -> int:
        """Get the canonical embedding dimension for this session."""
        embedder = self._get_local_embedder()
        if embedder is not None:
            return embedder.dim
        return DEFAULT_EMBEDDING_DIM

    def _normalize_dim(self, vector: list[float], target_dim: int) -> list[float]:
        """Pad or truncate a vector to the target dimension."""
        if len(vector) == target_dim:
            return vector
        if len(vector) > target_dim:
            return vector[:target_dim]
        return vector + [0.0] * (target_dim - len(vector))

    async def _embed(self, text: str) -> list[float]:
        """
        Generate embedding with 3-tier fallback.
        All outputs are normalized to the canonical dimension.

          1. NIM cloud (optional, 1536-dim → truncated to canonical)
          2. sentence-transformers local (optional, 384-dim canonical)
          3. Hash-based (always available, canonical dim)
        """
        target_dim = self._get_canonical_dim()

        # ── Tier 1: NIM cloud ──
        if self.nim_key:
            try:
                http = await self._get_http()
                resp = await http.post(
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
                raw = data["data"][0]["embedding"]
                self._embedding_backend = "nim"
                return self._normalize_dim(raw, target_dim)
            except Exception as e:
                logger.debug(f"NIM embedding failed: {e}")

        # ── Tier 2: Local sentence-transformers ──
        embedder = self._get_local_embedder()
        if embedder is not None:
            try:
                embedding = embedder.embed(text)
                self._embedding_backend = "local_st"
                logger.debug(
                    f"Local embedding: dim={len(embedding)}, "
                    f"backend=sentence-transformers"
                )
                return self._normalize_dim(embedding, target_dim)
            except Exception as e:
                logger.debug(f"Local embedding failed: {e}")

        # ── Tier 3: Hash-based fallback (always available) ──
        self._embedding_backend = "hash"
        return self._hash_embedding(text, target_dim)

    def _hash_embedding(self, text: str, dim: int | None = None) -> list[float]:
        """Deterministic hash-based embedding (no API needed, low quality)."""
        d = dim or self._get_canonical_dim()
        rng = np.random.RandomState(abs(hash(text)) % (2**31))
        emb = rng.normal(0, 0.1, d).tolist()
        mag = np.linalg.norm(emb)
        return [v / mag for v in emb] if mag > 0 else emb

    async def _search_qdrant(self, query_vector: list[float], k: int) -> list[dict]:
        """Search Qdrant collection."""
        http = await self._get_http()
        resp = await http.post(
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

        target_dim = self._get_canonical_dim()
        q = np.array(query_vector)
        scored = []
        for pt in self._local_store:
            v = np.array(pt.get("vector", [0.0] * target_dim))
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
            http = await self._get_http()
            resp = await http.get(f"{self.qdrant_url}/collections")
            return resp.status_code == 200
        except Exception:
            return False

    async def close(self):
        """Close HTTP client."""
        if self._http and not self._http.is_closed:
            await self._http.aclose()
