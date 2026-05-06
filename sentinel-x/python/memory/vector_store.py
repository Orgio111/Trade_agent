"""
FAISS-backed Vector Store for Strategy Memory.
Stores reasoning chains from every trade and retrieves the most similar
historical setups for R-Mem (Reasoning-Driven Hierarchical Memory).
"""
from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

EMBED_DIM = 1024  # nvidia/nv-embedqa-e5-v5 output dimension


@dataclass
class StrategyMemory:
    session_id:      str
    symbol:          str
    timestamp:       datetime
    reasoning_chain: str          # full debate + decision JSON
    outcome:         str          # "WIN" | "LOSS" | "BLOCKED"
    pnl_pct:         float
    embedding:       np.ndarray   # shape (1024,)
    metadata:        dict[str, Any] = field(default_factory=dict)


class FAISSStrategyMemory:
    """
    Persistent FAISS index storing trade reasoning chains as embeddings.
    Supports k-NN retrieval to find similar historical setups.
    """

    def __init__(self, index_path: str = "/data/strategy_memory.faiss") -> None:
        self._path  = Path(index_path)
        self._meta_path = self._path.with_suffix(".meta.pkl")
        self._index: Any = None
        self._memories: list[StrategyMemory] = []
        self._load()

    def _load(self) -> None:
        try:
            import faiss  # type: ignore[import]
            if self._path.exists():
                self._index = faiss.read_index(str(self._path))
                if self._meta_path.exists():
                    with open(self._meta_path, "rb") as f:
                        self._memories = pickle.load(f)
                log.info("Loaded FAISS index: %d vectors", self._index.ntotal)
            else:
                self._index = faiss.IndexFlatIP(EMBED_DIM)  # inner-product (cosine after norm)
                log.info("Created new FAISS index (dim=%d)", EMBED_DIM)
        except ImportError:
            log.warning("faiss not installed — using numpy fallback")
            self._index = None

    def add(self, memory: StrategyMemory) -> None:
        self._memories.append(memory)
        vec = memory.embedding.astype(np.float32)
        # L2-normalize for cosine similarity via inner product
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        if self._index is not None:
            self._index.add(vec.reshape(1, -1))
        self._persist()
        log.debug("Memory stored: %s %s %s PnL=%.2f%%",
                  memory.symbol, memory.outcome, memory.session_id[:8], memory.pnl_pct * 100)

    def search(self, query_embedding: np.ndarray, k: int = 5) -> list[StrategyMemory]:
        """Retrieve k most-similar historical setups."""
        if not self._memories:
            return []

        vec = query_embedding.astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        if self._index is not None and self._index.ntotal > 0:
            k = min(k, self._index.ntotal)
            _, indices = self._index.search(vec.reshape(1, -1), k)
            return [self._memories[i] for i in indices[0] if 0 <= i < len(self._memories)]

        # Numpy fallback: brute-force cosine similarity
        embeddings = np.stack([m.embedding for m in self._memories])
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normed = embeddings / (norms + 1e-10)
        scores = normed @ vec
        top_k  = np.argsort(-scores)[:k]
        return [self._memories[i] for i in top_k]

    def get_win_rate(self, symbol: str, lookback: int = 50) -> float:
        relevant = [m for m in self._memories if m.symbol == symbol and m.outcome != "BLOCKED"]
        relevant = relevant[-lookback:]
        if not relevant:
            return 0.55
        return sum(1 for m in relevant if m.outcome == "WIN") / len(relevant)

    def _persist(self) -> None:
        try:
            import faiss
            self._path.parent.mkdir(parents=True, exist_ok=True)
            if self._index is not None:
                faiss.write_index(self._index, str(self._path))
            with open(self._meta_path, "wb") as f:
                pickle.dump(self._memories, f)
        except Exception as exc:
            log.error("Failed to persist FAISS index: %s", exc)

    def __len__(self) -> int:
        return len(self._memories)
