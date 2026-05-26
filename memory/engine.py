"""
TurboVec — Compressed Semantic Memory Engine.

Architecture
────────────
  ┌─────────────────────────────────────────┐
  │            TurboVecEngine               │
  │  ┌──────────┐  ┌──────────────────────┐ │
  │  │ FAISS    │  │  Metadata Store      │ │
  │  │ Index    │  │  (in-memory dict +   │ │
  │  │ IVFFlat  │  │   JSON on disk)      │ │
  │  │ + PQ     │  │                      │ │
  │  └──────────┘  └──────────────────────┘ │
  └─────────────────────────────────────────┘

Design decisions:
  - Product Quantisation (PQ) achieves >10× compression vs raw float32.
  - IVF + nprobe tuning keeps search <10 ms for 100 K+ vectors.
  - Cosine similarity via inner product on L2-normalised vectors.
  - Metadata stored as line-delimited JSON (append-only, memory-map friendly).
  - Online indexing: add() accepts single vectors without full rebuild.
  - Periodic re-train of IVF centroids via rebuild() for high-throughput sessions.

Performance targets:
  - Retrieval latency:  < 10 ms (99th pctile)
  - Memory compression: > 10× (raw 512-dim f32 = 2 KB → PQ 8× 8-bit = 128 B)
  - Throughput:         > 10 000 inserts / s
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from core.config import get_settings
from core.models import Side
from memory.models import MemoryEntry, MemoryType, SearchResult

logger = logging.getLogger(__name__)

# ── Guarded FAISS import ──────────────────────────────────────────────────────
try:
    import faiss  # type: ignore[import]

    _HAVE_FAISS = True
except ImportError:
    _HAVE_FAISS = False
    logger.warning("faiss not installed — TurboVec will use numpy fallback (slower, no compression)")


# ── Defaults ───────────────────────────────────────────────────────────────────
_DEFAULT_DIM = 768  # nv-embedqa-e5-v5 output dimension
_PQ_CODE_SIZE = 32  # bytes per vector after PQ (768 → 32 = 24× compression)
_IVF_NCENTROIDS = 100
_NPROBE = 10


class TurboVecEngine:
    """Compressed vector store with fast ANN search and metadata persistence."""

    def __init__(
        self,
        dimension: int = _DEFAULT_DIM,
        index_dir: str | None = None,
        ncentroids: int = _IVF_NCENTROIDS,
        nprobe: int = _NPROBE,
        code_size: int = _PQ_CODE_SIZE,
    ) -> None:
        self.dimension = dimension
        self.code_size = code_size
        self.ncentroids = ncentroids
        self.nprobe = nprobe

        cfg = get_settings()
        self._index_dir = Path(index_dir or cfg.turbovec_index_dir)
        self._index_dir.mkdir(parents=True, exist_ok=True)

        self._index_path = self._index_dir / "turbovec.index"
        self._meta_path = self._index_dir / "turbovec_meta.jsonl"
        self._seq_path = self._index_dir / "turbovec_seq.txt"

        # In-memory metadata
        self._metadata: list[dict[str, Any]] = []
        self._id_to_idx: dict[int, int] = {}  # FAISS external ID → metadata index

        # FAISS index (lazy-loaded)
        self._index: faiss.Index | None = None
        self._is_trained = False
        self._next_id: int = 0

        # Numpy fallback (used when FAISS unavailable, or as buffer before IVF training)
        self._vectors_np: list[np.ndarray] = []

        self._load_existing()

    # ── Persistence ────────────────────────────────────────────────────────────

    def _load_existing(self) -> None:
        """Load pre-existing index, metadata, and numpy vectors from disk."""
        # Load FAISS index
        if self._index_path.exists():
            try:
                self._index = faiss.read_index(str(self._index_path))
                self._is_trained = self._index.is_trained
                logger.info("Loaded FAISS index from %s (%d vectors)", self._index_path, self._index.ntotal)
            except Exception as exc:
                logger.warning("Failed to load FAISS index: %s — starting fresh", exc)
                self._index = None

        # Load numpy fallback vectors (used when FAISS was unavailable or buffer before IVF)
        np_path = self._index_dir / "turbovec_np.npy"
        if np_path.exists():
            try:
                arr = np.load(str(np_path))
                self._vectors_np = [arr[i] for i in range(arr.shape[0])]
                logger.info("Loaded %d numpy fallback vectors", len(self._vectors_np))
            except Exception as exc:
                logger.warning("Failed to load numpy vectors: %s", exc)

        # Load metadata
        if self._meta_path.exists():
            with open(self._meta_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self._metadata.append(json.loads(line))
            logger.info("Loaded %d metadata entries", len(self._metadata))

        # Load sequence counter
        if self._seq_path.exists():
            with open(self._seq_path, "r") as f:
                self._next_id = int(f.read().strip() or "0")

        # Rebuild id map from stored IDs in metadata
        for idx, meta in enumerate(self._metadata):
            fid = meta.get("_faiss_id")
            if fid is not None:
                self._id_to_idx[fid] = idx

    def _save_seq(self) -> None:
        with open(self._seq_path, "w") as f:
            f.write(str(self._next_id))

    def _append_metadata(self, entry: MemoryEntry, faiss_id: int) -> None:
        meta = {
            "_faiss_id": faiss_id,
            "memory_id": entry.memory_id,
            "memory_type": entry.memory_type.value,
            "symbol": entry.symbol,
            "timestamp": entry.timestamp.isoformat(),
            "side": entry.side.value if entry.side else None,
            "pnl_pct": entry.pnl_pct,
            "win": entry.win,
            "consensus_score": entry.consensus_score,
            "confidence_score": entry.confidence_score,
            "metadata": entry.metadata,
        }
        self._metadata.append(meta)
        self._id_to_idx[faiss_id] = len(self._metadata) - 1

        # Append to JSONL
        with open(self._meta_path, "a") as f:
            f.write(json.dumps(meta, default=str) + "\n")

    # ── Index management ───────────────────────────────────────────────────────

    def _ensure_index(self, vectors: np.ndarray) -> None:
        """Lazy-init and optionally train the FAISS index."""
        if self._index is not None and self._is_trained:
            return

        if not _HAVE_FAISS:
            logger.warning("FAISS not available — using numpy brute-force (no compression)")
            return

        n, d = vectors.shape
        if n < self.ncentroids:
            # Too few vectors for IVF — use flat index
            logger.info("Using flat (brute-force) index — only %d vectors", n)
            self._index = faiss.IndexFlatIP(d)
            self._is_trained = True
            return

        # IVF with Product Quantisation for compression
        quantiser = faiss.IndexFlatIP(d)
        code_size = min(self.code_size, d // 4)  # ensure PQ is not absurd
        self._index = faiss.IndexIVFPQ(quantiser, d, self.ncentroids, code_size, 8)

        logger.info("Training FAISS IVF+P Q index (%d centroids, %d bytes PQ code)", self.ncentroids, code_size)
        self._index.train(vectors)
        self._is_trained = True

    def _normalise(self, vectors: np.ndarray) -> np.ndarray:
        """L2-normalise vectors in-place for cosine similarity via inner product."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return vectors / norms

    # ── Public API ─────────────────────────────────────────────────────────────

    async def add(self, entry: MemoryEntry) -> int:
        """Add a memory entry (with its vector) to the store.

        Returns the FAISS external ID assigned to this vector.

        On first insert, creates a flat IP index immediately so all vectors
        go into FAISS from the start. When enough vectors accumulate and
        an IVF+P Q index is configured, training and migration happen in
        a thread to avoid blocking the event loop.
        """
        if entry.vector is None:
            raise ValueError("Cannot add memory entry without a vector")

        # Generate memory_id if empty
        if not entry.memory_id:
            entry.memory_id = str(uuid.uuid4())

        # Validate dimension
        if len(entry.vector) != self.dimension:
            raise ValueError(
                f"Vector dimension {len(entry.vector)} does not match "
                f"engine dimension {self.dimension}. Check TURBOVEC_DIMENSION config "
                f"or the NIM embedding model output."
            )

        vec = np.array([entry.vector], dtype=np.float32)
        vec = self._normalise(vec)
        faiss_id = self._next_id
        self._next_id += 1

        if _HAVE_FAISS:
            # Lazy-init: create flat index on first insert
            if self._index is None:
                flat = faiss.IndexFlatIP(self.dimension)
                # IndexIDMap enables add_with_ids for flat IP index
                self._index = faiss.IndexIDMap(flat)
                self._is_trained = True

            if self._is_trained:
                self._index.add_with_ids(vec, np.array([faiss_id], dtype=np.int64))
            else:
                # IVF not yet trained — accumulate, then batch-train on thread
                self._vectors_np.append(vec[0])
                if len(self._vectors_np) >= self.ncentroids:
                    batch = np.array(self._vectors_np, dtype=np.float32)
                    batch = self._normalise(batch)
                    await asyncio.to_thread(self._ensure_index, batch)
                    if self._is_trained:
                        ids_arr = np.array(
                            [self._metadata[i].get("_faiss_id", i) for i in range(len(self._metadata))],
                            dtype=np.int64,
                        )
                        self._index.add_with_ids(batch, ids_arr)
                        self._vectors_np.clear()
        else:
            self._vectors_np.append(vec[0])

        self._append_metadata(entry, faiss_id)
        self._save_seq()
        return faiss_id

    def search(
        self,
        query_vector: list[float] | np.ndarray,
        k: int = 10,
        memory_type_filter: MemoryType | None = None,
        symbol_filter: str | None = None,
    ) -> list[SearchResult]:
        """Search for the *k* most similar memory entries.

        Supports optional filtering by memory type and symbol.
        Filtering is applied post-search (metadata scan), so request
        a larger ``k`` when filters are active.
        """
        # Validate dimensions
        qv = query_vector if isinstance(query_vector, list) else query_vector.tolist()
        if len(qv) != self.dimension:
            raise ValueError(
                f"Query vector dimension {len(qv)} does not match "
                f"engine dimension {self.dimension}. Check TURBOVEC_DIMENSION config."
            )

        query = np.array([query_vector], dtype=np.float32)
        query = self._normalise(query)

        raw_results: list[tuple[float, int]] = []

        if _HAVE_FAISS and self._index is not None and self._index.ntotal > 0:
            self._index.nprobe = self.nprobe
            similarities, ids = self._index.search(query, min(k * 3, self._index.ntotal))
            for sim, fid in zip(similarities[0], ids[0]):
                if fid == -1:
                    continue
                raw_results.append((float(sim), int(fid)))
        elif self._vectors_np:
            # Numpy brute-force fallback
            query_flat = query[0]
            n = len(self._vectors_np)
            sims = np.dot(self._vectors_np, query_flat)
            top_k = min(k * 3, n)
            top_indices = np.argpartition(sims, -top_k)[-top_k:]
            top_order = top_indices[np.argsort(-sims[top_indices])]
            for idx in top_order:
                fid = self._metadata[idx].get("_faiss_id", idx)
                raw_results.append((float(sims[idx]), int(fid)))
        else:
            return []

        # Build results with metadata
        results: list[SearchResult] = []
        seen_ids: set[str] = set()

        for sim, fid in raw_results:
            idx = self._id_to_idx.get(fid)
            if idx is None:
                continue
            meta = self._metadata[idx]
            mid = meta.get("memory_id", "")
            if mid in seen_ids:
                continue
            seen_ids.add(mid)

            # Apply filters
            if memory_type_filter and meta.get("memory_type") != memory_type_filter.value:
                continue
            if symbol_filter and meta.get("symbol") != symbol_filter:
                continue

            # Parse timestamp and make it UTC-aware (JSONL stores naive ISO strings)
            ts_str = meta.get("timestamp", datetime.now(timezone.utc).isoformat())
            try:
                ts = datetime.fromisoformat(ts_str)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
            except Exception:
                ts = datetime.now(timezone.utc)

            entry = MemoryEntry(
                memory_id=mid,
                memory_type=MemoryType(meta.get("memory_type", "market_state")),
                symbol=meta.get("symbol", ""),
                timestamp=ts,
                side=Side(meta["side"]) if meta.get("side") else None,
                pnl_pct=meta.get("pnl_pct"),
                win=meta.get("win"),
                consensus_score=meta.get("consensus_score"),
                confidence_score=meta.get("confidence_score"),
                metadata=meta.get("metadata", {}),
            )
            results.append(SearchResult(memory=entry, similarity=sim, rank=len(results) + 1))

            if len(results) >= k:
                break

        return results

    def save(self) -> None:
        """Persist the FAISS index (and numpy fallback vectors) to disk."""
        if _HAVE_FAISS and self._index is not None:
            faiss.write_index(self._index, str(self._index_path))
            logger.info("Saved FAISS index (%d vectors) to %s", self._index.ntotal, self._index_path)
        # Save numpy fallback vectors so they survive restart
        if self._vectors_np:
            np_path = self._index_dir / "turbovec_np.npy"
            np.save(str(np_path), np.array(self._vectors_np, dtype=np.float32))

    async def rebuild(self) -> None:
        """Rebuild the IVF centroids and re-index all vectors.

        Call periodically (e.g. every 10 000 inserts) for optimal index quality.
        Runs training in a thread to avoid blocking the event loop.
        """
        if not _HAVE_FAISS:
            return

        # Collect all vectors from metadata IDs
        all_vectors = []
        all_ids = []
        for idx, meta in enumerate(self._metadata):
            fid = meta.get("_faiss_id")
            if fid is not None:
                path = self._index_dir / f"vec_{fid}.npy"
                if path.exists():
                    all_vectors.append(np.load(str(path)))
                    all_ids.append(fid)

        if not all_vectors:
            logger.info("No saved vectors to rebuild index with")
            return

        vecs = np.array(all_vectors, dtype=np.float32)
        await asyncio.to_thread(self._ensure_index, vecs)
        if self._index is not None and self._is_trained:
            self._index.reset()
            self._index.add_with_ids(vecs, np.array(all_ids, dtype=np.int64))
            self.save()
            logger.info("Rebuilt index with %d vectors", len(all_ids))

    @property
    def size(self) -> int:
        """Number of vectors currently indexed."""
        if _HAVE_FAISS and self._index is not None:
            return self._index.ntotal
        return len(self._vectors_np)

    @property
    def compression_ratio(self) -> float:
        """Estimated compression ratio vs raw float32 storage."""
        if not _HAVE_FAISS or self._index is None:
            return 1.0
        raw_bytes = self.dimension * 4  # float32
        # For IVF + PQ
        if hasattr(self._index, "pq"):
            stored_bytes = self._index.pq.code_size
            ratio = raw_bytes / stored_bytes if stored_bytes > 0 else 1.0
            return round(ratio, 1)
        return 1.0


# ── Global singleton ──────────────────────────────────────────────────────────
_engine: TurboVecEngine | None = None


def get_engine() -> TurboVecEngine:
    """Return the global TurboVec engine singleton."""
    global _engine
    if _engine is None:
        _engine = TurboVecEngine()
    return _engine
