"""
Hierarchical Memory System — Phase 1 Self-Evolving Core.

Two-tier memory with temporal decay and pattern abstraction:
  ┌─────────────────────────────────────────────────────┐
  │  Short-Term Memory (STM)  — last 100 trades          │
  │  • Full reasoning chain + signals                    │
  │  • Fast O(1) retrieval by session_id                │
  │  • Decays rapidly (half-life: 24h)                  │
  ├─────────────────────────────────────────────────────┤
  │  Long-Term Memory (LTM)   — FAISS + pattern DB      │
  │  • Abstract patterns extracted from STM clusters    │
  │  • Slow decay (half-life: 30d)                      │
  │  • Generalizes across symbols                       │
  └─────────────────────────────────────────────────────┘
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np

from memory.vector_store import FAISSStrategyMemory, StrategyMemory
from core.nim_client import nim_embed, nim_json

log = logging.getLogger(__name__)

STM_CAPACITY    = 100
STM_HALF_LIFE_H = 24.0
LTM_HALF_LIFE_H = 24.0 * 30

_ABSTRACTOR_SYSTEM = """You are a trading pattern analyst. Given a cluster of similar trade setups,
extract the underlying abstract pattern:
{
  "pattern_name": "<descriptive 3-5 word name>",
  "market_condition": "<regime, volatility, trend direction>",
  "key_signals": ["<signal with threshold>", ...],
  "historical_win_rate": <float>,
  "expected_pnl_range": [<low>, <high>],
  "failure_triggers": ["<what causes this pattern to fail>", ...],
  "confidence": <0-1>
}"""


@dataclass
class STMEntry:
    session_id:      str
    symbol:          str
    timestamp:       datetime
    reasoning_chain: str
    signals:         dict[str, Any]
    outcome:         str   # WIN | LOSS | BLOCKED
    pnl_pct:         float
    embedding:       np.ndarray
    _created_ts:     float = field(default_factory=time.monotonic, init=False)

    def relevance_score(self) -> float:
        """Temporal decay: relevance halves every STM_HALF_LIFE_H hours."""
        age_h = (time.monotonic() - self._created_ts) / 3600.0
        base = 0.5 ** (age_h / STM_HALF_LIFE_H)
        outcome_weight = 1.0 if self.outcome == "LOSS" else 0.7
        return base * outcome_weight


@dataclass
class LTMPattern:
    pattern_id:      str
    pattern_name:    str
    market_condition: str
    key_signals:     list[str]
    win_rate:        float
    pnl_range:       tuple[float, float]
    failure_triggers: list[str]
    embedding:       np.ndarray
    last_updated:    datetime
    occurrence_count: int = 1

    def relevance_score(self) -> float:
        age_h = (datetime.utcnow() - self.last_updated).total_seconds() / 3600.0
        decay = 0.5 ** (age_h / LTM_HALF_LIFE_H)
        frequency_boost = min(np.log1p(self.occurrence_count) / 5.0, 0.3)
        return decay + frequency_boost


class HierarchicalMemory:
    """
    Two-tier trading memory with autonomous pattern abstraction.
    STM holds raw trade records; LTM holds abstracted patterns.
    """

    def __init__(self, faiss_store: FAISSStrategyMemory) -> None:
        self._stm: OrderedDict[str, STMEntry] = OrderedDict()
        self._ltm: list[LTMPattern] = []
        self._faiss = faiss_store
        self._abstraction_queue: list[STMEntry] = []
        self._CLUSTER_THRESHOLD = 10  # abstract after 10 similar memories

    async def add(
        self,
        session_id: str,
        symbol: str,
        reasoning_chain: str,
        signals: dict,
        outcome: str,
        pnl_pct: float,
    ) -> None:
        embeds = await nim_embed([reasoning_chain])
        vec = np.array(embeds[0], dtype=np.float32)

        entry = STMEntry(
            session_id=session_id,
            symbol=symbol,
            timestamp=datetime.utcnow(),
            reasoning_chain=reasoning_chain,
            signals=signals,
            outcome=outcome,
            pnl_pct=pnl_pct,
            embedding=vec,
        )

        # LRU eviction
        if len(self._stm) >= STM_CAPACITY:
            evicted_id, evicted = self._stm.popitem(last=False)
            self._abstraction_queue.append(evicted)

        self._stm[session_id] = entry
        # Also persist to FAISS for long-range retrieval
        self._faiss.add(StrategyMemory(
            session_id=session_id,
            symbol=symbol,
            timestamp=entry.timestamp,
            reasoning_chain=reasoning_chain,
            outcome=outcome,
            pnl_pct=pnl_pct,
            embedding=vec,
        ))

        # Trigger abstraction if queue is full
        if len(self._abstraction_queue) >= self._CLUSTER_THRESHOLD:
            asyncio.create_task(self._abstract_patterns())

    def retrieve_stm(self, query_vec: np.ndarray, k: int = 5) -> list[STMEntry]:
        """Return top-k STM entries by cosine similarity × relevance decay."""
        if not self._stm:
            return []
        entries = list(self._stm.values())
        embeddings = np.stack([e.embedding for e in entries])
        norm = np.linalg.norm(query_vec)
        q = query_vec / (norm + 1e-10)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normed = embeddings / (norms + 1e-10)
        cosine_scores = normed @ q
        relevance = np.array([e.relevance_score() for e in entries])
        combined = cosine_scores * 0.7 + relevance * 0.3
        top_k = np.argsort(-combined)[:k]
        return [entries[i] for i in top_k]

    def retrieve_ltm(self, query_vec: np.ndarray, k: int = 3) -> list[LTMPattern]:
        """Return top-k LTM patterns by cosine similarity × recency."""
        if not self._ltm:
            return []
        embeddings = np.stack([p.embedding for p in self._ltm])
        norm = np.linalg.norm(query_vec)
        q = query_vec / (norm + 1e-10)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normed = embeddings / (norms + 1e-10)
        cosine_scores = normed @ q
        relevance = np.array([p.relevance_score() for p in self._ltm])
        combined = cosine_scores * 0.6 + relevance * 0.4
        top_k = np.argsort(-combined)[:k]
        return [self._ltm[i] for i in top_k]

    async def _abstract_patterns(self) -> None:
        """Extract abstract patterns from the queued STM entries via NIM."""
        entries = self._abstraction_queue.copy()
        self._abstraction_queue.clear()

        wins   = [e for e in entries if e.outcome == "WIN"]
        losses = [e for e in entries if e.outcome == "LOSS"]

        for cluster, label in [(wins, "WIN cluster"), (losses, "LOSS cluster")]:
            if len(cluster) < 3:
                continue
            summary = "\n".join(
                f"[{e.symbol} {e.outcome} PnL={e.pnl_pct:.1%}] {e.reasoning_chain[:200]}"
                for e in cluster[:5]
            )
            result = await nim_json(
                [
                    {"role": "system", "content": _ABSTRACTOR_SYSTEM},
                    {"role": "user", "content": f"Cluster ({label}):\n{summary}"},
                ],
                temperature=0.1,
            )

            pattern_text = result.get("pattern_name", "") + " " + str(result.get("key_signals", []))
            embeds = await nim_embed([pattern_text])
            vec = np.array(embeds[0], dtype=np.float32)

            import uuid
            pattern = LTMPattern(
                pattern_id=str(uuid.uuid4()),
                pattern_name=result.get("pattern_name", "unknown"),
                market_condition=result.get("market_condition", ""),
                key_signals=result.get("key_signals", []),
                win_rate=float(result.get("historical_win_rate", 0.5)),
                pnl_range=tuple(result.get("expected_pnl_range", [0, 0])),
                failure_triggers=result.get("failure_triggers", []),
                embedding=vec,
                last_updated=datetime.utcnow(),
            )
            self._ltm.append(pattern)
            log.info("LTM pattern abstracted: '%s' (win_rate=%.1%)", pattern.pattern_name, pattern.win_rate)

    async def build_context(self, query_text: str, symbol: str) -> str:
        """Build a rich context string from both STM and LTM for the Supervisor."""
        embeds = await nim_embed([query_text])
        vec = np.array(embeds[0], dtype=np.float32)

        stm_hits = self.retrieve_stm(vec, k=3)
        ltm_hits = self.retrieve_ltm(vec, k=2)

        lines = ["=== Hierarchical Memory Context ==="]

        if stm_hits:
            lines.append("\n[Short-Term Memory — Recent Similar Setups]")
            for e in stm_hits:
                lines.append(
                    f"  • {e.symbol} {e.outcome} ({e.timestamp.date()}) "
                    f"PnL={e.pnl_pct:.1%} | {e.reasoning_chain[:150]}..."
                )

        if ltm_hits:
            lines.append("\n[Long-Term Memory — Abstract Patterns]")
            for p in ltm_hits:
                lines.append(
                    f"  • Pattern: '{p.pattern_name}' "
                    f"(win_rate={p.win_rate:.0%}, regime={p.market_condition})\n"
                    f"    Signals: {p.key_signals[:2]}\n"
                    f"    Failure triggers: {p.failure_triggers[:2]}"
                )

        return "\n".join(lines) if len(lines) > 1 else "No relevant memory found."
