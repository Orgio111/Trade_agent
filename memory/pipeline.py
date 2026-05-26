"""
Semantic Memory Pipeline — the bridge between live market data and TurboVec.

Workflow
────────
  Market Tick
       │
       ▼
  1. Build state text from prices, indicators, signals
       │
       ▼
  2. Embed state → query vector
       │
       ▼
  3. TurboVec search for similar historical states
       │
       ▼
  4. Rank by outcome similarity (profitable/failed)
       │
       ▼
  5. Format as RetrievalContext → inject into agent prompts
       │
       ▼
  Council deliberates with memory-augmented context

  ════════════════════════════════

  After trade closes:
  1. Embed market state at entry time
  2. Store with outcome (win/loss, PnL, confidence)
  3. TurboVec indexes for future retrieval
"""
from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np

from memory.embedding import EmbeddingPipeline, get_embedder
from memory.engine import TurboVecEngine, get_engine
from memory.models import (
    MemoryEntry,
    MemoryType,
    RetrievalContext,
    RetrievalPurpose,
    SearchResult,
)
from core.models import CouncilDecision, Side, TradeOutcome, TechnicalSignal

logger = logging.getLogger(__name__)

# ── Defaults ───────────────────────────────────────────────────────────────────
_DEFAULT_K = 8          # results to retrieve per query
_MIN_STATE_BARS = 20    # minimum closes needed to build a meaningful state


class SemanticMemoryPipeline:
    """End-to-end memory pipeline: embed → search → inject → store.

    Typical usage in the supervisor loop::

        pipeline = SemanticMemoryPipeline()

        # Before council deliberation:
        ctx = await pipeline.retrieve(symbol, closes, technical)
        # → inject ctx.format_for_prompt() into council system prompt

        # After trade closes:
        await pipeline.store_outcome(decision, outcome, closes)
    """

    def __init__(
        self,
        engine: TurboVecEngine | None = None,
        embedder: EmbeddingPipeline | None = None,
    ) -> None:
        self._engine = engine or get_engine()
        self._embedder = embedder or get_embedder()
        self._running = False
        self._last_compact_time = 0.0
        self._compact_interval_s = 3600  # compact index every hour

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Initialize the pipeline (no-op, engine is lazy)."""
        self._running = True
        logger.info(
            "SemanticMemoryPipeline started — engine has %d vectors (%.1f× compression)",
            self._engine.size,
            self._engine.compression_ratio,
        )

    async def stop(self) -> None:
        """Persist the index and shut down."""
        self._running = False
        self._engine.save()
        logger.info("SemanticMemoryPipeline stopped — saved %d vectors", self._engine.size)

    # ── Retrieval ──────────────────────────────────────────────────────────────

    async def retrieve(
        self,
        symbol: str,
        closes: np.ndarray,
        volumes: np.ndarray | None = None,
        technical: TechnicalSignal | None = None,
        k: int = _DEFAULT_K,
        purpose: RetrievalPurpose = RetrievalPurpose.TRADE_VALIDATION,
    ) -> RetrievalContext:
        """Search TurboVec for market states similar to the current one.

        Returns a ``RetrievalContext`` with results and a pre-formatted
        prompt block ready for injection into agent deliberation.
        """
        if len(closes) < _MIN_STATE_BARS:
            return RetrievalContext(purpose=purpose, symbol=symbol)

        # 1. Build and embed current state
        query_vec = await self._embedder.embed_market_state(
            symbol=symbol,
            closes=closes,
            volumes=volumes,
            rsi=technical.rsi_14 if technical else None,
            ema_fast=technical.ema_fast if technical else None,
            ema_slow=technical.ema_slow if technical else None,
            atr=technical.atr_14 if technical else None,
            macd=technical.macd if technical else None,
            macd_signal=technical.macd_signal if technical else None,
            bb_upper=technical.bb_upper if technical else None,
            bb_lower=technical.bb_lower if technical else None,
            volume_ratio=technical.volume_ratio if technical else None,
            trend=technical.trend.value if technical else None,
        )

        # 2. Search for similar states (same symbol, all types)
        results = self._engine.search(
            query_vector=query_vec.tolist(),
            k=k * 2,  # request extra for filtering
            symbol_filter=symbol,
        )

        # 3. Analyse outcome distribution
        n_win = sum(1 for r in results if r.memory.win is True)
        n_loss = sum(1 for r in results if r.memory.win is False)
        win_pnls = [r.memory.pnl_pct for r in results if r.memory.win is True and r.memory.pnl_pct is not None]
        loss_pnls = [r.memory.pnl_pct for r in results if r.memory.win is False and r.memory.pnl_pct is not None]

        # 4. Compute memory age span
        ages = []
        now = datetime.now(timezone.utc)
        for r in results:
            delta = (now - r.memory.timestamp).total_seconds() / 3600
            ages.append(delta)
        max_age = max(ages) if ages else None

        return RetrievalContext(
            purpose=purpose,
            symbol=symbol,
            query_vector=query_vec.tolist(),
            results=results[:k],
            n_similar_profitable=n_win,
            n_similar_failed=n_loss,
            avg_win_pnl=float(np.mean(win_pnls)) if win_pnls else None,
            avg_loss_pnl=float(np.mean(loss_pnls)) if loss_pnls else None,
            memory_age_hours=max_age,
        )

    # ── Storage ────────────────────────────────────────────────────────────────

    async def store_market_state(
        self,
        symbol: str,
        closes: np.ndarray,
        volumes: np.ndarray | None = None,
        technical: TechnicalSignal | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> str | None:
        """Store the current market state as a retrievable memory.

        Returns the memory_id if stored, None if insufficient data.
        """
        if len(closes) < _MIN_STATE_BARS:
            return None

        vec = await self._embedder.embed_market_state(
            symbol=symbol,
            closes=closes,
            volumes=volumes,
            rsi=technical.rsi_14 if technical else None,
            ema_fast=technical.ema_fast if technical else None,
            ema_slow=technical.ema_slow if technical else None,
            atr=technical.atr_14 if technical else None,
            macd=technical.macd if technical else None,
            macd_signal=technical.macd_signal if technical else None,
            bb_upper=technical.bb_upper if technical else None,
            bb_lower=technical.bb_lower if technical else None,
            volume_ratio=technical.volume_ratio if technical else None,
            trend=technical.trend.value if technical else None,
            extra_metadata=extra_metadata,
        )

        entry = MemoryEntry(
            memory_id="",
            memory_type=MemoryType.MARKET_STATE,
            symbol=symbol,
            timestamp=datetime.now(timezone.utc),
            vector=vec.tolist(),
            metadata=extra_metadata or {},
        )
        await self._engine.add(entry)
        await self._maybe_compact()
        return entry.memory_id

    async def store_trade_outcome(
        self,
        decision: CouncilDecision,
        outcome: TradeOutcome | None,
        closes: np.ndarray,
        volumes: np.ndarray | None = None,
    ) -> str | None:
        """Store a completed trade outcome as a retrievable memory.

        This is the most important storage operation — it builds the
        knowledge base of "what happened when the market looked like X".
        """
        if len(closes) < _MIN_STATE_BARS:
            return None

        # Embed market state at decision time
        vec = await self._embedder.embed_market_state(
            symbol=decision.symbol,
            closes=closes,
            volumes=volumes,
            rsi=decision.technical.rsi_14 if decision.technical else None,
            ema_fast=decision.technical.ema_fast if decision.technical else None,
            ema_slow=decision.technical.ema_slow if decision.technical else None,
            atr=decision.technical.atr_14 if decision.technical else None,
            macd=decision.technical.macd if decision.technical else None,
            macd_signal=decision.technical.macd_signal if decision.technical else None,
            bb_upper=decision.technical.bb_upper if decision.technical else None,
            bb_lower=decision.technical.bb_lower if decision.technical else None,
            volume_ratio=decision.technical.volume_ratio if decision.technical else None,
            trend=decision.technical.trend.value if decision.technical else None,
            extra_metadata={
                "session_id": decision.session_id,
                "consensus_score": decision.consensus_score,
                "rationale": decision.rationale,
                "bull_score": decision.bull_score,
                "bear_score": decision.bear_score,
            },
        )

        is_win = outcome.win if outcome else None
        pnl = outcome.pnl_pct if outcome else None

        memory_type = (
            MemoryType.PROFITABLE_SETUP
            if is_win is True
            else MemoryType.FAILED_SETUP if is_win is False else MemoryType.EXECUTED_TRADE
        )

        entry = MemoryEntry(
            memory_id="",
            memory_type=memory_type,
            symbol=decision.symbol,
            timestamp=decision.timestamp,
            vector=vec.tolist(),
            side=decision.final_side,
            pnl_pct=pnl,
            win=is_win,
            consensus_score=decision.consensus_score,
            metadata={
                "session_id": decision.session_id,
                "rationale": decision.rationale,
                "entry_price": outcome.entry_price if outcome else None,
                "exit_price": outcome.exit_price if outcome else None,
                "quantity": outcome.quantity if outcome else None,
                "holding_period_s": outcome.holding_period_s if outcome else None,
            },
        )
        await self._engine.add(entry)
        logger.info(
            "Stored %s memory for %s %s win=%s pnl=%+.2f%%",
            memory_type.value,
            decision.symbol,
            decision.final_side.value,
            is_win,
            (pnl or 0) * 100,
        )

        await self._maybe_compact()
        return entry.memory_id

    async def store_risk_event(
        self,
        symbol: str,
        event_type: str,
        var_95: float,
        var_99: float,
        drawdown_pct: float,
        extra: dict[str, Any] | None = None,
    ) -> str | None:
        """Store a risk event (VaR breach, drawdown spike, kill-switch)."""
        vec = await self._embedder.embed_risk_event(
            symbol=symbol,
            event_type=event_type,
            var_95=var_95,
            var_99=var_99,
            drawdown_pct=drawdown_pct,
            extra=extra,
        )
        entry = MemoryEntry(
            memory_id="",
            memory_type=MemoryType.RISK_EVENT,
            symbol=symbol,
            timestamp=datetime.now(timezone.utc),
            vector=vec.tolist(),
            metadata={"event_type": event_type, "var_95": var_95, "var_99": var_99, "drawdown_pct": drawdown_pct, **(extra or {})},
        )
        await self._engine.add(entry)
        return entry.memory_id

    # ── Maintenance ────────────────────────────────────────────────────────────

    async def _maybe_compact(self) -> None:
        """Periodically compact/re-train the FAISS index for optimal performance."""
        now = time.time()
        if now - self._last_compact_time > self._compact_interval_s:
            self._last_compact_time = now
            logger.info("Compacting TurboVec index (%d vectors)...", self._engine.size)
            await self._engine.rebuild()
            self._engine.save()
            logger.info("TurboVec compacted")


# ── Global singleton ──────────────────────────────────────────────────────────
_pipeline: SemanticMemoryPipeline | None = None


def get_memory_pipeline() -> SemanticMemoryPipeline:
    """Return the global SemanticMemoryPipeline singleton."""
    global _pipeline
    if _pipeline is None:
        _pipeline = SemanticMemoryPipeline()
    return _pipeline
