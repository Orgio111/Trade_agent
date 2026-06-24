"""Brain #4b: NautilusTrader-powered Order Flow Imbalance Brain.

Extends microstructure analysis with NautilusTrader's Rust-core L2/L3
orderbook reconstruction. When NautilusTrader is available, uses real
depth imbalance data; otherwise falls back to tick-rule OFI.

Weight in Go orchestrator: 0.12 (slightly higher than pure microstructure).
"""
from __future__ import annotations

import logging
from collections import deque
from typing import Optional

import numpy as np

from .base_brain import BaseBrain, BrainSignal

logger = logging.getLogger(__name__)

# Attempt NautilusTrader import — optional dependency
_NAUTILUS_AVAILABLE = False
try:
    from orchestrator.nautilus_bridge.orderbook_engine import (
        OrderbookEngine,
        OrderbookMetrics,
    )
    _NAUTILUS_AVAILABLE = True
except ImportError:
    OrderbookEngine = None  # type: ignore[assignment,misc]
    OrderbookMetrics = None  # type: ignore[assignment,misc]


class OrderFlowNautilusBrain(BaseBrain):
    """Order Flow Imbalance brain powered by NautilusTrader orderbook.

    Uses L2/L3 depth data when available (via NautilusTrader Rust core),
    otherwise falls back to trade-tick OFI classification (like MicrostructureBrain).

    Key enhancements over pure microstructure:
      - Real bid/ask depth imbalance (not tick-rule estimated)
      - Liquidity gap detection across orderbook levels
      - Depth-weighted mid-price shift detection
      - Spread compression signals
    """

    @property
    def brain_id(self) -> str:
        return "orderflow_nautilus"

    def __init__(self) -> None:
        self._lookback = 100
        self._cum_ofi: deque[float] = deque(maxlen=self._lookback)
        self._trades: deque[dict] = deque(maxlen=self._lookback * 2)
        self._ob_metrics: deque[dict] = deque(maxlen=self._lookback)
        self._nautilus_engine: Optional[OrderbookEngine] = None  # type: ignore[valid-type]

        if _NAUTILUS_AVAILABLE and OrderbookEngine is not None:
            logger.info("[orderflow_nautilus] NautilusTrader available — using L2/L3 depth")

    async def warmup(self) -> None:
        """Initialize NautilusTrader orderbook engine if available."""
        if _NAUTILUS_AVAILABLE and OrderbookEngine is not None:
            try:
                self._nautilus_engine = OrderbookEngine()
                await self._nautilus_engine.start()
                logger.info("[orderflow_nautilus] NautilusTrader OrderbookEngine started")
            except Exception as exc:
                logger.warning("[orderflow_nautilus] OrderbookEngine failed: %s — fallback to tick OFI", exc)
                self._nautilus_engine = None
        else:
            logger.info("[orderflow_nautilus] NautilusTrader unavailable — using tick-rule OFI")

    async def cooldown(self) -> None:
        """Shutdown NautilusTrader engine."""
        if self._nautilus_engine:
            await self._nautilus_engine.stop()

    async def compute_score(self, symbol: str) -> BrainSignal:
        """Compute orderflow signal with NautilusTrader depth if available."""
        # Try NautilusTrader depth-based analysis first
        if self._nautilus_engine and symbol in self._nautilus_engine.latest_metrics:
            return self._compute_from_depth(symbol)

        # Fallback: trade-tick OFI (same logic as MicrostructureBrain)
        return self._compute_from_trades(symbol)

    def push_orderbook_snapshot(
        self,
        symbol: str,
        bids: list,
        asks: list,
    ) -> None:
        """Push L2 orderbook snapshot for manual feeding (testing / non-Nautilus path)."""
        if self._nautilus_engine:
            metrics = self._nautilus_engine.compute_imbalance(bids, asks)
            self._ob_metrics.append({
                "imbalance": metrics.bid_ask_imbalance,
                "buy_pressure": metrics.buy_pressure,
                "sell_pressure": metrics.sell_pressure,
                "spread_bps": metrics.spread_bps,
                "gap": metrics.liquidity_gap,
                "dw_mid": metrics.depth_weighted_mid,
            })

    def push_trade(self, price: float, qty: float, side: str, ts: int = 0) -> None:
        """Push a trade tick for OFI computation (fallback path)."""
        trade = {"price": price, "qty": qty, "side": side, "ts": ts}
        self._trades.append(trade)
        if side in ("buy", "B"):
            self._cum_ofi.append(qty)
        elif side in ("sell", "S"):
            self._cum_ofi.append(-qty)
        else:
            if len(self._trades) >= 2:
                prev = list(self._trades)[-2]
                if price > prev["price"]:
                    self._cum_ofi.append(qty)
                elif price < prev["price"]:
                    self._cum_ofi.append(-qty)
                else:
                    self._cum_ofi.append(0.0)

    # ── Private helpers ──────────────────────────────────────

    def _compute_from_depth(self, symbol: str) -> BrainSignal:
        """Compute signal from NautilusTrader L2/L3 orderbook metrics."""
        metrics = self._nautilus_engine.latest_metrics[symbol]
        self._ob_metrics.append({
            "imbalance": metrics.bid_ask_imbalance,
            "buy_pressure": metrics.buy_pressure,
            "sell_pressure": metrics.sell_pressure,
            "spread_bps": metrics.spread_bps,
            "gap": metrics.liquidity_gap,
            "dw_mid": metrics.depth_weighted_mid,
        })

        # Imbalance-driven score: imbalance ∈ [-1, +1] already
        score = metrics.bid_ask_imbalance

        # Boost if liquidity gap detected (thin levels = potential breakout)
        if metrics.liquidity_gap > 2.0:
            score *= 1.3  # amplify signal on gap
            logger.debug("[orderflow_nautilus] Liquidity gap=%.2f — boosting signal", metrics.liquidity_gap)

        # Reduce confidence if spread is wide (low liquidity = noisy)
        confidence = 0.8
        if metrics.spread_bps > 10:
            confidence *= 0.7
            logger.debug("[orderflow_nautilus] Wide spread=%.1fbps — reducing confidence", metrics.spread_bps)

        # Direction from imbalance sign
        direction = 1 if score > 0.05 else (-1 if score < -0.05 else 0)

        return BrainSignal(
            brain_id=self.brain_id,
            symbol=symbol,
            score=float(np.clip(score, -1.0, 1.0)),
            confidence=float(confidence),
            weight=0.12,
            direction=direction,
            metadata={
                "source": "nautilus_depth",
                "bid_ask_imbalance": round(metrics.bid_ask_imbalance, 6),
                "buy_pressure": round(metrics.buy_pressure, 4),
                "sell_pressure": round(metrics.sell_pressure, 4),
                "spread_bps": round(metrics.spread_bps, 2),
                "liquidity_gap": round(metrics.liquidity_gap, 4),
            },
        )

    def _compute_from_trades(self, symbol: str) -> BrainSignal:
        """Fallback: compute from trade-tick OFI (same as MicrostructureBrain)."""
        if len(self._trades) < 20:
            return BrainSignal(
                brain_id=self.brain_id,
                symbol=symbol,
                score=0.0,
                confidence=0.1,
                metadata={"reason": "insufficient_trades"},
            )

        ofi_values = list(self._cum_ofi)[-self._lookback:] if self._cum_ofi else []
        if not ofi_values:
            return BrainSignal(
                brain_id=self.brain_id,
                symbol=symbol,
                score=0.0,
                confidence=0.1,
            )

        ofi_array = np.array(ofi_values)
        raw_ofi = float(ofi_array[-1]) if len(ofi_array) > 0 else 0.0

        ofi_mean = float(np.mean(ofi_array[-50:])) if len(ofi_array) >= 50 else 0.0
        ofi_std = float(np.std(ofi_array[-50:])) if len(ofi_array) >= 50 else 1.0
        if ofi_std < 1e-8:
            ofi_std = 1.0

        z_score = (raw_ofi - ofi_mean) / ofi_std
        score = float(np.tanh(z_score * 0.5))
        confidence = min(0.85, 0.3 + abs(z_score) * 0.1)
        direction = 1 if score > 0.05 else (-1 if score < -0.05 else 0)

        return BrainSignal(
            brain_id=self.brain_id,
            symbol=symbol,
            score=float(np.clip(score, -1.0, 1.0)),
            confidence=float(confidence),
            weight=0.10,
            direction=direction,
            metadata={
                "source": "tick_ofi_fallback",
                "ofi": round(raw_ofi, 6),
                "ofi_z": round(z_score, 3),
                "trade_count": len(self._trades),
            },
        )
