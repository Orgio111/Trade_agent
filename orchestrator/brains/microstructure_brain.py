"""Brain #4: Microstructure / Order Flow Imbalance (OFI) Analyzer.

Computes real-time order flow imbalance from trade data.
Weight in Go orchestrator: 0.10.
"""

from __future__ import annotations

import logging
import os
from collections import deque

import numpy as np

from .base_brain import BaseBrain, BrainSignal

logger = logging.getLogger(__name__)


class MicrostructureBrain(BaseBrain):
    """Microstructure / Order Flow Imbalance brain.

    Tracks trade-by-trade order flow: classifies trades as buyer-initiated
    (at ask) or seller-initiated (at bid), accumulates imbalance, and
    generates a directional score.
    """

    @property
    def brain_id(self) -> str:
        return "microstructure"

    def __init__(self) -> None:
        self._lookback = int(os.getenv("OFI_LOOKBACK", "100"))
        self._trades: deque[dict] = deque(maxlen=self._lookback * 2)
        self._cum_ofi: deque[float] = deque(maxlen=self._lookback)

    async def compute_score(self, symbol: str) -> BrainSignal:
        if len(self._trades) < 20:
            return BrainSignal(
                brain_id=self.brain_id,
                symbol=symbol,
                score=0.0,
                confidence=0.1,
                metadata={"reason": "insufficient_trades"},
            )

        # Compute OFI
        ofi_values = list(self._cum_ofi[-self._lookback:]) if self._cum_ofi else []
        if not ofi_values:
            return BrainSignal(
                brain_id=self.brain_id,
                symbol=symbol,
                score=0.0,
                confidence=0.1,
            )

        ofi_array = np.array(ofi_values)
        raw_ofi = float(ofi_array[-1]) if len(ofi_array) > 0 else 0.0

        # Normalize OFI: compare recent to rolling mean/std
        ofi_mean = float(np.mean(ofi_array[-50:])) if len(ofi_array) >= 50 else 0.0
        ofi_std = float(np.std(ofi_array[-50:])) if len(ofi_array) >= 50 else 1.0
        if ofi_std < 1e-8:
            ofi_std = 1.0

        z_score = (raw_ofi - ofi_mean) / ofi_std
        # Map z-score to [-1, +1] via tanh saturation
        score = float(np.tanh(z_score * 0.5))

        # Confidence based on trade count and z-score magnitude
        confidence = min(0.9, 0.3 + abs(z_score) * 0.1)

        # Check for absorption (high volume but price doesn't move → absorption)
        absorption = self._detect_absorption()
        if absorption:
            # Invert signal if absorption detected (smart money absorbing)
            score *= -0.5
            confidence *= 0.7

        return BrainSignal(
            brain_id=self.brain_id,
            symbol=symbol,
            score=float(np.clip(score, -1.0, 1.0)),
            confidence=float(confidence),
            metadata={
                "ofi": round(raw_ofi, 6),
                "ofi_z": round(z_score, 3),
                "trade_count": len(self._trades),
                "absorption": absorption,
            },
        )

    def push_trade(self, price: float, qty: float, side: str, ts: int = 0) -> None:
        """Push a new trade tick for OFI computation."""
        trade = {"price": price, "qty": qty, "side": side, "ts": ts}
        self._trades.append(trade)

        # Update cumulative OFI
        if side in ("buy", "B"):
            self._cum_ofi.append(qty)
        elif side in ("sell", "S"):
            self._cum_ofi.append(-qty)
        else:
            # Classify by tick rule: if price > last price → buy
            if len(self._trades) >= 2:
                prev = list(self._trades)[-2]
                if price > prev["price"]:
                    self._cum_ofi.append(qty)
                elif price < prev["price"]:
                    self._cum_ofi.append(-qty)
                else:
                    self._cum_ofi.append(0.0)

    def _detect_absorption(self) -> bool:
        """Detect if large orders are being absorbed without price movement."""
        if len(self._trades) < 30:
            return False
        recent = list(self._trades)[-30:]
        total_vol = sum(t["qty"] for t in recent)
        prices = [t["price"] for t in recent]
        price_range = max(prices) - min(prices)
        avg_price = sum(prices) / len(prices)
        if avg_price == 0:
            return False
        # High volume but small price range → absorption
        range_pct = price_range / avg_price
        vol_threshold = float(os.getenv("OFI_ABSORPTION_VOL", "10.0"))
        return total_vol > vol_threshold and range_pct < 0.001

    async def warmup(self) -> None:
        logger.info(f"[microstructure] Ready (lookback={self._lookback})")
