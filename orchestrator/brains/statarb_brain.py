"""Brain #8: Statistical Arbitrage / Funding Rate Analyzer.

Detects mean-reversion opportunities and monitors perpetual funding rates.
Weight in Go orchestrator: 0.10.
"""

from __future__ import annotations

import logging
import os
from collections import deque

import numpy as np

from .base_brain import BaseBrain, BrainSignal

logger = logging.getLogger(__name__)


class StatArbBrain(BaseBrain):
    """Statistical arbitrage and funding rate brain.

    Components:
      1. Z-score mean reversion: When price deviates >2σ from moving average,
         bet on reversion (contrarian signal).
      2. Funding rate: Persistent positive funding → overleveraged longs → bearish.
         Persistent negative funding → overleveraged shorts → bullish.
      3. Basis spread: Spot vs perp basis for arbitrage detection.
    """

    @property
    def brain_id(self) -> str:
        return "statarb_funding"

    def __init__(self) -> None:
        self._lookback = int(os.getenv("STATARB_LOOKBACK", "100"))
        self._prices: deque[float] = deque(maxlen=self._lookback)
        self._funding_rates: deque[float] = deque(maxlen=72)  # ~3 days at 8h intervals
        self._basis_spreads: deque[float] = deque(maxlen=50)
        self._z_threshold = float(os.getenv("STATARB_Z_THRESHOLD", "2.0"))

    async def warmup(self) -> None:
        logger.info(f"[statarb_funding] Ready (z_thresh={self._z_threshold})")

    async def compute_score(self, symbol: str) -> BrainSignal:
        z_score = self._mean_reversion_score()
        funding_score = self._funding_rate_score()
        basis_score = self._basis_score()

        # Weighted combination
        combined = 0.45 * z_score + 0.35 * funding_score + 0.20 * basis_score

        confidence = 0.25
        if len(self._prices) >= 50:
            confidence += 0.2
        if len(self._funding_rates) >= 8:
            confidence += 0.15
        if len(self._basis_spreads) >= 10:
            confidence += 0.1
        confidence = min(0.85, confidence)

        has_data = len(self._prices) >= 20 or len(self._funding_rates) >= 3

        return BrainSignal(
            brain_id=self.brain_id,
            symbol=symbol,
            score=float(np.clip(combined, -1.0, 1.0)),
            confidence=confidence if has_data else 0.1,
            metadata={
                "z_score": round(z_score, 4),
                "funding_score": round(funding_score, 4),
                "basis_score": round(basis_score, 4),
                "data_available": has_data,
            },
        )

    def push_price(self, price: float) -> None:
        """Push a new closing price tick."""
        self._prices.append(price)

    def push_funding_rate(self, rate: float) -> None:
        """Push a funding rate (decimal, e.g. 0.0001 = 0.01%)."""
        self._funding_rates.append(rate)

    def push_basis_spread(self, spread_pct: float) -> None:
        """Push basis spread (perp_price - spot_price) / spot_price in decimal."""
        self._basis_spreads.append(spread_pct)

    def _mean_reversion_score(self) -> float:
        """Z-score based mean reversion signal.

        High positive z-score (price above mean) → sell (expect reversion down).
        High negative z-score (price below mean) → buy (expect reversion up).
        """
        if len(self._prices) < 30:
            return 0.0

        prices = np.array(list(self._prices)[-self._lookback:])
        window = prices[-30:]  # 30-period lookback for stats
        mean = float(np.mean(window))
        std = float(np.std(window))

        if std < 1e-10:
            return 0.0

        z = (prices[-1] - mean) / std

        # Invert: high z → sell signal (contrarian)
        if abs(z) < self._z_threshold:
            return 0.0  # no edge below threshold

        # Scale and invert
        score = -float(np.tanh(z * 0.5))  # contrarian: sell overbought, buy oversold
        return score

    def _funding_rate_score(self) -> float:
        """Funding rate contrarian signal.

        Accumulated positive funding = overcrowded longs → bearish.
        Accumulated negative funding = overcrowded shorts → bullish.
        """
        if len(self._funding_rates) < 3:
            return 0.0

        recent = list(self._funding_rates)[-8:]  # last day at 8h intervals
        avg_funding = float(np.mean(recent))

        # Scale: 0.01% avg funding → mild bearish; 0.05%+ → strong bearish
        # Funding is in decimal form (0.0001 = 0.01%)
        score = -float(np.tanh(avg_funding * 500))  # invert: positive funding → sell
        return score

    def _basis_score(self) -> float:
        """Basis spread signal.

        Large positive basis (perp > spot) → market is bullish leveraged → could be top.
        Large negative basis (perp < spot) → market is bearish leveraged → could be bottom.
        Extreme basis → contrarian signal.
        """
        if len(self._basis_spreads) < 5:
            return 0.0

        recent = list(self._basis_spreads)[-10:]
        avg_basis = float(np.mean(recent))

        # Contrarian: extreme positive basis → bearish, extreme negative → bullish
        score = -float(np.tanh(avg_basis * 100))
        return score
