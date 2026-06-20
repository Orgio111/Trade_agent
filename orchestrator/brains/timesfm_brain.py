"""Brain #1: TimesFM Time-Series Forecaster.

Uses Google's TimesFM foundation model for zero-shot time-series forecasting.
Direction output: long (+1), short (-1), or hold (0).
Weight in Go orchestrator: 0.25 (highest — primary forecaster).
"""

from __future__ import annotations

import logging
import os

import numpy as np

from .base_brain import BaseBrain, BrainSignal

logger = logging.getLogger(__name__)


class TimesFMBrain(BaseBrain):
    """TimesFM-based time-series forecasting brain.

    Loads the checkpoint lazily (the 200M model is ~800MB).
    Falls back to a simple momentum heuristic if TimesFM is unavailable.
    """

    DIRECTION_THRESHOLD = 0.002  # 0.2% expected return → directional signal

    @property
    def brain_id(self) -> str:
        return "timesfm"

    def __init__(self) -> None:
        self._forecaster = None
        self._load_error: str | None = None
        self._price_buffer: list[float] = []
        self._max_context = int(os.getenv("TIMESFM_MAX_CONTEXT", "512"))

    async def warmup(self) -> None:
        """Load TimesFM checkpoint on startup."""
        try:
            from orchestrator.timesfm_forecaster import TimesFMForecaster

            checkpoint = os.getenv(
                "TIMESFM_CHECKPOINT", "google/timesfm-2.5-200m-pytorch"
            )
            self._forecaster = TimesFMForecaster(
                checkpoint=checkpoint,
                max_context=self._max_context,
                direction_threshold=self.DIRECTION_THRESHOLD,
            )
            if not self._forecaster.available:
                self._load_error = self._forecaster._load_error
                logger.warning(f"TimesFM unavailable: {self._load_error}")
        except ImportError as e:
            self._load_error = f"TimesFM package missing: {e}"
            logger.warning(f"TimesFM unavailable: {self._load_error}")

    async def compute_score(self, symbol: str) -> BrainSignal:
        """Forecast next price direction and emit a score."""
        # Collect recent prices (placeholder: use buffer or fetch via CCXT)
        prices = self._get_recent_prices(symbol)

        if self._forecaster is not None and self._forecaster.available and len(prices) >= 20:
            result = self._forecaster.forecast(prices, horizon=12)
            if result.available:
                direction_map = {"long": 1.0, "short": -1.0, "hold": 0.0}
                score = direction_map.get(result.direction, 0.0) * result.confidence
                return BrainSignal(
                    brain_id=self.brain_id,
                    symbol=symbol,
                    score=score,
                    confidence=result.confidence,
                    metadata={
                        "model": result.model,
                        "expected_return": result.expected_return,
                        "horizon": result.horizon,
                    },
                )

        # Fallback: simple momentum heuristic
        return self._momentum_fallback(symbol, prices)

    def _get_recent_prices(self, symbol: str) -> list[float]:
        """Get recent prices from internal buffer or external source.

        In production, this is fed by the Go data engine via shared memory.
        For now, returns the internal buffer (populated externally).
        """
        return self._price_buffer[-self._max_context :]

    def push_price(self, price: float) -> None:
        """Append a new price tick to the internal buffer."""
        self._price_buffer.append(price)
        if len(self._price_buffer) > self._max_context * 2:
            self._price_buffer = self._price_buffer[-self._max_context :]

    def _momentum_fallback(self, symbol: str, prices: list[float]) -> BrainSignal:
        """Simple momentum fallback when TimesFM is unavailable."""
        if len(prices) < 20:
            return BrainSignal(
                brain_id=self.brain_id,
                symbol=symbol,
                score=0.0,
                confidence=0.1,
                metadata={"fallback": True, "reason": "insufficient_data"},
            )

        # 20-period rate of change
        roc = (prices[-1] - prices[-20]) / prices[-20] if prices[-20] != 0 else 0.0
        # Map ROC to [-1, +1] with soft saturation
        score = np.tanh(roc * 50)  # amplify small moves, cap at ±1
        confidence = min(0.5, abs(roc) * 25)

        return BrainSignal(
            brain_id=self.brain_id,
            symbol=symbol,
            score=float(score),
            confidence=float(confidence),
            metadata={"fallback": True, "roc_20": round(roc, 6)},
        )
