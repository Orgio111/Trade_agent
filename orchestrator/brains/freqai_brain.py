"""Brain #2: FreqAI / XGBoost Technical Signals.

Uses frequency-domain features + XGBoost for short-term price prediction.
Weight in Go orchestrator: 0.15.
"""

from __future__ import annotations

import logging
import os
import time

import numpy as np

from .base_brain import BaseBrain, BrainSignal

logger = logging.getLogger(__name__)


class FreqAIBrain(BaseBrain):
    """FreqAI-inspired technical indicator brain.

    Computes RSI, MACD, Bollinger Band position, and volume profile
    from recent OHLCV data. Uses either a pre-trained XGBoost model
    or a rule-based scoring system as fallback.
    """

    @property
    def brain_id(self) -> str:
        return "freqai"

    def __init__(self) -> None:
        self._model = None
        self._ohlcv_buffer: list[dict] = []
        self._max_bars = 200

    async def warmup(self) -> None:
        """Attempt to load pre-trained XGBoost model."""
        model_path = os.getenv("FREQAI_MODEL_PATH", "")
        if model_path and os.path.exists(model_path):
            try:
                import joblib
                self._model = joblib.load(model_path)
                logger.info("[freqai] Loaded XGBoost model from %s", model_path)
            except ImportError:
                logger.warning("[freqai] joblib not installed, using rule-based scoring")

    async def compute_score(self, symbol: str) -> BrainSignal:
        bars = self._ohlcv_buffer[-100:]
        if len(bars) < 30:
            return BrainSignal(
                brain_id=self.brain_id,
                symbol=symbol,
                score=0.0,
                confidence=0.1,
                metadata={"reason": "insufficient_ohlcv"},
            )

        closes = np.array([b["close"] for b in bars])
        highs = np.array([b["high"] for b in bars])
        lows = np.array([b["low"] for b in bars])
        volumes = np.array([b["volume"] for b in bars])

        # Compute technical indicators
        rsi = self._compute_rsi(closes, 14)
        macd_hist = self._compute_macd(closes)
        bb_pos = self._compute_bb_position(closes, 20)
        vol_score = self._compute_volume_score(volumes, 20)

        # Rule-based scoring (fallback when no XGBoost model)
        score = 0.0
        confidence = 0.0

        # RSI signal
        if rsi < 30:
            score += 0.3  # oversold → buy
            confidence += 0.2
        elif rsi > 70:
            score -= 0.3  # overbought → sell
            confidence += 0.2
        else:
            score += (50 - rsi) / 100  # mild mean reversion
            confidence += 0.05

        # MACD histogram signal
        if macd_hist > 0:
            score += min(0.2, macd_hist / closes[-1] * 1000)
        else:
            score -= min(0.2, abs(macd_hist) / closes[-1] * 1000)

        # Bollinger Band position
        if bb_pos < 0.2:
            score += 0.15  # near lower band → buy
        elif bb_pos > 0.8:
            score -= 0.15  # near upper band → sell

        # Volume confirmation
        confidence += vol_score * 0.2

        # Clamp and normalize
        score = float(np.clip(score, -1.0, 1.0))
        confidence = float(np.clip(confidence, 0.1, 0.95))

        # Try XGBoost model if available
        if self._model is not None:
            try:
                features = self._extract_features(closes, highs, lows, volumes)
                model_score = float(self._model.predict([features])[0])
                # Blend rule-based and model: 60% model, 40% rules
                score = 0.6 * model_score + 0.4 * score
                confidence = min(0.95, confidence + 0.15)
            except Exception as e:
                logger.warning("[freqai] Model prediction failed: %s", e)

        return BrainSignal(
            brain_id=self.brain_id,
            symbol=symbol,
            score=score,
            confidence=confidence,
            metadata={
                "rsi": round(rsi, 2),
                "macd_hist": round(float(macd_hist), 4),
                "bb_pos": round(bb_pos, 3),
                "model_used": self._model is not None,
            },
        )

    def push_ohlcv(self, bar: dict) -> None:
        """Push a new OHLCV bar into the buffer."""
        self._ohlcv_buffer.append(bar)
        if len(self._ohlcv_buffer) > self._max_bars:
            self._ohlcv_buffer = self._ohlcv_buffer[-self._max_bars:]

    def _compute_rsi(self, closes: np.ndarray, period: int = 14) -> float:
        deltas = np.diff(closes[-period - 1 :])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains) if len(gains) > 0 else 0
        avg_loss = np.mean(losses) if len(losses) > 0 else 1e-10
        rs = avg_gain / avg_loss
        return float(100 - 100 / (1 + rs))

    def _compute_macd(self, closes: np.ndarray) -> float:
        if len(closes) < 26:
            return 0.0
        ema12 = self._ema(closes, 12)
        ema26 = self._ema(closes, 26)
        macd_line = ema12 - ema26
        signal_line = macd_line * 0.2  # simplified signal
        return float(macd_line - signal_line)

    def _compute_bb_position(self, closes: np.ndarray, period: int = 20) -> float:
        if len(closes) < period:
            return 0.5
        window = closes[-period:]
        mean = np.mean(window)
        std = np.std(window)
        if std == 0:
            return 0.5
        return float((closes[-1] - (mean - 2 * std)) / (4 * std))

    def _compute_volume_score(self, volumes: np.ndarray, period: int = 20) -> float:
        if len(volumes) < period:
            return 0.3
        avg_vol = np.mean(volumes[-period:])
        recent_vol = volumes[-1]
        if avg_vol == 0:
            return 0.3
        ratio = recent_vol / avg_vol
        return float(min(1.0, ratio / 3))

    def _ema(self, data: np.ndarray, period: int) -> float:
        if len(data) < period:
            return float(data[-1]) if len(data) > 0 else 0.0
        multiplier = 2 / (period + 1)
        ema = float(np.mean(data[:period]))
        for price in data[period:]:
            ema = (price - ema) * multiplier + ema
        return ema

    def _extract_features(self, closes, highs, lows, volumes) -> list[float]:
        """Extract feature vector for XGBoost model."""
        rsi = self._compute_rsi(closes, 14)
        macd = self._compute_macd(closes)
        bb = self._compute_bb_position(closes, 20)
        vol_score = self._compute_volume_score(volumes, 20)
        roc5 = (closes[-1] - closes[-6]) / closes[-6] if len(closes) > 5 else 0.0
        roc10 = (closes[-1] - closes[-11]) / closes[-11] if len(closes) > 10 else 0.0
        hl_spread = (highs[-1] - lows[-1]) / closes[-1] if closes[-1] > 0 else 0.0
        return [rsi / 100, macd, bb, vol_score, roc5, roc10, hl_spread]
