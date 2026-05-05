"""Technical Analysis Agent: RSI, EMA, ATR, MACD, Bollinger Bands."""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime

import numpy as np

from core.config import get_settings
from core.messaging import MsgType, get_bus
from core.models import Side, TechnicalSignal
from core.observability import AGENT_LATENCY, SIGNAL_COUNTER

logger = logging.getLogger(__name__)


def _ema(prices: np.ndarray, period: int) -> np.ndarray:
    alpha = 2.0 / (period + 1)
    out = np.empty_like(prices)
    out[0] = prices[0]
    for i in range(1, len(prices)):
        out[i] = alpha * prices[i] + (1 - alpha) * out[i - 1]
    return out


def _rsi(prices: np.ndarray, period: int = 14) -> float:
    if len(prices) < period + 1:
        return 50.0
    deltas = np.diff(prices[-(period + 1) :])
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = gains.mean() + 1e-10
    avg_loss = losses.mean() + 1e-10
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> float:
    if len(highs) < period + 1:
        return float(np.std(closes[-20:]) * 2) if len(closes) >= 2 else 1.0
    tr = np.maximum(
        highs[-period:] - lows[-period:],
        np.maximum(
            np.abs(highs[-period:] - closes[-(period + 1) : -1]),
            np.abs(lows[-period:] - closes[-(period + 1) : -1]),
        ),
    )
    return float(tr.mean())


def _bollinger(prices: np.ndarray, period: int = 20, std_dev: float = 2.0) -> tuple[float, float]:
    if len(prices) < period:
        mid = prices.mean()
        std = prices.std()
    else:
        mid = prices[-period:].mean()
        std = prices[-period:].std()
    return float(mid + std_dev * std), float(mid - std_dev * std)


def _macd(
    prices: np.ndarray,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[float, float]:
    if len(prices) < slow:
        return 0.0, 0.0
    ema_fast = _ema(prices, fast)
    ema_slow = _ema(prices, slow)
    macd_line = ema_fast - ema_slow
    sig_line = _ema(macd_line, signal)
    return float(macd_line[-1]), float(sig_line[-1])


class TechnicalAgent:
    """Computes technical indicators and emits TechnicalSignal to the bus."""

    async def analyze(
        self,
        symbol: str,
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        volumes: np.ndarray,
    ) -> TechnicalSignal:
        with AGENT_LATENCY.labels(agent="technical").time():
            rsi = _rsi(closes)
            ema_fast = float(_ema(closes, 12)[-1])
            ema_slow = float(_ema(closes, 26)[-1])
            atr = _atr(highs, lows, closes)
            macd, macd_sig = _macd(closes)
            bb_upper, bb_lower = _bollinger(closes)
            vol_ratio = float(volumes[-1] / (volumes[-20:].mean() + 1e-10)) if len(volumes) >= 20 else 1.0

            # Scoring: +1 bull, -1 bear per indicator
            score = 0.0
            score += 1.0 if ema_fast > ema_slow else -1.0
            score += 1.0 if rsi < 30 else (-1.0 if rsi > 70 else 0.0)
            score += 1.0 if macd > macd_sig else -1.0
            score += 0.5 if vol_ratio > 1.5 else 0.0

            norm_score = score / 3.5  # normalize to [-1, 1]
            trend = Side.BUY if norm_score > 0.15 else (Side.SELL if norm_score < -0.15 else Side.HOLD)
            confidence = min(abs(norm_score), 1.0)

            signal = TechnicalSignal(
                symbol=symbol,
                timestamp=datetime.utcnow(),
                rsi_14=rsi,
                ema_fast=ema_fast,
                ema_slow=ema_slow,
                atr_14=atr,
                macd=macd,
                macd_signal=macd_sig,
                bb_upper=bb_upper,
                bb_lower=bb_lower,
                volume_ratio=vol_ratio,
                trend=trend,
                confidence=confidence,
            )

        cfg = get_settings()
        bus = await get_bus()
        await bus.publish(
            cfg.stream_signals, MsgType.TECHNICAL_SIGNAL, signal.model_dump(mode="json")
        )
        SIGNAL_COUNTER.labels(agent="technical", symbol=symbol, side=trend.value).inc()
        logger.debug("Technical signal %s → %s (conf=%.2f)", symbol, trend.value, confidence)
        return signal
