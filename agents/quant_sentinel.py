"""
Quant Agent — Volatility regime detection and statistical edge scoring.

Ported from Sentinel-X (sentinel-x/python/agents/quant.py).

Provides:
- Hurst exponent for trend/mean-reversion detection
- Garman-Klass volatility (efficient, uses OHLC)
- Realised volatility (close-to-close)
- Regime classification (LOW / MEDIUM / HIGH / CRISIS)
- Statistical edge scoring
- Tradeability assessment via NIM LLM through the scheduler circuit breaker
"""

from __future__ import annotations

import logging
from datetime import datetime
from enum import Enum

import numpy as np

from core.nim_client import nim_json
from core.scheduler import TaskType, route

log = logging.getLogger(__name__)


class VolatilityRegime(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRISIS = "CRISIS"


# ── Statistical indicators (CPU) ──────────────────────────────────────────────


def hurst_exponent(series: np.ndarray, max_lag: int = 20) -> float:
    """Hurst exponent H.

    - H > 0.5 → trending
    - H ≈ 0.5 → random walk
    - H < 0.5 → mean-reverting
    """
    lags = range(2, min(max_lag, len(series) // 2))
    tau = [np.std(np.diff(series, n)) for n in lags]
    if len(tau) < 2 or min(tau) < 1e-10:
        return 0.5
    poly = np.polyfit(np.log([*lags]), np.log(tau), 1)
    return float(poly[0])


def realised_volatility(returns: np.ndarray, window: int = 20) -> float:
    """Annualised realised volatility from close-to-close returns."""
    if len(returns) < window:
        return float(returns.std() * np.sqrt(252))
    return float(returns[-window:].std() * np.sqrt(252))


def garman_klass_vol(
    highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
) -> float:
    """Garman-Klass volatility estimator — more efficient than close-to-close.

    Uses OHLC data to estimate volatility with ~7× the efficiency of
    close-to-close under ideal conditions.
    """
    if len(highs) < 2:
        return 0.0
    log_hl = np.log(highs / (lows + 1e-10)) ** 2 * 0.5
    log_co = np.log(closes[1:] / (closes[:-1] + 1e-10)) ** 2 * (2 * np.log(2) - 1)
    n = min(len(log_hl), len(log_co))
    gk = (log_hl[:n] - log_co[:n]).mean()
    return float(np.sqrt(max(gk, 0.0) * 252))


def classify_regime(rv: float) -> VolatilityRegime:
    if rv < 0.15:
        return VolatilityRegime.LOW
    elif rv < 0.40:
        return VolatilityRegime.MEDIUM
    elif rv < 0.80:
        return VolatilityRegime.HIGH
    return VolatilityRegime.CRISIS


# ── CPU-only statistical pass ─────────────────────────────────────────────────


async def _cpu_quant_analysis(
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
) -> dict:
    """NumPy-only statistical computation (always CPU)."""
    returns = np.diff(np.log(closes + 1e-10))
    hurst = hurst_exponent(closes)
    rv = realised_volatility(returns)
    gk_vol = garman_klass_vol(highs, lows, closes)
    regime = classify_regime(rv)

    # Statistical edge: prefer trending regimes
    edge_score = hurst - 0.5 if hurst > 0.5 else 0.0

    return {
        "hurst": hurst,
        "realized_vol": rv,
        "gk_vol": gk_vol,
        "regime": regime.value,
        "edge_score": float(edge_score),
    }


# ── GPU LLM interpretation ────────────────────────────────────────────────────

_QUANT_SYSTEM = """You are a quantitative researcher specialising in volatility regimes and market microstructure.
Given statistical metrics, output:
{
  "regime_assessment": "<1 sentence>",
  "tradeable": true | false,
  "confidence_adjustment": <float -0.2 to 0.2>,  // adjust the council's confidence
  "reasoning": "<2 sentences>"
}
Conservative bias: mark non-trending, high-volatility regimes as not tradeable."""


async def _gpu_quant_llm(stats: dict, symbol: str) -> dict:
    """NIM LLM interprets statistical regime output (GPU path)."""
    return await nim_json(
        [
            {"role": "system", "content": _QUANT_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Symbol: {symbol}\n"
                    f"Hurst={stats['hurst']:.3f} | RV={stats['realized_vol']:.1%} | "
                    f"GK_Vol={stats['gk_vol']:.1%} | Regime={stats['regime']}\n"
                    "Assess tradeability."
                ),
            },
        ],
        temperature=0.1,
    )


def _cpu_quant_fallback(stats: dict) -> dict:
    """Lightweight CPU fallback when GPU circuit breaker is open."""
    regime = stats.get("regime", "MEDIUM")
    tradeable = regime not in ("CRISIS",)
    hurst = stats.get("hurst", 0.5)
    adj = (hurst - 0.5) * 0.3
    return {
        "regime_assessment": f"CPU fallback: {regime} regime detected",
        "tradeable": tradeable,
        "confidence_adjustment": float(adj),
        "reasoning": "GPU unavailable — using statistical heuristic.",
    }


# ── Public API ────────────────────────────────────────────────────────────────


class QuantSentinelAgent:
    """Enhanced quant agent with Hurst, GK-Vol, regime detection, and GPU/CPU routing.

    Usage::

        agent = QuantSentinelAgent()
        result = await agent.analyze("BTC/USDT", closes, highs, lows)
        # result contains: hurst, realised_vol, gk_vol, regime, edge_score,
        #                 tradeable, confidence_adjustment, reasoning
    """

    async def analyze(
        self,
        symbol: str,
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
    ) -> dict:
        # 1. Route indicator computation to CPU (always CPU)
        stats = await route(
            TaskType.INDICATOR,
            gpu_fn=lambda: _cpu_quant_analysis(closes, highs, lows),
            cpu_fn=lambda: _cpu_quant_analysis(closes, highs, lows),
        )

        # 2. Route LLM interpretation through the circuit breaker
        llm_result = await route(
            TaskType.LLM,
            gpu_fn=lambda: _gpu_quant_llm(stats, symbol),
            cpu_fn=lambda: _cpu_quant_fallback(stats),
        )

        return {
            **stats,
            **llm_result,
            "symbol": symbol,
            "timestamp": datetime.utcnow().isoformat(),
        }
