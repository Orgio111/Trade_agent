"""
Quant Agent: Volatility regime detection and statistical edge scoring.
Routes heavy computation (RL inference) to GPU; indicators to CPU.
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
    LOW    = "LOW"
    MEDIUM = "MEDIUM"
    HIGH   = "HIGH"
    CRISIS = "CRISIS"


def _hurst_exponent(series: np.ndarray, max_lag: int = 20) -> float:
    """Hurst exponent H: >0.5 trending, ~0.5 random walk, <0.5 mean-reverting."""
    lags = range(2, min(max_lag, len(series) // 2))
    tau  = [np.std(np.diff(series, n)) for n in lags]
    if len(tau) < 2 or min(tau) < 1e-10:
        return 0.5
    poly = np.polyfit(np.log([*lags]), np.log(tau), 1)
    return float(poly[0])


def _realized_volatility(returns: np.ndarray, window: int = 20) -> float:
    """Annualized realized volatility."""
    if len(returns) < window:
        return float(returns.std() * np.sqrt(252))
    return float(returns[-window:].std() * np.sqrt(252))


def _garman_klass_vol(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> float:
    """Garman-Klass volatility — more efficient than close-to-close."""
    if len(highs) < 2:
        return 0.0
    log_hl = (np.log(highs / (lows + 1e-10)) ** 2) * 0.5
    log_co = (np.log(closes[1:] / (closes[:-1] + 1e-10)) ** 2) * (2 * np.log(2) - 1)
    n = min(len(log_hl), len(log_co))
    gk = (log_hl[:n] - log_co[:n]).mean()
    return float(np.sqrt(gk * 252))


async def _cpu_quant_analysis(
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
) -> dict:
    """CPU path: pure NumPy statistical computations."""
    returns = np.diff(np.log(closes + 1e-10))
    hurst   = _hurst_exponent(closes)
    rv      = _realized_volatility(returns)
    gk_vol  = _garman_klass_vol(highs, lows, closes)

    # Regime classification
    if rv < 0.15:
        regime = VolatilityRegime.LOW
    elif rv < 0.40:
        regime = VolatilityRegime.MEDIUM
    elif rv < 0.80:
        regime = VolatilityRegime.HIGH
    else:
        regime = VolatilityRegime.CRISIS

    # Statistical edge: prefer trending regimes
    edge_score = hurst - 0.5 if hurst > 0.5 else 0.0

    return {
        "hurst": hurst,
        "realized_vol": rv,
        "gk_vol": gk_vol,
        "regime": regime.value,
        "edge_score": float(edge_score),
    }


_QUANT_SYSTEM = """You are a quantitative researcher specializing in volatility regimes and market microstructure.
Given statistical metrics, output:
{
  "regime_assessment": "<1 sentence>",
  "tradeable": true | false,
  "confidence_adjustment": <float -0.2 to 0.2>,  // adjust the council's confidence
  "reasoning": "<2 sentences>"
}
Conservative bias: mark non-trending, high-volatility regimes as not tradeable."""


async def _gpu_quant_llm(stats: dict, symbol: str) -> dict:
    """GPU path: NIM LLM interprets the statistical output."""
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


class QuantAgent:
    async def analyze(self, symbol: str, closes: np.ndarray, highs: np.ndarray, lows: np.ndarray) -> dict:
        # Route indicator computation to CPU
        stats = await route(
            TaskType.INDICATOR,
            gpu_fn=lambda: _cpu_quant_analysis(closes, highs, lows),
            cpu_fn=lambda: _cpu_quant_analysis(closes, highs, lows),
        )
        # Route LLM interpretation to GPU
        llm_result = await route(
            TaskType.LLM,
            gpu_fn=lambda: _gpu_quant_llm(stats, symbol),
            cpu_fn=lambda: _cpu_quant_fallback(stats),
        )
        return {**stats, **llm_result, "symbol": symbol, "timestamp": datetime.utcnow().isoformat()}


async def _cpu_quant_fallback(stats: dict) -> dict:
    """Lightweight CPU fallback when GPU circuit breaker is open."""
    regime = stats.get("regime", "MEDIUM")
    tradeable = regime not in ("CRISIS",)
    hurst = stats.get("hurst", 0.5)
    adj = (hurst - 0.5) * 0.3
    return {
        "regime_assessment": f"CPU fallback: {regime} regime detected",
        "tradeable": tradeable,
        "confidence_adjustment": adj,
        "reasoning": "GPU unavailable — using statistical heuristic.",
    }
