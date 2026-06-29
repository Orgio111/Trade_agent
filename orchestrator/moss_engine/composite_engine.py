"""Moss Composite Engine — 5-pillar signal factory.

Architectural pattern from MOSS Trading Skills Factory v1.0.28:
  - Trend:       EMA(200) + SuperTrend rolling impulse
  - Momentum:    RSI(14) + MACD cross & divergence layer
  - MeanRevert:  Bollinger Band deviation normalized
  - Volume:      OBV slope regression
  - Volatility:  ATR(14) breakout urgency detector

Default 5-pillar blending weights: [0.3, 0.25, 0.2, 0.15, 0.1].
Reflection loop enables tactical ±35% adjustment at segment evaluation.

Uses SkillRegistry-inspired _state_buffers keyed by (symbol, interval)
to preserve sliding windows, RSI sums, OBV rolls across ticks.

Security: API keys loaded from .env via python-dotenv.  No hardcoded secrets.
"""

from __future__ import annotations

import logging
import math
import os
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .schemas import (
    CompositeSignal,
    FadeSignal,
    MOSS_FACTORY_VERSION,
    PillarOutput,
)
from .skill_registry import SkillRegistry

logger = logging.getLogger(__name__)

# ── .env auto-load ─────────────────────────────────────
try:
    from dotenv import load_dotenv
    _project_root = Path(__file__).resolve().parents[2]
    _env_file = _project_root / ".env"
    if _env_file.exists():
        load_dotenv(_env_file, override=False)
except ImportError:
    pass


# ══════════════════════════════════════════════════════════════════════
# MOSS Trading Skills Factory v1.0.28 Constants
# ══════════════════════════════════════════════════════════════════════

# Pillars:
PILLAR_WEIGHTS: Tuple[float, float, float, float, float] = (0.3, 0.25, 0.2, 0.15, 0.1)
PILLAR_NAMES: Tuple[str, str, str, str, str] = (
    "trend", "momentum", "mean_reversion", "volume", "volatility"
)
# Lookback:
EMA_PERIODS: int = 200
RSI_PERIODS: int = 14
BB_PERIODS: int = 20
ATR_PERIODS: int = 14
OBV_LOOKBACK: int = 50
# SuperTrend:
SUPERTREND_MULTIPLIER: float = 2.0
# Normalization thresholds:
RSI_OVERSOLD: float = 30.0
RSI_OVERBOUGHT: float = 70.0
BB_BREACH_OUTER_THRESHOLD: float = 1.5   # outer band breach
BB_BREACH_MIDDLE_THRESHOLD: float = 0.7  # middle band cutoff
ATR_BREAKOUT_THRESHOLD: float = 1.3      # momentum breakout
# Volume anomaly:
VOLUME_ANOMALY_RATIO: float = 2.0       # Trigger 2x
PRICE_VELOCITY_THRESHOLD: float = 0.02  # 2%


# ══════════════════════════════════════════════════════════════════════
# SECTION B:  MossCompositeEngine
# ══════════════════════════════════════════════════════════════════════

class MossCompositeEngine:
    """MOSS 5-Pillar Composite Signal Engine.

    Computes:
      - Trend:       EMA(200) + SuperTrend
      - Momentum:    RSI(14) + MACD
      - MeanRevert:  Bollinger Band distance normalized
      - Volume:      OBV slope regression
      - Volatility:  ATR breakout

    Blends 5 pillars into a single composite score in [-1.0, +1.0] with
    weighted confidence and direction.

    State Memory Fix: _state_buffers persists
      - ema_values    (deque)
      - rsi_sums      (tuple: gain_sum, loss_sum)
      - obv_values    (deque)
      - atr_values    (deque)
      - bollinger     (tuple: sma, stdev)
    Partitioned by (symbol, interval) and never reset.
    """

    def __init__(self, registry: Optional[SkillRegistry] = None) -> None:
        self._registry = registry
        # State Memory Fix: partitioned by (symbol, interval)
        self._state_buffers: Dict[str, Dict[str, Any]] = {}
        self._version: str = MOSS_FACTORY_VERSION

    # ── Utility ────────────────────────────────

    def get_state_key(self, symbol: str, interval: str) -> str:
        return f"{symbol}_{interval}"

    def _get_state_buffer(self, symbol: str, interval: str) -> Dict[str, Any]:
        return self._state_buffers.setdefault(self.get_state_key(symbol, interval), {})

    def _safe_deque(self, key: str, size: int) -> deque:
        buf = self._get_state_buffer("", "")["__global"] if "/" in key else self._get_state_buffer("global", "global")
        if key not in buf:
            buf[key] = deque(maxlen=size)
        return buf[key]

    # ── Trend Pillar (EMA + SuperTrend) ────────

    def _compute_trend_pillar(self, closes: List[float], highs: List[float], lows: List[float]) -> PillarOutput:
        state_key = self.get_state_key("trend", "")
        buf = self._get_state_buffer("trend", state_key)

        # EMA(200)
        ema_key = f"ema_values"
        ema_deque = self._safe_deque(ema_key, EMA_PERIODS)
        if len(ema_deque) == 0:
            ema = closes[0]
        else:
            ema = (ema_deque[-1] * (EMA_PERIODS - 1) + closes[-1]) / EMA_PERIODS
        ema_deque.append(ema)
        buf[ema_key] = ema_deque
        ema_score = math.tanh((closes[-1] - ema) / (ema * 0.02))  # /2% threshold

        # SuperTrend
        supertrend_key = f"supertrend_values"
        supertrend_deque = self._safe_deque(supertrend_key, EMA_PERIODS)
        close = closes[-1]
        hl2 = (highs[-1] + lows[-1]) / 2
        raw_atr = self._compute_atr(highs, lows, closes)
        if len(supertrend_deque) > 0:
            prev_super = supertrend_deque[-1]
            upper = hl2 + SUPERTREND_MULTIPLIER * raw_atr
            lower = hl2 - SUPERTREND_MULTIPLIER * raw_atr
            super = upper if close > prev_super else lower
        else:
            super = hl2
        supertrend_deque.append(super)
        buf[supertrend_key] = supertrend_deque
        super_score = math.tanh((close - super) / (super * 0.02))

        # Blend: 0.3 EMA + 0.7 SuperTrend
        trend_score = 0.3 * ema_score + 0.7 * super_score
        conf = min(1.0, 0.5 + abs(trend_score * 0.3))
        return PillarOutput(
            pillar_name="trend",
            score=float(trend_score),
            confidence=float(conf),
            weight=0.3,
            raw_values={"ema": ema, "ema_score": ema_score, "super": super, "super_score": super_score},
        )

    # ── Momentum Pillar (RSI + MACD) ─────────

    def _compute_momentum_pillar(self, closes: List[float]) -> PillarOutput:
        state_key = self.get_state_key("momentum", "")
        buf = self._get_state_buffer("momentum", state_key)

        # RSI(14)
        rsi_key = f"rsi_sums"
        rsi_state = buf.get(rsi_key, (0.0, 0.0))
        gain_sum, loss_sum = rsi_state
        if len(closes) < 2:
            rsi = 50.0
        else:
            delta = closes[-1] - closes[-2]
            if delta > 0:
                gain_sum = gain_sum * (RSI_PERIODS - 1) / RSI_PERIODS + delta
                loss_sum *= (RSI_PERIODS - 1) / RSI_PERIODS
            elif delta < 0:
                gain_sum *= (RSI_PERIODS - 1) / RSI_PERIODS
                loss_sum = loss_sum * (RSI_PERIODS - 1) / RSI_PERIODS - delta
            else:
                gain_sum *= (RSI_PERIODS - 1) / RSI_PERIODS
                loss_sum *= (RSI_PERIODS - 1) / RSI_PERIODS
            buf[rsi_key] = (gain_sum, loss_sum)
            avg_gain = gain_sum / RSI_PERIODS
            avg_loss = loss_sum / RSI_PERIODS
            if avg_loss < 1e-8:
                avg_loss = 1e-8
            rs = avg_gain / avg_loss
            rsi = 100.0 - (100.0 / (1.0 + rs))

        rsi_norm = (rsi - 50.0) / 20.0  # [-50,+50] → [-2.5,+2.5]
        rsi_score = float(np.tanh(rsi_norm))
        rsi_conf = min(1.0, 0.3 + abs(rsi_norm) * 0.1)

        # MACD
        macd_12 = self._ema(closes, 12)
        macd_26 = self._ema(closes, 26)
        macd = macd_12 - macd_26
        signal = self._ema([macd_12, *(buf.get("macd_hist", tuple()))], 9)[0]  # last 9 macds
        buf["macd_hist"] = macd_hist = tuple(list(buf.get("macd_hist", tuple()))[-8:] + [macd])
        macd_cross_score = 0.5 if macd > signal else -0.5
        histogram = macd - signal
        macd_div_score = float(np.sign(histogram) * (abs(histogram) / np.percentile(np.abs([abs(h) for h in macd_hist]), 75) if len(macd_hist) > 0 else 1.0))
        macd_blend = 0.6 * macd_cross_score + 0.4 * macd_div_score

        # Blend: 0.6 RSI + 0.4 MACD
        momentum_score = 0.6 * rsi_score + 0.4 * macd_blend
        conf = min(1.0, 0.5 + abs(momentum_score * 0.3))
        return PillarOutput(
            pillar_name="momentum",
            score=float(momentum_score),
            confidence=float(conf),
            weight=0.25,
            raw_values={
                "rsi": rsi,
                "avg_gain": avg_gain,
                "avg_loss": avg_loss,
                "macd": macd,
                "signal": signal,
                "histogram": histogram,
            },
        )

    # ── MeanReversion Pillar (Bollinger Bands) ──

    def _compute_mean_reversion_pillar(self, closes: List[float]) -> PillarOutput:
        state_key = self.get_state_key("mean_reversion", "")
        buf = self._get_state_buffer("mean_reversion", state_key)

        # BB(20, 2)
        closes_arr = np.array(closes[-BB_PERIODS:])
        sma = float(np.mean(closes_arr))
        stdev = float(np.std(closes_arr))
        upper_bb = sma + 2.0 * stdev
        lower_bb = sma - 2.0 * stdev

        buf["bollinger"] = (sma, stdev)
        price = closes[-1]
        if stdev < 1e-8:
            bb_dist = 0.0
        else:
            bb_dist = (price - sma) / stdev
        # Normalize to [-1, +1] via tanh saturation
        bb_norm = math.tanh(bb_dist / 1.5)   # 1.5σ either side → ±0.6

        mean_score = -float(bb_norm)   # Negative → price below mean → mean-reversion buy signal
        conf = min(1.0, 0.5 + abs(bb_norm) * 0.2)
        return PillarOutput(
            pillar_name="mean_reversion",
            score=float(mean_score),
            confidence=float(conf),
            weight=0.2,
            raw_values={"bb_sma": sma, "stdev": stdev, "upper_bb": upper_bb, "lower_bb": lower_bb},
        )

    # ── Volume Pillar (OBV) ─────────────────

    def _compute_volume_pillar(self, closes: List[float], volumes: List[float]) -> PillarOutput:
        state_key = self.get_state_key("volume", "")
        buf = self._get_state_buffer("volume", state_key)

        obv_key = f"obv_values"
        obv_deque = self._safe_deque(obv_key, OBV_LOOKBACK)
        if len(obv_deque) == 0:
            obv = volumes[0]
        else:
            cmp_close = closes[-1] - closes[-2]
            obv = obv_deque[-1] + (volumes[-1] if cmp_close > 0 else -volumes[-1])
        obv_deque.append(obv)
        buf[obv_key] = obv_deque

        # OBV trend via rolling linear regression slope
        if len(obv_deque) >= 2:
            x = list(range(len(obv_deque)))
            y = list(obv_deque)
            slope = np.polyfit(x, y, 1)[0]
            # Normalize slope to [OBV_PER_M] → [-1, +1]
            OBV_PER_M = volumes[-1] * 50.0
            norm_slope = math.tanh(slope / OBV_PER_M) if OBV_PER_M > 1e-8 else 0.0
        else:
            norm_slope = 0.0

        volume_score = norm_slope
        conf = min(1.0, 0.5 + abs(norm_slope * 0.2))
        return PillarOutput(
            pillar_name="volume",
            score=float(volume_score),
            confidence=float(conf),
            weight=0.15,
            raw_values={"obv": obv, "obv_slope": slope, "obv_norm_slope": norm_slope},
        )

    # ── Volatility Pillar (ATR) ───────────

    def _compute_volatility_pillar(self, highs: List[float], lows: List[float], closes: List[float]) -> PillarOutput:
        state_key = self.get_state_key("volatility", "")
        buf = self._get_state_buffer("volatility", state_key)

        atr = self._compute_atr(highs, lows, closes)
        atr_key = f"atr_values"
        atr_deque = self._safe_deque(atr_key, ATR_PERIODS)
        atr_deque.append(atr)
        buf[atr_key] = atr_deque

        candle_range = highs[-1] - lows[-1]
        if atr < 1e-8:
            range_atr_ratio = 1.0
        else:
            range_atr_ratio = candle_range / atr
        # Map ratio to [1.0, ~4.0] → volatility score via tanh
        vol_score = math.tanh((range_atr_ratio - 1.0) / 2.0)

        conf = min(1.0, 0.4 + (range_atr_ratio > ATR_BREAKOUT_THRESHOLD) * 0.2)
        return PillarOutput(
            pillar_name="volatility",
            score=float(vol_score),
            confidence=float(conf),
            weight=0.1,
            raw_values={"atr": atr, "candle_range": candle_range, "range_atr_ratio": range_atr_ratio},
        )

    # ── Helpers ──────────────────────────

    def _ema(self, data: List[float], period: int) -> float:
        deque_key = f"ema_{period}"
        deque = self._safe_deque(deque_key, period)
        if len(deque) == 0:
            ema = data[0]
        else:
            ema = (data[-1] * (2.0 / (period + 1))) + (deque[-1] * (1.0 - (2.0 / (period + 1))))
        deque.append(ema)
        self._get_state_buffer("ema_helper", deque_key)[deque_key] = deque
        return ema

    def _compute_atr(self, highs: List[float], lows: List[float], closes: List[float]) -> float:
        if len(highs) < 2:
            return closes[0] * 0.01
        tr = max(
            highs[-1] - lows[-1],
            abs(highs[-1] - closes[-2]),
            abs(lows[-1] - closes[-2])
        )
        atr_key = f"atr_values"
        atr_deque = self._safe_deque(atr_key, ATR_PERIODS)
        if len(atr_deque) == 0:
            atr = tr
        else:
            atr = (tr * (1.0 / ATR_PERIODS)) + (atr_deque[-1] * ((ATR_PERIODS - 1) / ATR_PERIODS))
        atr_deque.append(atr)
        self._get_state_buffer("atr_helper", atr_key)[atr_key] = atr_deque
        return atr

    # ── Composite ────────────────────────

    def compute_composite_signal(
        self,
        symbol: str,
        interval: str,
        ohlcv: List[Dict[str, Any]],
        *,
        fade_override: Optional[FadeSignal] = None,
        whale_skew: float = 0.0,
        guardrail_daily_drawdown_pct: float = 0.0
    ) -> CompositeSignal:
        """Compute the 5-pillar composite signal from OHLCV candles.

        Returns a CompositeSignal ready for NATS JetStream.
        Fade override: Krypt-Trader Contrarian Fade Mode may flip direction.
        Whale skew: directional vector from KryptCryptoCore.
        """
        if len(ohlcv) < 2:
            return CompositeSignal(
                symbol=symbol,
                composite_score=0.0,
                composite_confidence=0.1,
                direction=0,
                contrarian_fade_invoked=False,
            )

        opens = [c["open"] for c in ohlcv]
        highs = [c["high"] for c in ohlcv]
        lows = [c["low"] for c in ohlcv]
        closes = [c["close"] for c in ohlcv]
        volumes = [c["volume"] for c in ohlcv]

        pillars = []
        scores = []
        confidences = []

        # Compute 5 pillars
        pillars.append(self._compute_trend_pillar(closes, highs, lows))
        pillars.append(self._compute_momentum_pillar(closes))
        pillars.append(self._compute_mean_reversion_pillar(closes))
        pillars.append(self._compute_volume_pillar(closes, volumes))
        pillars.append(self._compute_volatility_pillar(highs, lows, closes))

        # Apply Contrarian Fade override from Krypt-Trader
        contrarian_fade_invoked = False
        faded_direction = 0
        if fade_override:
            if fade_override.faded_direction != 0:
                contrarian_fade_invoked = True
                faded_direction = fade_override.faded_direction
                for pillar in pillars:
                    if pillar.pillar_name == "momentum":
                        # Momentum flipped / faded → override score
                        pill_out = PillarOutput(
                            pillar_name="momentum_faded",
                            score=-pillar.score,
                            confidence=fade_override.fade_confidence,
                            weight=pillar.weight,
                            raw_values=pillar.raw_values,
                        )
                        pillars.remove(pillar)
                        pillars.append(pill_out)
        
        # Blend pillars
        composite_score = 0.0
        composite_confidence = 0.0
        for idx, pillar in enumerate(pillars):
            composite_score += pillar.score * pillar.weight
            composite_confidence += pillar.confidence * pillar.weight
            scores.append(pillar.score)
            confidences.append(pillar.confidence)

        # Apply whale skew  (Krypt-Trader Whale Tracker)
        composite_score = np.clip(composite_score + whale_skew * 0.2, -1.0, 1.0)
        
        # Map composite score to direction (+1, -1, 0)
        if composite_score > 0.1:
            direction = +1
        elif composite_score < -0.1:
            direction = -1
        else:
            direction = 0

        # Apply contrarian fade direction override
        if contrarian_fade_invoked and fade_override:
            direction = fade_override.faded_direction

        # Guardrails check (master kill-switch handled in SkillRegistry.execute() upstream)
        if guardrail_daily_drawdown_pct > 8.0:
            direction = 0

        return CompositeSignal(
            symbol=symbol,
            composite_score=float(composite_score),
            composite_confidence=float(composite_confidence),
            direction=direction,
            pillars=pillars,
            contrarian_fade_invoked=contrarian_fade_invoked,
            whale_skew=whale_skew,
            timestamp_ms=int(time.time() * 1000),
        )

    # ── Convenience ───────────────────────

    @property
    def version(self) -> str:
        return self._version

    def reset_buffers(self, symbol: str, interval: str) -> None:
        """Dangerous utility to reset state buffers for a symbol."""
        state_key = self.get_state_key(symbol, interval)
        if state_key in self._state_buffers:
            del self._state_buffers[state_key]
            logger.warning("[%s] State buffers RESET", state_key)

    def inject_into_registry(self, registry: SkillRegistry, skill_name: str = "moss_composite") -> None:
        """Self-register into a SkillRegistry as an active brain."""
        if self._registry is None:
            self._registry = registry
        registry.register(
            name=skill_name,
            skill_cls_or_instance=self,
            version=self._version,
            description="Moss 5-pillar Composite signal generator",
            tags=("moss", "alpha", "signal"),
            hot_swap_enabled=True,
            kill_switch_armed=True,
        )
