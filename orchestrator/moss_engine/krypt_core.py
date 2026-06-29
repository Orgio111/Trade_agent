"""KryptCryptoCore — Krypt-Trader-inspired microstructure engine.

Heuristics:
  1. Whale Tracker / Order Flow Imbalance (OFI)
     - Sliding 100-trade window
     - Side = "buy" → +qty, "sell" → -qty
     - Net-volume variance score → directional skew
     - Urgent vector: $100k+ notional → pass skew to composite

  2. 15-Minute Momentum Scanner
     - Current bar volume > 2.0x rolling 20-period avg → volume anomaly
     - Expansionary price velocity → gate trigger

  3. Contrarian Fade Mode
     - RSI extreme + price beyond outer BB → invert composite direction
     - "Fading the crowd" → mean-reversion signal

Security: No API keys required (uses incoming trades from Binance WebSocket).
"""

from __future__ import annotations

import logging
import math
import os
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .schemas import (
    CompositeSignal,
    FadeSignal,
    MomentumScanResult,
    MOSS_FACTORY_VERSION,
    PillarOutput,
    WhaleEvent,
)
from .skill_registry import MossSkill, SkillRegistry

logger = logging.getLogger(__name__)

# ── Krypt-Trader Constants ─────────────

# Whale Tracker
OFI_WINDOW: int = 100               # rolling 100 trades
MIN_WHALE_NOTIONAL: float = 100_000  # $100k => urgent vector

# Momentum Scanner
VOLUME_LOOKBACK: int = 20           # 20 periods rolling
VOLUME_RATIO_THRESHOLD: float = 2.0 # current volume > 2.0x rolling avg
PRICE_VELOCITY_THRESHOLD: float = 0.02 # 2% candle

# Contrarian Fade
RSI_EXTREME_HIGH: float = 80.0      # overbought
RSI_EXTREME_LOW: float = 20.0       # oversold
BB_MULTIPLIER: float = 2.0          # outer band multiplier
BB_PERIODS: int = 20                 # bollinger period


# ══════════════════════════════════════════════════════════════════════
# MODULE 1:  KryptCryptoCore
# ══════════════════════════════════════════════════════════════════════

class KryptCryptoCore(MossSkill):
    """Krypt-Trader-inspired microstructure engine.

    Exposes 3 real-time heuristics:
    - Whale Tracker: OFI directional skew
    - 15m Momentum Scanner: volume+velocity anomaly
    - Contrarian Fade: RSI+BB inversion

    Stateless design — persistent state buffers managed upstream.
    """

    @property
    def skill_name(self) -> str:
        return "krypt_crypto_core"

    def __init__(self) -> None:
        self._version: str = MOSS_FACTORY_VERSION

    # ── Whale Tracker / OFI ───────────────

    def _compute_ofi(self, trades: List[Dict[str, Any]], symbol_price: float = 0.0) -> Tuple[float, float, List[WhaleEvent]]:
        """Order Flow Imbalance: rolling 100-trade variance score + whale events.

        Returns (ofi_score: float ∈ [-1.0, +1.0], directional_skew: float, whale_events)
        """
        window = min(OFI_WINDOW, len(trades))
        trades_window = trades[-window:]
        
        # Classify trades and compute OFI
        ofi_values: List[float] = []
        whale_events: List[WhaleEvent] = []
        
        for trade in trades_window:
            price = float(trade.get("p", symbol_price))
            qty = float(trade.get("q", 0.0))
            side = trade.get("side", trade.get("m", "buy")).lower()
            notional = price * qty
            
            # OFI accumulation
            if side in ("buy", "B"):
                ofi_delta = +qty
            elif side in ("sell", "A"):
                ofi_delta = -qty
            else:
                ofi_delta = 0.0
            ofi_values.append(ofi_delta)
            
            # Whale check
            if notional >= MIN_WHALE_NOTIONAL:
                whale_events.append(
                    WhaleEvent(
                        symbol=trade.get("symbol", ""),
                        price=price,
                        quantity=qty,
                        notional_usd=notional,
                        side=side,
                        is_urgent=True,
                    )
                )

        if not ofi_values:
            return 0.0, 0.0, []
        
        # OFI rolling variance
        ofi_array = np.array(ofi_values)
        rolling_mean = float(np.mean(ofi_array))
        rolling_var = float(np.var(ofi_array))
        
        # Normalize variance to directional skew (-1..+1)
        skew_factor = math.tanh(rolling_mean / math.sqrt(rolling_var + 1e-8)) if rolling_var > 1e-8 else 0.0
        
        # Map variance → score
        ofi_score = float(np.tanh((rolling_var - np.percentile(ofi_values, 50)) / 1000.0))
        
        return ofi_score, skew_factor, whale_events

    def _whale_tracker_invoke(
        self,
        symbol: str,
        trades: List[Dict[str, Any]],
        price: float = 0.0
    ) -> Tuple[float, float, List[WhaleEvent]]:
        """Entry point for Whale Tracker."""
        return self._compute_ofi(trades, price)

    # ── 15m Momentum Scanner ────────────

    def _momentum_scanner_invoke(
        self,
        symbol: str,
        interval: str,
        ohlcv: List[Dict[str, Any]],
        volumes: List[float],
        closes: List[float]
    ) -> MomentumScanResult:
        """15-minute momentum scanner: volume anomaly + price velocity."""
        if len(volumes) < VOLUME_LOOKBACK:
            return MomentumScanResult(
                symbol=symbol,
                timeframe=interval,
                current_volume=0.0,
                avg_volume_20=0.0,
                volume_ratio=0.0,
                price_velocity=0.0,
                momentum_triggered=False,
            )

        rolling_avg = float(np.mean(volumes[-VOLUME_LOOKBACK:]))
        current_volume = volumes[-1]
        volume_ratio = current_volume / (rolling_avg + 1e-8)
        
        # Price velocity
        open_price = ohlcv[-1]["open"]
        close_price = closes[-1]
        price_velocity = abs((close_price - open_price) / open_price)

        # Trigger gate
        volume_trigger = volume_ratio >= VOLUME_RATIO_THRESHOLD
        velocity_trigger = price_velocity >= PRICE_VELOCITY_THRESHOLD
        momentum_triggered = volume_trigger and velocity_trigger

        return MomentumScanResult(
            symbol=symbol,
            timeframe=interval,
            current_volume=current_volume,
            avg_volume_20=rolling_avg,
            volume_ratio=volume_ratio,
            price_velocity=price_velocity,
            momentum_triggered=momentum_triggered,
        )

    # ── Contrarian Fade Mode ─────────────

    def _fade_detector_invoke(
        self,
        symbol: str,
        closes: List[float],
        rsi_value: float,
        bb_upper: float,
        bb_lower: float,
        price: float
    ) -> FadeSignal:
        """Contrarian Fade detector: RSI extreme + BB breach → invert composite."""
        # RSI extreme
        rsi_low = rsi_value <= RSI_EXTREME_LOW
        rsi_high = rsi_value >= RSI_EXTREME_HIGH
        rsi_extreme = rsi_low or rsi_high
        
        # BB breach
        bb_high_breach = price > bb_upper          # outer BB high
        bb_low_breach = price < bb_lower           # outer BB low
        bb_breach = bb_high_breach or bb_low_breach
        
        # Original composite direction sign (-1, +1)
        # (assumed from momentum pillar sign)
        original_direction = +1 if rsi_value > 50 else -1
        
        # Fade logic
        if (rsi_extreme and bb_breach):
            faded_direction = -original_direction
        else:
            faded_direction = 0
        
        # Confidence mapping
        if (rsi_low and bb_low_breach) or (rsi_high and bb_high_breach):
            fade_conf = 0.9
        elif rsi_extreme or bb_breach:
            fade_conf = 0.6
        else:
            fade_conf = 0.0

        return FadeSignal(
            symbol=symbol,
            rsi_extreme=rsi_extreme,
            bb_breach=bb_breach,
            original_direction=original_direction,
            faded_direction=faded_direction,
            fade_confidence=fade_conf,
        )

    # ── MossSkill Interface ─────────────

    def execute(
        self,
        symbol: str,
        interval: str,
        ohlcv: List[Dict[str, Any]],
        trades: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """KryptCryptoCore skill execution.

        Returns composite-ready microstructure dict:
          {
            "ofi_score": float,
            "directional_skew": float,
            "whale_events": List[WhaleEvent],
            "momentum_scan": MomentumScanResult,
            "fade_signal": FadeSignal
          }
        """
        if len(ohlcv) < 2:
            return {
                "ofi_score": 0.0,
                "directional_skew": 0.0,
                "whale_events": [],
                "momentum_scan": None,
                "fade_signal": None
            }

        # Extract arrays
        opens = [c["open"] for c in ohlcv]
        highs = [c["high"] for c in ohlcv]
        lows = [c["low"] for c in ohlcv]
        closes = [c["close"] for c in ohlcv]
        volumes = [c["volume"] for c in ohlcv]

        # Whale Tracker (OFI)
        trades_data = trades or []
        price = closes[-1]
        ofi_score, directional_skew, whale_events = self._whale_tracker_invoke(
            symbol, trades_data, price
        )

        # Momentum Scanner
        momentum_scan = self._momentum_scanner_invoke(
            symbol, interval, ohlcv, volumes, closes
        )

        # Contrarian Fade
        closes_arr = np.array(closes[-BB_PERIODS:])
        sma = float(np.mean(closes_arr))
        stdev = float(np.std(closes_arr))
        bb_upper = sma + BB_MULTIPLIER * stdev
        bb_lower = sma - BB_MULTIPLIER * stdev

        rsi_val = self._compute_rsi(closes[-14:])  # RSI(14)
        fade_signal = self._fade_detector_invoke(
            symbol, closes, rsi_val, bb_upper, bb_lower, closes[-1]
        )

        # Package results
        return {
            "ofi_score": ofi_score,
            "directional_skew": directional_skew,
            "whale_events": [ev.to_dict() for ev in whale_events],
            "momentum_scan": momentum_scan.to_dict() if momentum_scan else None,
            "fade_signal": fade_signal.to_dict() if fade_signal else None,
        }
    
    def _compute_rsi(self, closes: List[float]) -> float:
        """Standalone RSI(14) computation."""
        if len(closes) < 2:
            return 50.0
        
        deltas = np.diff(closes)
        gains = deltas.clip(min=0)
        losses = (-deltas).clip(min=0)
        
        avg_gain = float(np.mean(gains)) if len(gains) > 0 else 0.0
        avg_loss = float(np.mean(losses)) if len(losses) > 0 else 0.0
        if avg_loss < 1e-8:
            avg_loss = 1e-8
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi

    def inject_into_registry(self, registry: SkillRegistry) -> None:
        """Self-register into SkillRegistry."""
        registry.register(
            name="krypt_core",
            skill_cls_or_instance=self,
            version=self._version,
            description="Krypt-Trader crypto microstructure engine",
            tags=("krypt", "market_structure", "alpha"),
            hot_swap_enabled=True,
        )

    @property
    def version(self) -> str:
        return self._version
