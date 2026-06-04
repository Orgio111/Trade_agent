"""
Quantex strategy engine — rule-based and AI-driven strategies.
Phase 1: EMA Crossover + RSI filter strategy.
"""
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd


@dataclass
class Signal:
    direction: str  # "long", "short", "hold"
    confidence: float  # 0.0 to 1.0
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profits: list[dict] = field(default_factory=list)
    reason: str = ""
    metadata: dict = field(default_factory=dict)


class StrategyEngine:
    """
    Base strategy engine with rule-based and ML strategies.
    Phase 1: EMA Crossover + RSI filter with ATR-based SL/TP.
    """

    def __init__(
        self,
        ema_fast: int = 11,
        ema_slow: int = 92,
        rsi_period: int = 14,
        rsi_overbought: float = 62.0,
        rsi_oversold: float = 37.0,
        atr_multiplier_sl: float = 1.63,
        atr_multiplier_tp1: float = 1.90,
        atr_multiplier_tp2: float = 2.84,
    ):
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.atr_multiplier_sl = atr_multiplier_sl
        self.atr_multiplier_tp1 = atr_multiplier_tp1
        self.atr_multiplier_tp2 = atr_multiplier_tp2

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute technical indicators for the strategy."""
        result = df.copy()

        # EMAs
        result["ema_fast"] = result["close"].ewm(span=self.ema_fast, adjust=False).mean()
        result["ema_slow"] = result["close"].ewm(span=self.ema_slow, adjust=False).mean()

        # RSI
        delta = result["close"].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta.where(delta < 0, 0.0))
        avg_gain = gain.ewm(span=self.rsi_period, adjust=False).mean()
        avg_loss = loss.ewm(span=self.rsi_period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        result["rsi"] = 100 - (100 / (1 + rs))

        # ATR
        high_low = result["high"] - result["low"]
        high_close = (result["high"] - result["close"].shift()).abs()
        low_close = (result["low"] - result["close"].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        result["atr"] = tr.rolling(window=14).mean()

        # Crossover signal
        result["ema_cross"] = 0
        result.loc[result["ema_fast"] > result["ema_slow"], "ema_cross"] = 1
        result.loc[result["ema_fast"] <= result["ema_slow"], "ema_cross"] = -1

        # Previous cross for detecting changes
        result["ema_cross_prev"] = result["ema_cross"].shift(1)
        result["crossover_buy"] = (result["ema_cross"] == 1) & (result["ema_cross_prev"] == -1)
        result["crossover_sell"] = (result["ema_cross"] == -1) & (result["ema_cross_prev"] == 1)

        return result

    def generate_signal(self, df: pd.DataFrame) -> Signal:
        """
        Generate trading signal from latest data.
        Returns a Signal with direction, confidence, and risk levels.
        """
        if len(df) < self.ema_slow + 5:
            return Signal(direction="hold", confidence=0.0, reason="Insufficient data")

        df = self.compute_indicators(df)
        latest = df.iloc[-1]
        prev = df.iloc[-2]

        # Base confidence starts at 0.5
        confidence = 0.5
        reasons = []

        # Check for EMA crossover
        if latest["crossover_buy"]:
            direction = "long"
            reasons.append("EMA crossover BUY")
            confidence += 0.15
        elif latest["crossover_sell"]:
            direction = "short"
            reasons.append("EMA crossover SELL")
            confidence += 0.15
        else:
            # No crossover — check trend alignment
            if latest["ema_cross"] == 1 and latest["close"] > latest["ema_slow"]:
                direction = "long"
                confidence += 0.05
                reasons.append("Uptrend")
            elif latest["ema_cross"] == -1 and latest["close"] < latest["ema_slow"]:
                direction = "short"
                confidence += 0.05
                reasons.append("Downtrend")
            else:
                return Signal(
                    direction="hold",
                    confidence=0.3,
                    reason="No clear trend",
                )

        # RSI filter
        rsi = latest.get("rsi", 50)
        if direction == "long":
            if rsi > self.rsi_overbought:
                confidence -= 0.15
                reasons.append(f"RSI overbought ({rsi:.0f})")
            elif rsi < self.rsi_oversold:
                confidence += 0.10
                reasons.append(f"RSI oversold ({rsi:.0f})")
        elif direction == "short":
            if rsi < self.rsi_oversold:
                confidence -= 0.15
                reasons.append(f"RSI oversold ({rsi:.0f})")
            elif rsi > self.rsi_overbought:
                confidence += 0.10
                reasons.append(f"RSI overbought ({rsi:.0f})")

        # Volume confirmation
        vol_sma = df["volume"].rolling(20).mean()
        if len(df) >= 20:
            vol_ratio = latest["volume"] / vol_sma.iloc[-1]
            if vol_ratio > 1.5:
                confidence += 0.10
                reasons.append(f"Volume spike ({vol_ratio:.1f}x)")
            elif vol_ratio < 0.5:
                confidence -= 0.10
                reasons.append(f"Low volume ({vol_ratio:.1f}x)")

        # Calculate SL/TP levels
        atr = latest.get("atr", latest["close"] * 0.02)
        entry_price = latest["close"]

        if direction == "long":
            stop_loss = entry_price - (atr * self.atr_multiplier_sl)
            tp1 = entry_price + (atr * self.atr_multiplier_tp1)
            tp2 = entry_price + (atr * self.atr_multiplier_tp2)
        else:
            stop_loss = entry_price + (atr * self.atr_multiplier_sl)
            tp1 = entry_price - (atr * self.atr_multiplier_tp1)
            tp2 = entry_price - (atr * self.atr_multiplier_tp2)

        # Clamp confidence
        confidence = max(0.0, min(1.0, confidence))

        # Minimum confidence threshold
        if confidence < 0.4:
            return Signal(
                direction="hold",
                confidence=confidence,
                reason=f"Confidence too low: {', '.join(reasons)}",
            )

        return Signal(
            direction=direction,
            confidence=round(confidence, 4),
            entry_price=entry_price,
            stop_loss=round(stop_loss, 2),
            take_profits=[
                {"level": 1, "price": round(tp1, 2), "qty_pct": 0.5, "trail": False},
                {"level": 2, "price": round(tp2, 2), "qty_pct": 0.5, "trail": True, "trail_dist": atr * 0.5},
            ],
            reason=", ".join(reasons),
            metadata={
                "rsi": round(rsi, 2),
                "atr": round(atr, 2),
                "ema_fast": round(latest.get("ema_fast", 0), 2),
                "ema_slow": round(latest.get("ema_slow", 0), 2),
            },
        )


# Composite confidence scorer from multiple signal sources
def composite_confidence(signals: dict) -> float:
    """
    Merge multiple signal sources into a single confidence score.

    score = Σ(weight_i × signal_i) / Σ(weight_i)
    """
    weights = {
        "technical": 0.40,
        "momentum": 0.25,
        "volume": 0.20,
        "sentiment": 0.15,
    }

    score = 0.0
    total_weight = 0.0

    for signal_name, weight in weights.items():
        if signal_name in signals:
            score += weight * signals[signal_name]
            total_weight += weight

    if total_weight == 0:
        return 0.5

    return score / total_weight
