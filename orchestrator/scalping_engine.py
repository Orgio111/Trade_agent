"""
QUANTEX Scalping Engine — Ultra-fast CPU-only trading decision engine.

Target: <5ms logic, no LLM/VLM/RAG, deterministic, 5-candle window.

Architecture:
  ┌─────────────────────────────────────────────────────────────┐
  │                  Scalping Decision Engine                   │
  │                                                              │
  │  Input:  current_price + last_5_candles + volatility +      │
  │          trend + support/resistance zones                    │
  │                                                              │
  │  Logic:  momentum_breakout → BUY                            │
  │          sharp_rejection_at_resistance → SELL                │
  │          low_volatility → HOLD                              │
  │                                                              │
  │  Output: {action, confidence, size_pct, reason}             │
  └─────────────────────────────────────────────────────────────┘

Rules:
  - Respond in <200ms (actual target: <5ms)
  - Use only current candle + last 5 candles summary
  - Do NOT analyze full history
  - Do NOT explain in detail
  - Prioritize speed over completeness

Usage:
    engine = ScalpingEngine()
    decision = engine.decide(
        current_price=42050.0,
        last_5_candles=[...],
        volatility="medium",
        trend="bullish",
        support=41900.0,
        resistance=42200.0,
    )
    # decision = {"action": "BUY", "confidence": 78, "size_pct": 2.5, "reason": "MOMENTUM_BREAKOUT"}
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Optional

# ── Prometheus Metrics (optional, graceful fallback) ────────
try:
    from prometheus_client import Histogram, Counter, Gauge

    SCALPING_DECISION_LATENCY = Histogram(
        "scalping_decision_latency_ms",
        "Scalping engine decision latency in milliseconds",
        buckets=[0.1, 0.5, 1, 2, 5, 10, 20, 50, 100],
    )
    SCALPING_SIGNALS_TOTAL = Counter(
        "scalping_signals_total",
        "Scalping engine signal counts by type",
        ["signal"],  # "momentum_breakout", "rejection_support", "rejection_resistance", "volume_spike", "trend_aligned"
    )
    SCALPING_PROMETHEUS_AVAILABLE = True
except ImportError:
    SCALPING_PROMETHEUS_AVAILABLE = False


# ── Constants ───────────────────────────────────────────────

CONFIDENCE_THRESHOLD = 50       # Minimum confidence to trigger trade
HIGH_VOLATILITY_THRESHOLD = 0.02  # 2% ATR/price = high volatility
LOW_VOLATILITY_THRESHOLD = 0.005  # 0.5% ATR/price = low volatility
MOMENTUM_BREAKOUT_THRESHOLD = 0.003  # 0.3% price move = breakout
REJECTION_WICK_RATIO = 0.6     # Wick > 60% of candle = rejection
TREND_ALIGNMENT_THRESHOLD = 0.002  # 0.2% above/below = trend aligned
MAX_SIZE_PCT = 5.0             # Maximum position size %
MIN_SIZE_PCT = 0.5             # Minimum position size %
HIGH_VOL_SIZE_REDUCTION = 0.5  # Reduce size by 50% in high vol


# ── Data Classes ────────────────────────────────────────────

@dataclass
class ScalpCandle:
    """Minimal candle data for scalping."""
    open: float
    high: float
    low: float
    close: float
    volume: float
    timestamp: float = 0.0

    @property
    def body_size(self) -> float:
        return abs(self.close - self.open)

    @property
    def range(self) -> float:
        return self.high - self.low

    @property
    def upper_wick(self) -> float:
        return self.high - max(self.open, self.close)

    @property
    def lower_wick(self) -> float:
        return min(self.open, self.close) - self.low

    @property
    def is_bullish(self) -> bool:
        return self.close > self.open

    @property
    def is_bearish(self) -> bool:
        return self.close < self.open

    @property
    def wick_ratio(self) -> float:
        """Upper wick as ratio of total range."""
        if self.range == 0:
            return 0.0
        return self.upper_wick / self.range

    @property
    def lower_wick_ratio(self) -> float:
        """Lower wick as ratio of total range."""
        if self.range == 0:
            return 0.0
        return self.lower_wick / self.range

    @property
    def body_ratio(self) -> float:
        """Body as ratio of total range."""
        if self.range == 0:
            return 0.0
        return self.body_size / self.range


@dataclass
class ScalpDecision:
    """Strict scalping decision output."""
    action: str  # "BUY" | "SELL" | "HOLD"
    confidence: int  # 0-100
    size_pct: float  # 0-5
    reason: str  # Short keyword

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        import json
        return json.dumps(self.to_dict())


# ── Scalping Engine ─────────────────────────────────────────

# Pre-compute field names for fast dict filtering in hot path
_SCALP_CANDLE_FIELDS = {f.name for f in __import__('dataclasses').fields(ScalpCandle)}


class ScalpingEngine:
    """
    Ultra-fast CPU-only scalping decision engine.

    All logic is pure computation — no I/O, no LLM, no network calls.
    Target latency: <5ms for decision logic alone.
    """

    def __init__(
        self,
        confidence_threshold: int = CONFIDENCE_THRESHOLD,
        max_size_pct: float = MAX_SIZE_PCT,
        high_vol_reduction: float = HIGH_VOL_SIZE_REDUCTION,
    ):
        self.confidence_threshold = confidence_threshold
        self.max_size_pct = max_size_pct
        self.high_vol_reduction = high_vol_reduction
        self._decision_count = 0

    def decide(
        self,
        current_price: float,
        last_5_candles: list[dict],
        volatility: str = "medium",
        trend: str = "neutral",
        support: float = 0.0,
        resistance: float = 0.0,
    ) -> ScalpDecision:
        """
        Make a scalping decision in <5ms.

        Args:
            current_price: Current market price
            last_5_candles: List of last 5 OHLCV dicts (most recent last)
            volatility: "low", "medium", "high"
            trend: "bullish", "bearish", "neutral"
            support: Key support level (0 = not provided)
            resistance: Key resistance level (0 = not provided)

        Returns:
            ScalpDecision with action, confidence, size_pct, reason
        """
        t0 = time.perf_counter()
        self._decision_count += 1

        # ── Guard: insufficient data ──
        if not last_5_candles or current_price <= 0:
            return ScalpDecision(
                action="HOLD",
                confidence=0,
                size_pct=0,
                reason="NO_DATA",
            )

        # ── Convert to ScalpCandle objects ──
        # Filter to only known fields to avoid TypeError from extra keys
        candles = [ScalpCandle(**{k: v for k, v in c.items() if k in _SCALP_CANDLE_FIELDS}) for c in last_5_candles[-5:]]
        current = candles[-1]

        # ── Step 1: Compute volatility metric ──
        vol_level = self._classify_volatility(candles, current_price)

        # ── Step 2: Detect momentum breakout ──
        momentum_signal = self._detect_momentum(candles)

        # ── Step 3: Detect rejection at S/R ──
        rejection_signal = self._detect_rejection(
            candles, current_price, support, resistance
        )

        # ── Step 4: Trend alignment check ──
        trend_aligned = self._check_trend_alignment(candles, trend)

        # ── Step 5: Volume confirmation ──
        volume_spike = self._detect_volume_spike(candles)

        # ── Step 6: Fuse signals ──
        decision = self._fuse_signals(
            momentum=momentum_signal,
            rejection=rejection_signal,
            trend_aligned=trend_aligned,
            volume_spike=volume_spike,
            volatility=vol_level,
            current_price=current_price,
            support=support,
            resistance=resistance,
        )

        # ── Record signal type metrics ──
        if SCALPING_PROMETHEUS_AVAILABLE:
            if momentum_signal:
                SCALPING_SIGNALS_TOTAL.labels(signal="momentum_breakout").inc()
            if rejection_signal == "BUY":
                SCALPING_SIGNALS_TOTAL.labels(signal="rejection_support").inc()
            elif rejection_signal == "SELL":
                SCALPING_SIGNALS_TOTAL.labels(signal="rejection_resistance").inc()
            if volume_spike:
                SCALPING_SIGNALS_TOTAL.labels(signal="volume_spike").inc()
            if trend_aligned:
                SCALPING_SIGNALS_TOTAL.labels(signal="trend_aligned").inc()

        elapsed_ms = (time.perf_counter() - t0) * 1000
        decision.reason = f"{decision.reason}|{elapsed_ms:.1f}ms"

        # ── Record Prometheus metrics ──
        if SCALPING_PROMETHEUS_AVAILABLE:
            SCALPING_DECISION_LATENCY.observe(elapsed_ms)

        return decision

    def _classify_volatility(self, candles: list[ScalpCandle], price: float) -> str:
        """Classify volatility from recent candle ranges."""
        if price <= 0:
            return "medium"

        avg_range = sum(c.range for c in candles) / len(candles)
        vol_pct = avg_range / price

        if vol_pct > HIGH_VOLATILITY_THRESHOLD:
            return "high"
        elif vol_pct < LOW_VOLATILITY_THRESHOLD:
            return "low"
        return "medium"

    def _detect_momentum(self, candles: list[ScalpCandle]) -> Optional[str]:
        """
        Detect momentum breakout from recent candles.

        Returns: "BUY", "SELL", or None
        """
        if len(candles) < 2:
            return None

        current = candles[-1]
        prev = candles[-2]

        # Momentum breakout: current candle breaks previous range with conviction
        price_change = (current.close - prev.close) / prev.close if prev.close > 0 else 0

        # Strong bullish momentum: close above prev high, green candle, good body
        if (current.close > prev.high and
            current.is_bullish and
            current.body_ratio > 0.5 and
            price_change > MOMENTUM_BREAKOUT_THRESHOLD):
            return "BUY"

        # Strong bearish momentum: close below prev low, red candle, good body
        if (current.close < prev.low and
            current.is_bearish and
            current.body_ratio > 0.5 and
            price_change < -MOMENTUM_BREAKOUT_THRESHOLD):
            return "SELL"

        # Acceleration: 3 consecutive candles in same direction
        if len(candles) >= 3:
            last3 = candles[-3:]
            if all(c.is_bullish for c in last3):
                # Check acceleration (each candle closes higher)
                if (last3[2].close > last3[1].close > last3[0].close and
                    price_change > MOMENTUM_BREAKOUT_THRESHOLD * 0.5):
                    return "BUY"
            elif all(c.is_bearish for c in last3):
                if (last3[2].close < last3[1].close < last3[0].close and
                    price_change < -MOMENTUM_BREAKOUT_THRESHOLD * 0.5):
                    return "SELL"

        return None

    def _detect_rejection(
        self,
        candles: list[ScalpCandle],
        current_price: float,
        support: float,
        resistance: float,
    ) -> Optional[str]:
        """
        Detect sharp rejection at support/resistance.

        Returns: "BUY" (rejection at support), "SELL" (rejection at resistance), or None
        """
        if not candles:
            return None

        current = candles[-1]

        # Rejection at resistance → SELL
        if resistance > 0:
            near_resistance = abs(current.high - resistance) / resistance < 0.002
            has_upper_wick = current.wick_ratio > REJECTION_WICK_RATIO
            if near_resistance and has_upper_wick:
                return "SELL"

        # Rejection at support → BUY
        if support > 0:
            near_support = abs(current.low - support) / support < 0.002
            has_lower_wick = current.lower_wick_ratio > REJECTION_WICK_RATIO
            if near_support and has_lower_wick:
                return "BUY"

        # Pin bar pattern (long wick, small body)
        if (current.body_ratio < 0.2 and
            (current.wick_ratio > 0.6 or current.lower_wick_ratio > 0.6)):
            # Bullish pin bar at low area
            if current.lower_wick_ratio > current.wick_ratio:
                return "BUY"
            # Bearish pin bar at high area
            return "SELL"

        return None

    def _check_trend_alignment(self, candles: list[ScalpCandle], trend: str) -> bool:
        """Check if recent price action aligns with the trend."""
        if len(candles) < 3 or trend == "neutral":
            return False

        current_price = candles[-1].close
        lookback_price = candles[-3].close

        if lookback_price <= 0:
            return False

        price_move = (current_price - lookback_price) / lookback_price

        if trend == "bullish" and price_move > TREND_ALIGNMENT_THRESHOLD:
            return True
        if trend == "bearish" and price_move < -TREND_ALIGNMENT_THRESHOLD:
            return True

        return False

    def _detect_volume_spike(self, candles: list[ScalpCandle]) -> bool:
        """Detect volume spike in current candle vs average."""
        if len(candles) < 3:
            return False

        avg_vol = sum(c.volume for c in candles[:-1]) / max(len(candles) - 1, 1)
        current_vol = candles[-1].volume

        if avg_vol <= 0:
            return False

        return current_vol > avg_vol * 1.5

    def _fuse_signals(
        self,
        momentum: Optional[str],
        rejection: Optional[str],
        trend_aligned: bool,
        volume_spike: bool,
        volatility: str,
        current_price: float,
        support: float,
        resistance: float,
    ) -> ScalpDecision:
        """
        Fuse all signals into a final decision.

        Scoring:
          momentum_breakout: +30 (BUY) or -30 (SELL)
          rejection:         +25 (BUY) or -25 (SELL)
          trend_aligned:     +15
          volume_spike:      +10
          high_volatility:   -20 confidence, size reduced
          low_volatility:    HOLD (no trade)
        """
        score = 0.0
        reasons = []

        # ── Momentum ──
        if momentum == "BUY":
            score += 30
            reasons.append("MOMENTUM_BREAKOUT")
        elif momentum == "SELL":
            score -= 30
            reasons.append("MOMENTUM_BREAKOUT")

        # ── Rejection ──
        if rejection == "BUY":
            score += 25
            reasons.append("REJECTION_SUPPORT")
        elif rejection == "SELL":
            score -= 25
            reasons.append("REJECTION_RESISTANCE")

        # ── Trend alignment ──
        if trend_aligned:
            score += 15 if score > 0 else -15
            reasons.append("TREND_ALIGNED")

        # ── Volume ──
        if volume_spike and score != 0:
            score *= 1.2
            reasons.append("VOLUME_SPIKE")

        # ── Volatility adjustment ──
        if volatility == "low":
            return ScalpDecision(
                action="HOLD",
                confidence=0,
                size_pct=0,
                reason="LOW_VOLatility",
            )

        if volatility == "high":
            reasons.append("HIGH_VOL_REDUCE")

        # ── Conflicting signals → HOLD ──
        if momentum and rejection and momentum != rejection:
            return ScalpDecision(
                action="HOLD",
                confidence=0,
                size_pct=0,
                reason="CONFLICTING_SIGNALS",
            )

        # ── No clear signal → HOLD ──
        if score == 0:
            return ScalpDecision(
                action="HOLD",
                confidence=0,
                size_pct=0,
                reason="NO_SIGNAL",
            )

        # ── Determine action and confidence ──
        if score > 0:
            action = "BUY"
        else:
            action = "SELL"

        # Map score to 0-100 confidence
        confidence = min(100, max(0, int(abs(score) * 1.2)))

        # ── Threshold check ──
        if confidence < self.confidence_threshold:
            return ScalpDecision(
                action="HOLD",
                confidence=confidence,
                size_pct=0,
                reason=f"BELOW_THRESHOLD|{'+'.join(reasons)}",
            )

        # ── Position sizing ──
        size_pct = self._compute_size(confidence, volatility)

        reason_str = "+".join(reasons) if reasons else "FUSED"

        return ScalpDecision(
            action=action,
            confidence=confidence,
            size_pct=round(size_pct, 2),
            reason=reason_str,
        )

    def _compute_size(self, confidence: int, volatility: str) -> float:
        """
        Compute position size based on confidence and volatility.

        Rules:
          - Base size = confidence / 100 * max_size
          - High volatility → reduce by 50%
          - Clamp to [MIN_SIZE, MAX_SIZE]
        """
        base_size = (confidence / 100.0) * self.max_size_pct

        if volatility == "high":
            base_size *= self.high_vol_reduction

        return max(MIN_SIZE_PCT, min(self.max_size_pct, base_size))

    def get_stats(self) -> dict:
        """Get engine statistics."""
        return {
            "total_decisions": self._decision_count,
            "confidence_threshold": self.confidence_threshold,
            "max_size_pct": self.max_size_pct,
        }


# ── Quick Convenience Function ──────────────────────────────

def scalp_decision(
    current_price: float,
    last_5_candles: list[dict],
    volatility: str = "medium",
    trend: str = "neutral",
    support: float = 0.0,
    resistance: float = 0.0,
) -> dict:
    """
    One-shot scalping decision. Returns strict JSON format.

    Usage:
        decision = scalp_decision(
            current_price=42050.0,
            last_5_candles=[{"open": 42000, "high": 42100, ...}],
        )
        # {"action": "BUY", "confidence": 78, "size_pct": 2.5, "reason": "MOMENTUM_BREAKOUT"}
    """
    engine = ScalpingEngine()
    result = engine.decide(
        current_price=current_price,
        last_5_candles=last_5_candles,
        volatility=volatility,
        trend=trend,
        support=support,
        resistance=resistance,
    )
    return result.to_dict()
