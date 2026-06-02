"""
QUANTEX Market Microstructure Engine — Institutional-grade order flow analysis.

Detects:
  - Spoofing / Quote stuffing manipulation
  - Hidden liquidity / Iceberg orders
  - Delta divergence and CVD (Cumulative Volume Delta)
  - Liquidation cascade risk
  - Order book imbalance patterns
  - Absorption vs rejection signals
"""

import time
import math
import statistics
from collections import deque
from typing import Optional

import numpy as np


class SpoofingDetector:
    """
    Detects spoofing and quote stuffing patterns in order book data.

    Spoofing = placing large orders with no intention to execute,
    to create false price pressure and trick other traders.

    Detection features:
      - Fast order cancellations (>80% cancel-to-order ratio)
      - Large orders appearing at key levels and vanishing
      - Repeated baiting patterns (same price level multiple times)
      - Abnormal order-to-trade ratios
    """

    def __init__(self, window_seconds: int = 60):
        self.window_seconds = window_seconds
        self._order_book_snapshots: deque = deque(maxlen=1000)
        self._cancel_tracker: dict[float, int] = {}  # price -> cancel count

    def record_snapshot(self, bids: list[list[float]], asks: list[list[float]],
                        timestamp: Optional[float] = None):
        """Record an order book snapshot for analysis."""
        t = timestamp or time.time()
        self._order_book_snapshots.append({
            "timestamp": t,
            "bids": [(p, q) for p, q in bids if q > 0],
            "asks": [(p, q) for p, q in asks if q > 0],
        })

    def record_cancellation(self, price: float, quantity: float):
        """Record an order cancellation at a price level."""
        self._cancel_tracker[price] = self._cancel_tracker.get(price, 0) + 1

    def analyze(self) -> dict:
        """
        Analyze recent order book data for spoofing patterns.

        Returns dict with:
          - spoofing_detected: boolean
          - confidence: 0-1 score
          - spoofing_levels: price levels with suspicious activity
          - cancel_to_order_ratio: overall ratio
          - reasoning: explanation
        """
        if len(self._order_book_snapshots) < 10:
            return {"spoofing_detected": False, "confidence": 0.0,
                    "spoofing_levels": [], "cancel_to_order_ratio": 0.0,
                    "reasoning": "Insufficient data"}

        # Analyze recent snapshots (last 60 seconds)
        cutoff = time.time() - self.window_seconds
        recent = [s for s in self._order_book_snapshots if s["timestamp"] > cutoff]

        if len(recent) < 5:
            return {"spoofing_detected": False, "confidence": 0.0,
                    "spoofing_levels": [], "cancel_to_order_ratio": 0.0,
                    "reasoning": "Insufficient recent data"}

        spoofing_score = 0.0
        spoofing_levels = []
        reasons = []

        # 1. Check for rapid order disappearance at levels
        level_persistence = {}
        for snap in recent:
            for p, q in snap["bids"][:5]:  # Top 5 bid levels
                rounded_p = round(p, 1)
                level_persistence[rounded_p] = level_persistence.get(rounded_p, 0) + 1
            for p, q in snap["asks"][:5]:  # Top 5 ask levels
                rounded_p = round(p, 1)
                level_persistence[rounded_p] = level_persistence.get(rounded_p, 0) + 1

        # Levels that appear and disappear rapidly
        for level, count in level_persistence.items():
            if count < len(recent) * 0.3:  # Present < 30% of time
                cancel_count = self._cancel_tracker.get(level, 0)
                if cancel_count > 3:
                    spoofing_levels.append({
                        "price": level,
                        "appearance_pct": count / len(recent) * 100,
                        "cancel_count": cancel_count,
                    })
                    spoofing_score += 0.1
                    if spoofing_score <= 0.3:
                        reasons.append(f"Level ${level:.1f}: appeared {count/len(recent)*100:.0f}% of time, {cancel_count} cancels")

        # 2. Cancel-to-order ratio
        total_cancels = sum(self._cancel_tracker.values())
        total_snapshots = len(recent)
        cancel_ratio = total_cancels / total_snapshots if total_snapshots > 0 else 0

        if cancel_ratio > 5:
            spoofing_score += 0.3
            reasons.append(f"High cancel ratio: {cancel_ratio:.1f}x vs snapshots")
        elif cancel_ratio > 3:
            spoofing_score += 0.15

        # 3. Large order disappearance (size drops > 50% between snaps)
        size_drops = 0
        for i in range(1, len(recent)):
            prev_bid_sizes = sum(q for _, q in recent[i-1]["bids"][:3])
            curr_bid_sizes = sum(q for _, q in recent[i]["bids"][:3])
            if prev_bid_sizes > 0 and curr_bid_sizes < prev_bid_sizes * 0.5:
                size_drops += 1

        if size_drops > len(recent) * 0.4:
            spoofing_score += 0.2
            reasons.append(f"Large order disappearance: {size_drops}/{len(recent)} snaps")

        return {
            "spoofing_detected": spoofing_score >= 0.35,
            "confidence": round(min(spoofing_score, 1.0), 4),
            "spoofing_levels": spoofing_levels[:5],
            "cancel_to_order_ratio": round(cancel_ratio, 2),
            "reasoning": "; ".join(reasons) if reasons else "No spoofing patterns detected",
        }


class HiddenLiquidityDetector:
    """
    Detects hidden/iceberg orders in the order book.

    Iceberg orders show only a portion of their true size.
    Detection methods:
      - Repeated fills at the same price level
      - Order book refill patterns after fills
      - Abnormal trade sizes at specific levels
      - Bid/ask stacking behavior
    """

    def __init__(self):
        self._trade_flow: deque = deque(maxlen=1000)
        self._refill_tracker: dict[float, list[float]] = {}

    def record_trade(self, price: float, quantity: float, side: str):
        """Record a trade for analysis."""
        self._trade_flow.append({
            "price": price,
            "quantity": quantity,
            "side": side,
            "timestamp": time.time(),
        })

    def record_refill(self, price: float, quantity: float):
        """Record an order book refill at a price level."""
        if price not in self._refill_tracker:
            self._refill_tracker[price] = []
        self._refill_tracker[price].append(quantity)
        # Keep only recent refills
        cutoff = time.time() - 300
        self._refill_tracker = {
            p: [q for q in qs if q > 0][-10:]
            for p, qs in self._refill_tracker.items()
        }

    def analyze(self) -> dict:
        """
        Analyze for hidden/iceberg order patterns.

        Returns dict with:
          - hidden_liquidity_detected: boolean
          - confidence: 0-1
          - iceberg_levels: price levels with suspected icebergs
          - reasoning: explanation
        """
        if len(self._trade_flow) < 20:
            return {"hidden_liquidity_detected": False, "confidence": 0.0,
                    "iceberg_levels": [], "reasoning": "Insufficient data"}

        iceberg_score = 0.0
        iceberg_levels = []
        reasons = []

        # 1. Look for refill patterns (order reappears after fill)
        for price, refills in self._refill_tracker.items():
            if len(refills) >= 3:
                # Consistent refills suggest iceberg
                mean_refill = statistics.mean(refills) if len(refills) > 1 else refills[0]
                refill_consistency = 1.0 - (statistics.stdev(refills) / mean_refill) if len(refills) > 1 and mean_refill > 0 else 0.5
                if refill_consistency > 0.7 and len(refills) >= 3:
                    iceberg_levels.append({
                        "price": round(price, 2),
                        "refill_count": len(refills),
                        "avg_refill_size": round(mean_refill, 6),
                        "consistency": round(refill_consistency, 4),
                    })
                    iceberg_score += 0.15

        if iceberg_levels:
            reasons.append(f"Refill patterns at {len(iceberg_levels)} levels")
            iceberg_score = min(1.0, iceberg_score + len(iceberg_levels) * 0.1)

        # 2. Check for abnormal trade size clusters
        recent_trades = list(self._trade_flow)[-50:]
        trade_sizes = [t["quantity"] for t in recent_trades]
        if trade_sizes:
            median_size = statistics.median(trade_sizes)
            # If many trades at same size, suggests iceberg slicing
            size_counts = {}
            for s in trade_sizes:
                rounded = round(s, 6)
                size_counts[rounded] = size_counts.get(rounded, 0) + 1
            # Same size appearing 3+ times = iceberg slices
            repeated_sizes = [(s, c) for s, c in size_counts.items() if c >= 3]
            if repeated_sizes:
                iceberg_score += 0.2
                reasons.append(f"Repeated trade sizes: {len(repeated_sizes)} clusters")

        return {
            "hidden_liquidity_detected": iceberg_score >= 0.3,
            "confidence": round(min(iceberg_score, 1.0), 4),
            "iceberg_levels": sorted(iceberg_levels, key=lambda x: x["refill_count"], reverse=True)[:5],
            "reasoning": "; ".join(reasons) if reasons else "No hidden liquidity detected",
        }


class DeltaCVDTracker:
    """
    Tracks Cumulative Volume Delta (CVD) and detects divergences.

    Delta = aggressive buy volume - aggressive sell volume
    CVD = cumulative sum of delta over time

    Divergence patterns:
      - Price making higher highs, CVD making lower highs (bearish)
      - Price making lower lows, CVD making higher lows (bullish)
      - Delta exhaustion: strong delta followed by reversal
    """

    def __init__(self):
        self._deltas: deque = deque(maxlen=10000)
        self._prices: deque = deque(maxlen=10000)
        self._cvd = 0.0

    def record_tick(self, price: float, delta: float):
        """Record a price tick with its delta value."""
        self._prices.append(price)
        self._deltas.append(delta)
        self._cvd += delta

    def get_cvd(self) -> float:
        """Get current cumulative delta."""
        return self._cvd

    def analyze_divergence(self, lookback: int = 100) -> dict:
        """
        Detect delta/price divergences.

        Returns dict with:
          - divergence: "bullish", "bearish", "none"
          - strength: 0-1
          - delta_trend: "rising", "falling", "neutral"
          - price_trend: "rising", "falling", "neutral"
          - reasoning: explanation
        """
        if len(self._deltas) < lookback or len(self._prices) < lookback:
            return {"divergence": "none", "strength": 0.0,
                    "delta_trend": "neutral", "price_trend": "neutral",
                    "reasoning": "Insufficient data"}

        prices = list(self._prices)[-lookback:]
        deltas = list(self._deltas)[-lookback:]
        cvd = np.cumsum(deltas)

        # Price trend (regression slope)
        price_slope = np.polyfit(range(len(prices)), prices, 1)[0]
        price_trend = "rising" if price_slope > 0 else "falling" if price_slope < 0 else "neutral"

        # CVD trend
        cvd_slope = np.polyfit(range(len(cvd)), cvd, 1)[0]
        delta_trend = "rising" if cvd_slope > 0 else "falling" if cvd_slope < 0 else "neutral"

        # Divergence detection
        divergence = "none"
        strength = 0.0
        reasons = []

        # Bullish divergence: price down, CVD up
        if price_trend == "falling" and delta_trend == "rising":
            divergence = "bullish"
            # Strength proportional to slope difference
            strength = min(1.0, abs(price_slope - cvd_slope) / (abs(price_slope) + abs(cvd_slope) + 1e-10) * 2)
            reasons.append(f"Bullish divergence: price falling ({price_slope:.4f}) but CVD rising ({cvd_slope:.4f})")

        # Bearish divergence: price up, CVD down
        elif price_trend == "rising" and delta_trend == "falling":
            divergence = "bearish"
            strength = min(1.0, abs(price_slope - cvd_slope) / (abs(price_slope) + abs(cvd_slope) + 1e-10) * 2)
            reasons.append(f"Bearish divergence: price rising ({price_slope:.4f}) but CVD falling ({cvd_slope:.4f})")

        # Delta exhaustion: strong delta in one direction, then reversal
        recent_deltas = deltas[-20:]
        rolling_delta_mean = np.mean(recent_deltas) if recent_deltas else 0
        if abs(rolling_delta_mean) > 2 * np.std(deltas[:-20]) if len(deltas) > 20 else 0:
            exhaustion_dir = "buying" if rolling_delta_mean > 0 else "selling"
            reasons.append(f"Delta exhaustion: extreme {exhaustion_dir} pressure")
            strength += 0.15

        return {
            "divergence": divergence,
            "strength": round(min(strength, 1.0), 4),
            "delta_trend": delta_trend,
            "price_trend": price_trend,
            "cvd_value": round(self._cvd, 4),
            "cvd_slope": round(cvd_slope, 6),
            "price_slope": round(price_slope, 6),
            "reasoning": "; ".join(reasons) if reasons else "No divergence detected",
        }

    def get_delta_signal(self) -> dict:
        """Generate a trading signal from delta analysis."""
        div = self.analyze_divergence()

        if div["divergence"] == "bullish" and div["strength"] > 0.5:
            return {
                "direction": "long",
                "confidence": round(0.5 + div["strength"] * 0.3, 4),
                "reason": div["reasoning"],
            }
        elif div["divergence"] == "bearish" and div["strength"] > 0.5:
            return {
                "direction": "short",
                "confidence": round(0.5 + div["strength"] * 0.3, 4),
                "reason": div["reasoning"],
            }
        return {
            "direction": "hold",
            "confidence": 0.3,
            "reason": f"No strong divergence. CVD trend: {div['delta_trend']}, Price trend: {div['price_trend']}",
        }


class LiquidationCascadePredictor:
    """
    Predicts the risk of liquidation cascades.

    A liquidation cascade occurs when falling prices trigger
    leveraged longs to be liquidated, creating more selling
    pressure, causing more liquidations.

    Detection features:
      - Liquidation cluster density
      - Open interest decay rate
      - Funding rate extremes
      - Leverage concentration
      - Price acceleration
    """

    def __init__(self):
        self._liquidations: deque = deque(maxlen=5000)

    def record_liquidation(self, symbol: str, side: str, quantity: float,
                           price: float, usd_value: float):
        """Record a liquidation event."""
        self._liquidations.append({
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "price": price,
            "usd_value": usd_value,
            "timestamp": time.time(),
        })

    def predict(self, market_data: dict) -> dict:
        """
        Predict liquidation cascade risk.

        Args:
            market_data: dict with open_interest, funding_rate,
                        price, volume, leverage_data

        Returns:
            dict with cascade_risk (0-1), severity, reasoning
        """
        risk_score = 0.0
        reasons = []
        cascade_levels = {
            0.0: "none",
            0.3: "low",
            0.5: "elevated",
            0.7: "high",
            0.9: "critical",
        }

        # 1. Recent liquidation cluster (last 5 minutes)
        now = time.time()
        recent = [l for l in self._liquidations if now - l["timestamp"] < 300]

        if recent:
            total_liq_usd = sum(l["usd_value"] for l in recent)
            liq_count = len(recent)
            # $10M+ in 5 min = elevated risk
            if total_liq_usd > 50_000_000:
                risk_score += 0.3
                reasons.append(f"High liquidation volume: ${total_liq_usd/1e6:.1f}M")
            elif total_liq_usd > 10_000_000:
                risk_score += 0.15

            if liq_count > 20:
                risk_score += 0.15
                reasons.append(f"Liquidation cluster: {liq_count} events")

            # Check if mostly longs being liquidated
            long_liqs = sum(1 for l in recent if l["side"] == "sell")  # Longs cause sells
            if long_liqs > len(recent) * 0.7 and long_liqs > 5:
                risk_score += 0.15
                reasons.append(f"Long squeeze: {long_liqs}/{len(recent)} liquidations are longs")

        # 2. Funding rate extreme
        funding_rate = market_data.get("funding_rate", 0)
        if abs(funding_rate) > 0.003:
            risk_score += 0.15
            reasons.append(f"Extreme funding rate: {funding_rate:.4f}")
        elif abs(funding_rate) > 0.001:
            risk_score += 0.05

        # 3. Open interest decay
        oi_current = market_data.get("open_interest", 0)
        oi_24h_ago = market_data.get("open_interest_24h_ago", oi_current)
        if oi_24h_ago > 0:
            oi_change = (oi_current - oi_24h_ago) / oi_24h_ago
            if oi_change < -0.10:
                risk_score += 0.2
                reasons.append(f"OI dropping: {oi_change:.1%} in 24h")
            elif oi_change < -0.05:
                risk_score += 0.1

        # 4. Price acceleration (convex move)
        price = market_data.get("price", 0)
        price_1h = market_data.get("price_1h_ago", price)
        price_4h = market_data.get("price_4h_ago", price)
        if price_4h > 0 and price_1h > 0:
            move_1h = (price - price_1h) / price_1h
            move_4h = (price - price_4h) / price_4h
            # If recent move is accelerating vs longer period
            if abs(move_1h) > abs(move_4h) * 1.5 and abs(move_1h) > 0.02:
                risk_score += 0.15
                direction = "down" if move_1h < 0 else "up"
                reasons.append(f"Price acceleration {direction}: 1h={move_1h:.2%} vs 4h={move_4h:.2%}")

        # Determine cascade severity
        severity = "none"
        for threshold, label in sorted(cascade_levels.items(), reverse=True):
            if risk_score >= threshold:
                severity = label
                break

        return {
            "cascade_risk": round(min(risk_score, 1.0), 4),
            "severity": severity,
            "reasoning": "; ".join(reasons) if reasons else "No cascade risk detected",
            "recent_liquidations_5m": len(recent),
            "recent_liquidation_volume_usd": round(sum(l["usd_value"] for l in recent), 2) if recent else 0,
        }


class OrderBookImbalanceAnalyzer:
    """
    Analyzes order book imbalance to predict short-term price direction.

    Imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)

    Signals:
      - Strong bid dominance (+0.5+): buying pressure
      - Strong ask dominance (-0.5-): selling pressure
      - Absorption: imbalance narrowing without price change
    """

    @staticmethod
    def analyze(bids: list[list[float]], asks: list[list[float]],
                depth_levels: int = 10) -> dict:
        """
        Analyze order book imbalance at multiple depth levels.

        Args:
            bids: [[price, quantity], ...]
            asks: [[price, quantity], ...]
            depth_levels: number of levels to analyze

        Returns:
            dict with imbalance metrics and signal
        """
        if not bids or not asks:
            return {"imbalance": 0.0, "signal": "neutral",
                    "bid_volume": 0.0, "ask_volume": 0.0}

        bid_vol = sum(q for _, q in bids[:depth_levels])
        ask_vol = sum(q for _, q in asks[:depth_levels])
        total_vol = bid_vol + ask_vol

        if total_vol <= 0:
            return {"imbalance": 0.0, "signal": "neutral",
                    "bid_volume": 0.0, "ask_volume": 0.0}

        imbalance = (bid_vol - ask_vol) / total_vol

        # Depth ratios at different levels
        level_1_bid = bids[0][1] if len(bids) > 0 else 0
        level_1_ask = asks[0][1] if len(asks) > 0 else 0
        level_5_bid = sum(q for _, q in bids[:5]) if len(bids) >= 5 else bid_vol
        level_5_ask = sum(q for _, q in asks[:5]) if len(asks) >= 5 else ask_vol

        # Top-heavy imbalance (more at first level)
        top_imbalance = (level_1_bid - level_1_ask) / (level_1_bid + level_1_ask + 1e-10)
        depth_5_imbalance = (level_5_bid - level_5_ask) / (level_5_bid + level_5_ask + 1e-10)

        # Signal generation
        if imbalance > 0.5 and top_imbalance > 0.3:
            signal = "bullish"
            confidence = min(0.9, 0.5 + abs(imbalance))
        elif imbalance < -0.5 and top_imbalance < -0.3:
            signal = "bearish"
            confidence = min(0.9, 0.5 + abs(imbalance))
        elif abs(imbalance) < 0.1:
            signal = "neutral"
            confidence = 0.3
        else:
            signal = "leaning_bullish" if imbalance > 0 else "leaning_bearish"
            confidence = 0.4 + abs(imbalance) * 0.3

        return {
            "imbalance": round(imbalance, 4),
            "top_imbalance": round(top_imbalance, 4),
            "depth_5_imbalance": round(depth_5_imbalance, 4),
            "bid_volume": round(bid_vol, 4),
            "ask_volume": round(ask_vol, 4),
            "bid_ask_ratio": round(bid_vol / (ask_vol + 1e-10), 4),
            "signal": signal,
            "confidence": round(min(confidence, 1.0), 4),
        }

    @staticmethod
    def detect_absorption(bids: list[list[float]], asks: list[list[float]],
                          trades: list[dict], window: int = 10) -> dict:
        """
        Detect absorption: price staying flat while large volume executes.

        Absorption = market makers absorbing aggressive flow.
        Often precedes a breakout.

        Returns dict with absorption_detected, strength, reasoning.
        """
        if not trades or len(trades) < window:
            return {"absorption_detected": False, "strength": 0.0, "reasoning": "Insufficient data"}

        recent_trades = trades[-window:]
        trade_volume = sum(t.get("quantity", 0) for t in recent_trades)
        buy_volume = sum(t.get("quantity", 0) for t in recent_trades if t.get("side") == "buy")
        sell_volume = sum(t.get("quantity", 0) for t in recent_trades if t.get("side") == "sell")

        # Current book imbalance
        imbalance_data = OrderBookImbalanceAnalyzer.analyze(bids, asks)

        # Absorption: high trade volume + stable price + neutral imbalance
        if trade_volume > 0 and imbalance_data["signal"] == "neutral":
            absorption_score = min(1.0, trade_volume * 0.1)
            if absorption_score > 0.4:
                return {
                    "absorption_detected": True,
                    "strength": round(absorption_score, 4),
                    "trade_volume": round(trade_volume, 4),
                    "buy_pressure": round(buy_volume / (trade_volume + 1e-10), 4),
                    "reasoning": f"Absorption: {trade_volume:.2f} units traded, book neutral",
                }

        return {"absorption_detected": False, "strength": 0.0, "reasoning": "No absorption pattern"}
