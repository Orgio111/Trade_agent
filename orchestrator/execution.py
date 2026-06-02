"""
QUANTEX Smart Execution Layer — Institutional order execution intelligence.

Features:
  - TWAP (Time-Weighted Average Price) execution
  - VWAP (Volume-Weighted Average Price) execution
  - Iceberg order splitting
  - Smart multi-venue routing
  - Slippage modeling and estimation
  - Implementation shortfall analysis

All execution algorithms minimize market impact and slippage.
"""

import time
import math
import asyncio
import statistics
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ExecutionPlan:
    """Complete execution plan with slice details."""
    symbol: str
    side: str  # "buy" or "sell"
    total_quantity: float
    algorithm: str  # "twap", "vwap", "iceberg", "market", "smart"
    slices: list[dict] = field(default_factory=list)
    estimated_slippage_bps: float = 0.0
    estimated_total_cost: float = 0.0
    duration_seconds: int = 0
    urgency: str = "normal"  # "low", "normal", "high"
    reasoning: str = ""


@dataclass
class VenueInfo:
    """Exchange venue information for routing decisions."""
    name: str
    maker_fee: float
    taker_fee: float
    liquidity_btc: float  # Estimated 1% market depth
    latency_ms: float
    reliability: float  # 0.0 to 1.0
    is_active: bool = True


class TWAPExecutor:
    """
    Time-Weighted Average Price execution.

    Splits a large order into equal time slices over a duration.
    Minimizes market impact by spreading execution evenly.
    """

    @staticmethod
    def plan(symbol: str, side: str, quantity: float,
             duration_minutes: int = 30, num_slices: int = 10) -> ExecutionPlan:
        """
        Create a TWAP execution plan.

        Args:
            symbol: Trading pair
            side: "buy" or "sell"
            quantity: Total quantity to execute
            duration_minutes: Total execution duration
            num_slices: Number of equal slices

        Returns:
            ExecutionPlan with slice details
        """
        if num_slices <= 0 or duration_minutes <= 0:
            return ExecutionPlan(
                symbol=symbol, side=side, total_quantity=quantity,
                algorithm="twap", slices=[],
                estimated_slippage_bps=0.5, estimated_total_cost=0.0,
                duration_seconds=0, urgency="high",
                reasoning="Single execution (no slicing needed)",
            )

        slice_qty = quantity / num_slices
        interval_seconds = (duration_minutes * 60) / num_slices
        slices = []

        for i in range(num_slices):
            slice_time = time.time() + (i * interval_seconds)
            slices.append({
                "slice": i + 1,
                "quantity": round(slice_qty, 8),
                "timestamp": slice_time,
                "delay_ms": int(i * interval_seconds * 1000),
                "type": "limit" if i < num_slices - 1 else "market",
                "reason": f"TWAP slice {i+1}/{num_slices}",
            })

        # Slippage estimation: TWAP reduces effective slippage
        estimated_slippage = 0.3 + (0.2 / math.sqrt(num_slices))
        estimated_cost = quantity * estimated_slippage / 10000  # Rough estimate

        return ExecutionPlan(
            symbol=symbol, side=side, total_quantity=quantity,
            algorithm="twap", slices=slices,
            estimated_slippage_bps=round(estimated_slippage, 2),
            estimated_total_cost=round(estimated_cost, 6),
            duration_seconds=duration_minutes * 60,
            urgency="normal",
            reasoning=f"TWAP: {num_slices} slices over {duration_minutes}min, estimated slippage {estimated_slippage:.1f}bps",
        )


class VWAPExecutor:
    """
    Volume-Weighted Average Price execution.

    Aligns execution with forecasted volume profile.
    More aggressive during high-volume periods for better fills.
    """

    @staticmethod
    def plan(symbol: str, side: str, quantity: float,
             volume_profile: Optional[list[float]] = None,
             num_slices: int = 12) -> ExecutionPlan:
        """
        Create a VWAP execution plan using volume profile.

        Args:
            symbol: Trading pair
            side: "buy" or "sell"
            quantity: Total quantity
            volume_profile: Expected volume distribution per slice (list of floats)
            num_slices: Number of slices

        Returns:
            ExecutionPlan with volume-weighted slices
        """
        if volume_profile is None:
            # Default: U-shaped volume profile (high at open/close)
            volume_profile = [1.2, 1.0, 0.8, 0.7, 0.6, 0.5, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2]

        if len(volume_profile) < num_slices:
            volume_profile = volume_profile[:num_slices] + [0.5] * (num_slices - len(volume_profile))
        if len(volume_profile) > num_slices:
            volume_profile = volume_profile[:num_slices]

        total_vol = sum(volume_profile)
        if total_vol <= 0:
            return TWAPExecutor.plan(symbol, side, quantity, duration_minutes=30, num_slices=num_slices)

        # Normalize weights
        weights = [v / total_vol for v in volume_profile]
        slices = []

        for i in range(num_slices):
            slice_qty = quantity * weights[i]
            slices.append({
                "slice": i + 1,
                "quantity": round(slice_qty, 8),
                "weight_pct": round(weights[i] * 100, 2),
                "volume_relative": volume_profile[i],
                "type": "limit",
                "reason": f"VWAP slice {i+1}/{num_slices} ({weights[i]*100:.1f}% of volume)",
            })

        estimated_slippage = 0.25 + (0.15 / math.sqrt(num_slices))

        return ExecutionPlan(
            symbol=symbol, side=side, total_quantity=quantity,
            algorithm="vwap", slices=slices,
            estimated_slippage_bps=round(estimated_slippage, 2),
            estimated_total_cost=round(quantity * estimated_slippage / 10000, 6),
            duration_seconds=60 * 60,  # ~1 hour VWAP
            urgency="normal",
            reasoning=f"VWAP: {num_slices} slices weighted by volume profile, est. slippage {estimated_slippage:.1f}bps",
        )


class IcebergExecutor:
    """
    Iceberg order execution — shows only a portion of the total order.

    Prevents revealing full order size to the market,
    reducing adverse selection and slippage.
    """

    @staticmethod
    def plan(symbol: str, side: str, total_quantity: float,
             display_size: float = 0.0, price_limit: Optional[float] = None,
             refresh_interval_ms: int = 1000) -> ExecutionPlan:
        """
        Create an iceberg execution plan.

        Args:
            symbol: Trading pair
            side: "buy" or "sell"
            total_quantity: Total quantity to execute
            display_size: Quantity to show per slice (0 = auto 10%)
            price_limit: Max/min price limit
            refresh_interval_ms: Time between slice refreshes

        Returns:
            ExecutionPlan with iceberg slices
        """
        if display_size <= 0:
            display_size = total_quantity * 0.10  # Show 10% by default

        num_slices = max(1, int(math.ceil(total_quantity / display_size)))
        slices = []

        for i in range(num_slices):
            remaining = total_quantity - sum(s["quantity"] for s in slices)
            qty = min(display_size, remaining)
            slices.append({
                "slice": i + 1,
                "quantity": round(qty, 8),
                "display_size": round(min(display_size, qty), 8),
                "type": "limit",
                "price_limit": price_limit,
                "refresh_delay_ms": refresh_interval_ms,
                "reason": f"Iceberg slice {i+1}/{num_slices} (showing {display_size:.6f})",
            })

        # Iceberg has higher slippage due to partial fills but better price
        estimated_slippage = 0.4 + (0.1 * num_slices / 20)

        return ExecutionPlan(
            symbol=symbol, side=side, total_quantity=total_quantity,
            algorithm="iceberg", slices=slices,
            estimated_slippage_bps=round(estimated_slippage, 2),
            estimated_total_cost=round(total_quantity * estimated_slippage / 10000, 6),
            duration_seconds=num_slices * refresh_interval_ms // 1000,
            urgency="low",
            reasoning=f"Iceberg: {num_slices} slices of {display_size:.6f}, slippage {estimated_slippage:.1f}bps",
        )


class SmartOrderRouter:
    """
    Intelligent multi-venue order routing.

    Routes orders to the best venue based on:
      - Available liquidity
      - Fee structure
      - Latency
      - Historical fill quality
    """

    def __init__(self):
        self.venues: dict[str, VenueInfo] = {
            "binance": VenueInfo("binance", 0.0002, 0.0004, 100.0, 50, 0.99, True),
            "bybit": VenueInfo("bybit", 0.0002, 0.0005, 80.0, 60, 0.98, True),
            "hyperliquid": VenueInfo("hyperliquid", 0.0003, 0.0005, 60.0, 30, 0.95, True),
        }

    def add_venue(self, venue: VenueInfo):
        """Add or update a venue."""
        self.venues[venue.name] = venue

    def route(self, order: dict) -> list[dict]:
        """
        Route an order to the optimal venue(s).

        For large orders, splits across multiple venues.
        For small orders, routes to a single best venue.

        Returns list of {venue, quantity, reason} allocations.
        """
        quantity = order.get("quantity", 0)
        side = order.get("side", "buy")
        urgency = order.get("urgency", "normal")

        # Filter active venues with sufficient liquidity
        active = {k: v for k, v in self.venues.items() if v.is_active}

        if not active:
            return [{"venue": "binance", "quantity": quantity, "reason": "Fallback to Binance"}]

        if quantity < 0.1:
            # Small order: single best venue
            score = {
                name: self._score_venue(v, order)
                for name, v in active.items()
            }
            best = max(score, key=score.get)
            return [{"venue": best, "quantity": quantity, "reason": f"Best score: {score[best]:.2f}"}]

        # Large order: split across venues
        scored = sorted(
            [(name, self._score_venue(v, order)) for name, v in active.items()],
            key=lambda x: x[1],
            reverse=True,
        )
        total_score = sum(s[1] for s in scored)
        if total_score <= 0:
            return [{"venue": "binance", "quantity": quantity, "reason": "Fallback"}]

        allocations = []
        remaining = quantity
        for name, score_val in scored:
            alloc = quantity * (score_val / total_score)
            alloc = min(alloc, remaining)
            if alloc > 0:
                allocations.append({
                    "venue": name,
                    "quantity": round(alloc, 8),
                    "score": round(score_val, 2),
                    "reason": f"{score_val/total_score*100:.0f}% allocation",
                })
                remaining -= alloc
            if remaining <= 0:
                break

        # Assign remaining to top venue
        if remaining > 0 and allocations:
            allocations[0]["quantity"] += remaining
            allocations[0]["quantity"] = round(allocations[0]["quantity"], 8)

        return allocations

    def _score_venue(self, venue: VenueInfo, order: dict) -> float:
        """Score a venue for a specific order. Higher = better."""
        score = 0.0

        # Fee score (lower is better)
        taker_fee = venue.taker_fee if order.get("type", "market") == "market" else venue.maker_fee
        fee_score = max(0, (0.001 - taker_fee) / 0.001) * 30
        score += fee_score

        # Liquidity score
        liq_score = min(1.0, venue.liquidity_btc / 100) * 30
        score += liq_score

        # Latency score (lower is better)
        lat_score = max(0, (200 - venue.latency_ms) / 200) * 20
        score += lat_score

        # Reliability score
        score += venue.reliability * 20

        return score


class SlippageEstimator:
    """
    Estimate execution slippage based on order size, liquidity, and volatility.
    """

    @staticmethod
    def estimate(quantity: float, price: float, volume_24h: float,
                 volatility_bps: float = 50, spread_bps: float = 3) -> dict:
        """
        Estimate expected slippage for an order.

        Uses a simplified Almgren-Chriss market impact model.

        Args:
            quantity: Order size in base currency
            price: Current market price
            volume_24h: 24-hour volume in quote currency
            volatility_bps: Daily volatility in basis points
            spread_bps: Current bid-ask spread in basis points

        Returns:
            dict with expected_slippage_bps, permanent_impact, temporary_impact
        """
        if volume_24h <= 0 or price <= 0:
            return {"expected_slippage_bps": spread_bps, "permanent_impact": 0, "temporary_impact": spread_bps}

        order_value = quantity * price
        participation_rate = order_value / volume_24h  # % of daily volume

        # Almgren-Chriss simplified model
        # Permanent impact: ~0.142 * sigma * (Q/V)^0.5
        sigma = volatility_bps / 10000  # Convert to decimal
        participation = max(0.001, min(1.0, participation_rate))
        permanent_impact = 0.142 * sigma * (participation ** 0.5)

        # Temporary impact: spread + 0.142 * sigma * |Q/V|^0.6 + permanent
        temporary_impact = (spread_bps / 10000) + 0.142 * sigma * (participation ** 0.6)

        total_impact_bps = (permanent_impact + temporary_impact) * 10000

        return {
            "expected_slippage_bps": round(total_impact_bps, 2),
            "permanent_impact_bps": round(permanent_impact * 10000, 2),
            "temporary_impact_bps": round(temporary_impact * 10000, 2),
            "participation_rate": round(participation * 100, 4),
            "order_value_usd": round(order_value, 2),
        }


class ExecutionOrchestrator:
    """
    Unified execution orchestrator — chooses the best algorithm,
    routes to the best venue, and manages execution lifecycle.
    """

    def __init__(self):
        self.router = SmartOrderRouter()
        self.slippage = SlippageEstimator()

    def create_execution_plan(self, order: dict) -> ExecutionPlan:
        """
        Create optimal execution plan for an order.

        Algorithm selection:
          - Small orders (< 0.1 BTC): market if urgent, limit if patient
          - Medium orders (0.1-1.0 BTC): TWAP
          - Large orders (1.0-10 BTC): VWAP with iceberg
          - Very large (> 10 BTC): Iceberg across multiple venues

        Args:
            order: dict with symbol, side, quantity, price, urgency, volume_24h

        Returns:
            ExecutionPlan with algorithm, slices, routing, and cost estimates
        """
        symbol = order.get("symbol", "")
        side = order.get("side", "buy")
        qty = order.get("quantity", 0)
        price = order.get("price", 0)
        urgency = order.get("urgency", "normal")
        volume_24h = order.get("volume_24h", 10_000_000)  # Default $10M

        # Estimate slippage
        slippage_est = self.slippage.estimate(qty, price, volume_24h)

        # Route venues
        route_allocations = self.router.route(order)

        # Choose algorithm based on order size and urgency
        if qty < 0.01 or urgency == "high":
            # Small/urgent: market or TWAP with few slices
            plan = TWAPExecutor.plan(symbol, side, qty,
                                     duration_minutes=5 if urgency == "high" else 15,
                                     num_slices=3 if urgency == "high" else 6)
            plan.algorithm = "market" if urgency == "high" else "twap"
        elif qty < 0.5:
            # Medium: TWAP
            plan = TWAPExecutor.plan(symbol, side, qty,
                                     duration_minutes=30, num_slices=10)
        elif qty < 5.0:
            # Large: VWAP with volume profile
            plan = VWAPExecutor.plan(symbol, side, qty, num_slices=12)
        else:
            # Very large: Iceberg across venues
            plan = IcebergExecutor.plan(symbol, side, qty, display_size=qty * 0.05)
            plan.algorithm = "iceberg_multi"

        # Update plan with routing and slippage estimates
        plan.estimated_slippage_bps = slippage_est["expected_slippage_bps"]
        plan.estimated_total_cost = (qty * price * slippage_est["expected_slippage_bps"] / 10000
                                     if price > 0 else 0)
        plan.reasoning += f" | Routing: {route_allocations}"

        return plan


# Convenience function for getting the best execution plan
def get_execution_plan(order: dict) -> ExecutionPlan:
    """Quick entry point to get an execution plan for an order."""
    orch = ExecutionOrchestrator()
    return orch.create_execution_plan(order)
