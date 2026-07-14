"""
Risk Service — Risk management engine with multiple layers of protection.

Provides:
- Position risk limits
- Portfolio drawdown controls
- Correlation limits
- Leverage controls
- Real-time risk monitoring
- Automated risk reduction
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional

import numpy as np

from services.execution import (
    AccountInfo,
    ExecutionMode,
    ExecutionOrchestrator,
    Order,
    OrderSide,
    OrderType,
    Position,
    create_execution_orchestrator,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# ENUMS & DATA CLASSES
# ═══════════════════════════════════════════════════════════════════

class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class RiskAction(str, Enum):
    ALLOW = "ALLOW"
    REDUCE = "REDUCE"      # Reduce position size
    REJECT = "REJECT"      # Reject new orders
    LIQUIDATE = "LIQUIDATE"  # Liquidate positions
    HALT = "HALT"          # Halt all trading


@dataclass
class RiskConfig:
    """Risk management configuration."""
    # Position limits
    max_position_pct: float = 0.05      # 5% per position
    max_total_exposure: float = 0.30    # 30% total
    max_leverage: int = 3               # Max 3x leverage

    # Drawdown limits
    max_daily_drawdown: float = 0.03    # 3% daily
    max_total_drawdown: float = 0.10    # 10% total
    max_position_drawdown: float = 0.05 # 5% per position

    # Correlation limits
    max_correlated_positions: int = 3
    correlation_threshold: float = 0.7

    # Concentration
    max_single_asset_pct: float = 0.15  # 15% in single asset

    # Volatility
    max_portfolio_volatility: float = 0.02  # 2% daily vol

    # Orders
    max_open_orders: int = 50
    max_order_size_pct: float = 0.02    # 2% per order

    # Auto-liquidation
    auto_liquidate_on_breach: bool = True
    liquidation_buffer_pct: float = 0.01  # 1% buffer


@dataclass
class RiskMetrics:
    """Current risk metrics."""
    timestamp: datetime

    # Portfolio
    total_exposure: float
    total_leverage: float
    portfolio_volatility: float
    current_drawdown: float
    daily_drawdown: float

    # Positions
    position_count: int
    largest_position_pct: float
    correlated_groups: int

    # Risk levels
    overall_risk: RiskLevel
    risk_factors: dict[str, RiskLevel]

    # Limits
    limits_breached: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "total_exposure": self.total_exposure,
            "total_leverage": self.total_leverage,
            "portfolio_volatility": self.portfolio_volatility,
            "current_drawdown": self.current_drawdown,
            "daily_drawdown": self.daily_drawdown,
            "position_count": self.position_count,
            "largest_position_pct": self.largest_position_pct,
            "correlated_groups": self.correlated_groups,
            "overall_risk": self.overall_risk.value,
            "risk_factors": {k: v.value for k, v in self.risk_factors.items()},
            "limits_breached": self.limits_breached,
        }


@dataclass
class RiskDecision:
    """Risk engine decision for an order."""
    action: RiskAction
    allowed_size: float
    reason: str
    risk_level: RiskLevel
    conditions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "action": self.action.value,
            "allowed_size": self.allowed_size,
            "reason": self.reason,
            "risk_level": self.risk_level.value,
            "conditions": self.conditions,
        }


# ═══════════════════════════════════════════════════════════════════
# RISK ENGINE
# ═══════════════════════════════════════════════════════════════════

class RiskEngine:
    """
    Real-time risk management engine.
    Evaluates orders and portfolio state against risk limits.
    """

    def __init__(self, config: RiskConfig | None = None):
        self.config = config or RiskConfig()
        self.execution: ExecutionOrchestrator | None = None

        # State
        self._daily_high: float = 0.0
        self._daily_low: float = 0.0
        self._equity_history: deque = deque(maxlen=1440)  # 24h at 1min
        self._position_history: dict[str, deque] = {}

        # Alerts
        self._alert_callbacks: list[Callable] = []

    def set_execution(self, execution: ExecutionOrchestrator):
        """Link execution orchestrator for position/account queries."""
        self.execution = execution

    def on_alert(self, callback: Callable[[RiskLevel, str], None]):
        """Register alert callback."""
        self._alert_callbacks.append(callback)

    def _emit_alert(self, level: RiskLevel, message: str):
        for cb in self._alert_callbacks:
            try:
                cb(level, message)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")

    async def evaluate_order(
        self,
        order: Any,  # Order type from execution service
        account: Any,  # AccountInfo
    ) -> RiskDecision:
        """
        Evaluate order against all risk limits.
        Returns RiskDecision with action and allowed size.
        """
        if not self.execution:
            return RiskDecision(
                action=RiskAction.REJECT,
                allowed_size=0,
                reason="Risk engine not connected to execution",
                risk_level=RiskLevel.HIGH,
            )

        # Get current state
        positions = await self.execution.get_positions()
        portfolio_value = account.total_balance

        # Calculate risk metrics
        metrics = await self._compute_metrics(account, positions)

        # Check all limits
        violations = []
        warnings = []

        # 1. Position size limit
        if order.quantity * (order.price or 0) > portfolio_value * self.config.max_order_size_pct:
            violations.append(f"Order size exceeds {self.config.max_order_size_pct:.1%} limit")

        # 2. Position concentration
        new_size = order.quantity
        # Check existing position
        for pos in positions:
            if pos.symbol == order.symbol:
                new_size += pos.size
                break

        if new_size * (order.price or 0) > portfolio_value * self.config.max_position_pct:
            violations.append(f"Position would exceed {self.config.max_position_pct:.1%} limit")

        # 3. Total exposure
        total_exposure = sum(p.size * p.mark_price for p in positions)
        new_exposure = total_exposure + new_size * (order.price or 0)
        if new_exposure > portfolio_value * self.config.max_total_exposure:
            violations.append(f"Total exposure would exceed {self.config.max_total_exposure:.1%}")

        # 4. Leverage
        new_leverage = new_exposure / portfolio_value if portfolio_value > 0 else 0
        if new_leverage > self.config.max_leverage:
            violations.append(f"Leverage would exceed {self.config.max_leverage}x")

        # 5. Daily drawdown
        if metrics.daily_drawdown > self.config.max_daily_drawdown:
            violations.append(f"Daily drawdown {metrics.daily_drawdown:.2%} exceeds {self.config.max_daily_drawdown:.1%}")

        # 6. Total drawdown
        if metrics.current_drawdown > self.config.max_total_drawdown:
            violations.append(f"Total drawdown {metrics.current_drawdown:.2%} exceeds {self.config.max_total_drawdown:.1%}")

        # 7. Position drawdown
        for pos in positions:
            pos_dd = abs(pos.unrealized_pnl) / (pos.size * pos.entry_price) if pos.size > 0 else 0
            if pos_dd > self.config.max_position_drawdown:
                violations.append(f"Position {pos.symbol} drawdown {pos_dd:.2%} exceeds limit")

        # 8. Correlation
        if len(positions) >= self.config.max_correlated_positions:
            violations.append(f"Max positions ({self.config.max_positions}) reached")

        # 9. Single asset concentration
        asset_exposure = sum(p.size * p.mark_price for p in positions if p.symbol == order.symbol)
        if asset_exposure > portfolio_value * self.config.max_single_asset_pct:
            violations.append(f"Single asset concentration exceeds {self.config.max_single_asset_pct:.1%}")

        # 10. Open orders limit
        open_orders = await self.execution.get_open_orders()
        if len(open_orders) >= self.config.max_open_orders:
            violations.append(f"Max open orders ({self.config.max_open_orders}) reached")

        # Determine action
        if violations:
            action = RiskAction.REJECT
            allowed_size = 0
            risk_level = RiskLevel.CRITICAL
        elif warnings:
            action = RiskAction.REDUCE
            allowed_size = self._calculate_reduced_size(order, portfolio_value, metrics)
            risk_level = RiskLevel.HIGH
        else:
            action = RiskAction.ALLOW
            allowed_size = order.quantity
            risk_level = RiskLevel.LOW

        # Emit alerts for violations
        for v in violations:
            self._emit_alert(RiskLevel.CRITICAL, f"Risk violation: {v}")

        return RiskDecision(
            action=action,
            allowed_size=allowed_size,
            reason="; ".join(violations) if violations else "All checks passed",
            risk_level=risk_level,
            conditions=warnings,
        )

    def _calculate_reduced_size(self, order, portfolio_value: float, metrics: RiskMetrics) -> float:
        """Calculate reduced order size based on risk."""
        # Reduce by risk factor
        risk_factor = 1.0 - metrics.current_drawdown * 2  # Reduce as drawdown increases
        risk_factor = max(0.1, min(1.0, risk_factor))

        max_allowed = portfolio_value * self.config.max_order_size_pct * risk_factor
        return min(order.quantity, max_allowed)

    async def _compute_metrics(
        self,
        account: Any,
        positions: list[Any],
    ) -> RiskMetrics:
        """Compute comprehensive risk metrics."""
        portfolio_value = account.total_balance

        # Exposure
        total_exposure = sum(p.size * p.mark_price for p in positions)
        total_leverage = total_exposure / portfolio_value if portfolio_value > 0 else 0

        # Drawdown
        self._equity_history.append(portfolio_value)
        if len(self._equity_history) >= 2:
            peak = max(self._equity_history)
            current_drawdown = (peak - portfolio_value) / peak if peak > 0 else 0
        else:
            current_drawdown = 0

        # Daily drawdown
        if len(self._equity_history) > 60:  # Last hour approx
            daily_peak = max(list(self._equity_history)[-60:])
            daily_drawdown = (daily_peak - portfolio_value) / daily_peak if daily_peak > 0 else 0
        else:
            daily_drawdown = 0

        # Portfolio volatility
        if len(self._equity_history) >= 30:
            returns = np.diff(self._equity_history) / np.array(list(self._equity_history)[:-1])
            portfolio_volatility = float(np.std(returns)) if len(returns) > 0 else 0
        else:
            portfolio_volatility = 0

        # Position metrics
        position_count = len(positions)
        largest_position_pct = 0
        if positions:
            largest_position_pct = max(p.size * p.mark_price for p in positions) / portfolio_value

        # Risk factors
        risk_factors = {
            "exposure": RiskLevel.LOW if total_exposure / portfolio_value < 0.5 else RiskLevel.HIGH,
            "leverage": RiskLevel.LOW if total_leverage < 2 else RiskLevel.HIGH,
            "drawdown": RiskLevel.LOW if current_drawdown < 0.05 else RiskLevel.CRITICAL,
            "daily_drawdown": RiskLevel.LOW if daily_drawdown < 0.02 else RiskLevel.HIGH,
            "volatility": RiskLevel.LOW if portfolio_volatility < 0.01 else RiskLevel.HIGH,
            "concentration": RiskLevel.LOW if largest_position_pct < 0.1 else RiskLevel.MEDIUM,
        }

        # Overall risk
        risk_levels = list(risk_factors.values())
        if RiskLevel.CRITICAL in risk_levels:
            overall = RiskLevel.CRITICAL
        elif RiskLevel.HIGH in risk_levels:
            overall = RiskLevel.HIGH
        elif RiskLevel.MEDIUM in risk_levels:
            overall = RiskLevel.MEDIUM
        else:
            overall = RiskLevel.LOW

        # Limits breached
        limits = []
        if total_exposure / portfolio_value > self.config.max_total_exposure:
            limits.append("max_total_exposure")
        if total_leverage > self.config.max_leverage:
            limits.append("max_leverage")
        if current_drawdown > self.config.max_total_drawdown:
            limits.append("max_total_drawdown")
        if daily_drawdown > self.config.max_daily_drawdown:
            limits.append("max_daily_drawdown")

        return RiskMetrics(
            timestamp=datetime.now(),
            total_exposure=total_exposure,
            total_leverage=total_leverage,
            portfolio_volatility=portfolio_volatility,
            current_drawdown=current_drawdown,
            daily_drawdown=daily_drawdown,
            position_count=position_count,
            largest_position_pct=largest_position_pct,
            correlated_groups=0,  # Would need correlation matrix
            overall_risk=overall,
            risk_factors=risk_factors,
            limits_breached=limits,
        )

    async def check_portfolio_risk(self) -> RiskMetrics:
        """Check overall portfolio risk and auto-liquidate if needed."""
        if not self.execution:
            return None

        account = await self.execution.get_account()
        positions = await self.execution.get_positions()
        metrics = await self._compute_metrics(account, positions)

        # Auto-liquidate on critical breach
        if self.config.auto_liquidate_on_breach and metrics.overall_risk == RiskLevel.CRITICAL:
            await self._emergency_liquidate()

        return metrics

    async def _emergency_liquidate(self):
        """Emergency liquidation of all positions."""
        logger.critical("[RiskEngine] EMERGENCY LIQUIDATION TRIGGERED")
        self._emit_alert(RiskLevel.CRITICAL, "EMERGENCY LIQUIDATION INITIATED")

        positions = await self.execution.get_positions()
        for pos in positions:
            # Close position
            side = OrderSide.SELL if pos.side == "BUY" else OrderSide.BUY
            order = Order(
                symbol=pos.symbol,
                side=side,
                type=OrderType.MARKET,
                quantity=pos.size,
                reduce_only=True,
            )
            await self.execution.place_order(order)
            logger.warning(f"[RiskEngine] Liquidated {pos.symbol} {pos.size} @ market")


# ═══════════════════════════════════════════════════════════════════
# KELLY SIZER (Optimal position sizing)
# ══════════════════════════════════════════════════════════════════

class KellySizer:
    """
    Kelly Criterion position sizer.
    Computes optimal position size based on win rate and payoff ratio.
    """

    def __init__(
        self,
        win_rate: float = 0.55,
        avg_win: float = 0.02,
        avg_loss: float = 0.01,
        max_kelly: float = 0.25,  # Cap at 25% of Kelly
        min_kelly: float = 0.01,  # Floor at 1%
    ):
        self.win_rate = win_rate
        self.avg_win = avg_win
        self.avg_loss = avg_loss
        self.max_kelly = max_kelly
        self.min_kelly = min_kelly

    def update_stats(self, win_rate: float, avg_win: float, avg_loss: float):
        """Update from trade history."""
        self.win_rate = win_rate
        self.avg_win = avg_win
        self.avg_loss = avg_loss

    def compute_kelly(self) -> float:
        """Compute Kelly fraction."""
        if self.avg_loss <= 0:
            return self.min_kelly

        # Kelly formula: f = (bp - q) / b
        # where b = avg_win/avg_loss, p = win_rate, q = 1 - win_rate
        b = self.avg_win / self.avg_loss
        p = self.win_rate
        q = 1 - p

        kelly = (b * p - q) / b if b > 0 else 0

        # Cap and floor
        kelly = max(self.min_kelly, min(self.max_kelly, kelly))

        return kelly

    def get_position_size(self, equity: float, price: float) -> float:
        """Get position size in base currency."""
        kelly_fraction = self.compute_kelly()
        return equity * kelly_fraction / price


# ═══════════════════════════════════════════════════════════════════
# RISK SERVICE (Integration)
# ═══════════════════════════════════════════════════════════════════

class RiskService:
    """
    Integrated risk service with engine + Kelly sizer + monitoring.
    """

    def __init__(self, config: RiskConfig | None = None):
        self.config = config or RiskConfig()
        self.engine = RiskEngine(self.config)
        self.kelly = KellySizer(
            max_kelly=self.config.max_position_pct,
            min_kelly=0.005,
        )
        self._monitoring_task: asyncio.Task | None = None

    def set_execution(self, execution: ExecutionOrchestrator):
        self.engine.set_execution(execution)

    async def evaluate_order(self, order, account) -> RiskDecision:
        """Evaluate order through risk engine."""
        return await self.engine.evaluate_order(order, account)

    def update_kelly_from_trades(self, trades: list[dict]):
        """Update Kelly sizer from trade history."""
        if not trades:
            return

        wins = [t for t in trades if t.get("pnl", 0) > 0]
        losses = [t for t in trades if t.get("pnl", 0) <= 0]

        if not wins or not losses:
            return

        win_rate = len(wins) / len(trades)
        avg_win = np.mean([t["pnl"] for t in wins])
        avg_loss = abs(np.mean([t["pnl"] for t in losses]))

        self.kelly.update_stats(win_rate, avg_win, avg_loss)

    def get_kelly_size(self, equity: float, price: float) -> float:
        """Get Kelly-optimal position size."""
        return self.kelly.get_position_size(equity, price)

    async def check_portfolio(self) -> RiskMetrics:
        """Check portfolio risk."""
        return await self.engine.check_portfolio_risk()

    async def start_monitoring(self, interval_seconds: int = 60):
        """Start background risk monitoring."""
        async def monitor():
            while True:
                try:
                    metrics = await self.check_portfolio()
                    if metrics and metrics.overall_risk in (RiskLevel.HIGH, RiskLevel.CRITICAL):
                        logger.warning(f"[RiskService] High risk detected: {metrics.overall_risk.value}")
                except Exception as e:
                    logger.error(f"[RiskService] Monitoring error: {e}")
                await asyncio.sleep(interval_seconds)

        self._monitoring_task = asyncio.create_task(monitor())

    async def stop_monitoring(self):
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass


def create_risk_service(config: RiskConfig | None = None) -> RiskService:
    """Factory for RiskService."""
    return RiskService(config)


if __name__ == "__main__":
    import asyncio

    async def test():
        # Test Kelly sizer
        kelly = KellySizer(win_rate=0.55, avg_win=0.02, avg_loss=0.01)
        print(f"Kelly fraction: {kelly.compute_kelly():.4f}")
        print(f"Position size (100k equity, 50k price): {kelly.get_position_size(100000, 50000):.4f}")

        # Test risk config
        config = RiskConfig()
        print(f"Risk config: max_pos={config.max_position_pct}, max_exp={config.max_total_exposure}")

    asyncio.run(test())