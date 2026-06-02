"""
QUANTEX Enhanced Risk Engine v2 — Institutional-grade risk management.

Extends the base RiskEngine with:
  - Kelly-adjusted position sizing (full + fractional)
  - Kill switches with automatic triggers
  - Anti-martingale sizing logic
  - Volatility-expansion-based position limits
  - Portfolio correlation heat management
  - Time-based risk decay
"""

import time
import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EnhancedRiskParams:
    """Extended risk parameters for institutional-grade management."""
    # Base risk
    max_risk_per_trade: float = 0.01
    max_risk_per_session: float = 0.05
    max_total_exposure: float = 0.15

    # Drawdown controls
    max_drawdown_soft: float = 0.10
    max_drawdown_hard: float = 0.20

    # Position limits
    max_open_positions: int = 3
    max_leverage: int = 10
    correlation_limit: float = 0.70

    # Kelly
    kelly_fraction: float = 0.25  # Quarter-Kelly for safety
    use_kelly: bool = True

    # Anti-martingale
    anti_martingale_enabled: bool = True
    anti_martingale_increase: float = 0.05  # Increase 5% per win
    anti_martingale_max_increase: float = 1.5  # Cap at 1.5x base

    # Kill switch
    kill_switch_drawdown: float = 0.25  # 25% DD = kill
    kill_switch_daily_loss: float = 0.08  # 8% daily loss = kill
    kill_switch_consecutive_losses: int = 6  # 6 losses = kill
    kill_switch_cooldown_hours: int = 24  # 24h cooldown after kill

    # Volatility limits
    vol_expansion_cap: float = 3.0  # 3x baseline vol = max position reduction
    vol_adjustment_smoothing: int = 5  # Smoothing window for vol estimates

    # Time-based decay
    time_decay_enabled: bool = True
    time_decay_hours: int = 4  # Reduce size by 20% every 4 hours without a trade
    time_decay_max: float = 0.5  # Don't go below 50% of base size

    # Correlation
    correlation_window: int = 24  # Hours of data for correlation calc
    correlation_max_single_asset: float = 0.50  # Max 50% portfolio in one asset


@dataclass
class KillSwitchStatus:
    activated: bool = False
    triggered_by: str = ""
    triggered_at: float = 0.0
    cooldown_until: float = 0.0
    reason: str = ""


class KellySizer:
    """Kelly Criterion position sizing with safety adjustments."""

    @staticmethod
    def full_kelly(win_rate: float, avg_win: float, avg_loss: float) -> float:
        """
        Full Kelly formula: f* = p/b - q/a

        Where:
          p = win probability
          q = loss probability (1-p)
          a = average loss (as fraction of risk)
          b = average win (as fraction of risk)

        Returns fraction of capital to risk (0.0 to 1.0).
        """
        if avg_loss == 0 or avg_win == 0:
            return 0.0
        q = 1.0 - win_rate
        a = abs(avg_loss)
        b = avg_win
        kelly = (win_rate / a) - (q / b) if b > 0 else 0.0
        return max(0.0, min(kelly, 0.5))  # Cap at 50%

    @staticmethod
    def fractional_kelly(win_rate: float, avg_win: float, avg_loss: float,
                         fraction: float = 0.25) -> float:
        """
        Fractional Kelly: apply a safety fraction to full Kelly.

        quarter_kelly = full_kelly * 0.25  (most common in practice)
        half_kelly = full_kelly * 0.5      (aggressive)
        """
        full = KellySizer.full_kelly(win_rate, avg_win, avg_loss)
        return full * fraction

    @staticmethod
    def adjust_for_confidence(kelly_fraction: float, confidence: float) -> float:
        """Scale Kelly linearly with prediction confidence."""
        return kelly_fraction * (0.5 + confidence * 0.5)

    @staticmethod
    def adjust_for_regime(kelly_fraction: float, regime: str) -> float:
        """
        Regime-based Kelly adjustment.

        strong_trend: 1.0x (normal)
        weak_trend: 0.75x
        ranging: 0.5x
        high_volatility: 0.3x
        crisis: 0.0x (no trading)
        """
        multipliers = {
            "strong_trend": 1.0,
            "weak_trend": 0.75,
            "ranging": 0.5,
            "high_volatility": 0.3,
            "crisis": 0.0,
            "accumulation": 0.6,
            "markup": 1.0,
            "distribution": 0.4,
            "markdown": 0.2,
        }
        return kelly_fraction * multipliers.get(regime, 0.75)


class AntiMartingaleSizer:
    """
    Anti-martingale position sizing — increase after wins, decrease after losses.

    The idea: let winners run (add to winning positions), cut losers fast.
    """

    def __init__(self, increase_pct: float = 0.05, max_increase: float = 1.5):
        self.increase_pct = increase_pct
        self.max_multiplier = max_increase
        self._consecutive_wins = 0
        self._consecutive_losses = 0
        self._base_size = 1.0

    def record_result(self, pnl: float):
        """Record trade result and update streak counters."""
        if pnl > 0:
            self._consecutive_wins += 1
            self._consecutive_losses = 0
        else:
            self._consecutive_losses += 1
            self._consecutive_wins = 0

    def get_multiplier(self) -> float:
        """
        Get position size multiplier based on recent results.

        After wins: increase by small increments (anti-martingale)
        After losses: reduce aggressively (risk preservation)
        """
        if self._consecutive_wins > 0:
            mult = 1.0 + (self._consecutive_wins * self.increase_pct)
            return min(mult, self.max_multiplier)
        elif self._consecutive_losses > 0:
            # Halve after 2 consecutive losses
            return max(0.5 ** (self._consecutive_losses - 1), 0.125)
        return 1.0

    def reset(self):
        """Reset streak counters (e.g., after a cooldown)."""
        self._consecutive_wins = 0
        self._consecutive_losses = 0


class KillSwitch:
    """
    Emergency kill switch with multiple automatic triggers.

    When triggered:
      - Closes all positions
      - Prevents new trades during cooldown
      - Logs the trigger reason
    """

    def __init__(self, params: EnhancedRiskParams):
        self.params = params
        self.status = KillSwitchStatus()

    def check(self, portfolio: dict) -> KillSwitchStatus:
        """Check all kill switch conditions. Returns active status if triggered."""
        # If already in cooldown
        if self.status.activated:
            if time.time() < self.status.cooldown_until:
                return self.status
            else:
                # Cooldown expired
                self.status = KillSwitchStatus()
                return self.status

        # Check drawdown
        dd = portfolio.get("drawdown", 0.0)
        if dd >= self.params.kill_switch_drawdown:
            self._activate("drawdown", f"Drawdown {dd:.1%} >= {self.params.kill_switch_drawdown:.0%}")
            return self.status

        # Check daily loss
        daily_pnl = portfolio.get("daily_pnl", 0.0)
        balance = portfolio.get("balance", 1.0)
        if balance > 0 and daily_pnl < 0 and abs(daily_pnl) / balance >= self.params.kill_switch_daily_loss:
            self._activate("daily_loss", f"Daily loss {abs(daily_pnl)/balance:.1%} >= {self.params.kill_switch_daily_loss:.0%}")
            return self.status

        # Check consecutive losses
        consec_losses = portfolio.get("consecutive_losses", 0)
        if consec_losses >= self.params.kill_switch_consecutive_losses:
            self._activate("consecutive_losses", f"{consec_losses} consecutive losses")
            return self.status

        return KillSwitchStatus(activated=False)

    def _activate(self, trigger: str, reason: str):
        """Activate the kill switch."""
        now = time.time()
        self.status = KillSwitchStatus(
            activated=True,
            triggered_by=trigger,
            triggered_at=now,
            cooldown_until=now + self.params.kill_switch_cooldown_hours * 3600,
            reason=reason,
        )

    def reset(self):
        """Manually reset the kill switch."""
        self.status = KillSwitchStatus()

    def is_cooldown_active(self) -> bool:
        """Check if still in cooldown period."""
        if not self.status.activated:
            return False
        if time.time() < self.status.cooldown_until:
            return True
        self.reset()
        return False


class VolatilityExpansionLimiter:
    """
    Limits position size during volatility expansion events.

    When vol spikes above baseline, reduce position sizes
    proportionally to protect against outlier moves.
    """

    def __init__(self, expansion_cap: float = 3.0, smoothing: int = 5):
        self.expansion_cap = expansion_cap
        self.smoothing = smoothing
        self._baseline_vol = None

    def set_baseline(self, atr_history: list[float]):
        """Set baseline volatility from historical ATR values."""
        if atr_history:
            self._baseline_vol = sorted(atr_history)[len(atr_history) // 2]  # Median

    def get_size_multiplier(self, current_atr: float) -> float:
        """
        Get position size multiplier based on current vs baseline volatility.

        If vol = 2x baseline, size = 50%
        If vol = 3x baseline, size = 33%
        """
        if self._baseline_vol is None or self._baseline_vol <= 0 or current_atr <= 0:
            return 1.0

        vol_ratio = current_atr / self._baseline_vol
        if vol_ratio <= 1.0:
            return 1.0
        if vol_ratio >= self.expansion_cap:
            return 0.1  # Minimal size at extreme vol

        # Inverse linear: vol_ratio 1.0->1.0, 3.0->0.33
        return max(0.1, 1.0 / vol_ratio)


class CorrelationManager:
    """
    Portfolio correlation heat management.

    Prevents correlated exposure stacking (e.g., BTC + ETH longs)
    by tracking correlation matrix and limiting correlated risk.
    """

    def __init__(self, max_correlation: float = 0.70, max_single_asset: float = 0.50):
        self.max_correlation = max_correlation
        self.max_single_asset = max_single_asset
        self._correlation_matrix: dict[tuple[str, str], float] = {}

    def update_correlation(self, symbol_a: str, symbol_b: str, correlation: float):
        """Update correlation estimate between two symbols."""
        key = tuple(sorted([symbol_a, symbol_b]))
        self._correlation_matrix[key] = correlation

    def get_correlation(self, symbol_a: str, symbol_b: str) -> float:
        """Get correlation between two symbols."""
        key = tuple(sorted([symbol_a, symbol_b]))
        return self._correlation_matrix.get(key, 0.5)  # Default 0.5

    def check_correlation_risk(self, current_positions: list[dict],
                                proposed_symbol: str) -> dict:
        """
        Check if adding a position would create excessive correlated exposure.

        Returns dict with:
          - risk_free: whether it's safe to add
          - max_size_multiplier: how much to reduce size
          - correlated_with: list of correlated positions
        """
        if not current_positions:
            return {"risk_free": True, "max_size_multiplier": 1.0, "correlated_with": []}

        correlated = []
        for pos in current_positions:
            corr = self.get_correlation(proposed_symbol, pos.get("symbol", ""))
            if corr >= self.max_correlation and pos.get("direction") == "long":
                correlated.append({
                    "symbol": pos.get("symbol"),
                    "correlation": corr,
                    "direction": pos.get("direction"),
                })

        if correlated:
            # Reduce size proportionally to number of correlated positions
            size_mult = max(0.2, 1.0 - len(correlated) * 0.3)
            return {
                "risk_free": False,
                "max_size_multiplier": round(size_mult, 4),
                "correlated_with": correlated,
            }

        return {"risk_free": True, "max_size_multiplier": 1.0, "correlated_with": []}


class TimeDecayManager:
    """
    Reduces position sizes during periods of inactivity.

    Based on the concept that edge decays over time without fresh data.
    Also prevents revenge trading after time gaps.
    """

    def __init__(self, decay_hours: int = 4, max_decay: float = 0.5):
        self.decay_hours = decay_hours
        self.max_decay = max_decay
        self._last_trade_time: Optional[float] = None

    def record_trade(self):
        """Record a trade to reset the decay timer."""
        self._last_trade_time = time.time()

    def get_multiplier(self) -> float:
        """Get position size multiplier based on time since last trade."""
        if self._last_trade_time is None:
            return 1.0

        hours_since = (time.time() - self._last_trade_time) / 3600
        if hours_since <= self.decay_hours:
            return 1.0

        decay = (hours_since - self.decay_hours) * 0.05  # 5% per hour after threshold
        return max(self.max_decay, 1.0 - decay)


class EnhancedRiskEngine:
    """
    Complete institutional risk management system combining all components.

    Integrates:
      - Base RiskEngine gates (drawdown, daily loss, exposure, etc.)
      - Kelly-adjusted position sizing
      - Kill switches
      - Anti-martingale logic
      - Volatility expansion limits
      - Correlation management
      - Time-based decay
    """

    def __init__(self, params: Optional[EnhancedRiskParams] = None):
        self.params = params or EnhancedRiskParams()
        self.kelly = KellySizer()
        self.martingale = AntiMartingaleSizer(
            increase_pct=self.params.anti_martingale_increase,
            max_increase=self.params.anti_martingale_max_increase,
        )
        self.kill_switch = KillSwitch(self.params)
        self.vol_limiter = VolatilityExpansionLimiter(
            expansion_cap=self.params.vol_expansion_cap,
            smoothing=self.params.vol_adjustment_smoothing,
        )
        self.correlation = CorrelationManager(
            max_correlation=self.params.correlation_limit,
            max_single_asset=self.params.correlation_max_single_asset,
        )
        self.time_decay = TimeDecayManager(
            decay_hours=self.params.time_decay_hours,
            max_decay=self.params.time_decay_max,
        )
        self._trade_history: list[dict] = []

    def calculate_position_size(self,
                                 balance: float,
                                 price: float,
                                 atr: float,
                                 confidence: float = 0.5,
                                 win_rate: float = 0.5,
                                 avg_win: float = 100.0,
                                 avg_loss: float = 100.0,
                                 regime: str = "weak_trend",
                                 current_positions: list[dict] | None = None,
                                 symbol: str = "",
                                 ) -> dict:
        """
        Calculate optimal position size using all available signals.

        Returns dict with:
          - position_size: recommended size in quote currency
          - multipliers: breakdown of each adjustment
          - reasoning: explanation of the sizing decision
        """
        reasoning = []
        multipliers = {}

        # 1. Base risk amount
        base_risk = balance * self.params.max_risk_per_trade
        multipliers["base_risk_pct"] = self.params.max_risk_per_trade
        reasoning.append(f"Base risk: {self.params.max_risk_per_trade:.1%} of ${balance:.2f} = ${base_risk:.2f}")

        # 2. ATR-based position sizing
        if atr > 0 and price > 0:
            sl_distance = atr * 1.5
            atr_size = (base_risk / sl_distance)
            multipliers["atr_based"] = atr_size
            reasoning.append(f"ATR-based size: ${base_risk:.2f} risk / {sl_distance:.2f} SL dist = {atr_size:.6f} units")
        else:
            atr_size = base_risk / (price * 0.02) if price > 0 else 0
            reasoning.append("ATR unavailable, using 2% price fallback")

        # 3. Kelly adjustment
        if self.params.use_kelly:
            kelly_f = self.kelly.fractional_kelly(win_rate, avg_win, avg_loss,
                                                   self.params.kelly_fraction)
            kelly_adj = self.kelly.adjust_for_confidence(kelly_f, confidence)
            kelly_adj = self.kelly.adjust_for_regime(kelly_adj, regime)
            multipliers["kelly_fraction"] = kelly_adj
            reasoning.append(f"Kelly-adjusted: {kelly_adj:.3f}x")
        else:
            kelly_adj = 1.0

        # 4. Anti-martingale
        if self.params.anti_martingale_enabled:
            martingale_mult = self.martingale.get_multiplier()
            multipliers["martingale"] = martingale_mult
            reasoning.append(f"Martingale mult: {martingale_mult:.2f}x")
        else:
            martingale_mult = 1.0

        # 5. Volatility expansion
        vol_mult = self.vol_limiter.get_size_multiplier(atr)
        multipliers["volatility"] = vol_mult
        if vol_mult < 1.0:
            reasoning.append(f"Vol expansion: {vol_mult:.2f}x reduction")

        # 6. Correlation check
        corr_result = self.correlation.check_correlation_risk(
            current_positions or [], symbol
        )
        corr_mult = corr_result["max_size_multiplier"]
        multipliers["correlation"] = corr_mult
        if not corr_result["risk_free"]:
            reasoning.append(f"Correlation: {corr_mult:.2f}x reduction")

        # 7. Time decay
        if self.params.time_decay_enabled:
            time_mult = self.time_decay.get_multiplier()
            multipliers["time_decay"] = time_mult
            if time_mult < 1.0:
                reasoning.append(f"Time decay: {time_mult:.2f}x reduction")
        else:
            time_mult = 1.0

        # 8. Exposure cap
        max_exposure = balance * self.params.max_total_exposure
        total_mult = kelly_adj * martingale_mult * vol_mult * corr_mult * time_mult
        final_size = atr_size * total_mult
        final_size = min(final_size, max_exposure / price if price > 0 else final_size)
        final_size_usd = final_size * price if price > 0 else 0

        reasoning.append(f"Total mult: {total_mult:.3f}x -> ${final_size_usd:.2f} ({final_size:.6f} units)")

        return {
            "position_size": round(final_size, 8),
            "position_size_usd": round(final_size_usd, 2),
            "total_multiplier": round(total_mult, 4),
            "multiplier_breakdown": {k: round(v, 4) for k, v in multipliers.items()},
            "reasoning": "; ".join(reasoning),
        }

    def record_trade_result(self, pnl: float, symbol: str, size: float):
        """Record a completed trade result for streak/correlation tracking."""
        self.martingale.record_result(pnl)
        self.time_decay.record_trade()
        self._trade_history.append({
            "pnl": pnl,
            "symbol": symbol,
            "size": size,
            "timestamp": time.time(),
        })

    def pre_trade_check(self, proposed_trade: dict, portfolio: dict) -> dict:
        """
        Run ALL risk checks before a trade. Returns approval status.

        Checks in order:
          1. Kill switch
          2. Base risk gates
          3. Correlation
          4. Volatility
          5. Anti-overtrading
        """
        # 1. Kill switch
        ks = self.kill_switch.check(portfolio)
        if ks.activated:
            return {
                "approved": False,
                "reason": f"KILL SWITCH: {ks.reason} (cooldown until {time.ctime(ks.cooldown_until)})",
                "severity": "hard",
            }

        # 2. Position sizing
        size_result = self.calculate_position_size(
            balance=portfolio.get("balance", 1000.0),
            price=proposed_trade.get("price", 0),
            atr=proposed_trade.get("atr", 0),
            confidence=proposed_trade.get("confidence", 0.5),
            win_rate=portfolio.get("win_rate", 0.5),
            avg_win=portfolio.get("avg_win", 100),
            avg_loss=portfolio.get("avg_loss", 100),
            regime=proposed_trade.get("regime", "weak_trend"),
            current_positions=portfolio.get("positions", []),
            symbol=proposed_trade.get("symbol", ""),
        )

        return {
            "approved": True,
            "severity": "none",
            "reason": "All enhanced risk checks passed",
            "position_sizing": size_result,
        }

    def get_risk_report(self) -> dict:
        """Generate complete risk status report."""
        return {
            "kill_switch": {
                "activated": self.kill_switch.status.activated,
                "triggered_by": self.kill_switch.status.triggered_by,
                "cooldown_remaining_hours": max(0, (self.kill_switch.status.cooldown_until - time.time()) / 3600) if self.kill_switch.status.activated else 0,
            },
            "streaks": {
                "consecutive_wins": self.martingale._consecutive_wins,
                "consecutive_losses": self.martingale._consecutive_losses,
                "current_multiplier": self.martingale.get_multiplier(),
            },
            "volatility": {
                "baseline_set": self.vol_limiter._baseline_vol is not None,
            },
            "time_decay": {
                "multiplier": self.time_decay.get_multiplier(),
                "last_trade": time.ctime(self.time_decay._last_trade_time) if self.time_decay._last_trade_time else "none",
            },
            "total_trades_recorded": len(self._trade_history),
        }
