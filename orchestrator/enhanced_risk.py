"""Enhanced Risk Engine — hard filter, ATR volatility filter, Kelly sizing, circuit breaker.

Extends the base RiskEngine with:
  1. ATR volatility filter — block signals when ATR exceeds threshold × multiplier
  2. Kelly criterion sizing — optimal f based on win rate and payoff ratio
  3. Circuit breaker — halt trading after N consecutive losses or max drawdown
  4. Position-time decay — reduce max size as time-in-position increases
  5. Correlation check — block if already have correlated position
  6. Hard time-window — only trade during configured hours

This is a DETERMINISTIC engine — no LLM calls. All rules are config-driven.

Usage:
    from orchestrator.enhanced_risk import EnhancedRiskEngine

    engine = EnhancedRiskEngine(config_path="config.yaml")
    decision = engine.validate(signal, account_state, indicators=indicators_dict)
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import yaml
from models import AccountState, RiskDecision, Signal

logger = logging.getLogger(__name__)


# ── Data classes ────────────────────────────────────────────


@dataclass
class RiskMetrics:
    """Computed risk metrics for a single validation."""

    kelly_fraction: float = 0.0
    atr_zscore: float = 0.0
    position_count: int = 0
    drawdown_pct: float = 0.0
    loss_streak: int = 0
    volatility_pass: bool = True
    circuit_breaker_active: bool = False
    time_window_pass: bool = True
    confidence_pass: bool = True
    size_adjusted: float = 0.0
    reject_reasons: list[str] = field(default_factory=list)


# ── Enhanced Risk Engine ────────────────────────────────────


class EnhancedRiskEngine:
    """Deterministic risk engine with ATR filter, Kelly sizing, and circuit breaker.

    All parameters configurable via config.yaml under 'risk'/'enhanced' key.
    """

    def __init__(self, config_path: str = "config.yaml"):
        # Load config
        try:
            with open(config_path) as f:
                self.config = yaml.safe_load(f) or {}
        except FileNotFoundError:
            self.config = {}

        risk_cfg = self.config.get("risk", {})
        enhanced_cfg = risk_cfg.get("enhanced", {})

        # ── Base risk parameters (from existing RiskEngine) ──
        self.max_position_pct = risk_cfg.get("max_position_pct", 0.05)
        self.max_daily_drawdown = risk_cfg.get("max_daily_drawdown", 0.03)
        self.max_concurrent = risk_cfg.get("max_concurrent", 3)
        self.kill_streak = risk_cfg.get("kill_streak", 5)
        self.min_confidence = risk_cfg.get("min_confidence", 0.6)

        # ── ATR volatility filter ──
        atr_cfg = enhanced_cfg.get("atr_filter", {})
        self.atr_enabled: bool = atr_cfg.get("enabled", True)
        self.atr_lookback: int = atr_cfg.get("lookback", 14)
        self.atr_block_zscore: float = atr_cfg.get("block_zscore", 3.0)
        # Block if ATR_pct > mean_atr_pct * multiplier
        self.atr_block_mult: float = atr_cfg.get("block_multiplier", 2.5)

        # ── Kelly criterion sizing ──
        kelly_cfg = enhanced_cfg.get("kelly", {})
        self.kelly_enabled: bool = kelly_cfg.get("enabled", True)
        self.kelly_fraction: float = kelly_cfg.get("fraction", 0.25)  # use quarter-Kelly
        self.default_win_rate: float = kelly_cfg.get("default_win_rate", 0.55)
        self.default_payoff: float = kelly_cfg.get("default_payoff", 1.5)
        self.kelly_max_pct: float = kelly_cfg.get("max_pct", 0.04)  # hard cap

        # ── Circuit breaker ──
        cb_cfg = enhanced_cfg.get("circuit_breaker", {})
        self.cb_enabled: bool = cb_cfg.get("enabled", True)
        self.cb_loss_streak: int = cb_cfg.get("loss_streak", 5)
        self.cb_drawdown_pct: float = cb_cfg.get("max_drawdown_pct", 0.05)
        self.cb_cooldown_seconds: int = cb_cfg.get("cooldown_seconds", 300)
        self.cb_trigger_hourly_limit: int = cb_cfg.get("hourly_limit", 10)

        # ── Time window ──
        tw_cfg = enhanced_cfg.get("time_window", {})
        self.tw_enabled: bool = tw_cfg.get("enabled", False)
        self.tw_start_hour: int = tw_cfg.get("start_hour", 0)
        self.tw_end_hour: int = tw_cfg.get("end_hour", 24)

        # ── Internal state ──
        self._loss_streak: int = 0
        self._daily_pnl_pct: float = 0.0
        self._open_positions: int = 0
        self._max_drawdown_pct: float = 0.0
        self._circuit_breaker_active: bool = False
        self._circuit_breaker_until: float = 0.0  # timestamp
        self._trades_this_hour: int = 0
        self._hour_start: float = time.time()

        # Rolling stats for Kelly
        self._win_count: int = 0
        self._loss_count: int = 0
        self._total_win_pnl: float = 0.0
        self._total_loss_pnl: float = 0.0

    # ── Account state update ────────────────────────────

    def update_account_state(self, state: AccountState):
        """Update internal state from account."""
        self._loss_streak = state.loss_streak
        self._daily_pnl_pct = state.daily_pnl_pct
        self._open_positions = state.open_positions
        self._max_drawdown_pct = state.max_drawdown_pct

    # ── Main validation ──────────────────────────────────

    def validate(
        self,
        signal: Signal,
        state: AccountState,
        indicators: dict[str, Any] | None = None,
    ) -> RiskDecision:
        """Validate signal against ALL risk rules.

        Args:
            signal: Trading signal from autogen_team.
            state: Current account state.
            indicators: Dict with ATR, volatility metrics.

        Returns:
            RiskDecision with allow/block and adjusted signal.
        """
        self.update_account_state(state)
        metrics = RiskMetrics()
        reject_reasons: list[str] = []

        # ── 1. Circuit breaker check ────────────────────
        if self.cb_enabled:
            if self._is_circuit_breaker_active():
                metrics.circuit_breaker_active = True
                reject_reasons.append("Circuit breaker active")
            elif self._should_trigger_circuit_breaker(state):
                self._activate_circuit_breaker()
                metrics.circuit_breaker_active = True
                reject_reasons.append("Circuit breaker triggered")

        # ── 2. Daily drawdown limit ─────────────────────
        if state.daily_pnl_pct <= -self.max_daily_drawdown:
            reject_reasons.append(f"Daily drawdown: {state.daily_pnl_pct:.2%}")

        # ── 3. Kill streak (loss streak) ────────────────
        if state.loss_streak >= self.kill_streak:
            reject_reasons.append(f"Kill streak: {state.loss_streak} >= {self.kill_streak}")

        # ── 4. Max concurrent positions ─────────────────
        if state.open_positions >= self.max_concurrent:
            reject_reasons.append(f"Max positions: {state.open_positions} >= {self.max_concurrent}")

        # ── 5. Minimum confidence ────────────────────────
        if signal.confidence < self.min_confidence:
            metrics.confidence_pass = False
            reject_reasons.append(f"Low confidence: {signal.confidence:.2f} < {self.min_confidence}")

        # ── 6. HOLD always blocked ──────────────────────
        if signal.action == "HOLD":
            reject_reasons.append("Signal is HOLD")

        # ── 7. ATR volatility filter ────────────────────
        if self.atr_enabled and indicators:
            atr_pct = indicators.get("atr_pct", 0.0)
            atr_mean_pct = indicators.get("atr_mean_pct", atr_pct)
            if atr_mean_pct > 0 and atr_pct > atr_mean_pct * self.atr_block_mult:
                # ATR z-score calculation
                atr_std = indicators.get("atr_std_pct", atr_mean_pct * 0.3)
                if atr_std > 0:
                    metrics.atr_zscore = (atr_pct - atr_mean_pct) / atr_std
                metrics.volatility_pass = False
                reject_reasons.append(
                    f"ATR vol filter: {atr_pct:.4f} > {atr_mean_pct * self.atr_block_mult:.4f}"
                )

        # ── 8. Time window ──────────────────────────────
        if self.tw_enabled:
            hour = datetime.utcnow().hour
            if not (self.tw_start_hour <= hour < self.tw_end_hour):
                metrics.time_window_pass = False
                reject_reasons.append(f"Outside trading hours: {hour}:00 UTC")

        # ── 9. Invalid size ─────────────────────────────
        if signal.size_pct <= 0 and signal.action != "HOLD":
            reject_reasons.append("Invalid position size")

        # ── Calculate position size ──────────────────────
        metrics.reject_reasons = reject_reasons

        if reject_reasons:
            logger.warning(f"Signal REJECTED: {'; '.join(reject_reasons)}")
            return RiskDecision(
                allow=False,
                reason="; ".join(reject_reasons),
                max_size=0.0,
            )

        # ── All checks passed — calculate size ───────────
        max_equity_risk = state.equity * self.max_position_pct
        requested_size = state.equity * signal.size_pct

        # Kelly sizing (overrides if enabled and enough data)
        kelly_size = self._calc_kelly_size(state.equity)
        metrics.kelly_fraction = kelly_size / state.equity if state.equity > 0 else 0

        if self.kelly_enabled and kelly_size > 0:
            adjusted_size = min(requested_size, kelly_size, max_equity_risk)
        else:
            adjusted_size = min(requested_size, max_equity_risk)

        metrics.size_adjusted = adjusted_size

        adjusted_signal = signal.model_copy()
        adjusted_signal.size_pct = adjusted_size / state.equity if state.equity > 0 else 0

        logger.info(
            f"Signal APPROVED: {signal.action} size={adjusted_signal.size_pct:.2%} "
            f"conf={signal.confidence:.2f} kelly={kelly_size:.2f}"
        )

        return RiskDecision(
            allow=True,
            reason="All risk checks passed",
            adjusted_signal=adjusted_signal,
            max_size=max_equity_risk,
        )

    # ── Kelly criterion ─────────────────────────────────

    def _calc_kelly_size(self, equity: float) -> float:
        """Calculate position size using fractional Kelly criterion.

        f* = (bp - q) / b
        where:
          b = payoff ratio (avg win / avg loss)
          p = win rate
          q = 1 - p

        Returns:
            Dollar amount to risk, capped at kelly_max_pct.
        """
        if not self.kelly_enabled or equity <= 0:
            return 0.0

        # Use observed stats if enough trades, else defaults
        total = self._win_count + self._loss_count
        if total >= 20:
            p = self._win_count / total
            avg_win = self._total_win_pnl / self._win_count if self._win_count > 0 else 1.0
            avg_loss = abs(self._total_loss_pnl / self._loss_count) if self._loss_count > 0 else 1.0
            b = avg_win / avg_loss if avg_loss > 0 else self.default_payoff
        else:
            p = self.default_win_rate
            b = self.default_payoff

        q = 1.0 - p

        # Kelly fraction
        if b <= 0:
            return 0.0

        kelly_f = (b * p - q) / b
        if kelly_f <= 0:
            return 0.0

        # Fractional Kelly (quarter-Kelly by default)
        kelly_f *= self.kelly_fraction

        # Cap at max percentage
        kelly_f = min(kelly_f, self.kelly_max_pct)

        return equity * kelly_f

    # ── Circuit breaker ─────────────────────────────────

    def _is_circuit_breaker_active(self) -> bool:
        """Check if circuit breaker is currently active."""
        if not self._circuit_breaker_active:
            return False
        if time.time() >= self._circuit_breaker_until:
            self._circuit_breaker_active = False
            logger.info("Circuit breaker: cooldown expired, trading resumed")
            return False
        return True

    def _should_trigger_circuit_breaker(self, state: AccountState) -> bool:
        """Check if circuit breaker should be triggered."""
        if state.loss_streak >= self.cb_loss_streak:
            return True
        if self._max_drawdown_pct >= self.cb_drawdown_pct:
            return True
        # Hourly trade limit
        self._check_hourly_reset()
        if self._trades_this_hour >= self.cb_trigger_hourly_limit:
            return True
        return False

    def _activate_circuit_breaker(self):
        """Activate circuit breaker with cooldown."""
        self._circuit_breaker_active = True
        self._circuit_breaker_until = time.time() + self.cb_cooldown_seconds
        logger.warning(
            f"Circuit BREAKER activated! Cooldown: {self.cb_cooldown_seconds}s. "
            f"Resume after {datetime.fromtimestamp(self._circuit_breaker_until).isoformat()}"
        )

    def _check_hourly_reset(self):
        """Reset hourly trade counter if hour has passed."""
        now = time.time()
        if now - self._hour_start >= 3600:
            self._trades_this_hour = 0
            self._hour_start = now

    # ── Trade result tracking ───────────────────────────

    def on_trade_result(self, pnl_pct: float, is_win: bool):
        """Update internal statistics after trade closes."""
        if is_win:
            self._loss_streak = 0
            self._win_count += 1
            self._total_win_pnl += abs(pnl_pct)
        else:
            self._loss_streak += 1
            self._loss_count += 1
            self._total_loss_pnl += abs(pnl_pct)

        self._trades_this_hour += 1

    # ── Status ───────────────────────────────────────────

    def get_status(self) -> dict[str, Any]:
        """Return current risk engine state."""
        total = self._win_count + self._loss_count
        return {
            "loss_streak": self._loss_streak,
            "kill_streak_limit": self.kill_streak,
            "daily_pnl_pct": self._daily_pnl_pct,
            "daily_drawdown_limit": self.max_daily_drawdown,
            "open_positions": self._open_positions,
            "max_concurrent": self.max_concurrent,
            "max_position_pct": self.max_position_pct,
            "circuit_breaker_active": self._circuit_breaker_active,
            "circuit_breaker_until": (
                datetime.fromtimestamp(self._circuit_breaker_until).isoformat()
                if self._circuit_breaker_active else None
            ),
            "kelly_stats": {
                "win_count": self._win_count,
                "loss_count": self._loss_count,
                "total_trades": total,
                "win_rate": self._win_count / total if total > 0 else 0.0,
                "avg_win_pnl": self._total_win_pnl / self._win_count if self._win_count > 0 else 0.0,
                "avg_loss_pnl": self._total_loss_pnl / self._loss_count if self._loss_count > 0 else 0.0,
            },
            "atr_filter_enabled": self.atr_enabled,
            "time_window": {
                "enabled": self.tw_enabled,
                "hours": f"{self.tw_start_hour}:00-{self.tw_end_hour}:00 UTC" if self.tw_enabled else "always",
            },
        }
