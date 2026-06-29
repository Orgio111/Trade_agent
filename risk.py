"""Risk Engine - Hard rules, no LLM dependency."""

from typing import Optional, Dict, Any
from models import Signal, SignalAction, RiskDecision, AccountState, OrderSide
import logging

logger = logging.getLogger(__name__)


class RiskEngine:
    """
    Hard-coded risk rules - deterministic, no LLM.
    All limits configurable via config.yaml
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        import yaml
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        risk_cfg = self.config.get("risk", {})
        
        self.max_position_pct = risk_cfg.get("max_position_pct", 0.05)
        self.max_daily_drawdown = risk_cfg.get("max_daily_drawdown", 0.03)
        self.max_concurrent = risk_cfg.get("max_concurrent", 3)
        self.kill_streak = risk_cfg.get("kill_streak", 5)
        self.min_confidence = risk_cfg.get("min_confidence", 0.6)
        self.volatility_filter = risk_cfg.get("volatility_filter", True)
        self.atr_threshold_mult = risk_cfg.get("atr_threshold_mult", 2.5)
        
        # Internal state
        self._loss_streak = 0
        self._daily_pnl_pct = 0.0
        self._open_positions = 0
        self._max_drawdown_pct = 0.0
    
    def update_account_state(self, state: AccountState):
        """Update internal state from account."""
        self._loss_streak = state.loss_streak
        self._daily_pnl_pct = state.daily_pnl_pct
        self._open_positions = state.open_positions
        self._max_drawdown_pct = state.max_drawdown_pct
    
    def validate(self, signal: Signal, state: AccountState) -> RiskDecision:
        """
        Validate signal against all risk rules.
        Returns RiskDecision with allow/block and adjusted signal.
        """
        self.update_account_state(state)
        
        # 1. Daily drawdown limit
        if state.daily_pnl_pct <= -self.max_daily_drawdown:
            logger.warning(f"Daily drawdown limit hit: {state.daily_pnl_pct:.2%} <= -{self.max_daily_drawdown:.2%}")
            return RiskDecision(
                allow=False,
                reason=f"Daily drawdown limit exceeded: {state.daily_pnl_pct:.2%}",
                max_size=0.0
            )
        
        # 2. Kill streak
        if state.loss_streak >= self.kill_streak:
            logger.warning(f"Kill streak triggered: {state.loss_streak} >= {self.kill_streak}")
            return RiskDecision(
                allow=False,
                reason=f"Loss streak limit: {state.loss_streak} consecutive losses",
                max_size=0.0
            )
        
        # 3. Max concurrent positions
        if state.open_positions >= self.max_concurrent:
            logger.warning(f"Max concurrent positions: {state.open_positions} >= {self.max_concurrent}")
            return RiskDecision(
                allow=False,
                reason=f"Max concurrent positions reached: {state.open_positions}",
                max_size=0.0
            )
        
        # 4. Minimum confidence
        if signal.confidence < self.min_confidence:
            logger.warning(f"Signal confidence too low: {signal.confidence:.2f} < {self.min_confidence}")
            return RiskDecision(
                allow=False,
                reason=f"Confidence below minimum: {signal.confidence:.2f} < {self.min_confidence}",
                max_size=0.0
            )
        
        # 5. Volatility filter (ATR-based)
        if self.volatility_filter and signal.reasoning:
            # This would be checked with ATR from context
            # For now, we check if signal has ATR info in reasoning
            pass
        
        # 6. Signal validation
        if signal.action == SignalAction.HOLD:
            return RiskDecision(
                allow=False,
                reason="Signal is HOLD",
                max_size=0.0
            )
        
        if signal.size_pct <= 0:
            return RiskDecision(
                allow=False,
                reason="Invalid position size",
                max_size=0.0
            )
        
        # All checks passed - adjust position size
        max_equity_risk = state.equity * self.max_position_pct
        requested_size = state.equity * signal.size_pct
        adjusted_size = min(requested_size, max_equity_risk)
        
        adjusted_signal = signal.model_copy()
        adjusted_signal.size_pct = adjusted_size / state.equity if state.equity > 0 else 0
        
        logger.info(f"Signal validated: {signal.action.value} {adjusted_signal.size_pct:.2%} equity (confidence: {signal.confidence:.2f})")
        
        return RiskDecision(
            allow=True,
            reason="All risk checks passed",
            adjusted_signal=adjusted_signal,
            max_size=max_equity_risk
        )
    
    def on_trade_result(self, pnl_pct: float, is_win: bool):
        """Update internal streak counters after trade closes."""
        if is_win:
            self._loss_streak = 0
        else:
            self._loss_streak += 1
    
    def get_status(self) -> Dict[str, Any]:
        return {
            "loss_streak": self._loss_streak,
            "kill_streak_limit": self.kill_streak,
            "daily_pnl_pct": self._daily_pnl_pct,
            "daily_drawdown_limit": self.max_daily_drawdown,
            "open_positions": self._open_positions,
            "max_concurrent": self.max_concurrent,
            "max_position_pct": self.max_position_pct,
        }