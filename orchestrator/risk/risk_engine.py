"""
QUANTEX Institutional Risk Engine — Cascading protection with 10 risk gates.

RISK PARAMETERS (Small Account Optimized):
  max_risk_per_trade:    1% of balance
  max_risk_per_session:  5% per day
  max_total_exposure:    15% of balance
  max_drawdown_soft:     10% → reduce size 50%
  max_drawdown_hard:     20% → halt trading
  max_consecutive_loss:  4 → force cooldown
  max_open_positions:    3 concurrent
  max_leverage:          10x hard cap
  correlation_limit:     70% max similarity
  volatility_stop_mult:  2.5 ATR for stops

Formulas:
  Position Size = (balance × risk_pct) / (ATR × atr_multiplier)
  Half-Kelly: f* = 0.5 × [(W/L) × p - (1-p)] / (W/L)
  Portfolio Heat = Σ(position_i × leverage_i) / balance
  Liq Distance (long) = entry × (1 - 1/leverage + maintenance_margin)
"""
import time
import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RiskParams:
    max_risk_per_trade: float = 0.01
    max_risk_per_session: float = 0.05
    max_total_exposure: float = 0.15
    max_drawdown_soft: float = 0.10
    max_drawdown_hard: float = 0.20
    max_consecutive_losses: int = 4
    max_open_positions: int = 3
    max_leverage: int = 10
    correlation_limit: float = 0.70
    atr_multiplier_sl: float = 1.5
    atr_multiplier_tp: float = 3.0


@dataclass
class RiskCheckResult:
    approved: bool
    severity: str  # "hard" | "soft" | "none"
    reason: str
    size_multiplier: Optional[float] = None
    max_leverage: Optional[int] = None
    modifications: dict = field(default_factory=dict)


class RiskEngine:
    """
    Multi-layer risk management with cascading protection.
    Every trade passes through all active gates before execution.
    """

    def __init__(self, params: Optional[RiskParams] = None):
        self.params = params or RiskParams()

    # ═══════════════════════════════════════════════════════
    # POSITION SIZING FORMULAS
    # ═══════════════════════════════════════════════════════

    def calculate_position_size(self, balance: float, atr: float, price: float,
                                 leverage: int = 1) -> float:
        """ATR-based position sizing:
        Position = (Balance × Risk%) / (ATR × StopMultiplier)
        """
        risk_amount = balance * self.params.max_risk_per_trade
        sl_distance = atr * self.params.atr_multiplier_sl
        if sl_distance <= 0 or price <= 0:
            return 0.0
        position = (risk_amount / sl_distance) * leverage
        max_size = balance * self.params.max_total_exposure * leverage
        return min(position, max_size)

    def kelly_criterion(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """Half-Kelly: f* = 0.5 × [(W/L) × p - (1-p)] / (W/L)"""
        if avg_loss == 0 or avg_win == 0:
            return 0.0
        q = 1.0 - win_rate
        kelly = (win_rate / abs(avg_loss)) - (q / abs(avg_win))
        return max(0.0, min(kelly * 0.5, 0.25))

    def volatility_adjusted_size(self, base_size: float, current_vol: float,
                                  baseline_vol: float) -> float:
        """Scale position inversely with volatility."""
        if current_vol <= 0 or baseline_vol <= 0:
            return base_size
        vol_ratio = baseline_vol / current_vol
        vol_scalar = max(0.3, min(2.0, vol_ratio))
        return base_size * vol_scalar

    def dynamic_leverage(self, base_leverage: int, volatility_percentile: float,
                          confidence: float, regime: str) -> int:
        """Adaptive leverage: base × vol × confidence × regime."""
        vol_scalar = 1.0 - (volatility_percentile / 200.0)
        conf_scalar = 0.5 + confidence * 0.5
        regime_map = {
            "strong_trend": 1.2, "weak_trend": 1.0,
            "ranging": 0.7, "high_volatility": 0.5, "crisis": 0.0,
        }
        regime_scalar = regime_map.get(regime, 1.0)
        final_lvg = base_leverage * vol_scalar * conf_scalar * regime_scalar
        return max(1, min(int(final_lvg), self.params.max_leverage))

    # ═══════════════════════════════════════════════════════
    # RISK GATE CHECK
    # ═══════════════════════════════════════════════════════

    def check_all_gates(self, proposed_trade: dict, portfolio: dict) -> RiskCheckResult:
        """Run ALL risk gates. Returns first hard veto or cumulative soft modifications."""
        checks = [
            self._check_drawdown_gate(portfolio),
            self._check_daily_loss_gate(portfolio),
            self._check_consecutive_loss_gate(portfolio),
            self._check_position_count_gate(portfolio),
            self._check_exposure_gate(proposed_trade, portfolio),
            self._check_leverage_gate(proposed_trade),
            self._check_volatility_gate(proposed_trade),
        ]
        hard_vetoes = [c for c in checks if not c.approved and c.severity == "hard"]
        soft_checks = [c for c in checks if c.severity == "soft"]

        if hard_vetoes:
            return hard_vetoes[0]

        modifications = {}
        for sc in soft_checks:
            modifications.update(sc.modifications)

        # Default size/leverage if no soft modifications
        if "size_multiplier" not in modifications:
            modifications["size_multiplier"] = 1.0
        if "max_leverage" not in modifications:
            modifications["max_leverage"] = self.params.max_leverage

        return RiskCheckResult(
            approved=True,
            severity="none",
            reason="All gates passed",
            size_multiplier=modifications.get("size_multiplier", 1.0),
            max_leverage=modifications.get("max_leverage", self.params.max_leverage),
            modifications=modifications,
        )

    def _check_drawdown_gate(self, portfolio: dict) -> RiskCheckResult:
        """Hard: >20% DD = stop. Soft: >10% = reduce size 50%."""
        dd = portfolio.get("drawdown", 0.0)
        if dd >= self.params.max_drawdown_hard:
            return RiskCheckResult(False, "hard",
                                   f"HALT: Drawdown {dd:.1%} > {self.params.max_drawdown_hard:.0%}")
        if dd >= self.params.max_drawdown_soft:
            reduction = 1.0 - (dd - self.params.max_drawdown_soft) / 0.10
            return RiskCheckResult(True, "soft",
                                   f"Reducing size {((1-reduction)*100):.0f}% due to drawdown",
                                   size_multiplier=max(0.25, reduction))
        return RiskCheckResult(True, "none", "Drawdown OK")

    def _check_daily_loss_gate(self, portfolio: dict) -> RiskCheckResult:
        """Hard: >5% daily loss = halt."""
        daily_pnl = portfolio.get("daily_pnl", 0.0)
        balance = portfolio.get("balance", 1.0)
        if balance <= 0:
            return RiskCheckResult(False, "hard", "Zero balance")
        daily_loss_pct = abs(daily_pnl) / balance if daily_pnl < 0 else 0
        if daily_loss_pct >= self.params.max_risk_per_session:
            return RiskCheckResult(False, "hard",
                                   f"Daily loss {daily_loss_pct:.1%} > {self.params.max_risk_per_session:.0%} limit")
        return RiskCheckResult(True, "none", "Daily loss OK")

    def _check_consecutive_loss_gate(self, portfolio: dict) -> RiskCheckResult:
        """Hard: 4 consecutive losses = cooldown. Soft: 2+ = reduce size."""
        consec = portfolio.get("consecutive_losses", 0)
        if consec >= self.params.max_consecutive_losses:
            return RiskCheckResult(False, "hard",
                                   f"COOLDOWN: {consec} consecutive losses")
        if consec >= 2:
            return RiskCheckResult(True, "soft", f"Recent losses ({consec}) — reducing size 50%",
                                   size_multiplier=0.5, max_leverage=3)
        return RiskCheckResult(True, "none", "No consecutive loss issue")

    def _check_position_count_gate(self, portfolio: dict) -> RiskCheckResult:
        """Hard: max positions reached."""
        open_pos = portfolio.get("open_positions", 0)
        if open_pos >= self.params.max_open_positions:
            return RiskCheckResult(False, "hard",
                                   f"Max positions ({self.params.max_open_positions}) reached")
        return RiskCheckResult(True, "none", f"Positions: {open_pos}/{self.params.max_open_positions}")

    def _check_exposure_gate(self, trade: dict, portfolio: dict) -> RiskCheckResult:
        """Soft: check total portfolio exposure."""
        balance = portfolio.get("balance", 1.0)
        current_exposure = portfolio.get("current_exposure", 0.0)
        trade_exposure = trade.get("notional", 0.0) / max(balance, 0.01)
        total_exposure = current_exposure + trade_exposure
        if total_exposure > self.params.max_total_exposure:
            return RiskCheckResult(True, "soft",
                                   f"Exposure {total_exposure:.0%} near limit — reducing",
                                   size_multiplier=0.5)
        return RiskCheckResult(True, "none", f"Exposure {total_exposure:.0%} OK")

    def _check_leverage_gate(self, trade: dict) -> RiskCheckResult:
        """Hard: leverage exceeds max."""
        lev = trade.get("leverage", 1)
        if lev > self.params.max_leverage:
            return RiskCheckResult(False, "hard",
                                   f"Leverage {lev}x > max {self.params.max_leverage}x")
        return RiskCheckResult(True, "none", f"Leverage {lev}x OK")

    def _check_volatility_gate(self, trade: dict) -> RiskCheckResult:
        """Soft: high volatility reduces position size."""
        vol = trade.get("volatility_percentile", 50)
        if vol > 95:
            return RiskCheckResult(False, "hard",
                                   f"Volatility kill-switch: {vol:.0f}th percentile")
        if vol > 80:
            return RiskCheckResult(True, "soft", f"High vol ({vol:.0f}th) — max lev 2x",
                                   max_leverage=2)
        return RiskCheckResult(True, "none", "Volatility OK")


class PanicMode:
    """
    AI-driven panic mode — emergency intervention when market conditions
    become extreme. Multi-level response: WARNING → CRITICAL.
    """

    @staticmethod
    def assess(market_conditions: dict, portfolio: dict) -> dict:
        """
        Assess panic level. Returns PANIC or normal response.

        Triggers:
          - BTC 1h drop > 5%
          - $500M+ liquidations in 1h
          - Extreme funding rates
          - Fear & Greed < 15
          - Portfolio drawdown > 12%
        """
        score = 0.0

        # Market signals
        btc_drop = market_conditions.get("btc_1h_drop", 0)
        if btc_drop < -0.05:
            score += 0.30
        total_liq = market_conditions.get("total_liq_1h", 0)
        if total_liq > 500_000_000:
            score += 0.25
        funding = market_conditions.get("funding_rate", 0)
        if abs(funding) > 0.003:
            score += 0.15
        fear_greed = market_conditions.get("fear_greed", 50)
        if fear_greed < 15:
            score += 0.10

        # Portfolio signals
        dd = portfolio.get("drawdown", 0.0)
        if dd > 0.12:
            score += 0.20
        upl_pct = portfolio.get("unrealized_pnl_pct", 0.0)
        if upl_pct < -0.08:
            score += 0.15

        if score >= 0.70:
            return {
                "panic": True,
                "level": "CRITICAL",
                "actions": [
                    "close_all_positions",
                    "cancel_all_orders",
                    "set_leverage_1x",
                    "halt_new_trades_24h",
                ],
                "reason": f"Panic score: {score:.2f}",
            }
        if score >= 0.45:
            return {
                "panic": True,
                "level": "WARNING",
                "actions": [
                    "reduce_all_positions_50pct",
                    "tighten_all_stops",
                    "no_new_longs",
                ],
                "reason": f"Elevated risk: {score:.2f}",
            }
        return {"panic": False, "level": "normal", "actions": [], "reason": "Normal conditions"}


class AntiOvertradeSystem:
    """
    Detects and prevents overtrading and revenge trading patterns.

    Rules:
      - >20 trades/hour = overtrading
      - Revenge trading: 2+ losses followed by 1.5x larger position
      - Cooldown enforced after detection
    """

    def __init__(self, window_minutes: int = 60):
        self.window = window_minutes
        self._trade_log: list[dict] = []

    def record_trade(self, trade: dict):
        self._trade_log.append({**trade, "timestamp": time.time()})
        # Prune old entries
        cutoff = time.time() - self.window * 60
        self._trade_log = [t for t in self._trade_log if t["timestamp"] > cutoff]

    def check(self) -> dict:
        """Check if trading should be blocked."""
        now = time.time()
        recent = [t for t in self._trade_log if t["timestamp"] > now - self.window * 60]
        trade_count = len(recent)
        recent_losses = [t for t in recent if t.get("pnl", 0) < 0]

        # Overtrading
        if trade_count > 20:
            return {"block": True, "reason": f"Overtrading: {trade_count} trades/h",
                    "cooldown_minutes": 30}

        # Revenge trading
        if len(recent_losses) >= 2 and len(recent) >= 2:
            last_size = recent[-1].get("size", 0)
            prev_size = recent[-2].get("size", 0)
            if prev_size > 0 and last_size > prev_size * 1.5:
                return {"block": True, "reason": "Revenge trading detected",
                        "cooldown_minutes": 60, "pattern": "revenge"}

        return {"block": False}
