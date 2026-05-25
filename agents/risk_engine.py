"""
Risk Engine: Value at Risk (Historical Simulation), CVaR, Fractional Kelly
Criterion, ATR-based stop-loss, and position sizing.

Optionally delegates to the Rust Risk Engine (sentinel-x) when configured.
"""
from __future__ import annotations

import logging
from datetime import datetime

import numpy as np
from scipy.stats import norm

from core.config import get_settings
from core.messaging import MsgType, get_bus
from core.models import CouncilDecision, RiskReport, Side
from core.observability import AGENT_LATENCY, VAR_GAUGE
from core.sentinelx_bridge import validate_trade as rust_validate

logger = logging.getLogger(__name__)


def historical_var(
    returns: np.ndarray,
    confidence: float = 0.99,
) -> tuple[float, float]:
    """Returns (VaR, CVaR) as positive loss fractions using historical simulation."""
    if len(returns) < 30:
        return 0.05, 0.08  # conservative defaults when data is thin

    sorted_r = np.sort(returns)
    idx = int((1 - confidence) * len(sorted_r))
    var = float(-sorted_r[max(idx - 1, 0)])
    cvar = float(-sorted_r[:max(idx, 1)].mean())
    return max(var, 0.001), max(cvar, var)


def parametric_var(
    returns: np.ndarray,
    confidence: float = 0.99,
) -> float:
    """Parametric (Gaussian) VaR — cross-check against historical."""
    mu, sigma = float(returns.mean()), float(returns.std())
    z = norm.ppf(1 - confidence)
    return float(-(mu + z * sigma))


def fractional_kelly(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    fraction: float = 0.25,
) -> float:
    """
    Kelly fraction = (W/L * p - (1-p)) / (W/L)
    Returns fraction of capital to risk; clipped to [0, 1].
    """
    if avg_loss < 1e-10:
        return 0.0
    odds = avg_win / avg_loss
    kelly = (odds * win_rate - (1.0 - win_rate)) / odds
    kelly = max(kelly, 0.0)
    return kelly * fraction


class RiskEngine:
    """Gatekeeper: approves/rejects trades and sizes positions.

    When ``sentinelx_risk_addr`` is configured, delegates VaR / Kelly / stop
    calculations to the Rust Risk Engine (sentinel-x).  Falls back to the
    pure-Python implementation on gRPC failure so the system is never blocked.
    """

    async def evaluate(
        self,
        decision: CouncilDecision,
        historical_returns: np.ndarray,
        current_price: float,
        portfolio_equity: float,
        win_rate: float = 0.55,
        avg_win_pct: float = 0.015,
        avg_loss_pct: float = 0.010,
    ) -> RiskReport:
        cfg = get_settings()

        # ── Try Rust engine first ────────────────────────────────────────────
        atr = decision.technical.atr_14 if decision.technical else current_price * 0.02
        rust_result = await rust_validate(
            decision, historical_returns, current_price, portfolio_equity, atr,
        )

        with AGENT_LATENCY.labels(agent="risk").time():
            if rust_result is not None:
                # Rust engine succeeded — use its results
                approved = rust_result.approved
                rejection_reason = rust_result.rejection_reason
                var_95 = rust_result.var_95
                var_99 = rust_result.var_99
                cvar_99 = rust_result.cvar_99
                kelly_raw = rust_result.kelly_raw
                kelly_frac = rust_result.kelly_fractional
                position_size_usd = rust_result.position_size_usd
                position_size_units = rust_result.position_size_units
                stop_loss_price = rust_result.stop_loss_price
                take_profit_price = rust_result.take_profit_price
            else:
                # ── Pure-Python fallback ────────────────────────────────────────
                var_95, _ = historical_var(historical_returns, 0.95)
                var_99, cvar_99 = historical_var(historical_returns, cfg.var_confidence)
                parametric_99 = parametric_var(historical_returns, cfg.var_confidence)

                # Blend historical and parametric VaR (conservative: take max)
                var_99 = max(var_99, parametric_99)

                kelly_raw = fractional_kelly(win_rate, avg_win_pct, avg_loss_pct, 1.0)
                kelly_frac = fractional_kelly(
                    win_rate, avg_win_pct, avg_loss_pct, cfg.kelly_fraction
                )

                # Position size: min of Kelly and hard cap
                size_pct = min(kelly_frac, cfg.max_position_pct)
                position_size_usd = portfolio_equity * size_pct
                position_size_units = position_size_usd / current_price if current_price > 0 else 0

                # ATR-based stop / take-profit
                stop_multiplier = cfg.atr_stop_multiplier
                if decision.final_side == Side.BUY:
                    stop_loss_price = current_price - stop_multiplier * atr
                    take_profit_price = current_price + stop_multiplier * 1.5 * atr
                else:
                    stop_loss_price = current_price + stop_multiplier * atr
                    take_profit_price = current_price - stop_multiplier * 1.5 * atr

                # Approval logic
                rejection_reason: str | None = None
                if decision.consensus_score < cfg.min_consensus_score:
                    rejection_reason = (
                        f"Low consensus score {decision.consensus_score:.2f} "
                        f"< threshold {cfg.min_consensus_score:.2f}"
                    )
                elif var_99 > 0.15:
                    rejection_reason = f"VaR99 too high: {var_99:.1%}"
                elif position_size_usd < 1.0:
                    rejection_reason = "Kelly sizing too small — insufficient edge"

                approved = rejection_reason is None

        report = RiskReport(
            symbol=decision.symbol,
            timestamp=datetime.utcnow(),
            session_id=decision.session_id,
            var_95=var_95,
            var_99=var_99,
            cvar_99=cvar_99,
            kelly_raw=kelly_raw,
            kelly_fractional=kelly_frac,
            position_size_usd=position_size_usd,
            position_size_units=position_size_units,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            max_drawdown_pct=var_99,
            approved=approved,
            rejection_reason=rejection_reason,
        )

        bus = await get_bus()
        await bus.publish(
            cfg.stream_risk, MsgType.RISK_REPORT, report.model_dump(mode="json")
        )
        VAR_GAUGE.labels(symbol=decision.symbol, confidence="99").set(var_99)
        VAR_GAUGE.labels(symbol=decision.symbol, confidence="95").set(var_95)

        if approved:
            logger.info(
                "Risk APPROVED %s | VaR99=%.2f%% | size=%.2f | stop=%.4f",
                decision.symbol,
                var_99 * 100,
                position_size_usd,
                stop_loss_price,
            )
        else:
            logger.warning(
                "Risk REJECTED %s: %s",
                decision.symbol,
                rejection_reason,
            )
        return report
