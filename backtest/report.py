"""Performance report computation for backtest results."""

from __future__ import annotations

import logging
import math
from datetime import timezone

import numpy as np

from backtest.models import BacktestConfig, BacktestResult, BacktestTrade

logger = logging.getLogger(__name__)


def compute_performance(
    config: BacktestConfig,
    trades: list[BacktestTrade],
    equity_curve: list[float],
    equity_timestamps: list[datetime],
    errors: list[str] | None = None,
) -> BacktestResult:
    """Compute all performance metrics from a backtest run.

    Parameters
    ----------
    config : BacktestConfig
        The configuration used for this run.
    trades : list[BacktestTrade]
        All completed trades (with exit prices and PnL set).
    equity_curve : list[float]
        Equity value at each bar (mark-to-market).
    equity_timestamps : list[datetime]
        Timestamps corresponding to each equity value.
    errors : list[str], optional
        Any errors encountered during the run.

    Returns
    -------
    BacktestResult
        Populated with all computed metrics.
    """
    result = BacktestResult(
        config=config,
        trades=trades,
        equity_curve=equity_curve,
        equity_timestamps=equity_timestamps,
        errors=errors or [],
    )

    if not equity_curve:
        return result

    initial = config.initial_capital
    final = equity_curve[-1]
    result.final_equity = final
    result.total_return_pct = (final / initial - 1.0) * 100.0

    # ── Annualised return ─────────────────────────────────────────────────────
    n_bars = len(equity_curve)
    if n_bars >= 2:
        total_years = _bars_per_year(config.timeframe)
        if total_years > 0:
            ann_return = (final / initial) ** (1.0 / total_years) - 1.0
        else:
            ann_return = 0.0
    else:
        ann_return = 0.0

    # ── Log returns for Sharpe / Sortino ──────────────────────────────────────
    eq = np.array(equity_curve, dtype=np.float64)
    log_ret = np.diff(np.log(np.maximum(eq, 1.0)))
    n_ret = len(log_ret)

    if n_ret >= 2:
        factor = _bars_per_year(config.timeframe)

        # Sharpe ratio (annualised)
        mean_ret = float(np.mean(log_ret))
        std_ret = float(np.std(log_ret, ddof=1))
        if std_ret > 1e-10:
            result.sharpe_ratio = (mean_ret / std_ret) * math.sqrt(factor)
        else:
            result.sharpe_ratio = 0.0

        # Sortino ratio (downside deviation only)
        downside = log_ret[log_ret < 0]
        if len(downside) > 0:
            downside_std = float(np.std(downside, ddof=1))
            if downside_std > 1e-10:
                result.sortino_ratio = (mean_ret / downside_std) * math.sqrt(factor)
            else:
                result.sortino_ratio = 0.0
    else:
        result.sharpe_ratio = 0.0
        result.sortino_ratio = 0.0

    # ── Max drawdown ──────────────────────────────────────────────────────────
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    if len(dd) > 0:
        result.max_drawdown_pct = float(np.min(dd)) * 100.0
    else:
        result.max_drawdown_pct = 0.0

    # Calmar ratio (ann. return / max drawdown)
    if result.max_drawdown_pct < -1e-6:
        result.calmar_ratio = ann_return / (abs(result.max_drawdown_pct) / 100.0)
    else:
        result.calmar_ratio = 0.0 if abs(ann_return) < 1e-10 else ann_return * 100.0

    # ── Trade-based metrics ───────────────────────────────────────────────────
    closed_trades = [t for t in trades if t.pnl is not None]
    result.total_trades = len(closed_trades)

    if closed_trades:
        wins = [t for t in closed_trades if t.pnl is not None and t.pnl > 0]
        losses = [t for t in closed_trades if t.pnl is not None and t.pnl <= 0]
        result.win_rate = len(wins) / len(closed_trades)

        gross_profit = sum(t.pnl for t in wins) if wins else 0.0
        gross_loss = abs(sum(t.pnl for t in losses)) if losses else 0.0
        result.profit_factor = (
            gross_profit / gross_loss if gross_loss > 1e-6 else float("inf")
        )

        # Average trade duration
        durations = []
        for t in closed_trades:
            if t.entry_time and t.exit_time:
                delta = (t.exit_time - t.entry_time).total_seconds() / 3600
                durations.append(delta)
        if durations:
            result.avg_trade_duration_h = float(np.mean(durations))

    return result


def _bars_per_year(timeframe: str) -> float:
    """Estimate the number of bars in a trading year for a given timeframe.

    Uses 365 days × 24 hours (crypto markets trade 24/7).
    """
    unit = timeframe[-1]
    try:
        num = int(timeframe[:-1]) if len(timeframe) > 1 else 1
    except ValueError:
        return 365 * 24  # default to hourly

    if unit == "m":
        return 365 * 24 * (60 // num)
    elif unit == "h":
        return 365 * 24 / num
    elif unit == "d":
        return 365 / num
    elif unit == "w":
        return 52 / num
    else:
        return 365 * 24  # hourly default
