"""
Walk-Forward Optimization (WFO): trains on a rolling window then tests on
the out-of-sample forward period. Produces Sharpe-optimal parameter sets.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable

import numpy as np
from scipy.optimize import differential_evolution

logger = logging.getLogger(__name__)


@dataclass
class WFOResult:
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    best_params: dict
    train_sharpe: float
    test_sharpe: float
    test_max_drawdown: float
    test_win_rate: float


@dataclass
class WFOConfig:
    train_months: int = 12
    test_months: int = 3
    step_months: int = 1
    param_bounds: dict = field(default_factory=dict)
    min_trades: int = 30


def sharpe_ratio(returns: np.ndarray, risk_free: float = 0.0, annualize: float = 252.0) -> float:
    if len(returns) < 2 or returns.std() < 1e-10:
        return 0.0
    excess = returns - risk_free / annualize
    return float(np.sqrt(annualize) * excess.mean() / excess.std())


def max_drawdown(equity_curve: np.ndarray) -> float:
    if len(equity_curve) < 2:
        return 0.0
    roll_max = np.maximum.accumulate(equity_curve)
    dd = (equity_curve - roll_max) / (roll_max + 1e-10)
    return float(dd.min())


class WalkForwardOptimizer:
    """
    Rolls a train/test window across historical data to find stable
    strategy parameters that generalize out-of-sample.
    """

    def __init__(
        self,
        backtest_fn: Callable[[np.ndarray, dict], np.ndarray],
        config: WFOConfig | None = None,
    ) -> None:
        """
        backtest_fn(price_series, params) → array of period returns
        """
        self._backtest = backtest_fn
        self._config = config or WFOConfig()

    def run(
        self,
        prices: np.ndarray,
        timestamps: list[datetime],
    ) -> list[WFOResult]:
        cfg = self._config
        results: list[WFOResult] = []
        n = len(prices)
        total_months = (timestamps[-1] - timestamps[0]).days // 30
        train_bars = cfg.train_months * 30
        test_bars = cfg.test_months * 30
        step_bars = cfg.step_months * 30

        cursor = 0
        while cursor + train_bars + test_bars <= n:
            train_prices = prices[cursor : cursor + train_bars]
            test_prices = prices[cursor + train_bars : cursor + train_bars + test_bars]

            best_params = self._optimize(train_prices)
            train_returns = self._backtest(train_prices, best_params)
            test_returns = self._backtest(test_prices, best_params)

            eq = np.cumprod(1 + test_returns)
            result = WFOResult(
                train_start=timestamps[cursor],
                train_end=timestamps[cursor + train_bars - 1],
                test_start=timestamps[cursor + train_bars],
                test_end=timestamps[min(cursor + train_bars + test_bars - 1, n - 1)],
                best_params=best_params,
                train_sharpe=sharpe_ratio(train_returns),
                test_sharpe=sharpe_ratio(test_returns),
                test_max_drawdown=max_drawdown(eq),
                test_win_rate=float((test_returns > 0).mean()),
            )
            results.append(result)
            logger.info(
                "WFO window %s→%s: train_sharpe=%.2f  test_sharpe=%.2f  test_dd=%.2f%%",
                result.train_start.date(),
                result.test_end.date(),
                result.train_sharpe,
                result.test_sharpe,
                result.test_max_drawdown * 100,
            )
            cursor += step_bars

        return results

    def _optimize(self, prices: np.ndarray) -> dict:
        cfg = self._config
        if not cfg.param_bounds:
            return {}

        bounds = list(cfg.param_bounds.values())
        keys = list(cfg.param_bounds.keys())

        def objective(x: np.ndarray) -> float:
            params = dict(zip(keys, x))
            returns = self._backtest(prices, params)
            if len(returns) < cfg.min_trades:
                return 1e6
            s = sharpe_ratio(returns)
            return -s  # minimize negative Sharpe

        result = differential_evolution(
            objective,
            bounds,
            maxiter=100,
            popsize=10,
            seed=42,
            tol=0.001,
        )
        return dict(zip(keys, result.x))
