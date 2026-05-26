"""Backtesting engine — replay historical OHLCV data through the supervisor
pipeline and compute performance metrics."""

from __future__ import annotations

from backtest.data_loader import fetch_ohlcv, load_data
from backtest.engine import BacktestEngine
from backtest.models import BacktestConfig, BacktestResult, BacktestTrade, BarRecord
from backtest.report import compute_performance

__all__ = [
    "BacktestEngine",
    "BacktestConfig",
    "BacktestResult",
    "BacktestTrade",
    "BarRecord",
    "fetch_ohlcv",
    "load_data",
    "compute_performance",
]
