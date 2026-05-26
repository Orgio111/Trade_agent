"""Backtesting data models: bars, trades, config, and results."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np


@dataclass
class BarRecord:
    """Single OHLCV bar used in historical replay."""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    def to_tuple(self) -> tuple:
        return (self.timestamp, self.open, self.high, self.low, self.close, self.volume)


@dataclass
class BacktestTrade:
    """A completed trade from a backtest run."""

    symbol: str
    side: str  # "BUY" | "SELL"
    entry_time: datetime
    exit_time: datetime | None = None
    entry_price: float = 0.0
    exit_price: float | None = None
    quantity: float = 0.0
    pnl: float | None = None
    pnl_pct: float | None = None
    exit_reason: str = "signal"  # signal | stop_loss | take_profit
    rationale: str = ""


@dataclass
class BacktestConfig:
    """Configuration for a backtest run."""

    symbols: list[str] = field(default_factory=lambda: ["BTC/USDT", "ETH/USDT"])
    start: str = "2024-01-01"
    end: str = "2024-12-31"
    initial_capital: float = 100_000.0
    exchange_id: str = "binance"
    timeframe: str = "1h"
    slippage_bps: float = 5.0
    stop_loss_pct: float = 0.05
    """Stop-loss threshold as fraction of entry price (e.g. 0.05 = 5%)."""
    features_enabled: bool = False
    """Enable feature extraction in backtest (requires tick/funding/OI data)."""
    max_cycles: int = 0
    """Limit bars processed (0 = all).  Useful for quick smoke tests."""


@dataclass
class BacktestResult:
    """Complete results from a backtest run."""

    config: BacktestConfig
    trades: list[BacktestTrade] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)
    equity_timestamps: list[datetime] = field(default_factory=list)
    peak_equity: float = 0.0
    final_equity: float = 0.0
    total_return_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    avg_trade_duration_h: float = 0.0
    calmar_ratio: float = 0.0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": {
                "symbols": self.config.symbols,
                "start": self.config.start,
                "end": self.config.end,
                "initial_capital": self.config.initial_capital,
                "timeframe": self.config.timeframe,
            },
            "summary": {
                "total_return_pct": round(self.total_return_pct, 2),
                "sharpe_ratio": round(self.sharpe_ratio, 3),
                "sortino_ratio": round(self.sortino_ratio, 3),
                "max_drawdown_pct": round(self.max_drawdown_pct, 2),
                "win_rate": round(self.win_rate, 3),
                "profit_factor": round(self.profit_factor, 3),
                "total_trades": self.total_trades,
                "final_equity": round(self.final_equity, 2),
                "peak_equity": round(self.peak_equity, 2),
                "calmar_ratio": round(self.calmar_ratio, 3),
            },
            "equity_curve": (
                [round(e, 2) for e in self.equity_curve] if self.equity_curve else []
            ),
            "errors": self.errors,
        }
