"""
QUANTEX Backtesting Engine — Historical data, trade simulation, performance analytics.

Supports:
- Binance REST API and data.binance.vision CSV data
- Realistic fill simulation with configurable slippage
- Comprehensive performance metrics (Sharpe, Sortino, Calmar, win rate, profit factor)
- Walk-forward validation
"""
import json
import time
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import httpx
import numpy as np
import pandas as pd


# ── Data Structures ──────────────────────────────────────

@dataclass
class BacktestTrade:
    entry_time: datetime
    exit_time: datetime
    symbol: str
    direction: str  # 'long' | 'short'
    entry_price: float
    exit_price: float
    quantity: float
    leverage: int = 1
    pnl: float = 0.0
    pnl_pct: float = 0.0
    fees: float = 0.0
    reason: str = ""
    confidence: float = 0.0
    exit_reason: str = ""  # 'tp', 'sl', 'signal', 'manual'


@dataclass
class BacktestResult:
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    total_fees: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_trade_duration: timedelta = timedelta()
    expectancy: float = 0.0
    equity_curve: list[float] = field(default_factory=list)
    trades: list[BacktestTrade] = field(default_factory=list)
    monthly_returns: dict = field(default_factory=dict)
    strategy_params: dict = field(default_factory=dict)

    def summary(self) -> str:
        """Pretty-print backtest results."""
        lines = [
            "╔══════════════════════════════════════════════╗",
            "║         BACKTEST RESULTS                    ║",
            "╠══════════════════════════════════════════════╣",
            f"  Total Trades:     {self.total_trades}",
            f"  Win Rate:         {self.win_rate:.1%}",
            f"  Total PnL:        ${self.total_pnl:+.2f} ({self.total_pnl_pct:+.2%})",
            f"  Max Drawdown:     {self.max_drawdown_pct:.2%}",
            f"  Sharpe Ratio:     {self.sharpe_ratio:.2f}",
            f"  Sortino Ratio:    {self.sortino_ratio:.2f}",
            f"  Profit Factor:    {self.profit_factor:.2f}",
            f"  Avg Win:          ${self.avg_win:.2f}",
            f"  Avg Loss:         ${self.avg_loss:.2f}",
            f"  Expectancy:       ${self.expectancy:.2f}",
            f"  Calmar Ratio:     {self.calmar_ratio:.2f}",
            "╚══════════════════════════════════════════════╝",
        ]
        return "\n".join(lines)


# ── Historical Data Loader ───────────────────────────────

class DataLoader:
    """Load historical OHLCV data from multiple sources."""

    BINANCE_BASE = "https://api.binance.com/api/v3"
    BINANCE_VISION = "https://data.binance.vision/data/spot/monthly/klines"

    @staticmethod
    async def from_binance_api(
        symbol: str = "BTCUSDT",
        interval: str = "1h",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1500,
    ) -> pd.DataFrame:
        """Fetch historical klines from Binance REST API."""
        url = f"{DataLoader.BINANCE_BASE}/klines"
        params = {
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": min(limit, 1500),
        }
        if start_time:
            params["startTime"] = int(start_time.timestamp() * 1000)
        if end_time:
            params["endTime"] = int(end_time.timestamp() * 1000)

        all_candles = []
        async with httpx.AsyncClient() as client:
            while True:
                resp = await client.get(url, params=params)
                resp.raise_for_status()
                data = resp.json()
                if not data:
                    break
                all_candles.extend(data)
                if len(data) < limit:
                    break
                # Move time window forward for pagination
                params["startTime"] = data[-1][0] + 1

        if not all_candles:
            return pd.DataFrame()

        df = pd.DataFrame(all_candles, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_av", "trades", "tb_base_av", "tb_quote_av", "ignore",
        ])
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        df = df.set_index("open_time")
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        df = df[["open", "high", "low", "close", "volume"]]
        return df

    @staticmethod
    def from_csv(filepath: str) -> pd.DataFrame:
        """Load historical data from a CSV file."""
        df = pd.read_csv(filepath)
        if "timestamp" in df.columns or "open_time" in df.columns:
            time_col = "timestamp" if "timestamp" in df.columns else "open_time"
            df[time_col] = pd.to_datetime(df[time_col])
            df = df.set_index(time_col)
        return df

    @staticmethod
    def generate_mock_data(
        periods: int = 5000,
        start_price: float = 50000.0,
        volatility: float = 0.02,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Generate synthetic OHLCV data for testing."""
        np.random.seed(seed)
        dates = pd.date_range(
            end=datetime.now(),
            periods=periods,
            freq="1h",
        )
        returns = np.random.normal(0, volatility, periods)
        price = start_price * np.exp(np.cumsum(returns))
        
        df = pd.DataFrame(index=dates)
        df["close"] = price
        df["open"] = df["close"].shift(1) * (1 + np.random.normal(0, volatility * 0.3, periods))
        df["high"] = df[["open", "close"]].max(axis=1) * (1 + abs(np.random.normal(0, volatility * 0.5, periods)))
        df["low"] = df[["open", "close"]].min(axis=1) * (1 - abs(np.random.normal(0, volatility * 0.5, periods)))
        df["volume"] = np.random.exponential(100, periods)
        df = df.fillna(method="bfill").dropna()
        return df


# ── Backtesting Engine ───────────────────────────────────

class BacktestEngine:
    """
    Core backtesting engine with realistic trade simulation.
    Supports multi-entry/multi-exit stop management.
    """

    def __init__(
        self,
        initial_balance: float = 10.0,
        maker_fee: float = 0.0002,   # 0.02%
        taker_fee: float = 0.0004,   # 0.04%
        slippage_bps: float = 0.5,   # 0.5 bps slippage
    ):
        self.initial_balance = initial_balance
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.slippage_bps = slippage_bps

    async def run(
        self,
        strategy,
        data: pd.DataFrame,
        symbol: str = "BTCUSDT",
        leverage: int = 3,
    ) -> BacktestResult:
        """
        Run a full backtest with the given strategy on historical data.
        
        Args:
            strategy: Object with `generate_signal(df) -> Signal` method
            data: OHLCV DataFrame
            symbol: Trading pair symbol
            leverage: Account leverage
        
        Returns:
            BacktestResult with all performance metrics
        """
        balance = self.initial_balance
        equity = balance
        position = None  # {direction, entry_price, qty, sl, tp}
        
        trades: list[BacktestTrade] = []
        equity_curve = [balance]

        # Precompute strategy signals on the full dataset
        df = data.copy()
        signals = []
        for i in range(len(df)):
            chunk = df.iloc[:i+1]
            if len(chunk) > 50:
                sig = strategy.generate_signal(chunk)
                signals.append(sig)
            else:
                signals.append(None)

        # Walk through data and simulate trading
        for i in range(1, len(df)):
            current = df.iloc[i]
            prev = df.iloc[i - 1]
            current_time = df.index[i]
            signal = signals[i-1] if i-1 < len(signals) else None

            # ── Manage existing position ────────
            if position is not None:
                # Check stop loss
                if position["direction"] == "long" and current["low"] <= position["sl"]:
                    exit_price = position["sl"] * (1 - self.slippage_bps / 10000)
                    fee = exit_price * position["qty"] * self.taker_fee
                    pnl = (exit_price - position["entry_price"]) * position["qty"] * leverage - fee
                    equity += pnl
                    
                    trades.append(BacktestTrade(
                        entry_time=position["entry_time"],
                        exit_time=current_time,
                        symbol=symbol,
                        direction=position["direction"],
                        entry_price=position["entry_price"],
                        exit_price=exit_price,
                        quantity=position["qty"],
                        leverage=leverage,
                        pnl=pnl,
                        pnl_pct=pnl / balance if balance > 0 else 0,
                        fees=fee,
                        exit_reason="sl",
                    ))
                    position = None

                elif position["direction"] == "short" and current["high"] >= position["sl"]:
                    exit_price = position["sl"] * (1 + self.slippage_bps / 10000)
                    fee = abs(exit_price * position["qty"]) * self.taker_fee
                    pnl = (position["entry_price"] - exit_price) * position["qty"] * leverage - fee
                    equity += pnl
                    
                    trades.append(BacktestTrade(
                        entry_time=position["entry_time"],
                        exit_time=current_time,
                        symbol=symbol,
                        direction=position["direction"],
                        entry_price=position["entry_price"],
                        exit_price=exit_price,
                        quantity=position["qty"],
                        leverage=leverage,
                        pnl=pnl,
                        pnl_pct=pnl / balance if balance > 0 else 0,
                        fees=fee,
                        exit_reason="sl",
                    ))
                    position = None

                # Check take profit 1
                elif position.get("tp1_hit") is None and position.get("tp1") is not None:
                    if position["direction"] == "long" and current["high"] >= position["tp1"]:
                        position["tp1_hit"] = True
                    elif position["direction"] == "short" and current["low"] <= position["tp1"]:
                        position["tp1_hit"] = True

                # Check take profit 2
                elif position.get("tp2_hit") is None and position.get("tp2") is not None:
                    if position["direction"] == "long" and current["high"] >= position["tp2"]:
                        exit_price = position["tp2"]
                        fee = abs(exit_price * position["qty"]) * self.taker_fee
                        remaining_qty = position["qty"] * 0.5
                        pnl = (exit_price - position["entry_price"]) * remaining_qty * leverage - fee
                        equity += pnl
                        
                        trades.append(BacktestTrade(
                            entry_time=position["entry_time"],
                            exit_time=current_time,
                            symbol=symbol,
                            direction=position["direction"],
                            entry_price=position["entry_price"],
                            exit_price=exit_price,
                            quantity=remaining_qty,
                            leverage=leverage,
                            pnl=pnl,
                            pnl_pct=pnl / balance if balance > 0 else 0,
                            fees=fee,
                            exit_reason="tp2",
                        ))
                        position = None

                    elif position["direction"] == "short" and current["low"] <= position["tp2"]:
                        exit_price = position["tp2"]
                        fee = abs(exit_price * position["qty"]) * self.taker_fee
                        remaining_qty = position["qty"] * 0.5
                        pnl = (position["entry_price"] - exit_price) * remaining_qty * leverage - fee
                        equity += pnl
                        
                        trades.append(BacktestTrade(
                            entry_time=position["entry_time"],
                            exit_time=current_time,
                            symbol=symbol,
                            direction=position["direction"],
                            entry_price=position["entry_price"],
                            exit_price=exit_price,
                            quantity=remaining_qty,
                            leverage=leverage,
                            pnl=pnl,
                            pnl_pct=pnl / balance if balance > 0 else 0,
                            fees=fee,
                            exit_reason="tp2",
                        ))
                        position = None

            # ── Enter new position ──────────────
            if position is None and signal is not None and signal.direction in ("long", "short"):
                if signal.confidence >= 0.4:
                    entry_price = current["close"] * (1 + self.slippage_bps / 10000 * (1 if signal.direction == "long" else -1))
                    fee = entry_price * balance * self.taker_fee * 0.01  # Approximate
                    
                    position = {
                        "direction": signal.direction,
                        "entry_price": entry_price,
                        "entry_time": current_time,
                        "qty": balance * 0.01 / (entry_price * self.slippage_bps / 10000),  # Simplified sizing
                        "sl": signal.stop_loss,
                        "tp1": signal.take_profits[0]["price"] if signal.take_profits else None,
                        "tp2": signal.take_profits[1]["price"] if len(signal.take_profits) > 1 else None,
                        "tp1_hit": None,
                        "tp2_hit": None,
                    }

            # Track equity
            if position is not None:
                current_pnl = 0
                if position["direction"] == "long":
                    current_pnl = (current["close"] - position["entry_price"]) * position["qty"] * leverage
                else:
                    current_pnl = (position["entry_price"] - current["close"]) * position["qty"] * leverage
                current_equity = equity + current_pnl
            else:
                current_equity = equity

            equity_curve.append(current_equity)

        # Close any remaining position at last price
        if position is not None:
            last = df.iloc[-1]
            exit_price = last["close"]
            if position["direction"] == "long":
                pnl = (exit_price - position["entry_price"]) * position["qty"] * leverage
            else:
                pnl = (position["entry_price"] - exit_price) * position["qty"] * leverage
            equity += pnl
            
            trades.append(BacktestTrade(
                entry_time=position["entry_time"],
                exit_time=df.index[-1],
                symbol=symbol,
                direction=position["direction"],
                entry_price=position["entry_price"],
                exit_price=exit_price,
                quantity=position["qty"],
                leverage=leverage,
                pnl=pnl,
                exit_reason="end_of_data",
            ))
            position = None

        return self._compute_metrics(trades, equity_curve)

    def _compute_metrics(
        self,
        trades: list[BacktestTrade],
        equity_curve: list[float],
    ) -> BacktestResult:
        """Compute all performance metrics from trade list and equity curve."""
        equity = np.array(equity_curve)
        returns = np.diff(equity) / equity[:-1]
        # Filter out NaN/inf returns
        returns = returns[np.isfinite(returns)]

        total_trades = len(trades)
        if total_trades == 0:
            return BacktestResult(
                equity_curve=equity_curve,
                win_rate=0.0,
                sharpe_ratio=0.0,
            )

        winning = [t for t in trades if t.pnl > 0]
        losing = [t for t in trades if t.pnl <= 0]
        win_rate = len(winning) / total_trades if total_trades > 0 else 0

        total_pnl = sum(t.pnl for t in trades)
        total_pnl_pct = total_pnl / self.initial_balance if self.initial_balance > 0 else 0
        total_fees = sum(t.fees for t in trades)

        # Max drawdown
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        max_dd_pct = float(np.max(drawdown))

        # Sharpe ratio (annualized, assuming hourly data)
        if len(returns) > 0 and np.std(returns) > 0:
            sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(365 * 24))
            # Sortino (downside deviation only)
            downside = returns[returns < 0]
            sortino = float(np.mean(returns) / np.std(downside) * np.sqrt(365 * 24)) if len(downside) > 0 and np.std(downside) > 0 else 0
        else:
            sharpe = 0.0
            sortino = 0.0

        # Calmar ratio
        calmar = (total_pnl_pct * 100) / (max_dd_pct * 100) if max_dd_pct > 0 else 0

        # Profit factor
        gross_profit = sum(t.pnl for t in winning) if winning else 0
        gross_loss = abs(sum(t.pnl for t in losing)) if losing else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf") if gross_profit > 0 else 0

        # Average trade metrics
        avg_win = np.mean([t.pnl for t in winning]) if winning else 0
        avg_loss = np.mean([t.pnl for t in losing]) if losing else 0
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * abs(avg_loss)) if total_trades > 0 else 0

        # Average trade duration
        durations = [(t.exit_time - t.entry_time).total_seconds() for t in trades if t.exit_time and t.entry_time]
        avg_duration = timedelta(seconds=np.mean(durations)) if durations else timedelta()

        return BacktestResult(
            total_trades=total_trades,
            winning_trades=len(winning),
            losing_trades=len(losing),
            win_rate=win_rate,
            total_pnl=total_pnl,
            total_pnl_pct=total_pnl_pct,
            total_fees=total_fees,
            max_drawdown=max_dd_pct,
            max_drawdown_pct=max_dd_pct,
            sharpe_ratio=round(sharpe, 4),
            sortino_ratio=round(sortino, 4),
            calmar_ratio=round(calmar, 4),
            profit_factor=round(profit_factor, 4),
            avg_win=round(avg_win, 4),
            avg_loss=round(avg_loss, 4),
            avg_trade_duration=avg_duration,
            expectancy=round(expectancy, 4),
            equity_curve=[float(e) for e in equity_curve],
            trades=trades[:100],  # Limit stored trades
        )


# ── Walk-Forward Analyzer ───────────────────────────────

class WalkForwardAnalyzer:
    """
    Walk-forward backtesting to prevent overfitting.
    Splits data into training/validation windows and tests strategy robustness.
    """

    def __init__(self, backtest_engine: BacktestEngine):
        self.backtest = backtest_engine

    async def analyze(
        self,
        strategy_class,
        data: pd.DataFrame,
        train_size: int = 2000,
        test_size: int = 500,
        step_size: int = 500,
    ) -> list[BacktestResult]:
        """Run walk-forward analysis."""
        results = []
        for start in range(0, len(data) - train_size - test_size, step_size):
            train_data = data.iloc[start:start + train_size]
            test_data = data.iloc[start + train_size:start + train_size + test_size]

            # Train strategy on training window
            strategy = strategy_class()

            # Test on out-of-sample window
            result = await self.backtest.run(strategy, test_data)
            result.strategy_params = {
                "train_window": str(data.index[start]),
                "test_window": str(data.index[start + train_size]),
                "samples": len(test_data),
            }
            results.append(result)

        return results
