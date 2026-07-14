"""
Backtest Engine for Strategy Evaluation — Historical Replay.

Core concept: Fast, accurate backtesting of StrategyDSL instances
on historical OHLCV data to compute fitness scores.

Architecture:
  StrategyDSL + Historical Data → BacktestEngine → Performance Metrics
                                              ↓
                                    RL Reward Score (fitness)
"""

from __future__ import annotations

import logging
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from orchestrator.rl_memory.reward_function import RewardConfig, compute_reward
from orchestrator.rl_memory.memory_store import TradeOutcome
from .seg_generator import StrategyDSL, IndicatorConfig, EntryRule, ExitRule, RiskConfig

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for backtest engine."""

    # Data
    data_dir: str = "data/historical"
    default_symbol: str = "BTCUSDT"
    default_timeframe: str = "1m"

    # Execution simulation
    initial_capital: float = 10000.0
    fee_bps: float = 2.0       # 2 bps = 0.02% per side
    slippage_bps: float = 1.0  # 1 bp slippage

    # Position sizing
    position_sizing: str = "fixed_pct"  # "fixed_pct", "kelly", "volatility"
    position_size_pct: float = 0.05     # 5% per trade

    # Risk limits
    max_open_positions: int = 1
    max_daily_drawdown: float = 0.05
    max_total_drawdown: float = 0.15

    # Performance
    min_trades_for_score: int = 10
    risk_free_rate: float = 0.02  # Annual

    # Output
    save_trades: bool = True
    save_equity_curve: bool = True
    results_dir: str = "backtest/results"


@dataclass
class BacktestResult:
    """Complete backtest result."""

    strategy_name: str
    symbol: str
    timeframe: str
    start_date: datetime
    end_date: datetime
    config: BacktestConfig

    # Trade log
    trades: list[TradeOutcome] = field(default_factory=list)

    # Equity curve
    equity_curve: list[float] = field(default_factory=list)
    timestamps: list[datetime] = field(default_factory=list)

    # Metrics
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    total_pnl_pct: float = 0.0
    total_pnl_abs: float = 0.0
    avg_pnl_pct: float = 0.0
    std_pnl_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    max_drawdown_duration: int = 0
    calmar_ratio: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0

    # RL-based fitness
    rl_fitness: float = 0.0
    avg_reward: float = 0.0

    # Metadata
    backtest_duration_ms: float = 0.0
    data_bars: int = 0

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "strategy_name": self.strategy_name,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "total_trades": self.total_trades,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": self.win_rate,
            "total_pnl_pct": self.total_pnl_pct,
            "total_pnl_abs": self.total_pnl_abs,
            "avg_pnl_pct": self.avg_pnl_pct,
            "std_pnl_pct": self.std_pnl_pct,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "max_drawdown_pct": self.max_drawdown_pct,
            "max_drawdown_duration": self.max_drawdown_duration,
            "calmar_ratio": self.calmar_ratio,
            "profit_factor": self.profit_factor,
            "expectancy": self.expectancy,
            "rl_fitness": self.rl_fitness,
            "avg_reward": self.avg_reward,
            "backtest_duration_ms": self.backtest_duration_ms,
            "data_bars": self.data_bars,
            "trades": [t.to_dict() for t in self.trades] if self.config.save_trades else [],
            "equity_curve": self.equity_curve if self.config.save_equity_curve else [],
        }


class FeatureEngine:
    """Technical indicator computation for backtesting."""

    @staticmethod
    def compute_all(df: pd.DataFrame, config: IndicatorConfig | None = None) -> pd.DataFrame:
        """Compute all technical indicators."""
        config = config or IndicatorConfig()
        df = df.copy()

        # RSI
        df["rsi"] = FeatureEngine._rsi(df["close"], config.rsi_period)

        # MACD
        macd_line, macd_signal, macd_hist = FeatureEngine._macd(
            df["close"], config.macd_fast, config.macd_slow, config.macd_signal
        )
        df["macd"] = macd_line
        df["macd_signal"] = macd_signal
        df["macd_hist"] = macd_hist

        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = FeatureEngine._bollinger_bands(
            df["close"], config.bb_period, config.bb_std
        )
        df["bb_upper"] = bb_upper
        df["bb_middle"] = bb_middle
        df["bb_lower"] = bb_lower
        df["bb_position"] = (df["close"] - bb_lower) / (bb_upper - bb_lower + 1e-10)
        df["bb_width"] = (bb_upper - bb_lower) / bb_middle

        # EMAs
        df["ema_fast"] = FeatureEngine._ema(df["close"], config.ema_fast)
        df["ema_slow"] = FeatureEngine._ema(df["close"], config.ema_slow)
        df["ema_trend"] = FeatureEngine._ema(df["close"], config.ema_trend)
        df["ema_5_cross_20"] = np.where(
            (df["ema_fast"] > df["ema_slow"]) & (df["ema_fast"].shift(1) <= df["ema_slow"].shift(1)),
            1,
            np.where(
                (df["ema_fast"] < df["ema_slow"]) & (df["ema_fast"].shift(1) >= df["ema_slow"].shift(1)),
                -1, 0
            )
        )

        # ATR
        df["atr"] = FeatureEngine._atr(df["high"], df["low"], df["close"], config.atr_period)

        # Volume
        df["vol_sma"] = df["volume"].rolling(config.volume_period).mean()
        df["vol_ratio"] = df["volume"] / df["vol_sma"]

        # Price action
        df["returns"] = df["close"].pct_change()
        df["log_returns"] = np.log(df["close"] / df["close"].shift(1))

        return df

    @staticmethod
    def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / (loss + 1e-10)
        return 100 - 100 / (1 + rs)

    @staticmethod
    def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        macd_signal = macd_line.ewm(span=signal, adjust=False).mean()
        macd_hist = macd_line - macd_signal
        return macd_line, macd_signal, macd_hist

    @staticmethod
    def _bollinger_bands(close: pd.Series, period: int = 20, std: float = 2.0) -> tuple:
        sma = close.rolling(period).mean()
        rolling_std = close.rolling(period).std()
        upper = sma + std * rolling_std
        lower = sma - std * rolling_std
        return upper, sma, lower

    @staticmethod
    def _ema(close: pd.Series, period: int) -> pd.Series:
        return close.ewm(span=period, adjust=False).mean()

    @staticmethod
    def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(period).mean()


class SignalEvaluator:
    """Evaluates entry/exit signals from StrategyDSL."""

    def __init__(self, strategy: StrategyDSL):
        self.strategy = strategy
        self.indicators = strategy.indicators

    def evaluate_entry(self, row: pd.Series, prev_row: pd.Series | None = None) -> tuple[bool, str]:
        """
        Check if entry conditions are met.

        Returns:
            (should_enter, direction) where direction is "long" or "short"
        """
        long_signals = 0
        short_signals = 0
        total_weight = 0

        for rule in self.strategy.entry_rules:
            weight = rule.weight
            total_weight += weight

            indicator_val = self._get_indicator_value(row, rule.indicator)
            if indicator_val is None:
                continue

            prev_val = None
            if prev_row is not None:
                prev_val = self._get_indicator_value(prev_row, rule.indicator)

            if self._check_condition(indicator_val, prev_val, rule.operator, rule.value):
                if rule.indicator in ["rsi", "bb_position"] and rule.operator in ["<", "<="]:
                    # Oversold / lower band → long
                    long_signals += weight
                elif rule.indicator in ["rsi", "bb_position"] and rule.operator in [">", ">="]:
                    # Overbought / upper band → short
                    short_signals += weight
                elif rule.indicator == "macd_hist":
                    if indicator_val > 0:
                        long_signals += weight
                    else:
                        short_signals += weight
                elif rule.indicator == "ema_cross":
                    if indicator_val > 0:
                        long_signals += weight
                    elif indicator_val < 0:
                        short_signals += weight
                elif rule.indicator == "volume_spike":
                    long_signals += weight  # Volume spike confirms direction

        # Decide based on weighted signals
        if long_signals > short_signals and long_signals > total_weight * 0.3:
            return True, "long"
        elif short_signals > long_signals and short_signals > total_weight * 0.3:
            return True, "short"
        return False, "hold"

    def evaluate_exit(
        self,
        row: pd.Series,
        entry_price: float,
        direction: str,
        bars_held: int,
        max_favorable: float,
        max_adverse: float,
    ) -> tuple[bool, str]:
        """
        Check if exit conditions are met.

        Returns:
            (should_exit, reason)
        """
        for rule in self.strategy.exit_rules:
            if rule.type == "tp":
                # Take profit
                if direction == "long":
                    pnl_pct = (row["close"] - entry_price) / entry_price
                else:
                    pnl_pct = (entry_price - row["close"]) / entry_price
                if pnl_pct >= rule.value:
                    return True, "tp"

            elif rule.type == "sl":
                # Stop loss
                if direction == "long":
                    pnl_pct = (row["close"] - entry_price) / entry_price
                else:
                    pnl_pct = (entry_price - row["close"]) / entry_price
                if pnl_pct <= -rule.value:
                    return True, "sl"

            elif rule.type == "trailing_sl":
                # Trailing stop
                trail_pct = rule.value
                if direction == "long":
                    if max_favorable > 0:
                        trail_price = entry_price * (1 + max_favorable * (1 - trail_pct))
                        if row["close"] <= trail_price:
                            return True, "trailing_sl"
                else:
                    if max_favorable > 0:
                        trail_price = entry_price * (1 - max_favorable * (1 - trail_pct))
                        if row["close"] >= trail_price:
                            return True, "trailing_sl"

            elif rule.type == "time":
                # Time-based exit
                if bars_held >= rule.value:
                    return True, "time"

            elif rule.type == "signal_reverse":
                # Reverse signal (re-evaluate entry rules for opposite direction)
                should_enter, new_dir = self.evaluate_entry(row)
                if should_enter and new_dir != direction:
                    return True, "signal_reverse"

        return False, "hold"

    def _get_indicator_value(self, row: pd.Series, indicator: str) -> float | None:
        """Extract indicator value from row."""
        mapping = {
            "rsi": "rsi",
            "macd_hist": "macd_hist",
            "bb_position": "bb_position",
            "ema_cross": "ema_5_cross_20",
            "volume_spike": "vol_ratio",
            "atr": "atr",
        }
        col = mapping.get(indicator)
        if col and col in row and not pd.isna(row[col]):
            return float(row[col])
        return None

    def _check_condition(
        self, current: float, previous: float | None, operator: str, value: float
    ) -> bool:
        """Check if condition is met."""
        if operator == ">":
            return current > value
        elif operator == "<":
            return current < value
        elif operator == ">=":
            return current >= value
        elif operator == "<=":
            return current <= value
        elif operator == "crosses_above" and previous is not None:
            return previous <= value and current > value
        elif operator == "crosses_below" and previous is not None:
            return previous >= value and current < value
        return False


class BacktestEngine:
    """
    Fast vectorized backtest engine for StrategyDSL evaluation.
    """

    def __init__(self, config: BacktestConfig | None = None):
        self.config = config or BacktestConfig()
        Path(self.config.results_dir).mkdir(parents=True, exist_ok=True)

    def load_data(
        self,
        symbol: str | None = None,
        timeframe: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        data_path: str | None = None,
    ) -> pd.DataFrame:
        """Load historical OHLCV data."""
        symbol = symbol or self.config.default_symbol
        timeframe = timeframe or self.config.default_timeframe

        if data_path and Path(data_path).exists():
            df = pd.read_csv(data_path)
        else:
            # Try default locations
            paths = [
                Path(self.config.data_dir) / f"{symbol}_{timeframe}.csv",
                Path(self.config.data_dir) / f"{symbol}.csv",
                Path("data") / f"{symbol}_{timeframe}.csv",
            ]
            df = None
            for p in paths:
                if p.exists():
                    df = pd.read_csv(p)
                    break

        if df is None:
            # Generate synthetic data for testing
            logger.warning(f"No data found for {symbol}, generating synthetic data")
            df = self._generate_synthetic_data(symbol, timeframe)

        # Standardize columns
        df.columns = [c.lower() for c in df.columns]
        required = ["timestamp", "open", "high", "low", "close", "volume"]
        for r in required:
            if r not in df.columns:
                raise ValueError(f"Missing required column: {r}")

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp").sort_index()

        # Filter date range
        if start:
            df = df[df.index >= start]
        if end:
            df = df[df.index <= end]

        return df

    def _generate_synthetic_data(
        self, symbol: str, timeframe: str, bars: int = 10000
    ) -> pd.DataFrame:
        """Generate synthetic OHLCV data for testing."""
        np.random.seed(42)
        dates = pd.date_range(end=datetime.now(), periods=bars, freq="1min")

        # Random walk with drift
        returns = np.random.normal(0.0001, 0.002, bars)
        close = 50000 * np.exp(np.cumsum(returns))

        # Generate OHLC from close
        noise = np.random.uniform(0.999, 1.001, (bars, 4))
        open_ = close * noise[:, 0]
        high = np.maximum.reduce([close, open_, close * noise[:, 1], close * noise[:, 2]])
        low = np.minimum.reduce([close, open_, close * noise[:, 3], close * noise[:, 4] if False else close * 0.999])
        volume = np.random.lognormal(10, 1, bars)

        return pd.DataFrame({
            "timestamp": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        })

    def run(
        self,
        strategy: StrategyDSL,
        data: pd.DataFrame | None = None,
        symbol: str | None = None,
        timeframe: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        data_path: str | None = None,
    ) -> BacktestResult:
        """
        Run backtest for a strategy.

        Args:
            strategy: StrategyDSL to test
            data: Pre-loaded DataFrame (optional)
            symbol, timeframe, start, end, data_path: Data loading params

        Returns:
            BacktestResult with all metrics
        """
        start_time = datetime.now()

        # Load data if not provided
        if data is None:
            data = self.load_data(symbol, timeframe, start, end, data_path)

        # Compute indicators
        data = FeatureEngine.compute_all(data, strategy.indicators)
        data = data.dropna()

        if len(data) < 100:
            logger.warning(f"Insufficient data after indicators: {len(data)} bars")
            return self._empty_result(strategy, symbol or "UNKNOWN", timeframe or "UNKNOWN")

        # Initialize
        evaluator = SignalEvaluator(strategy)
        equity = self.config.initial_capital
        position = None  # (direction, entry_price, entry_time, size, max_fav, max_adv)
        trades = []
        equity_curve = [equity]
        timestamps = [data.index[0]]

        # Iterate through bars
        for i in range(1, len(data)):
            row = data.iloc[i]
            prev_row = data.iloc[i - 1]
            current_price = row["close"]

            # Update equity curve
            if position:
                direction, entry_price, entry_time, size, max_fav, max_adv = position
                if direction == "long":
                    unrealized_pnl = (current_price - entry_price) / entry_price * size * equity
                    max_fav = max(max_fav, (current_price - entry_price) / entry_price)
                    max_adv = min(max_adv, (current_price - entry_price) / entry_price)
                else:
                    unrealized_pnl = (entry_price - current_price) / entry_price * size * equity
                    max_fav = max(max_fav, (entry_price - current_price) / entry_price)
                    max_adv = min(max_adv, (entry_price - current_price) / entry_price)
                position = (direction, entry_price, entry_time, size, max_fav, max_adv)
                equity_curve.append(equity + unrealized_pnl)
            else:
                equity_curve.append(equity)
            timestamps.append(row.name)

            # Check exit if in position
            if position:
                direction, entry_price, entry_time, size, max_fav, max_adv = position
                bars_held = i - data.index.get_loc(entry_time)
                should_exit, exit_reason = evaluator.evaluate_exit(
                    row, entry_price, direction, bars_held, max_fav, max_adv
                )

                # Also check risk limits
                if not should_exit:
                    # Max hold time
                    if bars_held >= strategy.risk.max_hold_bars:
                        should_exit, exit_reason = True, "max_hold"
                    # Daily drawdown check (simplified)
                    current_dd = (equity - self.config.initial_capital) / self.config.initial_capital
                    if current_dd < -self.config.max_daily_drawdown:
                        should_exit, exit_reason = True, "daily_dd_limit"

                if should_exit:
                    # Execute exit
                    exit_price = current_price
                    if direction == "long":
                        pnl_pct = (exit_price - entry_price) / entry_price
                        gross_pnl = pnl_pct * size * equity
                    else:
                        pnl_pct = (entry_price - exit_price) / entry_price
                        gross_pnl = pnl_pct * size * equity

                    # Fees and slippage
                    fees = 2 * self.config.fee_bps / 10000 * size * equity
                    slippage = self.config.slippage_bps / 10000 * size * equity
                    net_pnl = gross_pnl - fees - slippage

                    equity += net_pnl

                    # Create trade outcome
                    trade = TradeOutcome(
                        trade_id=f"{strategy.name}_{len(trades)}",
                        timestamp=row.name.to_pydatetime(),
                        symbol=symbol or self.config.default_symbol,
                        action="BUY" if direction == "long" else "SELL",
                        entry_price=entry_price,
                        exit_price=exit_price,
                        quantity=size,
                        pnl_pct=pnl_pct,
                        pnl_abs=net_pnl,
                        confidence=0.7,  # Placeholder
                        regime="backtest",
                        indicators={k: float(row[k]) for k in [
                            "rsi", "macd_hist", "bb_position", "ema_5_cross_20", "vol_ratio", "atr"
                        ] if k in row and not pd.isna(row[k])},
                        agent_reasoning=[f"entry: {strategy.name}", f"exit: {exit_reason}"],
                        risk_decision={},
                        execution_latency_ms=0,
                        status="FILLED",
                        max_drawdown_pct=abs(min(0, max_adv)),
                    )
                    trades.append(trade)
                    position = None

            # Check entry if not in position
            if not position and len(trades) < 1000:  # Safety limit
                should_enter, direction = evaluator.evaluate_entry(row, prev_row)

                if should_enter:
                    # Position sizing
                    if self.config.position_sizing == "fixed_pct":
                        size = self.config.position_size_pct
                    elif self.config.position_sizing == "kelly":
                        # Simplified Kelly (would need win rate from strategy)
                        size = min(self.config.position_size_pct * 2, 0.1)
                    else:  # volatility
                        atr_pct = row.get("atr", 0) / current_price if current_price > 0 else 0.01
                        size = min(self.config.position_size_pct / max(atr_pct * 100, 0.01), 0.1)

                    # Apply leverage
                    size *= strategy.risk.leverage
                    size = min(size, strategy.risk.max_position_pct * strategy.risk.leverage)

                    position = (direction, current_price, row.name, size, 0.0, 0.0)

        # Close any open position at end
        if position:
            direction, entry_price, entry_time, size, max_fav, max_adv = position
            exit_price = data.iloc[-1]["close"]
            if direction == "long":
                pnl_pct = (exit_price - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - exit_price) / entry_price

            gross_pnl = pnl_pct * size * equity
            fees = 2 * self.config.fee_bps / 10000 * size * equity
            slippage = self.config.slippage_bps / 10000 * size * equity
            net_pnl = gross_pnl - fees - slippage
            equity += net_pnl

            trade = TradeOutcome(
                trade_id=f"{strategy.name}_{len(trades)}",
                timestamp=data.index[-1].to_pydatetime(),
                symbol=symbol or self.config.default_symbol,
                action="BUY" if direction == "long" else "SELL",
                entry_price=entry_price,
                exit_price=exit_price,
                quantity=size,
                pnl_pct=pnl_pct,
                pnl_abs=net_pnl,
                confidence=0.7,
                regime="backtest",
                indicators={},
                agent_reasoning=[f"entry: {strategy.name}", "exit: end_of_data"],
                risk_decision={},
                execution_latency_ms=0,
                status="FILLED",
                max_drawdown_pct=abs(min(0, max_adv)),
            )
            trades.append(trade)

        # Compute metrics
        result = self._compute_metrics(
            strategy, symbol or self.config.default_symbol, timeframe or self.config.default_timeframe,
            data.index[0], data.index[-1], trades, equity_curve, timestamps,
            len(data), (datetime.now() - start_time).total_seconds() * 1000
        )

        # Save results
        self._save_result(result)

        return result

    def _compute_metrics(
        self,
        strategy: StrategyDSL,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
        trades: list[TradeOutcome],
        equity_curve: list[float],
        timestamps: list[datetime],
        data_bars: int,
        duration_ms: float,
    ) -> BacktestResult:
        """Compute all performance metrics."""

        result = BacktestResult(
            strategy_name=strategy.name,
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            config=self.config,
            trades=trades,
            equity_curve=equity_curve,
            timestamps=timestamps,
            data_bars=data_bars,
            backtest_duration_ms=duration_ms,
        )

        if not trades:
            return result

        # Basic metrics
        result.total_trades = len(trades)
        result.wins = sum(1 for t in trades if t.pnl_pct > 0)
        result.losses = result.total_trades - result.wins
        result.win_rate = result.wins / result.total_trades if result.total_trades > 0 else 0

        pnl_pcts = [t.pnl_pct for t in trades]
        pnl_abs = [t.pnl_abs for t in trades]

        result.total_pnl_pct = sum(pnl_pcts)
        result.total_pnl_abs = sum(pnl_abs)
        result.avg_pnl_pct = np.mean(pnl_pcts)
        result.std_pnl_pct = np.std(pnl_pcts) if len(pnl_pcts) > 1 else 0

        # Sharpe ratio (annualized)
        if result.std_pnl_pct > 0:
            # Assume 1m bars, 252 trading days * 390 minutes = ~98280 bars/year
            bars_per_year = 252 * 390 if timeframe == "1m" else 252 * 24 * 60 / int(timeframe[:-1]) if timeframe.endswith("m") else 252
            result.sharpe_ratio = (result.avg_pnl_pct / result.std_pnl_pct) * np.sqrt(bars_per_year)
        else:
            result.sharpe_ratio = 0

        # Sortino ratio (downside deviation)
        negative_returns = [r for r in pnl_pcts if r < 0]
        if negative_returns:
            downside_std = np.std(negative_returns)
            if downside_std > 0:
                result.sortino_ratio = (result.avg_pnl_pct / downside_std) * np.sqrt(bars_per_year)

        # Max drawdown
        equity_arr = np.array(equity_curve)
        peak = np.maximum.accumulate(equity_arr)
        drawdown = (peak - equity_arr) / peak
        result.max_drawdown_pct = float(np.max(drawdown)) if len(drawdown) > 0 else 0

        # Max drawdown duration
        in_dd = drawdown > 0
        dd_durations = []
        current = 0
        for v in in_dd:
            if v:
                current += 1
            elif current > 0:
                dd_durations.append(current)
                current = 0
        if current > 0:
            dd_durations.append(current)
        result.max_drawdown_duration = max(dd_durations) if dd_durations else 0

        # Calmar ratio
        if result.max_drawdown_pct > 0:
            # Annualized return
            total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0]
            years = data_bars / bars_per_year
            annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else total_return
            result.calmar_ratio = annual_return / result.max_drawdown_pct

        # Profit factor
        gross_profit = sum(p for p in pnl_abs if p > 0)
        gross_loss = abs(sum(p for p in pnl_abs if p < 0))
        result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0

        # Expectancy
        avg_win = np.mean([p for p in pnl_abs if p > 0]) if result.wins > 0 else 0
        avg_loss = np.mean([p for p in pnl_abs if p < 0]) if result.losses > 0 else 0
        result.expectancy = result.win_rate * avg_win + (1 - result.win_rate) * avg_loss

        # RL-based fitness
        rewards = [compute_reward(t, RewardConfig())[0] for t in trades]
        result.avg_reward = np.mean(rewards) if rewards else 0

        # Combine metrics into fitness (RL reward weighted + traditional metrics)
        result.rl_fitness = (
            result.avg_reward * 0.4 +
            result.sharpe_ratio * 0.2 +
            result.win_rate * 0.2 +
            (1 - result.max_drawdown_pct) * 0.1 +
            min(result.profit_factor / 3, 1.0) * 0.1
        )

        return result

    def _empty_result(self, strategy: StrategyDSL, symbol: str, timeframe: str) -> BacktestResult:
        """Return empty result for failed backtests."""
        return BacktestResult(
            strategy_name=strategy.name,
            symbol=symbol,
            timeframe=timeframe,
            start_date=datetime.now(),
            end_date=datetime.now(),
            config=self.config,
        )

    def _save_result(self, result: BacktestResult):
        """Save backtest result to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{result.strategy_name}_{timestamp}.json"
        path = Path(self.config.results_dir) / filename

        with open(path, "w") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)

        logger.info(f"[BacktestEngine] Saved result to {path}")

    def run_batch(
        self,
        strategies: list[StrategyDSL],
        data: pd.DataFrame | None = None,
        **kwargs,
    ) -> list[BacktestResult]:
        """Run backtest for multiple strategies."""
        results = []
        for strategy in strategies:
            try:
                result = self.run(strategy, data, **kwargs)
                results.append(result)

                # Update strategy fitness
                strategy.fitness = result.rl_fitness
                strategy.trades_count = result.total_trades
                strategy.win_rate = result.win_rate
                strategy.sharpe = result.sharpe_ratio
                strategy.max_drawdown = result.max_drawdown_pct
                strategy.total_pnl_pct = result.total_pnl_pct

            except Exception as e:
                logger.error(f"[BacktestEngine] Strategy {strategy.name} failed: {e}")

        return results


# Factory functions
def create_backtest_engine(config: BacktestConfig | None = None) -> BacktestEngine:
    """Factory for BacktestEngine."""
    return BacktestEngine(config)


def create_backtest_config(**kwargs) -> BacktestConfig:
    """Factory for BacktestConfig."""
    return BacktestConfig(**kwargs)


__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "BacktestEngine",
    "FeatureEngine",
    "SignalEvaluator",
    "create_backtest_engine",
    "create_backtest_config",
]