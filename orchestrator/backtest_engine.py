"""Backtest Engine — replay historical candles through autogen pipeline.

Replays candle data from CSV/JSON files through the full autogen_team →
enhanced_risk → paper execution pipeline, then produces a performance report.

Usage:
    from orchestrator.backtest_engine import BacktestEngine

    engine = BacktestEngine(executor=executor, data_path="data/btc_1m.csv")
    report = engine.run()

CSV format (expected columns):
    timestamp, open, high, low, close, volume
    - optional: rsi_14, atr_14, ema_9, ema_21, bb_width, atr_pct, atr_mean_pct, atr_std_pct
"""

from __future__ import annotations

import csv
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class BacktestTrade:
    """Record of a single backtest trade."""

    cycle: int = 0
    timestamp: str = ""
    action: str = "HOLD"
    confidence: float = 0.0
    size_pct: float = 0.0
    entry_price: float = 0.0
    rejected: bool = False
    reject_reason: str = ""
    pnl_pct: float = 0.0
    latency_ms: float = 0.0


@dataclass
class BacktestReport:
    """Summary backtest report."""

    total_cycles: int = 0
    completed: int = 0
    rejected: int = 0
    errors: int = 0
    buys: int = 0
    sells: int = 0
    holds: int = 0
    initial_equity: float = 10000.0
    final_equity: float = 10000.0
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    win_count: int = 0
    loss_count: int = 0
    win_rate: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    avg_latency_ms: float = 0.0
    trades: list[dict] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)


class BacktestEngine:
    """Replay historical candles through the autogen pipeline.

    Args:
        executor: AutoGenSignalExecutor instance (paper mode).
        data_path: Path to historical CSV/JSON file.
        sample_rate: Use every Nth candle (1=every, 5=every 5th).
        max_cycles: Max cycles to run (0=all).
    """

    def __init__(
        self,
        executor: Any,  # AutoGenSignalExecutor
        data_path: str,
        sample_rate: int = 1,
        max_cycles: int = 0,
    ):
        self.executor = executor
        self.data_path = data_path
        self.sample_rate = max(1, sample_rate)
        self.max_cycles = max_cycles

    def run(self) -> dict:
        """Run backtest and return report dict."""
        logger.info(f"Backtest: loading data from {self.data_path}")

        # Load candle data
        candles = self._load_data()
        if not candles:
            logger.error("No candle data loaded")
            return BacktestReport().model_dump() if hasattr(BacktestReport, 'model_dump') else vars(BacktestReport())

        logger.info(f"Backtest: {len(candles)} candles, sample_rate={self.sample_rate}")

        # Sample candles
        sampled = candles[:: self.sample_rate]
        if self.max_cycles > 0:
            sampled = sampled[: self.max_cycles]

        report = BacktestReport(initial_equity=self.executor.account.equity)
        report.equity_curve.append(report.initial_equity)

        latencies: list[float] = []
        equity = report.initial_equity
        peak_equity = equity
        max_dd = 0.0

        for i, candle in enumerate(sampled):
            logger.info(f"Backtest cycle {i+1}/{len(sampled)}")

            try:
                result = self.executor.execute_cycle(
                    candle_data=candle,
                    vlm_output=None,  # no VLM in backtest
                )

                report.total_cycles += 1
                latencies.append(result.total_latency_ms)

                # Extract trade info
                action = result.langgraph_signal.get("action", "HOLD")
                confidence = result.langgraph_signal.get("confidence", 0.0)

                trade = BacktestTrade(
                    cycle=i + 1,
                    timestamp=result.timestamp,
                    action=action,
                    confidence=confidence,
                    rejected=(result.status == "rejected"),
                    reject_reason=result.risk_decision.get("reason", ""),
                    latency_ms=result.total_latency_ms,
                )

                if result.status == "completed":
                    report.completed += 1
                    if action == "BUY":
                        report.buys += 1
                    elif action == "SELL":
                        report.sells += 1
                    else:
                        report.holds += 1
                elif result.status == "rejected":
                    report.rejected += 1
                elif result.status == "error":
                    report.errors += 1

                # Track equity
                if self.executor.paper_mode and isinstance(self.executor.executor, type(self.executor.executor)):
                    equity = self.executor.account.equity

                # Simulate PnL for executed trades (simplified)
                if result.status == "completed" and action != "HOLD":
                    # Simple PnL: assume 0.1% per winning trade, -0.15% per losing
                    # In real backtest, this would track actual position entry/exit
                    is_win = confidence > 0.7
                    trade_pnl = 0.001 if is_win else -0.0015
                    trade.pnl_pct = trade_pnl
                    equity *= (1 + trade_pnl * confidence)

                    if is_win:
                        report.win_count += 1
                    else:
                        report.loss_count += 1

                report.equity_curve.append(equity)

                # Track drawdown
                if equity > peak_equity:
                    peak_equity = equity
                dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
                if dd > max_dd:
                    max_dd = dd

                report.trades.append(vars(trade))

            except Exception as e:
                logger.error(f"Backtest cycle {i+1} error: {e}")
                report.errors += 1
                report.total_cycles += 1

        # ── Compute final report ────────────────────────
        report.final_equity = equity
        report.total_pnl = equity - report.initial_equity
        report.total_pnl_pct = (equity / report.initial_equity - 1) if report.initial_equity > 0 else 0.0
        total_trades = report.win_count + report.loss_count
        report.win_rate = report.win_count / total_trades if total_trades > 0 else 0.0
        report.max_drawdown_pct = max_dd
        report.avg_latency_ms = sum(latencies) / len(latencies) if latencies else 0.0

        # Sharpe ratio (simplified: using mean PnL / std of equity changes)
        if len(report.equity_curve) > 2:
            returns = [
                (report.equity_curve[i] / report.equity_curve[i - 1] - 1)
                for i in range(1, len(report.equity_curve))
            ]
            if returns:
                mean_r = sum(returns) / len(returns)
                std_r = (sum((r - mean_r) ** 2 for r in returns) / len(returns)) ** 0.5
                report.sharpe_ratio = (mean_r / std_r * (252 * 24 * 60) ** 0.5) if std_r > 0 else 0.0

        logger.info(
            f"Backtest complete: {report.total_cycles} cycles, "
            f"PnL={report.total_pnl:.2f} ({report.total_pnl_pct:.2%}), "
            f"win_rate={report.win_rate:.1%}, max_dd={report.max_drawdown_pct:.2%}"
        )

        return vars(report)

    # ── Data loading ────────────────────────────────────

    def _load_data(self) -> list[dict]:
        """Load candle data from CSV or JSON."""
        path = Path(self.data_path)
        if not path.exists():
            logger.error(f"Data file not found: {self.data_path}")
            return []

        if path.suffix == ".json":
            return self._load_json(path)
        elif path.suffix in (".csv", ".txt"):
            return self._load_csv(path)
        else:
            logger.error(f"Unsupported format: {path.suffix}")
            return []

    def _load_csv(self, path: Path) -> list[dict]:
        """Load candles from CSV file."""
        candles: list[dict] = []
        try:
            with open(path, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    candle = {
                        "symbol": row.get("symbol", "BTCUSDT"),
                        "price": float(row.get("close", 0)),
                        "ohlcv": {
                            "open": float(row.get("open", 0)),
                            "high": float(row.get("high", 0)),
                            "low": float(row.get("low", 0)),
                            "close": float(row.get("close", 0)),
                            "volume": float(row.get("volume", 0)),
                        },
                        "indicators": {
                            "rsi_14": float(row.get("rsi_14", 50)),
                            "atr_14": float(row.get("atr_14", 0)),
                            "atr_pct": float(row.get("atr_pct", 0)),
                            "atr_mean_pct": float(row.get("atr_mean_pct", 0)),
                            "atr_std_pct": float(row.get("atr_std_pct", 0)),
                            "ema_9": float(row.get("ema_9", 0)),
                            "ema_21": float(row.get("ema_21", 0)),
                            "bb_width": float(row.get("bb_width", 0)),
                        },
                        "regime": row.get("regime", "unknown"),
                    }
                    candles.append(candle)
        except Exception as e:
            logger.error(f"Failed to load CSV: {e}")
        return candles

    def _load_json(self, path: Path) -> list[dict]:
        """Load candles from JSON file."""
        try:
            with open(path) as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and "candles" in data:
                return data["candles"]
            return [data]
        except Exception as e:
            logger.error(f"Failed to load JSON: {e}")
            return []
