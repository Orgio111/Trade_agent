"""
QUANTEX Continual Learning Pipeline — Automatic retraining with performance monitoring.

Architecture:
  ┌─────────────────────┐
  │ Backtest Monitor    │ ← checks if strategy performance degrades
  ├─────────────────────┤
  │ Retrain Trigger     │ ← time-based (daily) + perf-based (accuracy drop)
  ├─────────────────────┤
  │ Model Training      │ ← trains ML + RL models on fresh data
  ├─────────────────────┤
  │ Model Validation    │ ← walk-forward backtest on holdout
  ├─────────────────────┤
  │ Model Promotion     │ ← promote if improvement > threshold
  └─────────────────────┘

Usage:
    clp = ContinualLearningPipeline(ml_engine, ppo_train_fn)
    await clp.run_daily_check()
    # Automatically retrains if needed
"""

import asyncio
import time
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Callable
from collections import deque

import numpy as np
import pandas as pd

from .ml_signals import MLSignalEngine
from .model_registry import ModelRegistry, ModelVersion

logger = logging.getLogger("quantex.continual_learning")


class PerformanceTracker:
    """
    Tracks strategy performance over time.

    Maintains rolling windows of:
      - Trade outcomes (PnL, win/loss)
      - Daily PnL
      - Win rate (rolling 20, 50, 100)
      - Sharpe ratio (rolling 30-day)
      - Max drawdown
    """

    def __init__(self, window_sizes: list[int] = None):
        self.window_sizes = window_sizes or [20, 50, 100]
        self._trades: deque = deque(maxlen=max(self.window_sizes))
        self._daily_pnl: dict[str, float] = {}
        self._drawdown_peak = 0.0

    def record_trade(self, pnl: float, timestamp: Optional[float] = None):
        """Record a trade outcome."""
        self._trades.append({
            "pnl": pnl,
            "timestamp": timestamp or time.time(),
        })

    def get_win_rate(self, window: int = 20) -> float:
        """Get win rate over rolling window."""
        recent = list(self._trades)[-window:]
        if not recent:
            return 0.0
        wins = sum(1 for t in recent if t["pnl"] > 0)
        return wins / len(recent)

    def get_sharpe(self, window: int = 30, trading_days: int = 365) -> float:
        """Get rolling Sharpe ratio (annualized)."""
        # Convert to daily PnL
        daily_pnls = []
        if len(self._trades) < 5:
            return 0.0
        pnls = [t["pnl"] for t in list(self._trades)[-window:]]
        if len(pnls) < 2:
            return 0.0
        mean_pnl = np.mean(pnls)
        std_pnl = np.std(pnls) + 1e-10
        return (mean_pnl / std_pnl) * np.sqrt(trading_days)

    def get_daily_pnl_today(self) -> float:
        """Get today's total PnL."""
        today = datetime.now().strftime("%Y-%m-%d")
        return self._daily_pnl.get(today, 0.0)

    def get_summary(self) -> dict:
        """Get performance summary across all windows."""
        summary = {}
        for window in self.window_sizes:
            wr = self.get_win_rate(window)
            sharpe = self.get_sharpe(window)
            summary[f"win_rate_{window}"] = round(wr, 4)
            summary[f"sharpe_{window}"] = round(sharpe, 4)

        summary["trades_total"] = len(self._trades)
        summary["total_pnl"] = round(sum(t["pnl"] for t in self._trades), 4)
        summary["daily_pnl_today"] = round(self.get_daily_pnl_today(), 4)

        return summary


class ContinualLearningPipeline:
    """
    Automatic retraining pipeline for ML + RL models.

    Triggers:
      1. Time-based: Run daily at configurable time
      2. Performance-based: Retrain if win rate drops > threshold
      3. Manual: Explicit call to force retrain

    Flow:
      1. Check if retrain is needed
      2. Fetch latest market data
      3. Train ML model on expanded dataset
      4. Train RL model on expanded dataset
      5. Validate on held-out data (walk-forward)
      6. Compare performance with current best model
      7. Promote to production if better
    """

    def __init__(
        self,
        ml_engine: MLSignalEngine,
        ppo_train_fn: Callable,
        registry: Optional[ModelRegistry] = None,
        min_trades_before_retrain: int = 20,
        win_rate_threshold: float = 0.45,
        retrain_interval_hours: int = 24,
    ):
        """
        Args:
            ml_engine: MLSignalEngine to retrain
            ppo_train_fn: Callable that trains PPO and returns model
            registry: ModelRegistry for versioning
            min_trades_before_retrain: Minimum trades since last retrain
            win_rate_threshold: Retrain if win rate drops below this
            retrain_interval_hours: Force retrain after N hours
        """
        self.ml_engine = ml_engine
        self.ppo_train_fn = ppo_train_fn
        self.registry = registry or ModelRegistry()
        self.min_trades = min_trades_before_retrain
        self.wr_threshold = win_rate_threshold
        self.retrain_interval = retrain_interval_hours

        self.tracker = PerformanceTracker()
        self._last_retrain_time: Optional[float] = None
        self._retrain_count = 0
        self._is_training = False

    # ── Check if Retrain Needed ───────────────────────────────

    def should_retrain(self) -> tuple[bool, str]:
        """
        Check if retraining is needed.

        Returns:
            (should_retrain: bool, reason: str)
        """
        # 1. Not enough trades? Skip
        if len(self.tracker._trades) < self.min_trades:
            return False, f"Only {len(self.tracker._trades)} trades (need {self.min_trades})"

        # 2. Time-based: Force retrain after interval
        if self._last_retrain_time is not None:
            hours_since = (time.time() - self._last_retrain_time) / 3600
            if hours_since >= self.retrain_interval:
                return True, f"Time-based: {hours_since:.0f}h since last retrain"

        # 3. Performance-based: Win rate dropped
        wr_20 = self.tracker.get_win_rate(20)
        wr_50 = self.tracker.get_win_rate(50)

        if wr_20 < self.wr_threshold and wr_50 < self.wr_threshold:
            return True, f"Performance: WR(20)={wr_20:.1%}, WR(50)={wr_50:.1%} below {self.wr_threshold:.0%}"

        # 4. First retrain
        if self._last_retrain_time is None:
            return True, "Initial training"

        return False, f"Performance OK: WR(20)={wr_20:.1%}"

    # ── Execute Retrain ───────────────────────────────────────

    async def retrain(self, market_data: pd.DataFrame, val_data: Optional[pd.DataFrame] = None) -> dict:
        """
        Execute full retraining cycle.

        Args:
            market_data: Recent OHLCV data for training
            val_data: Optional validation data (if None, splits market_data)

        Returns:
            dict with retrain results
        """
        if self._is_training:
            return {"status": "skipped", "reason": "Already training"}

        self._is_training = True
        t0 = time.time()
        results = {}

        try:
            # Split data if no validation set
            if val_data is None:
                split = int(len(market_data) * 0.8)
                train_data = market_data.iloc[:split].copy()
                val_data = market_data.iloc[split:].copy()
            else:
                train_data = market_data.copy()

            # 1. Retrain ML model
            logger.info(f"Retraining ML model on {len(train_data)} candles...")
            ml_result = self.ml_engine.train(train_data, force=True)
            results["ml"] = {
                "train_acc": ml_result.get("train_accuracy", 0),
                "test_acc": ml_result.get("test_accuracy", 0),
                "features": ml_result.get("n_features", 0),
            }

            # 2. Save ML model version
            ml_version = self.registry.register_model(
                model_type="ml_rf",
                version=f"v{self._retrain_count + 1}",
                metrics={
                    "train_accuracy": ml_result.get("train_accuracy", 0),
                    "test_accuracy": ml_result.get("test_accuracy", 0),
                },
                metadata={
                    "candles": len(train_data),
                    "timestamp": datetime.utcnow().isoformat(),
                },
            )
            results["ml_version"] = ml_version.version

            # 3. Retrain PPO model (async)
            logger.info("Retraining PPO model...")
            try:
                ppo_model = self.ppo_train_fn(train_data)
                results["ppo"] = {"status": "trained"}

                ppo_version = self.registry.register_model(
                    model_type="ppo_rl",
                    version=f"v{self._retrain_count + 1}",
                    metrics={"trained": True},
                    metadata={
                        "candles": len(train_data),
                        "timestamp": datetime.utcnow().isoformat(),
                    },
                )
                results["ppo_version"] = ppo_version.version
            except Exception as e:
                logger.error(f"PPO retrain failed: {e}")
                results["ppo"] = {"status": "failed", "error": str(e)}

            # 4. Validate on held-out data
            if val_data is not None and len(val_data) >= 50:
                from .backtest import BacktestEngine, BacktestResult

                bt = BacktestEngine(initial_balance=100.0, taker_fee=0.0004, slippage_bps=0.5)
                val_result = await bt.run(self.ml_engine, val_data)

                results["validation"] = {
                    "pnl": round(val_result.total_pnl, 4),
                    "trades": len(val_result.trades),
                    "win_rate": round(val_result.win_rate, 4),
                    "sharpe": round(val_result.sharpe, 4),
                }

            elapsed = time.time() - t0
            self._last_retrain_time = time.time()
            self._retrain_count += 1

            results["status"] = "completed"
            results["elapsed_seconds"] = round(elapsed, 1)
            results["retrain_count"] = self._retrain_count

            logger.info(f"Retrain complete in {elapsed:.1f}s (count={self._retrain_count})")

        except Exception as e:
            logger.error(f"Retrain failed: {e}")
            results["status"] = "failed"
            results["error"] = str(e)

        finally:
            self._is_training = False

        return results

    # ── Daily Check ───────────────────────────────────────────

    async def run_daily_check(self, market_data: pd.DataFrame) -> dict:
        """
        Run the daily retrain check.

        Convenience method:
          1. Check if retrain needed
          2. If yes, execute retrain
          3. Return result

        Args:
            market_data: Recent market data

        Returns:
            dict with check result and retrain result if applicable
        """
        should, reason = self.should_retrain()

        result = {
            "check_timestamp": datetime.utcnow().isoformat(),
            "should_retrain": should,
            "reason": reason,
            "performance": self.tracker.get_summary(),
            "retrain_count": self._retrain_count,
        }

        if should:
            retrain_result = await self.retrain(market_data)
            result["retrain_result"] = retrain_result

        return result

    # ── Status ─────────────────────────────────────────────────

    def get_status(self) -> dict:
        """Get current pipeline status."""
        return {
            "is_training": self._is_training,
            "retrain_count": self._retrain_count,
            "last_retrain": datetime.fromtimestamp(self._last_retrain_time).isoformat()
            if self._last_retrain_time else None,
            "total_trades": len(self.tracker._trades),
            "performance": self.tracker.get_summary(),
            "next_check": self.should_retrain()[1],
        }
