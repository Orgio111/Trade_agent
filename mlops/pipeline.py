"""
End-to-end MLOps pipeline orchestrator.

Ties together drift detection, walk-forward optimization, PPO retraining,
and the model registry into a single async ``run()`` entry point.
"""
from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

from core.config import get_settings
from core.observability import RETRAIN_COUNTER
from mlops.drift_detector import DriftDetector, DriftReport
from mlops.model_registry import ModelMetadata, list_models, save_model
from mlops.walk_forward import WFOConfig, WalkForwardOptimizer, sharpe_ratio, max_drawdown
from mlops.retraining import evaluate_ppo_model, retrain_ppo

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Result of a pipeline execution."""
    success: bool = False
    model_version: int | None = None
    drift_report: DriftReport | None = None
    wfo_results: list[dict] = field(default_factory=list)
    train_sharpe: float = 0.0
    test_sharpe: float = 0.0
    test_max_drawdown: float = 0.0
    test_win_rate: float = 0.0
    total_timesteps: int = 0
    training_duration_s: float = 0.0
    error: str | None = None
    psi_scores: dict[str, float] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════════
#  MLOpsPipeline
# ═══════════════════════════════════════════════════════════════════════════════

class MLOpsPipeline:
    """Orchestrates the full MLOps lifecycle: drift → WFO → retrain → register.

    Usage::

        pipeline = MLOpsPipeline(drift_detector, price_data)
        result = await pipeline.run()
        print(f"Model v{result.model_version} registered")
    """

    def __init__(
        self,
        drift_detector: DriftDetector,
        price_data: np.ndarray | None = None,
        feature_data: dict[str, np.ndarray] | None = None,
    ) -> None:
        """
        Args:
            drift_detector: Initialized DriftDetector with reference set.
            price_data: Historical price series for retraining (shape (N,)).
            feature_data: Optional dict of feature arrays for drift checking.
        """
        self._drift = drift_detector
        self._price_data = price_data
        self._feature_data = feature_data or {}

    @property
    def drift_detector(self) -> DriftDetector:
        return self._drift

    async def run(
        self,
        price_data: np.ndarray | None = None,
        feature_data: dict[str, np.ndarray] | None = None,
    ) -> int | None:
        """Execute the full pipeline.

        Steps:
        1. Check drift on current feature data
        2. Run Walk-Forward Optimization to find optimal training params
        3. Retrain PPO using the WFO-informed hyperparameters
        4. Register the trained model in the Model Registry
        5. Prune old models

        Returns the registered model version number, or None on failure.
        """
        cfg = get_settings()
        prices = price_data if price_data is not None else self._price_data
        features = feature_data if feature_data is not None else self._feature_data

        if prices is None or len(prices) < 500:
            logger.warning("Pipeline: insufficient price data (%s points) — skipping", len(prices) if prices is not None else 0)
            return None

        result = PipelineResult()

        try:
            # ── Step 1: Drift check ───────────────────────────────────────
            if features and self._drift.has_reference:
                drift_report = self._drift.check(features)
                result.drift_report = drift_report
                result.psi_scores = drift_report.psi_scores
                logger.info(
                    "Drift check: detected=%s  max_psi=%.4f  features=%s",
                    drift_report.drift_detected,
                    drift_report.max_psi,
                    drift_report.drift_features,
                )

            # ── Step 2: Run WFO on recent data ────────────────────────────
            wfo_results = await self._run_wfo(prices, cfg)
            result.wfo_results = [self._serialize_wfo(w) for w in wfo_results]

            # Use median test metrics from WFO as training targets
            if wfo_results:
                median_sharpe = float(np.median([w.test_sharpe for w in wfo_results]))
                median_dd = float(np.median([w.test_max_drawdown for w in wfo_results]))
                median_wr = float(np.median([w.test_win_rate for w in wfo_results]))
            else:
                median_sharpe = 0.0
                median_dd = 0.0
                median_wr = 0.0

            # ── Step 3: Retrain PPO ───────────────────────────────────────
            logger.info("Pipeline: starting PPO retraining (timesteps=%d)", cfg.mlops_retrain_timesteps)
            t0 = time.monotonic()
            trained_path = await retrain_ppo(prices, total_timesteps=cfg.mlops_retrain_timesteps)
            duration = time.monotonic() - t0
            result.training_duration_s = duration
            result.total_timesteps = cfg.mlops_retrain_timesteps
            logger.info("Pipeline: retraining completed in %.1fs — saved to %s", duration, trained_path)

            # ── Step 4: Evaluate PPO via gym environment ─────────────────
            model_eval = evaluate_ppo_model(
                model_path=trained_path,
                price_data=prices,
                train_split=0.8,
                n_episodes=20,
                episode_steps=50,
            )
            train_sharpe = model_eval["train_sharpe"]
            test_sharpe = model_eval["test_sharpe"]
            test_dd = model_eval["test_max_drawdown"]
            test_wr = model_eval["test_win_rate"]
            test_ir = model_eval.get("test_information_ratio", 0.0)
            test_calmar = model_eval.get("test_calmar_ratio", 0.0)
            test_max_losses = model_eval.get("test_max_consecutive_losses", 0)

            # ── Step 5: Register in Model Registry ────────────────────────
            metadata = ModelMetadata(
                version=0,  # assigned by save_model
                created_at=datetime.utcnow().isoformat(),
                model_type="ppo_execution",
                training_duration_s=duration,
                total_timesteps=cfg.mlops_retrain_timesteps,
                train_sharpe=train_sharpe,
                test_sharpe=test_sharpe,
                test_max_drawdown=test_dd,
                test_win_rate=test_wr,
                test_information_ratio=test_ir,
                test_calmar_ratio=test_calmar,
                test_max_consecutive_losses=test_max_losses,
                psi_scores=result.psi_scores,
                feature_count=40,
                notes=f"WFO-triggered retrain. "
                      f"WFO windows: {len(wfo_results)}. "
                      f"Drift: {result.drift_report.drift_detected if result.drift_report else False}",
            )
            version = save_model(trained_path, metadata)
            result.model_version = version
            result.train_sharpe = train_sharpe
            result.test_sharpe = test_sharpe
            result.test_max_drawdown = median_dd
            result.test_win_rate = median_wr
            result.success = True

            # ── Step 6: Prune old models ──────────────────────────────────
            from mlops.model_registry import prune_models
            pruned = prune_models(keep_last=10)
            if pruned:
                logger.info("Pruned %d old model versions", pruned)

            logger.info(
                "Pipeline complete: v%d registered (train_sharpe=%.2f, test_sharpe=%.2f, dd=%.2f%%, ir=%.2f, calmar=%.2f)",
                version, train_sharpe, test_sharpe, median_dd * 100, test_ir, test_calmar,
            )
            return version

        except Exception as exc:
            result.error = str(exc)
            result.success = False
            logger.error("Pipeline failed: %s", exc, exc_info=True)
            return None

    # ── WFO helper ────────────────────────────────────────────────────────────

    async def _run_wfo(
        self,
        prices: np.ndarray,
        cfg: Any,
    ) -> list:
        """Run walk-forward optimization in a thread pool."""
        from datetime import datetime as dt

        n = len(prices)
        timestamps = [dt.fromtimestamp(86400 * i) for i in range(n)]

        def _backtest_fn(series: np.ndarray, params: dict) -> np.ndarray:
            """Simple momentum-based backtest for WFO."""
            lookback = int(params.get("lookback", 20))
            threshold = params.get("threshold", 0.0)
            returns = np.diff(np.log(series + 1e-10))
            signals = np.zeros_like(returns)
            for i in range(lookback, len(returns)):
                ma = np.mean(returns[i - lookback:i])
                signals[i] = 1 if ma > threshold else -1 if ma < -threshold else 0
            strategy_returns = signals[:-1] * returns[1:]
            return np.clip(strategy_returns, -0.1, 0.1)

        wfo_config = WFOConfig(
            train_months=cfg.wfo_train_months or 12,
            test_months=cfg.wfo_test_months or 3,
            step_months=1,
            param_bounds={
                "lookback": (5, 60),
                "threshold": (0.0, 0.01),
            },
            min_trades=10,
        )

        optimizer = WalkForwardOptimizer(_backtest_fn, wfo_config)

        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            lambda: optimizer.run(prices, timestamps),
        )
        return results

    @staticmethod
    def _serialize_wfo(w: Any) -> dict:
        return {
            "train_start": str(w.train_start.date()),
            "train_end": str(w.train_end.date()),
            "test_start": str(w.test_start.date()),
            "test_end": str(w.test_end.date()),
            "best_params": w.best_params,
            "train_sharpe": w.train_sharpe,
            "test_sharpe": w.test_sharpe,
            "test_max_drawdown": w.test_max_drawdown,
            "test_win_rate": w.test_win_rate,
        }
