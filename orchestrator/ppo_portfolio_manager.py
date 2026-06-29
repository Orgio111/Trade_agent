"""
QUANTEX PPO Portfolio Manager — Reinforcement Learning capital allocation.

Replaces rule-based Kelly/Markowitz allocation with a trained PPO policy
that outputs optimal capital allocation weights based on market state.

Architecture:
  PortfolioAllocEnv (gym) → PPO (SB3) → allocation weights
    ↓                              ↓
  Market returns            Optimal capital allocation
  + portfolio state         vector for N assets

Usage:
    # Training
    manager = PPOPortfolioManager(n_assets=4)
    result = manager.train(returns_df, total_timesteps=100_000)

    # Inference
    weights = manager.allocate(market_features)

    # Fallback to Markowitz if PPO unavailable
    weights = manager.allocate_with_fallback(returns_df)
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .portfolio_optimizer import PortfolioAllocation, PortfolioOptimizer

logger = logging.getLogger("quantex.ppo_portfolio_manager")


@dataclass
class PortfolioManagerResult:
    """Result from PPOPortfolioManager.allocate()."""
    weights: dict[str, float]       # symbol → weight
    expected_return: float
    expected_risk: float
    sharpe_ratio: float
    method: str                     # "ppo", "markowitz", "kelly", "risk_parity"
    ppo_used: bool = False
    ppo_confidence: float = 0.0
    model_version: str = ""
    metadata: dict = field(default_factory=dict)


class PPOPortfolioManager:
    """
    PPO-based portfolio capital allocation manager.

    Wraps a trained Stable-Baselines3 PPO model to output optimal
    capital allocation weights. Falls back to Markowitz/Kelly when
    the PPO model is not trained or unavailable.

    The manager maintains rolling performance tracking so the PPO
    agent can learn from its own allocation decisions.

    Parameters
    ----------
    n_assets:
        Number of assets in the portfolio.
    asset_names:
        Names of the assets (for weight dict keys).
    model_dir:
        Directory to save/load PPO models.
    lookback:
        Rolling window for environment features.
    max_weight:
        Max allocation per asset.
    risk_free_rate:
        Annual risk-free rate.
    """

    def __init__(
        self,
        n_assets: int = 4,
        asset_names: Optional[list[str]] = None,
        model_dir: str | None = None,
        lookback: int = 20,
        max_weight: float = 0.5,
        risk_free_rate: float = 0.02,
    ):
        self.n_assets = n_assets
        self.asset_names = asset_names or [f"asset_{i}" for i in range(n_assets)]
        self.lookback = lookback
        self.max_weight = max_weight
        self.risk_free_rate = risk_free_rate

        # Model directory
        if model_dir is None:
            model_dir = str(Path(__file__).parent / "models" / "ppo_portfolio")
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # PPO model (lazy-loaded)
        self._model = None
        self._env = None
        self._trained = False
        self._train_count = 0
        self._model_path: str = ""

        # Fallback optimizer
        self._optimizer = PortfolioOptimizer(risk_free_rate=risk_free_rate)

        # Performance tracking for adaptive blends
        self._allocation_history: list[dict] = []
        self._ppo_performance: list[float] = []
        self._markowitz_performance: list[float] = []

    # ── Model Lifecycle ──────────────────────────────────────

    def _load_model(self, path: str | None = None) -> bool:
        """Load a trained PPO model from disk.

        Args:
            path: Path to model file. If None, tries latest.

        Returns:
            True if model loaded successfully.
        """
        try:
            from stable_baselines3 import PPO
        except ImportError:
            logger.warning("stable_baselines3 not installed")
            return False

        if path is None:
            # Auto-discover latest
            candidates = sorted(self.model_dir.glob("ppo_portfolio_*.zip"))
            if not candidates:
                return False
            path = str(candidates[-1])
            self._model_path = path

        try:
            self._model = PPO.load(path)
            self._trained = True
            logger.info("PPO portfolio model loaded from %s", path)
            return True
        except Exception as e:
            logger.warning("Failed to load PPO model: %s", e)
            return False

    def _create_env(self, returns_df: pd.DataFrame) -> object:
        """Create PortfolioAllocEnv from returns data."""
        from .rl.portfolio_env import PortfolioAllocEnv

        return PortfolioAllocEnv(
            returns_df=returns_df,
            asset_names=self.asset_names,
            lookback=self.lookback,
            max_weight=self.max_weight,
        )

    def train(
        self,
        returns_df: pd.DataFrame,
        total_timesteps: int = 100_000,
        force: bool = False,
    ) -> dict:
        """Train the PPO portfolio allocation model.

        Args:
            returns_df: DataFrame of asset returns (columns = symbols).
            total_timesteps: Number of training timesteps.
            force: Force retraining even if model exists.

        Returns:
            Training result dict with status and metrics.
        """
        if self._trained and not force:
            return {"status": "skipped", "reason": "Already trained"}

        try:
            from stable_baselines3 import PPO
            from stable_baselines3.common.vec_env import DummyVecEnv
            from stable_baselines3.common.callbacks import EvalCallback
        except ImportError:
            return {"status": "error", "reason": "stable_baselines3 not installed"}

        # Align column names to asset_names
        df = returns_df.copy()
        available = [c for c in self.asset_names if c in df.columns]
        if len(available) < self.n_assets:
            # Pad with synthetic returns
            for name in self.asset_names:
                if name not in df.columns:
                    df[name] = np.random.normal(0, 0.01, len(df))
            available = self.asset_names
        df = df[available]

        # Create environment
        env = self._create_env(df)
        eval_env = self._create_env(df)

        vec_env = DummyVecEnv([lambda: env])

        # Create or update PPO model
        if self._model is None:
            self._model = PPO(
                "MlpPolicy",
                vec_env,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
                verbose=0,
                device="auto",
            )
        else:
            self._model.set_env(vec_env)

        # Callbacks
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(self.model_dir / "best"),
            log_path=str(self.model_dir / "eval_logs"),
            eval_freq=max(1000, total_timesteps // 20),
            n_eval_episodes=5,
            deterministic=True,
            verbose=0,
        )

        t0 = time.time()
        self._model.learn(
            total_timesteps=total_timesteps,
            callback=eval_callback,
            progress_bar=True,
        )
        elapsed = time.time() - t0

        # Save model
        self._train_count += 1
        model_path = self.model_dir / f"ppo_portfolio_v{self._train_count}.zip"
        self._model.save(str(model_path))
        self._model_path = str(model_path)

        # Also save as latest
        latest_path = self.model_dir / "ppo_portfolio_latest"
        self._model.save(str(latest_path))

        self._trained = True
        logger.info(
            "PPO portfolio trained in %.1fs (v%d, %d steps, saved to %s)",
            elapsed, self._train_count, total_timesteps, model_path,
        )

        return {
            "status": "trained",
            "version": self._train_count,
            "elapsed_seconds": round(elapsed, 1),
            "total_timesteps": total_timesteps,
            "model_path": str(model_path),
            "n_assets": self.n_assets,
        }

    # ── Inference ────────────────────────────────────────────

    def allocate(
        self,
        returns_df: pd.DataFrame,
        confidence: float = 0.5,
    ) -> PortfolioManagerResult:
        """Allocate capital using the trained PPO model.

        If PPO model is not available, falls back to Markowitz.

        Args:
            returns_df: Recent asset returns DataFrame.
            confidence: External confidence signal (blended with PPO).

        Returns:
            PortfolioManagerResult with weights and metrics.
        """
        # Try PPO first
        if self._trained and self._model is not None:
            try:
                return self._allocate_ppo(returns_df, confidence)
            except Exception as e:
                logger.warning("PPO allocation failed: %s", e)

        # Fallback to Markowitz
        return self._allocate_markowitz(returns_df, "markowitz")

    def allocate_with_fallback(
        self,
        returns_df: pd.DataFrame,
        method: str = "markowitz",
    ) -> PortfolioManagerResult:
        """Force allocation using a specific fallback method.

        Args:
            returns_df: Asset returns DataFrame.
            method: "markowitz", "risk_parity", "kelly", "equal_weight"

        Returns:
            PortfolioManagerResult.
        """
        return self._allocate_markowitz(returns_df, method)

    def _allocate_ppo(
        self,
        returns_df: pd.DataFrame,
        confidence: float,
    ) -> PortfolioManagerResult:
        """Allocate using PPO model by running a single environment step."""
        # Create temporary env and set it to current state
        env = self._create_env(returns_df)

        # Use the last lookback window as observation
        obs, _ = env.reset()

        # Get action from PPO
        action, states = self._model.predict(obs, deterministic=True)

        # Execute action to get the resulting weights
        _, reward, terminated, truncated, info = env.step(action)

        ppo_weights = info.get("weights", {})
        ppo_portfolio_return = info.get("portfolio_value", 1.0) - 1.0
        ppo_sharpe = info.get("sharpe", 0.0)

        # Build weight dict matching asset_names
        weights = {}
        for i, name in enumerate(self.asset_names):
            if name in ppo_weights:
                weights[name] = ppo_weights[name]

        # Compute expected risk from returns
        if len(returns_df) >= 20:
            aligned_cols = [c for c in self.asset_names if c in returns_df.columns]
            if aligned_cols:
                recent = returns_df[aligned_cols].iloc[-20:]
                w_vec = np.array([weights.get(c, 0.0) for c in aligned_cols], dtype=np.float32)
                w_vec = w_vec / (np.sum(w_vec) + 1e-10)
                port_risk = float(np.sqrt(np.dot(w_vec.T, np.dot(recent.cov() * 252, w_vec))))
            else:
                port_risk = 0.0
        else:
            port_risk = 0.0

        # Blend PPO confidence with external confidence
        ppo_conf = min(1.0, max(0.0, (ppo_sharpe + 2.0) / 4.0))
        blended_conf = 0.7 * ppo_conf + 0.3 * confidence

        # Track PPO performance
        self._ppo_performance.append(ppo_portfolio_return)

        env.close()

        return PortfolioManagerResult(
            weights=weights,
            expected_return=round(float(ppo_portfolio_return), 4),
            expected_risk=round(port_risk, 4),
            sharpe_ratio=round(ppo_sharpe, 4),
            method="ppo",
            ppo_used=True,
            ppo_confidence=round(blended_conf, 4),
            model_version=f"v{self._train_count}",
        )

    def _allocate_markowitz(
        self,
        returns_df: pd.DataFrame,
        method: str,
    ) -> PortfolioManagerResult:
        """Fallback allocation using Markowitz/Risk Parity/Kelly.

        Args:
            returns_df: Asset returns DataFrame.
            method: Allocation method name.

        Returns:
            PortfolioManagerResult.
        """
        aligned_cols = [c for c in self.asset_names if c in returns_df.columns]
        if not aligned_cols or len(returns_df) < 10:
            # Equal weight fallback
            w = {name: 1.0 / self.n_assets for name in self.asset_names}
            return PortfolioManagerResult(
                weights=w,
                expected_return=0.0,
                expected_risk=0.0,
                sharpe_ratio=0.0,
                method="equal_weight",
            )

        df = returns_df[aligned_cols]

        if method == "risk_parity":
            result = self._optimizer.risk_parity_allocation(df, max_weight=self.max_weight)
        elif method == "kelly":
            # Kelly works on single assets, use Markowitz for multi-asset
            result = self._optimizer.markowitz_allocation(df, max_weight=self.max_weight)
        else:
            result = self._optimizer.markowitz_allocation(df, max_weight=self.max_weight)

        # Align output to all asset_names
        weights = {}
        for name in self.asset_names:
            weights[name] = result.weights.get(name, 0.0)

        # Track Markowitz performance
        if len(returns_df) > 1:
            last_return = float(np.dot(
                [weights.get(c, 0.0) for c in aligned_cols],
                df.iloc[-1].values,
            ))
            self._markowitz_performance.append(last_return)

        return PortfolioManagerResult(
            weights=weights,
            expected_return=round(result.expected_return, 4),
            expected_risk=round(result.expected_risk, 4),
            sharpe_ratio=round(result.sharpe_ratio, 4),
            method=result.method,
        )

    # ── Blended Allocation ───────────────────────────────────

    def allocate_blended(
        self,
        returns_df: pd.DataFrame,
        ppo_weight: float = 0.6,
    ) -> PortfolioManagerResult:
        """Blend PPO and Markowitz allocations.

        Args:
            returns_df: Asset returns DataFrame.
            ppo_weight: Weight given to PPO allocation (rest goes to Markowitz).

        Returns:
            PortfolioManagerResult with blended weights.
        """
        ppo_result = self.allocate(returns_df)
        mark_result = self._allocate_markowitz(returns_df, "markowitz")

        if not ppo_result.ppo_used:
            return mark_result

        blended_weights = {}
        for name in self.asset_names:
            ppo_w = ppo_result.weights.get(name, 0.0)
            mark_w = mark_result.weights.get(name, 0.0)
            blended_weights[name] = ppo_weight * ppo_w + (1 - ppo_weight) * mark_w

        # Normalize
        total = sum(blended_weights.values()) + 1e-10
        blended_weights = {k: v / total for k, v in blended_weights.items()}

        # Blend metrics
        blended_return = (
            ppo_weight * ppo_result.expected_return
            + (1 - ppo_weight) * mark_result.expected_return
        )
        blended_risk = (
            ppo_weight * ppo_result.expected_risk
            + (1 - ppo_weight) * mark_result.expected_risk
        )
        blended_sharpe = (
            blended_return / blended_risk if blended_risk > 0 else 0.0
        )

        return PortfolioManagerResult(
            weights=blended_weights,
            expected_return=round(blended_return, 4),
            expected_risk=round(blended_risk, 4),
            sharpe_ratio=round(blended_sharpe, 4),
            method="ppo_markowitz_blend",
            ppo_used=True,
            ppo_confidence=ppo_result.ppo_confidence,
            model_version=ppo_result.model_version,
            metadata={
                "ppo_weight": ppo_weight,
                "markowitz_weight": 1.0 - ppo_weight,
                "ppo_weights": ppo_result.weights,
                "markowitz_weights": mark_result.weights,
            },
        )

    # ── Utility ──────────────────────────────────────────────

    def is_trained(self) -> bool:
        """Check if PPO model is trained and loaded."""
        return self._trained and self._model is not None

    def get_status(self) -> dict:
        """Get current manager status."""
        return {
            "trained": self._trained,
            "train_count": self._train_count,
            "model_path": self._model_path,
            "n_assets": self.n_assets,
            "asset_names": self.asset_names,
            "ppo_allocations": len(self._ppo_performance),
            "markowitz_allocations": len(self._markowitz_performance),
        }

    def record_allocation_result(self, pnl: float, method: str = "ppo"):
        """Record allocation result for performance tracking.

        Args:
            pnl: Realized PnL from this allocation.
            method: Which method produced it ("ppo" or "markowitz").
        """
        self._allocation_history.append({
            "pnl": pnl,
            "method": method,
            "timestamp": time.time(),
        })

    def get_performance_summary(self) -> dict:
        """Get performance comparison between PPO and Markowitz."""
        def _stats(arr: list[float]) -> dict:
            if not arr:
                return {"mean": 0.0, "std": 0.0, "n": 0, "sharpe": 0.0}
            a = np.array(arr)
            mean = float(np.mean(a))
            std = float(np.std(a)) + 1e-10
            sharpe = mean / std * np.sqrt(252)
            return {
                "mean": round(mean, 6),
                "std": round(std, 6),
                "n": len(arr),
                "sharpe": round(sharpe, 4),
            }

        return {
            "ppo": _stats(self._ppo_performance),
            "markowitz": _stats(self._markowitz_performance),
            "total_allocations": len(self._allocation_history),
        }
