"""
QUANTEX Portfolio Allocation Environment — PPO-based capital allocation.

Replaces Kelly Criterion / Markowitz rule-based allocation with a
learned PPO policy that outputs optimal capital allocation weights.

Architecture:
  ┌─────────────────────────────────────────────┐
  │  State (~30+N*6 dims)                       │
  │  [market_features, portfolio_state,         │
  │   asset_features × N,                       │
  │   current_weights]                          │
  └──────────────────┬──────────────────────────┘
                     ▼
  ┌─────────────────────────────────────────────┐
  │  PPO Policy (MlpPolicy)                     │
  │  → Continuous action: allocation weights    │
  └──────────────────┬──────────────────────────┘
                     ▼
  ┌─────────────────────────────────────────────┐
  │  Reward = port_return - risk_penalty        │
  │          - turnover_penalty - dd_penalty    │
  └─────────────────────────────────────────────┘

State features per timestep:
  - Market: N_assets × 6 (returns_1, returns_5, volatility, momentum,
    volume_ratio, correlation_tier)
  - Portfolio: drawdown, current_sharpe, win_rate, consec_losses,
    total_pnl_pct, n_trades, current_exposure
  - Current weights: N_assets allocation vector

Action (continuous Box):
  - N_asset weights (softmax-normalized to sum ≈ 1.0)

Usage:
    env = PortfolioAllocEnv(n_assets=4, returns_df=df)
    obs, _ = env.reset()
    action = model.predict(obs)  # allocation weights
    obs, reward, done, _, info = env.step(action)
"""

from __future__ import annotations

from typing import Optional
from dataclasses import dataclass

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd


@dataclass
class AllocationStep:
    """Result of one allocation step."""
    weights: dict[str, float]       # symbol → weight
    portfolio_return: float
    portfolio_risk: float
    sharpe: float
    drawdown: float
    turnover: float
    reward: float


class PortfolioAllocEnv(gym.Env):
    """
    Gymnasium environment for PPO-based portfolio capital allocation.

    Learns to allocate capital across N assets to maximize risk-adjusted
    returns, with penalties for drawdown, turnover, and concentration risk.

    Parameters
    ----------
    returns_df:
        DataFrame with columns = asset symbols, rows = periodic returns.
    asset_names:
        List of asset symbol names. If None, uses returns_df.columns.
    lookback:
        Rolling window for feature computation.
    risk_free_rate:
        Annual risk-free rate for Sharpe computation.
    turnover_penalty:
        Penalty multiplier for weight changes (0 = no penalty).
    concentration_penalty:
        Penalty for having > max_weight in any single asset.
    max_weight:
        Maximum allocation to any single asset.
    min_weight:
        Minimum allocation (0 for long-only).
    reward_scale:
        Scale factor for rewards (helps PPO training stability).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        returns_df: pd.DataFrame,
        asset_names: Optional[list[str]] = None,
        lookback: int = 20,
        risk_free_rate: float = 0.02,
        turnover_penalty: float = 0.5,
        concentration_penalty: float = 0.3,
        max_weight: float = 0.5,
        min_weight: float = 0.0,
        reward_scale: float = 1.0,
    ):
        super().__init__()

        self.returns_df = returns_df.copy()
        self.asset_names = list(asset_names) if asset_names else list(returns_df.columns)
        self.n_assets = len(self.asset_names)
        self.lookback = lookback
        self.risk_free_rate = risk_free_rate
        self.turnover_penalty = turnover_penalty
        self.concentration_penalty = concentration_penalty
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.reward_scale = reward_scale

        # State dimensions per asset
        self._n_asset_features = 6  # ret_1, ret_5, volatility, momentum, vol_ratio, corr_tier
        # Portfolio-level features
        self._n_portfolio_features = 7  # dd, sharpe, wr, consec_losses, pnl_pct, trades, exposure

        # ── Observation space ──────────────────────────
        # [asset_features × N_assets] + [portfolio_features] + [current_weights]
        obs_dim = (
            self.n_assets * self._n_asset_features
            + self._n_portfolio_features
            + self.n_assets
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # ── Action space (continuous weights) ──────────
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.n_assets,),
            dtype=np.float32,
        )

        # ── Internal state ─────────────────────────────
        self._step = 0
        self._max_steps = len(returns_df) - lookback - 1
        self._current_weights = np.ones(self.n_assets, dtype=np.float32) / self.n_assets
        self._prev_weights = self._current_weights.copy()

        # Performance tracking
        self._portfolio_value = 1.0  # normalized start
        self._peak_value = 1.0
        self._returns_history: list[float] = []
        self._weight_history: list[np.ndarray] = [self._current_weights.copy()]
        self._consecutive_losses = 0
        self._total_trades = 0

        # Cache features
        self._cached_features: dict[int, np.ndarray] = {}

    # ── Feature Computation ──────────────────────────────────

    def _compute_asset_features(self, step: int) -> np.ndarray:
        """Compute asset-level features for all assets at given step.

        Returns (n_assets, n_asset_features) array.
        """
        if step in self._cached_features:
            return self._cached_features[step]

        features = np.zeros((self.n_assets, self._n_asset_features), dtype=np.float32)
        start = max(0, step - self.lookback)
        window = self.returns_df.iloc[start:step + 1]

        if len(window) < 5:
            self._cached_features[step] = features
            return features

        for i, asset in enumerate(self.asset_names):
            if asset not in window.columns:
                continue
            series = window[asset].values
            series = series[np.isfinite(series)]
            if len(series) < 5:
                continue

            # 0: 1-period return
            features[i, 0] = float(np.clip(series[-1], -0.1, 0.1) / 0.1)

            # 1: 5-period return
            if len(series) >= 5:
                features[i, 1] = float(np.clip(
                    (series[-1] - series[-5]) / (abs(series[-5]) + 1e-10),
                    -0.2, 0.2
                ) / 0.2)

            # 2: Volatility (std of returns)
            vol = float(np.std(series)) * np.sqrt(252)  # annualized
            features[i, 2] = float(np.clip(vol / 0.5, 0.0, 1.0))  # normalize to [0,1]

            # 3: Momentum (slope over window)
            if len(series) >= 10:
                x = np.arange(len(series), dtype=np.float32)
                slope = np.polyfit(x, series, 1)[0]
                features[i, 3] = float(np.clip(slope * 100, -1.0, 1.0))
            else:
                features[i, 3] = 0.0

            # 4: Volume-like: recent volatility ratio (quick vs slow)
            if len(series) >= 20:
                fast_vol = float(np.std(series[-5:]))
                slow_vol = float(np.std(series[-20:])) + 1e-10
                features[i, 4] = float(np.clip(fast_vol / slow_vol - 1.0, -1.0, 1.0))
            else:
                features[i, 4] = 0.0

            # 5: Correlation tier: average correlation with other assets
            if len(window.columns) > 1 and len(series) >= 10:
                corrs = []
                for j, other in enumerate(self.asset_names):
                    if j == i or other not in window.columns:
                        continue
                    other_series = window[other].values
                    other_series = other_series[np.isfinite(other_series)]
                    if len(other_series) >= 10:
                        min_len = min(len(series), len(other_series))
                        c = np.corrcoef(series[-min_len:], other_series[-min_len:])[0, 1]
                        if not np.isnan(c):
                            corrs.append(abs(c))
                avg_corr = float(np.mean(corrs)) if corrs else 0.5
                features[i, 5] = avg_corr
            else:
                features[i, 5] = 0.5

        self._cached_features[step] = features
        return features

    def _compute_portfolio_features(self) -> np.ndarray:
        """Compute portfolio-level features.

        Returns (n_portfolio_features,) array.
        """
        features = np.zeros(self._n_portfolio_features, dtype=np.float32)

        # 0: Drawdown
        dd = (self._peak_value - self._portfolio_value) / max(self._peak_value, 1e-10)
        features[0] = float(np.clip(dd, 0.0, 1.0))

        # 1: Rolling Sharpe (annualized)
        if len(self._returns_history) >= 10:
            recent = np.array(self._returns_history[-20:])
            mean_ret = float(np.mean(recent))
            std_ret = float(np.std(recent)) + 1e-10
            sharpe = (mean_ret - self.risk_free_rate / 252) / std_ret * np.sqrt(252)
            features[1] = float(np.clip(sharpe / 5.0, -1.0, 1.0))  # normalize
        else:
            features[1] = 0.0

        # 2: Win rate (rolling 20)
        if len(self._returns_history) >= 5:
            recent = self._returns_history[-20:] if len(self._returns_history) >= 20 else self._returns_history
            wins = sum(1 for r in recent if r > 0)
            features[2] = wins / max(len(recent), 1)
        else:
            features[2] = 0.5

        # 3: Consecutive losses
        features[3] = float(min(self._consecutive_losses / 10.0, 1.0))

        # 4: Total PnL %
        features[4] = float(np.clip(self._portfolio_value - 1.0, -0.5, 0.5) / 0.5)

        # 5: Number of rebalances (normalized)
        features[5] = float(min(self._total_trades / 100.0, 1.0))

        # 6: Current total exposure (sum of weights)
        features[6] = float(min(np.sum(self._current_weights), 1.0))

        return features

    def _build_observation(self) -> np.ndarray:
        """Build the full observation vector."""
        asset_features = self._compute_asset_features(self._step)
        portfolio_features = self._compute_portfolio_features()

        obs = np.concatenate([
            asset_features.flatten(),       # n_assets × n_asset_features
            portfolio_features,             # n_portfolio_features
            self._current_weights,          # n_assets
        ]).astype(np.float32)

        return obs

    # ── Reward Computation ────────────────────────────────────

    def _compute_reward(self, port_return: float) -> float:
        """Compute reward = port_return - penalties."""
        reward = port_return

        # Turnover penalty (changes weights too much = costly)
        turnover = float(np.sum(np.abs(self._current_weights - self._prev_weights)))
        reward -= self.turnover_penalty * turnover

        # Concentration penalty (any single asset > max_weight)
        max_w = float(np.max(self._current_weights))
        if max_w > self.max_weight:
            reward -= self.concentration_penalty * (max_w - self.max_weight) * 2.0

        # Drawdown penalty
        dd = self._peak_value - self._portfolio_value
        if dd > 0.05:  # more than 5% drawdown
            reward -= dd * 2.0

        # Consecutive losses penalty
        if self._consecutive_losses >= 3:
            reward -= self._consecutive_losses * 0.1

        return reward * self.reward_scale

    # ── Gymnasium API ────────────────────────────────────────

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset environment. Returns (obs, info)."""
        super().reset(seed=seed)

        self._step = self.lookback
        self._current_weights = np.ones(self.n_assets, dtype=np.float32) / self.n_assets
        self._prev_weights = self._current_weights.copy()
        self._portfolio_value = 1.0
        self._peak_value = 1.0
        self._returns_history = []
        self._weight_history = [self._current_weights.copy()]
        self._consecutive_losses = 0
        self._total_trades = 0
        self._cached_features = {}

        obs = self._build_observation()
        info = self._get_info()
        return obs, info

    def step(self, action: np.ndarray):
        """
        Execute allocation action.

        Args:
            action: Raw allocation weights (n_assets,). Will be normalized
                   to sum to 1.0 via softmax or clipping.

        Returns:
            (obs, reward, terminated, truncated, info)
        """
        # ── Process action into valid weights ──────────
        # Use softmax to ensure sum ≈ 1.0 and all positive
        action = np.asarray(action, dtype=np.float32).flatten()
        if len(action) != self.n_assets:
            action = np.ones(self.n_assets, dtype=np.float32) / self.n_assets

        exp_a = np.exp(action - np.max(action))
        weights = exp_a / (np.sum(exp_a) + 1e-10)

        # Apply max/min weight constraints
        weights = np.clip(weights, self.min_weight, self.max_weight)
        weights = weights / (np.sum(weights) + 1e-10)

        self._prev_weights = self._current_weights.copy()
        self._current_weights = weights
        self._weight_history.append(weights.copy())
        self._total_trades += 1

        # ── Compute portfolio return ──────────────────
        step_returns = self.returns_df.iloc[self._step].values
        if len(step_returns) != self.n_assets:
            # Alignment issue — zero return
            port_return = 0.0
        else:
            port_return = float(np.dot(weights, step_returns))

        self._portfolio_value *= (1.0 + port_return)
        self._peak_value = max(self._peak_value, self._portfolio_value)
        self._returns_history.append(port_return)

        if port_return < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0

        # ── Compute reward ────────────────────────────
        reward = self._compute_reward(port_return)

        # ── Step / done ───────────────────────────────
        self._step += 1
        terminated = False
        truncated = self._step >= self._max_steps

        # Bankruptcy termination
        if self._portfolio_value < 0.3:
            terminated = True
            reward -= 5.0  # bankruptcy penalty

        obs = self._build_observation()
        info = self._get_info()

        return obs, float(reward), terminated, truncated, info

    def render(self):
        """Print current allocation state."""
        print(f"Step {self._step}: Val=${self._portfolio_value:.4f} "
              f"DD={((self._peak_value - self._portfolio_value) / max(self._peak_value, 1e-10)):.2%} "
              f"Weights={np.round(self._current_weights, 3)}")

    def close(self):
        """Cleanup."""
        self._cached_features = {}
        self._returns_history.clear()
        self._weight_history.clear()

    # ── Helpers ──────────────────────────────────────────────

    def _get_info(self) -> dict:
        """Return current info dict."""
        dd = (self._peak_value - self._portfolio_value) / max(self._peak_value, 1e-10)
        recent_rets = np.array(self._returns_history[-20:]) if self._returns_history else np.array([0.0])
        sharpe = float(np.mean(recent_rets) / (np.std(recent_rets) + 1e-10) * np.sqrt(252)) if len(recent_rets) > 0 else 0.0

        return {
            "portfolio_value": round(self._portfolio_value, 4),
            "drawdown": round(dd, 4),
            "sharpe": round(sharpe, 4),
            "weights": {self.asset_names[i]: round(float(self._current_weights[i]), 4)
                       for i in range(self.n_assets)},
            "rebalances": self._total_trades,
            "consecutive_losses": self._consecutive_losses,
            "step": self._step,
        }

    def get_allocation_history(self) -> pd.DataFrame:
        """Get DataFrame of allocation weights over time."""
        if not self._weight_history:
            return pd.DataFrame()
        n = min(len(self._weight_history), len(self._returns_history) + 1)
        return pd.DataFrame(
            self._weight_history[:n],
            columns=self.asset_names,
        )

    @property
    def portfolio_value(self) -> float:
        return self._portfolio_value

    @property
    def current_weights(self) -> dict[str, float]:
        return {self.asset_names[i]: round(float(self._current_weights[i]), 4)
                for i in range(self.n_assets)}


# ── Factory function ────────────────────────────────────────

def make_portfolio_env(
    returns_df: pd.DataFrame,
    asset_names: Optional[list[str]] = None,
    lookback: int = 20,
    max_weight: float = 0.5,
) -> PortfolioAllocEnv:
    """
    Convenience factory for creating a PortfolioAllocEnv.

    Args:
        returns_df: DataFrame of asset returns (columns = symbols)
        asset_names: Subset of columns to use (default: all)
        lookback: Feature computation window
        max_weight: Max allocation per asset

    Returns:
        PortfolioAllocEnv instance
    """
    return PortfolioAllocEnv(
        returns_df=returns_df,
        asset_names=asset_names,
        lookback=lookback,
        max_weight=max_weight,
    )
