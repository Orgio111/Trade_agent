"""
QUANTEX Gymnasium Trading Environment — Proper gym.Env wrapper for Stable-Baselines3.

Wraps the existing TradingEnvironment into a Gymnasium v1-compatible interface:
  - observation_space: gym.spaces.Box(shape=(128,), dtype=float32)
  - action_space: gym.spaces.Discrete(4)
  - step() returns (obs, reward, terminated, truncated, info)
  - reset() returns (obs, info)

Actions:
  0 = HOLD        — do nothing
  1 = LONG        — open long position (only if flat)
  2 = SHORT       — open short position (only if flat)
  3 = CLOSE       — close existing position

Optional features (injected into observation):
  - obs[38:40]: ML signal (direction + confidence) from Random Forest
  - obs[64:96]: Multi-Timeframe features (1H, 4H, 1D context)
  - obs[96:128]: Deep Market Encoder (GPU LSTM+Transformer embedding)
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, TYPE_CHECKING

from .trading_env import TradingEnvironment

if TYPE_CHECKING:
    from ..multi_tf import MultiTimeframeEngine
    from ..deep_encoder import DeepMarketEncoder


class GymTradingEnv(gym.Env):
    """
    Gymnasium v1 wrapper around TradingEnvironment.

    Masks invalid actions so the agent can't scale-in when flat
    or open a new position while one is already open.
    Supports optional ML-in-obs, Multi-Timeframe, and Deep Encoder features.
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
        self,
        data: pd.DataFrame,
        initial_balance: float = 10.0,
        leverage: int = 3,
        maker_fee: float = 0.0002,
        taker_fee: float = 0.0004,
        lookback: int = 50,
        render_mode: Optional[str] = None,
        ml_engine: Optional[object] = None,  # MLSignalEngine for observation feature
        mtf_engine: Optional["MultiTimeframeEngine"] = None,  # Multi-Timeframe features
        deep_encoder: Optional["DeepMarketEncoder"] = None,   # GPU LSTM+Transformer encoder
    ):
        super().__init__()

        # Validate ml_engine at init time, not silently at runtime
        if ml_engine is not None and not hasattr(ml_engine, "predict"):
            raise TypeError("ml_engine must have a predict(df) method")

        self._env = TradingEnvironment(
            data=data,
            initial_balance=initial_balance,
            leverage=leverage,
            maker_fee=maker_fee,
            taker_fee=taker_fee,
            lookback=lookback,
        )

        self.render_mode = render_mode
        self.ml_engine = ml_engine

        # Attach MTF engine to underlying env
        if mtf_engine is not None:
            self._env.set_mtf_engine(mtf_engine)

        # Attach Deep Market Encoder (GPU) to underlying env
        if deep_encoder is not None:
            self._env.set_deep_encoder(deep_encoder)

        # ── Spaces ──────────────────────────────────────────
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._env.observation_space_shape,),
            dtype=np.float32,
        )

        # 4 discrete actions: hold, long, short, close
        self.action_space = spaces.Discrete(self._env.action_space_n)

        # Action masking helper
        self._valid_actions = np.ones(self._env.action_space_n, dtype=np.int8)

    # ── ML Signal Injection ────────────────────────────────

    def _inject_ml_signal(self):
        """
        Compute ML signal for current step and inject into observation[38:40].

        Uses the ML engine to predict direction + confidence from current
        data window, then calls set_ml_signal() on the underlying env.
        The observation is NOT rebuilt here — caller patches obs[38:40] directly.
        """
        if self.ml_engine is None:
            self._env.set_ml_signal(0.0, 0.0)
            return

        step = self._env.current_step
        if step >= len(self._env.data):
            self._env.set_ml_signal(0.0, 0.0)
            return

        # Build current data window for ML prediction
        current_df = self._env.data.iloc[:min(step + 1, len(self._env.data))]
        if len(current_df) < 50:
            self._env.set_ml_signal(0.0, 0.0)
            return

        ml_result = self.ml_engine.predict(current_df)
        direction_str = ml_result.get("direction", "hold")
        confidence = float(ml_result.get("confidence", 0.0))

        # Map direction to numeric: 1=long, -1=short, 0=hold
        if direction_str == "long":
            direction_num = 1.0
        elif direction_str == "short":
            direction_num = -1.0
        else:
            direction_num = 0.0

        self._env.set_ml_signal(direction_num, confidence)

    # ── Gymnasium API ────────────────────────────────────────

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset environment. Returns (obs, info)."""
        super().reset(seed=seed)
        obs = self._env.reset()
        self._update_valid_actions()
        self._inject_ml_signal()
        # Patch ML slots directly instead of rebuilding full observation
        obs[38] = self._env._ml_signal[0]
        obs[39] = self._env._ml_signal[1]
        info = self._get_info()
        return obs.astype(np.float32), info

    def step(self, action: int):
        """
        Execute action. Returns (obs, reward, terminated, truncated, info).

        Masks invalid actions (e.g. LONG when already in position) by
        treating them as HOLD with a small penalty.
        """
        # Clamp invalid actions to HOLD with a small penalty
        if self._valid_actions[action] == 0:
            action = 0  # HOLD
            penalty = -0.005
        else:
            penalty = 0.0

        obs, reward, done, info = self._env.step(action)
        reward += penalty

        # Inject ML signal and patch obs[38:40] directly (avoids full rebuild)
        self._inject_ml_signal()
        obs[38] = self._env._ml_signal[0]
        obs[39] = self._env._ml_signal[1]

        # Gymnasium v1: separate terminated (absorbing state) from truncated (time limit)
        terminated = done and self._env.current_step < len(self._env.data) - 1
        truncated = done and self._env.current_step >= len(self._env.data) - 1

        self._update_valid_actions()

        return obs.astype(np.float32), float(reward), terminated, truncated, info

    def render(self):
        """Print current state to console."""
        if self.render_mode == "human":
            self._env.render()

    def close(self):
        """Cleanup (no-op for this env)."""
        pass

    # ── Helpers ────────────────────────────────────────────

    def _update_valid_actions(self):
        """Update action mask based on current position state."""
        self._valid_actions = np.ones(self._env.action_space_n, dtype=np.int8)
        pos = self._env.position

        if pos != 0:
            # Can't open new position while in one
            self._valid_actions[1] = 0  # LONG
            self._valid_actions[2] = 0  # SHORT
        else:
            # Can't close when flat
            self._valid_actions[3] = 0  # CLOSE

    def _get_info(self) -> dict:
        """Return current info dict."""
        return {
            "balance": round(self._env.balance, 4),
            "position": self._env.position,
            "drawdown": round(
                (self._env.peak_balance - self._env.balance)
                / max(self._env.peak_balance, 1e-10),
                4,
            ),
            "trades": len(self._env.trades),
            "step": self._env.current_step,
            "valid_actions": self._valid_actions.copy(),
        }

    # ── Data access ─────────────────────────────────────────

    @property
    def trades(self) -> list[dict]:
        return self._env.trades

    @property
    def total_steps(self) -> int:
        return self._env._total_steps

    @property
    def balance(self) -> float:
        return self._env.balance

    @property
    def peak_balance(self) -> float:
        return self._env.peak_balance


# ── Factory function ────────────────────────────────────────

def make_gym_env(
    data: pd.DataFrame,
    initial_balance: float = 10.0,
    leverage: int = 3,
    lookback: int = 50,
    ml_engine: Optional[object] = None,
    mtf_engine: Optional["MultiTimeframeEngine"] = None,
    deep_encoder: Optional["DeepMarketEncoder"] = None,
) -> GymTradingEnv:
    """Convenience factory for creating a GymTradingEnv with optional ML, MTF, and Deep Encoder."""
    return GymTradingEnv(
        data=data,
        initial_balance=initial_balance,
        leverage=leverage,
        lookback=lookback,
        ml_engine=ml_engine,
        mtf_engine=mtf_engine,
        deep_encoder=deep_encoder,
    )
