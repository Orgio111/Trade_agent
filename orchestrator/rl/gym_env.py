"""
QUANTEX Gymnasium Trading Environment — Proper gym.Env wrapper for Stable-Baselines3.

Wraps the existing TradingEnvironment into a Gymnasium v1-compatible interface:
  - observation_space: gym.spaces.Box(shape=(64,), dtype=float32)
  - action_space: gym.spaces.Discrete(6)
  - step() returns (obs, reward, terminated, truncated, info)
  - reset() returns (obs, info)

Actions:
  0 = HOLD        — do nothing
  1 = LONG        — open long position (only if flat)
  2 = SHORT       — open short position (only if flat)
  3 = CLOSE       — close existing position
  4 = SCALE_IN    — add to existing position
  5 = SCALE_OUT   — reduce existing position
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Optional

from .trading_env import TradingEnvironment


class GymTradingEnv(gym.Env):
    """
    Gymnasium v1 wrapper around TradingEnvironment.

    Masks invalid actions so the agent can't scale-in when flat
    or open a new position while one is already open.
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
    ):
        super().__init__()

        self._env = TradingEnvironment(
            data=data,
            initial_balance=initial_balance,
            leverage=leverage,
            maker_fee=maker_fee,
            taker_fee=taker_fee,
            lookback=lookback,
        )

        self.render_mode = render_mode

        # ── Spaces ──────────────────────────────────────────
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._env.observation_space_shape,),
            dtype=np.float32,
        )

        # 6 discrete actions: hold, long, short, close, scale_in, scale_out
        self.action_space = spaces.Discrete(self._env.action_space_n)

        # Action masking helper
        self._valid_actions = np.ones(self._env.action_space_n, dtype=np.int8)

    # ── Gymnasium API ────────────────────────────────────────

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset environment. Returns (obs, info)."""
        super().reset(seed=seed)
        obs = self._env.reset()
        self._update_valid_actions()
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
            penalty = -0.01
        else:
            penalty = 0.0

        obs, reward, done, info = self._env.step(action)
        reward += penalty

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
            # Can't close, scale_in, scale_out when flat
            self._valid_actions[3] = 0  # CLOSE
            self._valid_actions[4] = 0  # SCALE_IN
            self._valid_actions[5] = 0  # SCALE_OUT

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
) -> GymTradingEnv:
    """Convenience factory for creating a GymTradingEnv."""
    return GymTradingEnv(
        data=data,
        initial_balance=initial_balance,
        leverage=leverage,
        lookback=lookback,
    )
