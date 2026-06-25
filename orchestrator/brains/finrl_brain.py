"""Brain #6: FinRL PPO Position Sizer via Kelly Criterion.

Uses reinforcement learning (PPO) to determine optimal position sizing.
Falls back to Kelly Criterion from win rate + avg win/loss ratio.
Weight in Go orchestrator: 0.10.
"""

from __future__ import annotations

import json
import logging
import os
from collections import deque
from pathlib import Path

import numpy as np

from .base_brain import BaseBrain, BrainSignal

logger = logging.getLogger(__name__)

# ── .env auto-load ─────────────────────────────────────
try:
    from dotenv import load_dotenv
    _project_root = Path(__file__).resolve().parents[2]
    _env_file = _project_root / ".env"
    if _env_file.exists():
        load_dotenv(_env_file, override=False)
except ImportError:
    pass


class FinRLBrain(BaseBrain):
    """FinRL PPO + Kelly Criterion position sizing brain.

    Primary: Trained PPO agent for position size recommendation.
    Fallback: Kelly Criterion = (p * b - q) / b where p=win_rate, q=1-p, b=avg_win/avg_loss.

    This brain's score represents the SIZING conviction, not direction.
    It amplifies or dampens the directional consensus from other brains.

    Features:
      - Auto-loads PPO model from disk or auto-trains from trade history
      - Persists trade results to JSON for survival across restarts
      - Online PPO training after each trade result
    """

    @property
    def brain_id(self) -> str:
        return "finrl_kelly"

    def __init__(self) -> None:
        self._ppo_agent = None
        self._trade_history: deque[dict] = deque(maxlen=500)
        self._max_risk_pct = float(os.getenv("MAX_RISK_PCT", "0.015"))  # 1.5%
        self.risk_pct = float(os.getenv("RISK_PCT_PER_TRADE", "0.015"))
        self._model_dir = os.getenv(
            "FINRL_MODEL_DIR",
            str(Path(__file__).resolve().parents[2] / "models" / "finrl"),
        )
        self._trade_log_path = Path(self._model_dir) / "trade_history.json"
        self._trained = False

    async def warmup(self) -> None:
        """Attempt to load trained PPO agent and past trade history."""
        # Load past trade history from disk
        self._load_trade_history()

        # Try loading existing PPO model
        model_path = os.getenv("FINRL_MODEL_PATH", "")
        if model_path and os.path.exists(model_path):
            try:
                from stable_baselines3 import PPO
                self._ppo_agent = PPO.load(model_path)
                self._trained = True
                logger.info("[finrl_kelly] PPO agent loaded from %s", model_path)
                return
            except ImportError:
                logger.warning("[finrl_kelly] stable_baselines3 not installed, using Kelly only")
            except Exception as e:
                logger.warning("[finrl_kelly] Model load failed: %s", e)

        # Auto-discover from model dir
        model_dir = Path(self._model_dir)
        model_file = model_dir / "ppo_finrl.zip"
        if model_file.exists():
            try:
                from stable_baselines3 import PPO
                self._ppo_agent = PPO.load(str(model_file))
                self._trained = True
                logger.info("[finrl_kelly] Loaded cached PPO from %s", model_file)
            except Exception as e:
                logger.warning("[finrl_kelly] Cached model load failed: %s", e)

        # Auto-train if enough trade history
        if len(self._trade_history) >= 50 and not self._trained:
            self._auto_train_ppo()

    async def compute_score(self, symbol: str) -> BrainSignal:
        """Compute position sizing recommendation.

        Score interpretation:
          Positive (0..1) → supports LONG, magnitude = sizing fraction
          Negative (-1..0) → supports SHORT, magnitude = sizing fraction
          Near 0 → reduce position / no edge
        """
        kelly_frac = self._compute_kelly()
        confidence = self._kelly_confidence()

        # If PPO agent available, blend with Kelly
        if self._ppo_agent is not None:
            try:
                obs = self._build_observation(symbol)
                action, _ = self._ppo_agent.predict(obs, deterministic=True)
                ppo_size = float(np.clip(action, -1.0, 1.0))
                # Blend: 50% PPO, 50% Kelly
                combined = 0.5 * ppo_size + 0.5 * kelly_frac
                confidence = min(0.95, confidence + 0.1)
                return BrainSignal(
                    brain_id=self.brain_id,
                    symbol=symbol,
                    score=float(np.clip(combined, -1.0, 1.0)),
                    confidence=confidence,
                    metadata={"kelly_frac": round(kelly_frac, 4), "ppo_size": round(ppo_size, 4)},
                )
            except Exception as e:
                logger.warning("[finrl_kelly] PPO predict failed: %s", e)

        # Pure Kelly
        return BrainSignal(
            brain_id=self.brain_id,
            symbol=symbol,
            score=float(np.clip(kelly_frac, -1.0, 1.0)),
            confidence=confidence,
            metadata={"kelly_frac": round(kelly_frac, 4), "method": "kelly_only"},
        )

    def push_trade_result(self, pnl_pct: float) -> None:
        """Record trade P&L for Kelly calculation and persist to disk."""
        self._trade_history.append({
            "pnl_pct": pnl_pct,
            "win": pnl_pct > 0,
        })
        # Persist to disk
        self._save_trade_history()

        # Online PPO update if agent loaded
        if self._ppo_agent is not None and len(self._trade_history) % 10 == 0:
            self._online_ppo_update()

    def _load_trade_history(self) -> None:
        """Load trade history from JSON file on disk."""
        if self._trade_log_path.exists():
            try:
                with open(self._trade_log_path, "r") as f:
                    data = json.load(f)
                for entry in data[-500:]:
                    self._trade_history.append(entry)
                logger.info("[finrl_kelly] Loaded %d trade results from %s",
                            len(self._trade_history), self._trade_log_path)
            except Exception as e:
                logger.warning("[finrl_kelly] Failed to load trade history: %s", e)

    def _save_trade_history(self) -> None:
        """Persist trade history to JSON file on disk."""
        try:
            self._trade_log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._trade_log_path, "w") as f:
                json.dump(list(self._trade_history), f, indent=2)
        except Exception as e:
            logger.debug("[finrl_kelly] Trade history save failed: %s", e)

    def _auto_train_ppo(self) -> None:
        """Auto-train PPO agent from accumulated trade history."""
        try:
            from stable_baselines3 import PPO
            from stable_baselines3.common.env_util import make_vec_env
        except ImportError:
            logger.debug("[finrl_kelly] stable_baselines3 not installed, skipping auto-train")
            return

        if len(self._trade_history) < 50:
            return

        try:
            # Build simple environment from trade history
            import gymnasium as gym
            from gymnasium import spaces

            class TradeEnv(gym.Env):
                """Minimal trading environment for PPO training from historical trades."""

                def __init__(self, trade_data):
                    super().__init__()
                    self.trade_data = trade_data
                    self.idx = 0
                    self.observation_space = spaces.Box(
                        low=-1, high=1, shape=(5,), dtype=np.float32
                    )
                    self.action_space = spaces.Box(
                        low=-1, high=1, shape=(1,), dtype=np.float32
                    )

                def reset(self, seed=None, options=None):
                    super().reset(seed=seed)
                    self.idx = 0
                    return self._get_obs(), {}

                def step(self, action):
                    if self.idx >= len(self.trade_data):
                        return self._get_obs(), 0.0, True, False, {}
                    trade = self.trade_data[self.idx]
                    actual_pnl = trade["pnl_pct"]
                    # Reward: high if action direction matches trade outcome
                    action_val = float(action[0]) if hasattr(action, '__len__') else float(action)
                    reward = actual_pnl * action_val * 10  # scale up for learning
                    self.idx += 1
                    done = self.idx >= len(self.trade_data) - 1
                    return self._get_obs(), reward, done, False, {}

                def _get_obs(self):
                    wins = [t for t in self.trade_data[:self.idx + 1] if t["win"]]
                    n = max(1, self.idx + 1)
                    wr = len(wins) / n
                    avg_win = np.mean([t["pnl_pct"] for t in wins]) if wins else 0.0
                    losses = [t for t in self.trade_data[:self.idx + 1] if not t["win"]]
                    avg_loss = np.mean([abs(t["pnl_pct"]) for t in losses]) if losses else 0.01
                    kelly = self._compute_kelly_inline(n, wr, avg_win, avg_loss)
                    return np.array([kelly, wr, avg_win, avg_loss, min(1.0, n / 500)],
                                    dtype=np.float32)

                @staticmethod
                def _compute_kelly_inline(n, wr, avg_win, avg_loss):
                    if avg_loss == 0:
                        return 0.0
                    b = avg_win / avg_loss
                    q = 1.0 - wr
                    kelly = (wr * b - q) / b
                    return float(np.clip(kelly * 0.5, -1, 1))

            trade_data = list(self._trade_history)
            env = make_vec_env(lambda: TradeEnv(trade_data), n_envs=1)

            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=3e-4,
                n_steps=256,
                batch_size=64,
                n_epochs=5,
                verbose=0,
            )
            model.learn(total_timesteps=5000)

            # Save
            model_dir = Path(self._model_dir)
            model_dir.mkdir(parents=True, exist_ok=True)
            model_file = model_dir / "ppo_finrl.zip"
            model.save(str(model_file))

            self._ppo_agent = model
            self._trained = True
            logger.info("[finrl_kelly] Auto-trained PPO on %d trades, saved to %s",
                        len(trade_data), model_file)
        except Exception as e:
            logger.warning("[finrl_kelly] Auto-train failed: %s", e)

    def _online_ppo_update(self) -> None:
        """Incrementally update PPO with new trade data (every 10 trades)."""
        if self._ppo_agent is None:
            return
        try:
            self._auto_train_ppo()
        except Exception as e:
            logger.debug("[finrl_kelly] Online PPO update failed: %s", e)

    def _compute_kelly(self) -> float:
        """Compute Kelly Criterion fraction from trade history."""
        if len(self._trade_history) < 20:
            return 0.0  # insufficient data → flat

        wins = [t["pnl_pct"] for t in self._trade_history if t["win"]]
        losses = [abs(t["pnl_pct"]) for t in self._trade_history if not t["win"]]

        if not wins or not losses:
            return 0.0

        p = len(wins) / len(self._trade_history)  # win rate
        q = 1.0 - p
        avg_win = float(np.mean(wins))
        avg_loss = float(np.mean(losses))

        if avg_loss == 0:
            return 0.0

        b = avg_win / avg_loss  # reward/risk ratio
        kelly = (p * b - q) / b  # Kelly fraction

        # Apply half-Kelly for safety (standard practice)
        half_kelly = kelly * 0.5

        # Cap at max risk
        return float(np.clip(half_kelly, -self._max_risk_pct / self.risk_pct, self._max_risk_pct / self.risk_pct))

    def _kelly_confidence(self) -> float:
        """Confidence based on number of trades and win rate stability."""
        n = len(self._trade_history)
        if n < 20:
            return 0.15
        if n < 50:
            return 0.4
        if n < 100:
            return 0.6
        return 0.8

    def _build_observation(self, symbol: str) -> np.ndarray:
        """Build observation vector for PPO agent."""
        # Simple state: [kelly_frac, win_rate, avg_win, avg_loss, trade_count_normalized]
        kelly = self._compute_kelly()
        wins = [t for t in self._trade_history if t["win"]]
        win_rate = len(wins) / max(1, len(self._trade_history))
        avg_win = float(np.mean([t["pnl_pct"] for t in wins])) if wins else 0.0
        avg_loss = float(np.mean([abs(t["pnl_pct"]) for t in self._trade_history if not t["win"]])) if len(self._trade_history) > len(wins) else 0.01
        count_norm = min(1.0, len(self._trade_history) / 500)

        return np.array([kelly, win_rate, avg_win, avg_loss, count_norm], dtype=np.float32)
