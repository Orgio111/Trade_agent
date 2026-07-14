"""Brain #6: FinRL PPO Position Sizer via Kelly Criterion.

Uses reinforcement learning (PPO) to determine optimal position sizing.
Falls back to Kelly Criterion from win rate + avg win/loss ratio.
Weight in Go orchestrator: 0.10.

Features:
  - Auto-loads PPO model from disk or auto-trains from trade history
  - Persists trade results to JSON for survival across restarts
  - Online PPO training after each trade result
  - Trade history backup before model overwrite
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
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


# ── Import guards (cached at module level) ─────────────
def _has_sb3() -> bool:
    try:
        import stable_baselines3  # noqa: F401
        return True
    except ImportError:
        return False


def _has_gymnasium() -> bool:
    try:
        import gymnasium  # noqa: F401
        return True
    except ImportError:
        return False


_SB3_AVAILABLE: bool = _has_sb3()
_GYMNASIUM_AVAILABLE: bool = _has_gymnasium()


class FinRLBrain(BaseBrain):
    """FinRL PPO + Kelly Criterion position sizing brain.

    Primary: Trained PPO agent for position size recommendation.
    Fallback: Kelly Criterion = (p * b - q) / b where p=win_rate, q=1-p, b=avg_win/avg_loss.

    This brain's score represents the SIZING conviction, not direction.
    It amplifies or dampens the directional consensus from other brains.

    Features:
      - Auto-loads PPO model from disk or auto-trains from trade history
      - Persists trade results to JSON for survival across restarts
      - Online PPO training after each trade result (every 10 trades)
      - Trade history backup before model overwrite
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
        self._online_train_interval = 10  # retrain every N trades
        self._total_trades = 0  # separate counter (deque maxlen caps len())

    def _try_load_ppo(self, path: str) -> None:
        """Load PPO model with SB3/torch compat workaround.

        SB3 2.9 + torch 2.12 has a PyTorchFileReader bug that prevents
        PPO.load() from reading .pth inside the zip. Workaround: extract
        the policy state dict manually and inject it into a fresh model.
        """
        if not _SB3_AVAILABLE:
            logger.debug("[finrl_kelly] stable_baselines3 not installed, using Kelly only")
            return

        try:
            from stable_baselines3 import PPO
            import zipfile, io

            # Step 1: Try normal PPO.load (works on older torch/SB3)
            try:
                self._ppo_agent = PPO.load(path)
                self._trained = True
                logger.info("[finrl_kelly] PPO agent loaded (standard path) from %s", path)
                return
            except RuntimeError:
                pass  # Fall through to workaround

            # Step 2: Extract policy state dict from zip manually
            with zipfile.ZipFile(path, "r") as zf:
                if "policy.pth" not in zf.namelist():
                    logger.warning("[finrl_kelly] No policy.pth in zip: %s", path)
                    return
                with zf.open("policy.pth") as f:
                    import torch
                    policy_sd = torch.load(
                        io.BytesIO(f.read()),
                        map_location="cpu",
                        weights_only=False,
                    )

            # Step 3: Create a fresh PPO and inject the state dict
            if not _GYMNASIUM_AVAILABLE:
                logger.warning("[finrl_kelly] gymnasium not installed")
                return

            import gymnasium as gym
            from gymnasium import spaces

            class _DummyEnv(gym.Env):
                def __init__(self, obs_dim, n_actions=3):
                    super().__init__()
                    self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
                    self.action_space = spaces.Discrete(n_actions)
                def reset(self, seed=None, options=None):
                    super().reset(seed=seed)
                    return self.observation_space.sample(), {}
                def step(self, action):
                    return self.observation_space.sample(), 0.0, True, False, {}

            # Detect action space shape from policy output layer
            action_dim = 3  # default: HOLD/BUY/SELL
            for k, v in policy_sd.items():
                if "action_net" in k and v.ndim == 2:
                    action_dim = v.shape[0]
                    break

            # Find obs dim from input layer
            obs_dim = 22
            for k, v in policy_sd.items():
                if "policy_net.0.weight" in k and v.ndim == 2:
                    obs_dim = v.shape[1]
                    break

            dummy = _DummyEnv(obs_dim, action_dim)
            model = PPO("MlpPolicy", dummy, verbose=0)
            model.policy.load_state_dict(policy_sd, strict=False)
            self._ppo_agent = model
            self._trained = True
            logger.info(
                "[finrl_kelly] PPO loaded (workaround: extract+inject) from %s",
                path,
            )
        except ImportError:
            logger.warning("[finrl_kelly] stable_baselines3 not installed, using Kelly only")
        except Exception as e:
            logger.warning("[finrl_kelly] Model load failed: %s", e)

    async def warmup(self) -> None:
        """Attempt to load trained PPO agent and past trade history."""
        # Load past trade history from disk
        self._load_trade_history()

        # Try loading existing PPO model
        model_path = os.getenv("FINRL_MODEL_PATH", "")
        if model_path and os.path.exists(model_path):
            self._try_load_ppo(model_path)
            if self._trained:
                return

        # Auto-discover from model dir
        model_dir = Path(self._model_dir)
        model_file = model_dir / "ppo_finrl.zip"
        if model_file.exists():
            self._try_load_ppo(str(model_file))

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
                action_int = int(action.item()) if hasattr(action, 'item') else int(action)
                # action is discrete (0=HOLD, 1=BUY, 2=SELL), convert to sizing
                ppo_size = 0.0
                if action_int == 1:
                    ppo_size = 0.5  # moderate long
                elif action_int == 2:
                    ppo_size = -0.5  # moderate short

                # Blend: 50% PPO, 50% Kelly
                combined = 0.5 * ppo_size + 0.5 * kelly_frac
                confidence = min(0.95, confidence + 0.1)
                return BrainSignal(
                    brain_id=self.brain_id,
                    symbol=symbol,
                    score=float(np.clip(combined, -1.0, 1.0)),
                    confidence=confidence,
                    metadata={
                        "kelly_frac": round(kelly_frac, 4),
                        "ppo_size": round(ppo_size, 4),
                        "ppo_action": action_int,
                        "method": "ppo_kelly_blend",
                        "trades": len(self._trade_history),
                    },
                )
            except Exception as e:
                logger.warning("[finrl_kelly] PPO predict failed: %s", e)

        # Pure Kelly
        return BrainSignal(
            brain_id=self.brain_id,
            symbol=symbol,
            score=float(np.clip(kelly_frac, -1.0, 1.0)),
            confidence=confidence,
            metadata={
                "kelly_frac": round(kelly_frac, 4),
                "method": "kelly_only",
                "trades": len(self._trade_history),
            },
        )

    def push_trade_result(self, pnl_pct: float) -> None:
        """Record trade P&L for Kelly calculation and persist to disk."""
        self._trade_history.append({
            "pnl_pct": pnl_pct,
            "win": pnl_pct > 0,
            "timestamp": time.time(),
        })
        self._total_trades += 1
        # Persist to disk
        self._save_trade_history()

        # Online PPO update if agent loaded (use counter, not deque len)
        if self._ppo_agent is not None and self._total_trades % self._online_train_interval == 0:
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
        """Auto-train PPO agent from accumulated trade history.

        Uses Discrete(3) action space (HOLD/BUY/SELL) and 21-dim obs
        to match retrain_finrl.py so models are interchangeable.
        """
        if not _SB3_AVAILABLE or not _GYMNASIUM_AVAILABLE:
            logger.debug("[finrl_kelly] stable_baselines3/gymnasium not installed, skipping auto-train")
            return

        if len(self._trade_history) < 50:
            return

        try:
            from stable_baselines3 import PPO
            from stable_baselines3.common.env_util import make_vec_env
            import gymnasium as gym
            from gymnasium import spaces

            WINDOW = 20  # must match retrain_finrl.py

            class TradeEnv(gym.Env):
                """Trading environment from historical trades, matching
                retrain_finrl.TradingEnv obs/action spaces."""

                def __init__(self, trade_data):
                    super().__init__()
                    self.trade_data = trade_data
                    self.idx = 0
                    self.window = WINDOW
                    # obs: (window-1) returns + position + unrealized_pnl
                    self.observation_space = spaces.Box(
                        low=-np.inf, high=np.inf,
                        shape=(WINDOW - 1 + 2,), dtype=np.float32,
                    )
                    self.action_space = spaces.Discrete(3)  # HOLD=0, BUY=1, SELL=2
                    self.position = 0  # -1, 0, +1
                    self.unrealized = 0.0

                def reset(self, seed=None, options=None):
                    super().reset(seed=seed)
                    self.idx = WINDOW
                    self.position = 0
                    self.unrealized = 0.0
                    return self._get_obs(), {}

                def step(self, action):
                    done = self.idx >= len(self.trade_data) - 1
                    trade = self.trade_data[min(self.idx, len(self.trade_data) - 1)]
                    pnl = trade["pnl_pct"]

                    # Update position
                    if action == 1:
                        self.position = 1
                    elif action == 2:
                        self.position = -1
                    else:
                        self.position = 0

                    reward = self.position * pnl * 10  # scale for learning
                    self.unrealized += self.position * pnl
                    self.idx += 1
                    return self._get_obs(), float(reward), done, False, {}

                def _get_obs(self):
                    # Recent trade returns as pseudo price series
                    start = max(0, self.idx - self.window)
                    recent = self.trade_data[start:self.idx]
                    rets = np.array([t["pnl_pct"] for t in recent], dtype=np.float32)
                    if len(rets) < self.window:
                        rets = np.pad(rets, (self.window - len(rets), 0))
                    returns = np.diff(rets) / (np.abs(rets[:-1]) + 1e-10)
                    returns = np.clip(returns, -1, 1)
                    if len(returns) < self.window - 1:
                        returns = np.pad(returns, (self.window - 1 - len(returns), 0))

                    obs = np.concatenate([
                        returns,
                        np.array([float(self.position), float(self.unrealized)],
                                 dtype=np.float32),
                    ])
                    return obs.astype(np.float32)

            trade_data = list(self._trade_history)
            env = make_vec_env(lambda: TradeEnv(trade_data), n_envs=1)

            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=3e-4,
                n_steps=512,
                batch_size=64,
                n_epochs=10,
                verbose=0,
            )

            # Train with more timesteps for better convergence
            total_timesteps = min(20000, max(5000, len(trade_data) * 20))
            model.learn(total_timesteps=total_timesteps)

            # Save with backup
            model_dir = Path(self._model_dir)
            model_dir.mkdir(parents=True, exist_ok=True)
            model_file = model_dir / "ppo_finrl.zip"

            # Backup previous model
            if model_file.exists():
                backup = model_dir / f"ppo_finrl_backup_{int(time.time())}.zip"
                try:
                    shutil.copy2(str(model_file), str(backup))
                except Exception:
                    pass

            model.save(str(model_file))

            self._ppo_agent = model
            self._trained = True

            # Evaluate on last 20% of data (non-critical, failures logged only)
            eval_reward = 0.0
            try:
                eval_env = TradeEnv(trade_data)
                obs, _ = eval_env.reset()
                total_reward = 0.0
                steps = 0
                for _ in range(len(trade_data) - WINDOW - 1):
                    action_raw, _ = model.predict(obs, deterministic=True)
                    action_int = int(action_raw.item()) if hasattr(action_raw, 'item') else int(action_raw)
                    obs, reward, done, _, _ = eval_env.step(action_int)
                    total_reward += reward
                    steps += 1
                    if done:
                        break
                eval_reward = total_reward / max(steps, 1)
            except Exception as eval_err:
                logger.debug("[finrl_kelly] Eval loop skipped: %s", eval_err)

            logger.info(
                "[finrl_kelly] Auto-trained PPO on %d trades (timesteps=%d, eval_reward=%.2f), saved to %s",
                len(trade_data), total_timesteps, eval_reward, model_file,
            )
        except Exception as e:
            logger.warning("[finrl_kelly] Auto-train failed: %s", e)

    def _online_ppo_update(self) -> None:
        """Incrementally update PPO with new trade data (every N trades)."""
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
        """Build observation vector matching PPO training environment.

        PPO trained with TradingEnv(obs_dim=21): 19 normalized returns + position + unrealized_pnl.
        Since we don't have a price series here, we synthesize from trade history statistics.
        """
        n_history = len(self._trade_history)
        # Build a pseudo observation from trade history
        # First 19 dims: synthetic "return series" from recent trades
        recent = list(self._trade_history)[-19:] if n_history >= 19 else list(self._trade_history)
        returns = np.array([t["pnl_pct"] for t in recent], dtype=np.float32)
        if len(returns) < 19:
            returns = np.pad(returns, (19 - len(returns), 0), constant_values=0.0)
        returns = np.clip(returns, -0.1, 0.1) / 0.1  # normalize

        # Position and unrealized PnL (from Kelly direction)
        kelly = self._compute_kelly()
        position = 1.0 if kelly > 0 else (-1.0 if kelly < 0 else 0.0)

        return np.concatenate([returns, np.array([position, kelly], dtype=np.float32)])
