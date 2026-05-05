"""
Ray Train/Tune retraining pipeline for the PPO execution agent.
Triggered automatically when drift detector fires PSI > threshold.
"""
from __future__ import annotations

import logging
import os
import tempfile
from typing import Any

import numpy as np

from core.config import get_settings
from core.observability import RETRAIN_COUNTER

logger = logging.getLogger(__name__)


def _build_env(price_data: np.ndarray, cfg: Any):
    """Build a gymnasium env for PPO execution training."""
    try:
        import gymnasium as gym  # type: ignore[import]
        from gymnasium import spaces

        class ExecutionEnv(gym.Env):
            metadata = {"render_modes": []}

            def __init__(self) -> None:
                super().__init__()
                self.observation_space = spaces.Box(
                    low=-np.inf, high=np.inf, shape=(40,), dtype=np.float32
                )
                self.action_space = spaces.Discrete(4)
                self._prices = price_data
                self._idx = 0
                self._entry_price = 0.0

            def reset(self, *, seed=None, options=None):
                super().reset(seed=seed)
                self._idx = np.random.randint(30, len(self._prices) - 10)
                self._entry_price = self._prices[self._idx]
                return self._obs(), {}

            def _obs(self) -> np.ndarray:
                window = self._prices[max(0, self._idx - 29) : self._idx + 1]
                if len(window) < 30:
                    window = np.pad(window, (30 - len(window), 0), mode="edge")
                returns = np.diff(window) / (window[:-1] + 1e-10)
                obs = np.zeros(40, dtype=np.float32)
                obs[:29] = returns.astype(np.float32)
                obs[29] = (self._prices[self._idx] - self._entry_price) / (self._entry_price + 1e-10)
                return obs

            def step(self, action: int):
                price = self._prices[min(self._idx, len(self._prices) - 1)]
                # Simulate slippage based on algo choice
                slippage = [0.001, 0.0005, 0.0003, 0.0002][action]
                reward = -slippage * 10_000  # negative slippage = reward
                self._idx += 1
                done = self._idx >= len(self._prices) - 1
                return self._obs(), reward, done, False, {}

        return ExecutionEnv()
    except ImportError as e:
        raise RuntimeError(f"gymnasium not installed: {e}") from e


async def retrain_ppo(price_data: np.ndarray, total_timesteps: int = 500_000) -> str:
    """
    Retrains the PPO execution agent using stable-baselines3 + Ray Tune.
    Returns path to the saved model.
    """
    cfg = get_settings()
    logger.info("Starting PPO retraining — timesteps=%d", total_timesteps)

    try:
        import ray  # type: ignore[import]
        from ray import tune  # type: ignore[import]
        from ray.air import RunConfig  # type: ignore[import]

        def train_fn(config: dict) -> None:
            from stable_baselines3 import PPO  # type: ignore[import]
            env = _build_env(price_data, cfg)
            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=config["lr"],
                n_steps=config["n_steps"],
                batch_size=config["batch_size"],
                gamma=config["gamma"],
                verbose=0,
            )
            model.learn(total_timesteps=total_timesteps)
            save_path = os.path.join(tune.get_trial_dir(), "ppo_model")
            model.save(save_path)
            tune.report({"mean_reward": -0.5})  # placeholder metric

        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, num_cpus=min(os.cpu_count() or 1, 8))

        tuner = tune.Tuner(
            train_fn,
            param_space={
                "lr": tune.loguniform(1e-5, 3e-3),
                "n_steps": tune.choice([512, 1024, 2048]),
                "batch_size": tune.choice([64, 128, 256]),
                "gamma": tune.uniform(0.95, 0.999),
            },
            tune_config=tune.TuneConfig(num_samples=4),
            run_config=RunConfig(name="ppo_retrain"),
        )
        results = tuner.fit()
        best_trial = results.get_best_result(metric="mean_reward", mode="max")
        best_model_path = os.path.join(best_trial.path, "ppo_model.zip")

        model_dir = os.path.dirname(cfg.ppo_model_path)
        os.makedirs(model_dir, exist_ok=True)
        import shutil
        shutil.copy2(best_model_path, cfg.ppo_model_path)
        logger.info("PPO retraining complete — saved to %s", cfg.ppo_model_path)
        RETRAIN_COUNTER.labels(model="ppo_execution").inc()
        return cfg.ppo_model_path

    except ImportError:
        # Fallback: single-threaded training without Ray
        logger.warning("Ray not available — falling back to direct SB3 training")
        from stable_baselines3 import PPO  # type: ignore[import]
        env = _build_env(price_data, cfg)
        model = PPO("MlpPolicy", env, learning_rate=3e-4, n_steps=1024, verbose=1)
        model.learn(total_timesteps=total_timesteps)
        os.makedirs(os.path.dirname(cfg.ppo_model_path), exist_ok=True)
        model.save(cfg.ppo_model_path.replace(".zip", ""))
        logger.info("PPO saved to %s", cfg.ppo_model_path)
        RETRAIN_COUNTER.labels(model="ppo_execution").inc()
        return cfg.ppo_model_path
