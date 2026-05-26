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


def _build_observation(prices: np.ndarray, idx: int, entry_price: float) -> np.ndarray:
    """Shared helper: build a 40-dim observation from a price window."""
    window = prices[max(0, idx - 29) : idx + 1]
    if len(window) < 30:
        window = np.pad(window, (30 - len(window), 0), mode="edge")
    returns = np.diff(window) / (window[:-1] + 1e-10)
    obs = np.zeros(40, dtype=np.float32)
    obs[:29] = returns.astype(np.float32)
    obs[29] = (prices[idx] - entry_price) / (entry_price + 1e-10)
    return obs


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
                return _build_observation(self._prices, self._idx, self._entry_price)

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


# ═══════════════════════════════════════════════════════════════════════════════
#  PPO Model Evaluation via Gym Environment
# ═══════════════════════════════════════════════════════════════════════════════

def _build_eval_env(price_data: np.ndarray, start_idx: int, episode_length: int):
    """Build a deterministic gym env slice for evaluation (no random reset)."""
    import gymnasium as gym  # type: ignore[import]
    from gymnasium import spaces

    class EvalEnv(gym.Env):
        """Non-random evaluation environment that walks forward sequentially."""
        metadata = {"render_modes": []}

        def __init__(self) -> None:
            super().__init__()
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(40,), dtype=np.float32
            )
            self.action_space = spaces.Discrete(4)
            self._prices = price_data
            self._start = start_idx
            self._length = min(episode_length, len(price_data) - start_idx - 10)
            self._idx = start_idx
            self._entry_price = price_data[start_idx]
            self._cumulative = 0.0

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self._idx = self._start
            self._entry_price = self._prices[self._idx]
            self._cumulative = 0.0
            return self._obs(), {}

        def _obs(self) -> np.ndarray:
            return _build_observation(self._prices, self._idx, self._entry_price)

        def step(self, action: int):
            slippage = [0.001, 0.0005, 0.0003, 0.0002][action]
            reward = -slippage * 10_000
            self._cumulative += reward
            self._idx += 1
            done = self._idx >= self._start + self._length - 1
            return self._obs(), reward, done, False, {}

        def get_total_reward(self) -> float:
            return self._cumulative

    return EvalEnv()


def evaluate_ppo_model(
    model_path: str,
    price_data: np.ndarray,
    train_split: float = 0.8,
    n_episodes: int = 20,
    episode_steps: int = 50,
) -> dict:
    """
    Evaluate a trained PPO model on in-sample and out-of-sample splits
    by running it deterministically through the gym execution environment.

    Returns
    -------
    dict with keys:
        train_sharpe, test_sharpe, test_max_drawdown, test_win_rate,
        test_information_ratio, test_calmar_ratio,
        test_max_consecutive_losses,
        train_cumulative_reward, test_cumulative_reward, n_episodes
    """
    from mlops.walk_forward import sharpe_ratio, max_drawdown

    _ZERO_RESULT = {
        "train_sharpe": 0.0,
        "test_sharpe": 0.0,
        "test_max_drawdown": 0.0,
        "test_win_rate": 0.0,
        "test_information_ratio": 0.0,
        "test_calmar_ratio": 0.0,
        "test_max_consecutive_losses": 0,
        "train_cumulative_reward": 0.0,
        "test_cumulative_reward": 0.0,
        "n_episodes": 0,
    }

    try:
        from stable_baselines3 import PPO  # type: ignore[import]
    except ImportError:
        logger.warning("evaluate_ppo_model: SB3 not available — returning zeros")
        return dict(_ZERO_RESULT)

    if len(price_data) < 500:
        logger.warning("evaluate_ppo_model: insufficient data (%d points)", len(price_data))
        return dict(_ZERO_RESULT)

    split_idx = int(len(price_data) * train_split)

    try:
        model = PPO.load(model_path)
    except Exception as exc:
        logger.warning("evaluate_ppo_model: failed to load model from %s: %s", model_path, exc)
        return dict(_ZERO_RESULT)

    def _run_episodes(data_slice: np.ndarray, n: int) -> list[float]:
        """Run n evaluation episodes and return list of cumulative rewards."""
        if len(data_slice) < episode_steps + 10:
            return [0.0]
        max_starts = len(data_slice) - episode_steps - 10
        # Evenly spaced starting positions
        starts = np.linspace(0, max_starts, min(n, max_starts + 1), dtype=int)
        episode_returns: list[float] = []
        for start in starts:
            env = _build_eval_env(data_slice, int(start), episode_steps)
            obs, _ = env.reset()
            done = False
            cum = 0.0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _, _ = env.step(int(action))
                cum += reward
            episode_returns.append(cum)
        return episode_returns

    # --- Evaluate on training split ---
    train_returns = _run_episodes(price_data[:split_idx], n_episodes)
    train_arr = np.array(train_returns, dtype=np.float64)
    train_sharpe_val = sharpe_ratio(train_arr) if len(train_arr) > 1 else 0.0

    # --- Evaluate on test split ---
    test_returns = _run_episodes(price_data[split_idx:], n_episodes)
    test_arr = np.array(test_returns, dtype=np.float64)
    test_sharpe_val = sharpe_ratio(test_arr) if len(test_arr) > 1 else 0.0

    # --- Compute drawdown on test equity curve ---
    test_equity = np.cumsum(test_arr) if len(test_arr) > 0 else np.array([0.0])
    test_dd = max_drawdown(test_equity)

    # --- Compute win rate: fraction of episodes with positive cumulative reward ---
    if len(test_arr) > 1:
        test_wr = float((test_arr > 0).mean())
    else:
        test_wr = 0.5

    # --- Compute information ratio: mean / std of episode returns ---
    if len(test_arr) > 1:
        test_ir = float(
            test_arr.mean() / (test_arr.std() + 1e-10)
        )
    else:
        test_ir = 0.0

    # --- Compute Calmar ratio: mean return / abs(max drawdown) ---
    calmar_denom = abs(test_dd) + 1e-10
    test_calmar = float(test_arr.mean() / calmar_denom) if len(test_arr) > 0 else 0.0

    # --- Compute max consecutive losses ---
    if len(test_arr) > 1:
        is_loss = (test_arr < 0).astype(int)
        # Find longest streak of 1s
        changes = np.diff(np.concatenate(([0], is_loss, [0])))
        run_starts = np.where(changes == 1)[0]
        run_ends = np.where(changes == -1)[0]
        if len(run_starts) > 0:
            max_consec = int((run_ends - run_starts).max())
        else:
            max_consec = 0
    else:
        max_consec = 0

    logger.info(
        "PPO evaluation complete: train_sharpe=%.3f test_sharpe=%.3f "
        "test_dd=%.2f%% wr=%.1f%% ir=%.2f calmar=%.2f max_losses=%d (episodes=%d)",
        train_sharpe_val, test_sharpe_val,
        test_dd * 100, test_wr * 100, test_ir, test_calmar, max_consec, len(test_arr),
    )

    return {
        "train_sharpe": train_sharpe_val,
        "test_sharpe": test_sharpe_val,
        "test_max_drawdown": test_dd,
        "test_win_rate": test_wr,
        "test_information_ratio": test_ir,
        "test_calmar_ratio": test_calmar,
        "test_max_consecutive_losses": max_consec,
        "train_cumulative_reward": float(train_arr.mean()) if len(train_arr) > 0 else 0.0,
        "test_cumulative_reward": float(test_arr.mean()) if len(test_arr) > 0 else 0.0,
        "n_episodes": len(test_arr),
    }


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
