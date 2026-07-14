"""QUANTEX Reinforcement Learning System — RL training, strategy evolution, portfolio allocation."""
from .trading_env import TradingEnvironment
from .strategy_evolver import StrategyEvolver, OptunaOptimizer

# Lazy import — gymnasium is optional (brains degrade gracefully without it)
try:
    from .gym_env import GymTradingEnv
except (ImportError, ModuleNotFoundError):
    GymTradingEnv = None  # type: ignore[assignment,misc]

__all__ = [
    "TradingEnvironment",
    "StrategyEvolver",
    "OptunaOptimizer",
    "GymTradingEnv",
]

