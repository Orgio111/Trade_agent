"""
QUANTEX PPO Portfolio Manager — Training Script.

Trains a PPO agent to allocate capital across N assets using
the PortfolioAllocEnv gym environment.

Usage:
    # Train with synthetic data (default)
    python -m orchestrator.train_ppo_portfolio --timesteps 100000

    # Train with historical returns data
    python -m orchestrator.train_ppo_portfolio --data path/to/returns.csv --timesteps 200000

    # Train and save model
    python -m orchestrator.train_ppo_portfolio --save models/ppo_portfolio --timesteps 50000

    # Quick benchmark (minimal timesteps)
    python -m orchestrator.train_ppo_portfolio --benchmark
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .ppo_portfolio_manager import PPOPortfolioManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("quantex.train_ppo_portfolio")


# ── Synthetic Data Generators ────────────────────────────────

def generate_synthetic_returns(
    n_assets: int = 4,
    n_periods: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic correlated asset returns for training.

    Creates N assets with realistic correlation structure:
    - Base market factor drives ~40% of returns
    - Each asset has individual noise component
    - Periodic volatility regimes (high/low)
    - Occasional outlier events

    Args:
        n_assets: Number of synthetic assets
        n_periods: Number of time periods
        seed: Random seed for reproducibility

    Returns:
        DataFrame with columns = asset symbols, index = period
    """
    rng = np.random.default_rng(seed)

    # Asset names
    asset_names = [f"ASSET_{i+1}" for i in range(n_assets)]

    # Common market factor
    market_factor = rng.normal(0, 0.005, n_periods)

    # Volatility regimes (alternating low/high)
    regime = np.where(np.arange(n_periods) % 50 < 15, 0.02, 0.005)

    # Asset-specific returns
    returns = np.zeros((n_periods, n_assets))
    for i in range(n_assets):
        # Correlated part (40% from market)
        correlated = 0.4 * market_factor

        # Idiosyncratic part (60% individual)
        unique = rng.normal(0, regime, n_periods)

        # Occasional outlier (1% probability)
        outlier_mask = rng.random(n_periods) < 0.01
        unique[outlier_mask] += rng.choice([-0.03, 0.03], size=outlier_mask.sum())

        returns[:, i] = correlated + unique

    return pd.DataFrame(returns, columns=asset_names)


# ── Real Data Loader ─────────────────────────────────────────

def load_returns_from_csv(path: str) -> pd.DataFrame:
    """Load returns data from a CSV file.

    Expected format: columns = asset symbols, rows = periodic returns.
    First column can be a date/time index (auto-detected).

    Args:
        path: Path to CSV file

    Returns:
        DataFrame with asset returns
    """
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    logger.info("Loaded returns: %d assets × %d periods", df.shape[1], df.shape[0])
    return df


# ── Training Runner ──────────────────────────────────────────

def run_training(
    returns_df: pd.DataFrame,
    n_assets: int = 4,
    asset_names: Optional[list[str]] = None,
    total_timesteps: int = 100_000,
    model_dir: str = "models/ppo_portfolio",
    force: bool = False,
    benchmark: bool = False,
) -> dict:
    """Run PPO portfolio training.

    Args:
        returns_df: Training returns data
        n_assets: Number of assets
        asset_names: Asset names (from returns_df columns if None)
        total_timesteps: Training timesteps
        model_dir: Model save directory
        force: Force retraining
        benchmark: Quick benchmark mode

    Returns:
        Training result dict
    """
    if asset_names is None:
        asset_names = list(returns_df.columns[:n_assets])

    if benchmark:
        total_timesteps = min(total_timesteps, 10_000)
        logger.info("BENCHMARK MODE: %d timesteps", total_timesteps)

    logger.info(
        "Training PPO Portfolio Manager: %d assets, %d timesteps",
        len(asset_names), total_timesteps,
    )

    manager = PPOPortfolioManager(
        n_assets=len(asset_names),
        asset_names=asset_names,
        model_dir=model_dir,
    )

    result = manager.train(
        returns_df=returns_df,
        total_timesteps=total_timesteps,
        force=force,
    )

    return result


# ── Evaluation ───────────────────────────────────────────────

def evaluate_model(
    manager: PPOPortfolioManager,
    returns_df: pd.DataFrame,
    n_episodes: int = 10,
) -> dict:
    """Evaluate trained PPO model against fallback methods.

    Args:
        manager: Trained PPOPortfolioManager
        returns_df: Evaluation returns data
        n_episodes: Number of evaluation episodes

    Returns:
        Performance comparison dict
    """
    logger.info("Evaluating PPO vs Markowitz (%d episodes)...", n_episodes)

    ppo_returns: list[float] = []
    mark_returns: list[float] = []

    for episode in range(n_episodes):
        # Use different slices of data for each episode
        start = np.random.randint(0, max(1, len(returns_df) - 100))
        eval_df = returns_df.iloc[start:start + 100]

        # PPO allocation
        ppo_result = manager.allocate(eval_df, confidence=0.5)

        # Markowitz allocation
        mark_result = manager.allocate_with_fallback(eval_df, method="markowitz")

        # Compute actual returns from the weights
        aligned = [c for c in manager.asset_names if c in eval_df.columns]
        if aligned and len(eval_df) > 1:
            ppo_ret = float(np.dot(
                [ppo_result.weights.get(c, 0.0) for c in aligned],
                eval_df[aligned].iloc[-1].values,
            ))
            mark_ret = float(np.dot(
                [mark_result.weights.get(c, 0.0) for c in aligned],
                eval_df[aligned].iloc[-1].values,
            ))
            ppo_returns.append(ppo_ret)
            mark_returns.append(mark_ret)

    comparison = {
        "n_episodes": n_episodes,
        "ppo_mean_return": float(np.mean(ppo_returns)) if ppo_returns else 0.0,
        "ppo_std_return": float(np.std(ppo_returns)) if ppo_returns else 0.0,
        "markowitz_mean_return": float(np.mean(mark_returns)) if mark_returns else 0.0,
        "markowitz_std_return": float(np.std(mark_returns)) if mark_returns else 0.0,
    }

    if comparison["markowitz_mean_return"] != 0:
        comparison["improvement_pct"] = round(
            (comparison["ppo_mean_return"] - comparison["markowitz_mean_return"])
            / abs(comparison["markowitz_mean_return"]) * 100,
            2,
        )
    else:
        comparison["improvement_pct"] = 0.0

    logger.info("Evaluation results:")
    for k, v in comparison.items():
        logger.info("  %s: %s", k, v)

    return comparison


# ── CLI ──────────────────────────────────────────────────────

def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train PPO portfolio allocation model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--data", type=str, default=None,
        help="Path to CSV returns file (default: synthetic data)",
    )
    parser.add_argument(
        "--assets", type=int, default=4,
        help="Number of assets (ignored if --data is provided)",
    )
    parser.add_argument(
        "--timesteps", type=int, default=100_000,
        help="Total training timesteps",
    )
    parser.add_argument(
        "--save", type=str, default="models/ppo_portfolio",
        help="Model save directory",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force retraining even if model exists",
    )
    parser.add_argument(
        "--benchmark", action="store_true",
        help="Quick benchmark mode (10K timesteps, no evaluation)",
    )
    parser.add_argument(
        "--eval", action="store_true",
        help="Run evaluation after training",
    )
    parser.add_argument(
        "--eval-episodes", type=int, default=10,
        help="Number of evaluation episodes",
    )

    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None):
    """Main entry point."""
    args = parse_args(argv)

    # Load or generate data
    if args.data:
        returns_df = load_returns_from_csv(args.data)
        n_assets = len(returns_df.columns)
    else:
        logger.info("Generating synthetic returns (%d assets, %d periods)...",
                     args.assets, 2000)
        returns_df = generate_synthetic_returns(n_assets=args.assets, n_periods=2000)
        n_assets = args.assets

    asset_names = list(returns_df.columns[:n_assets])

    # Phase 1: Train
    t0 = time.time()
    result = run_training(
        returns_df=returns_df,
        n_assets=n_assets,
        asset_names=asset_names,
        total_timesteps=args.timesteps,
        model_dir=args.save,
        force=args.force,
        benchmark=args.benchmark,
    )
    elapsed = time.time() - t0

    logger.info("Training completed in %.1fs", elapsed)
    logger.info("Result: %s", result)

    if result.get("status") == "error":
        logger.error("Training failed: %s", result.get("reason"))
        sys.exit(1)

    # Phase 2: Evaluate (optional)
    if args.eval and not args.benchmark:
        manager = PPOPortfolioManager(
            n_assets=n_assets,
            asset_names=asset_names,
            model_dir=args.save,
        )
        # Load the just-trained model
        manager._load_model()
        if manager.is_trained():
            eval_result = evaluate_model(
                manager=manager,
                returns_df=returns_df,
                n_episodes=args.eval_episodes,
            )
            logger.info("Evaluation: %s", eval_result)

    # Phase 3: Print allocation example
    if not args.benchmark:
        print("\n" + "=" * 60)
        print("PPO Portfolio Manager — Training Complete")
        print("=" * 60)
        print(f"  Status:        {result.get('status', '?')}")
        print(f"  Assets:        {n_assets}")
        print(f"  Timesteps:     {args.timesteps:,}")
        print(f"  Elapsed:       {elapsed:.1f}s")
        print(f"  Model saved:   {result.get('model_path', '?')}")
        print(f"  Model version: v{result.get('version', '?')}")

        if args.eval:
            print(f"\n  PPO Return:     {result.get('ppo_mean_return', '?')}")
            print(f"  Markowitz Ret:  {result.get('markowitz_mean_return', '?')}")

        print("\n  To use in production:")
        print(f"    from orchestrator.ppo_portfolio_manager import PPOPortfolioManager")
        print(f"    manager = PPOPortfolioManager(n_assets={n_assets})")
        print(f"    manager._load_model()")
        print(f"    weights = manager.allocate(returns_df)")
        print("=" * 60)


if __name__ == "__main__":
    main()
