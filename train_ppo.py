"""
QUANTEX RL Training Script — Stable-Baselines3 PPO on Trading Environment

Trains a PPO agent on simulated BTCUSDT market data with:

Features:
  - Proper Gymnasium v1 wrapper (terminated/truncated separation)
  - Invalid action masking (can't open position while in one)
  - TensorBoard logging for training metrics
  - Multi-episode evaluation loop
  - Equity curve and trade visualization
  - Model checkpoint saving/loading

Usage:
  python train_ppo.py                         # Quick test (10k steps)
  python train_ppo.py --total-timesteps 200000  # Full training
  python train_ppo.py --eval-only              # Load saved model & evaluate
  python train_ppo.py --tensorboard-log ./logs  # Custom log dir
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from typing import Union

# ── Project root setup ─────────────────────────────────────
_PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(_PROJECT_ROOT))


# ── Data Generation ─────────────────────────────────────────

def fetch_binance_data(
    symbol: str = "BTCUSDT",
    interval: str = "5m",
    days: int = 180,
) -> pd.DataFrame:
    """
    Fetch real historical OHLCV data from Binance REST API.

    Uses DataLoader.from_binance_api() with automatic pagination
    to fetch up to `days` of data at the given interval.

    Falls back to generate_training_data() if Binance API is unavailable.

    Args:
        symbol:   Trading pair (e.g. "BTCUSDT")
        interval: Kline interval ("5m", "15m", "1h", "4h", "1d")
        days:     How many days of historical data to fetch

    Returns:
        DataFrame with columns [open, high, low, close, volume]
    """
    from datetime import timedelta
    try:
        from orchestrator.backtest import DataLoader
        end = datetime.now()
        start = end - timedelta(days=days)

        data = asyncio.run(
            DataLoader.from_binance_api(
                symbol=symbol,
                interval=interval,
                start_time=start,
                end_time=end,
                limit=1500,
            )
        )
        if not data.empty:
            print(f"   Source: REAL BINANCE DATA ({len(data)} candles from {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')})")
            return data
    except Exception as e:
        print(f"   Binance API unavailable: {e}")

    print("   Falling back to synthetic data generation")
    return generate_training_data(periods=days * 288, start_price=50000.0, volatility=0.02)



def generate_training_data(
    periods: int = 5000,
    start_price: float = 50000.0,
    volatility: float = 0.02,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate realistic OHLCV data for RL training.

    Uses geometric Brownian motion with realistic volatility clustering.
    """
    rng = np.random.RandomState(seed)
    dates = pd.date_range(
        end=datetime.now(),
        periods=periods,
        freq="5min",
    )

    # Geometric Brownian Motion
    returns = rng.normal(0, volatility / np.sqrt(288), periods)  # ~288 5min bars/day
    # Add momentum (autocorrelation)
    for i in range(1, len(returns)):
        returns[i] += 0.05 * returns[i - 1]

    price = start_price * np.exp(np.cumsum(returns))
    price = np.maximum(price, start_price * 0.1)  # Prevent zero/negative

    # Generate OHLCV from close prices
    ohlc = []
    for i in range(periods):
        base = price[i]
        change_pct = rng.normal(0, volatility / 2 / np.sqrt(288))
        o = base * (1 + change_pct - 0.0005)
        c = base * (1 + change_pct + 0.0005)
        h = max(o, c) * (1 + abs(rng.normal(0, 0.001)))
        l = min(o, c) * (1 - abs(rng.normal(0, 0.001)))
        v = rng.exponential(100)
        ohlc.append([o, h, l, c, v])

    df = pd.DataFrame(
        ohlc,
        columns=["open", "high", "low", "close", "volume"],
        index=dates,
    )
    return df


def generate_eval_data(
    periods: int = 1000,
    start_price: float = 50000.0,
    seed: int = 99,
) -> pd.DataFrame:
    """Generate out-of-sample evaluation data with different seed."""
    return generate_training_data(periods=periods, start_price=start_price, seed=seed)


# ── Model Paths ─────────────────────────────────────────────

def get_model_dir() -> Path:
    """Get the model directory."""
    path = Path(__file__).parent / "models" / "rl"
    path.mkdir(parents=True, exist_ok=True)
    return path


# ── Training ────────────────────────────────────────────────

def train_ppo(args):
    """Train PPO agent on the trading environment."""
    print("=" * 60)
    print("  QUANTEX RL Training — PPO")
    print("=" * 60)

    # ── Generate data ──────────────────────────────────────
    print(f"\n Gathering training data ({args.total_timesteps:,} steps needed)...")
    if args.real_data:
        # Fetch real Binance data with enough samples for the requested timesteps
        needed_candles = args.total_timesteps + 200
        days_needed = max(needed_candles // 288 + 1, args.real_data_days)
        train_data = fetch_binance_data(days=days_needed, interval="5m")
    else:
        print("   Using synthetic data (use --real-data for Binance)")
        train_data = generate_training_data(periods=args.total_timesteps + 100, volatility=args.volatility)

    # Split: train / validation
    split = int(len(train_data) * 0.8)
    train_df = train_data.iloc[:split].copy()
    val_df = train_data.iloc[split:].copy()
    print(f"   Train: {len(train_df):,} candles ({train_df.index[0].strftime('%Y-%m-%d')} -> {train_df.index[-1].strftime('%Y-%m-%d')})")
    print(f"   Val:   {len(val_df):,} candles ({val_df.index[0].strftime('%Y-%m-%d')} -> {val_df.index[-1].strftime('%Y-%m-%d')})")

    # ── ML Engine (for observation feature) ────────────────
    from orchestrator.ml_signals import MLSignalEngine
    ml_engine = MLSignalEngine(min_training_samples=100, lookahead_periods=12)
    ml_loaded = ml_engine.load_model()
    if not ml_loaded:
        print("   Training ML model for observation features...")
        ml_engine.train(train_df, force=True)
    else:
        print(f"   ML engine loaded (trained {ml_engine._train_count}x)")

    # ── Multi-Timeframe Engine (optional) ──────────────────
    mtf_engine = None
    if args.mtf:
        print("\n   Initializing Multi-Timeframe engine...")
        higher_tfs = [tf.strip() for tf in args.mtf_intervals.split(",")]
        from orchestrator.multi_tf import fetch_mtf_data
        import asyncio
        mtf_engine = asyncio.run(
            fetch_mtf_data(
                symbol="BTCUSDT",
                base_interval="5m",
                higher_intervals=higher_tfs,
                days=args.real_data_days if args.real_data else 90,
            )
        )
        print(f"   MTF features: {mtf_engine.intervals} -> {mtf_engine.total_mtf_features} features")

    # ── Deep Market Encoder (GPU, optional) ────────────────
    deep_encoder = None
    if args.gpu_encoder:
        print("\n   Initializing Deep Market Encoder (GPU)...")
        from orchestrator.deep_encoder import DeepMarketEncoder
        deep_encoder = DeepMarketEncoder(
            encoding_dim=args.gpu_encoder_dim,
            device="auto",  # auto-detect CUDA
        )
        # Quick warm-up: run one forward pass with dummy data to init CUDA
        import numpy as np
        dummy_seq = np.random.randn(1, deep_encoder.seq_len, 8).astype(np.float32)
        deep_encoder.encode_batch(dummy_seq)
        device_name = deep_encoder.device.type
        print(f"   DeepEncoder on {device_name.upper()}: seq_len={deep_encoder.seq_len}, "
              f"lstm=128x2, transformer=128x4x2 -> encoding={args.gpu_encoder_dim}")

    # ── Create environment ─────────────────────────────────
    from orchestrator.rl.gym_env import GymTradingEnv

    env = GymTradingEnv(
        data=train_df,
        initial_balance=args.initial_balance,
        leverage=args.leverage,
        lookback=50,
        ml_engine=ml_engine,
        mtf_engine=mtf_engine,
        deep_encoder=deep_encoder,
    )

    # ── Callbacks ──────────────────────────────────────────
    from stable_baselines3.common.callbacks import (
        EvalCallback,
        StopTrainingOnRewardThreshold,
    )

    # Evaluation environment (validation set)
    eval_env = GymTradingEnv(
        data=val_df,
        initial_balance=args.initial_balance,
        leverage=args.leverage,
        lookback=50,
        ml_engine=ml_engine,
        mtf_engine=mtf_engine,
    )

    # Stop if mean reward threshold is met
    stop_callback = StopTrainingOnRewardThreshold(
        reward_threshold=args.reward_threshold,
        verbose=1,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(get_model_dir() / "best"),
        log_path=str(get_model_dir() / "eval_logs"),
        eval_freq=max(1000, args.total_timesteps // 20),
        n_eval_episodes=5,
        deterministic=True,
        callback_on_new_best=stop_callback,
        verbose=1,
    )

    # ── TensorBoard ────────────────────────────────────────
    tensorboard_log = args.tensorboard_log or str(get_model_dir() / "tensorboard")

    print(f"   Training set:   {len(train_df):,} candles")
    print(f"   Validation set: {len(val_df):,} candles")
    print(f"   Balance:        ${args.initial_balance:.2f}")
    print(f"   Leverage:       {args.leverage}x")
    print()

    # ── PPO Model ──────────────────────────────────────────
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    # Wrap in DummyVecEnv for SB3 compatibility
    vec_env = DummyVecEnv([lambda: env])

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=tensorboard_log,
        seed=args.seed,
        device="auto",
    )

    print(f"\n PPO Network: MlpPolicy")
    print(f"   Policy network: {model.policy}")
    print(f"   Learning rate:  {args.learning_rate}")
    print(f"   Batch size:     {args.batch_size}")
    print(f"   Steps per update: {args.n_steps}")
    print(f"   TensorBoard:    {tensorboard_log}")
    print()

    # ── Train ──────────────────────────────────────────────
    print(f" Training for {args.total_timesteps:,} timesteps...")

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=eval_callback,
        progress_bar=True,
    )

    # ── Save ───────────────────────────────────────────────
    model_path = get_model_dir() / f"ppo_trading_{args.total_timesteps}steps_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    model.save(str(model_path))
    print(f"\n Model saved to: {model_path}.zip")

    # Also save a canonical "latest" copy
    latest_path = get_model_dir() / "ppo_trading_latest"
    model.save(str(latest_path))
    print(f" Latest model:   {latest_path}.zip")

    env.close()
    eval_env.close()

    return model, model_path


# ── Evaluation ──────────────────────────────────────────────

def evaluate_ppo(
    model_or_path: Union[str, "PPO"],
    episodes: int = 5,
    render: bool = False,
):
    """Evaluate a trained PPO agent and show detailed results.

    Args:
        model_or_path: Either a path string to a saved model.zip,
                       or an in-memory PPO model object (avoids Windows file locking).
    """
    print("\n" + "=" * 60)
    print("  Evaluating PPO Agent")
    print("=" * 60)

    from orchestrator.rl.gym_env import GymTradingEnv
    from stable_baselines3 import PPO

    # Load model if path, otherwise use in-memory object
    if isinstance(model_or_path, str):
        model = PPO.load(model_or_path)
        print(f" Model loaded from: {model_or_path}")
    else:
        model = model_or_path
        print(" Using in-memory model (no disk reload)")

    # Generate evaluation data
    # Use real data for evaluation if training used real data, otherwise synthetic
    try:
        from orchestrator.backtest import DataLoader
        from datetime import timedelta
        eval_end = datetime.now()
        eval_start = eval_end - timedelta(days=14)
        eval_data = asyncio.run(
            DataLoader.from_binance_api(
                symbol="BTCUSDT",
                interval="5m",
                start_time=eval_start,
                end_time=eval_end,
                limit=1500,
            )
        )
        if eval_data.empty:
            raise ValueError("No data returned")
        print(f" Eval data: REAL Binance ({len(eval_data)} candles)")
    except Exception:
        eval_data = generate_eval_data(periods=2000)
        print(f" Eval data: synthetic ({len(eval_data)} candles)")

    all_episode_rewards = []
    all_trades = []
    final_balances = []

    for ep in range(episodes):
        from orchestrator.ml_signals import MLSignalEngine
        _ml_eval = MLSignalEngine()
        _ml_eval.load_model()
        env = GymTradingEnv(
            data=eval_data,
            initial_balance=10.0,
            leverage=3,
            lookback=50,
            ml_engine=_ml_eval,
            mtf_engine=None,  # Eval uses same obs dim as training — MTF slots stay 0 if not used
        )

        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        step_count = 0
        action_hist = {i: 0 for i in range(6)}
        first_30_actions = []

        while not done:
            action_raw, _ = model.predict(obs, deterministic=True)
            # Normalise action to plain int (handles 0-d numpy arrays)
            if isinstance(action_raw, np.ndarray) and action_raw.ndim >= 1:
                action_val = int(action_raw[0])
            else:
                action_val = int(action_raw)

            obs, reward, terminated, truncated, info = env.step(action_val)
            total_reward += reward
            action_hist[action_val] = action_hist.get(action_val, 0) + 1
            if step_count < 30:
                first_30_actions.append(action_val)
            step_count += 1
            done = terminated or truncated

            if render and ep == 0:
                env.render()

        all_episode_rewards.append(total_reward)
        final_balances.append(env.balance)
        all_trades.append(env.trades)

        pos = env._env.position
        qty = env._env.position_qty
        entry_p = env._env.entry_price

        print(f"   Episode {ep + 1}: Reward={total_reward:+.4f}  "
              f"Final Bal=${env.balance:.4f}  Trades={len(env.trades)}")
        if ep == 0:
            print(f"     Actions: H={action_hist.get(0,0)} L={action_hist.get(1,0)} S={action_hist.get(2,0)}"
                  f" C={action_hist.get(3,0)} SI={action_hist.get(4,0)} SO={action_hist.get(5,0)}")
            print(f"     First 30 actions: {first_30_actions}")
            print(f"     End state: pos={pos} qty={qty:.8f} entry=${entry_p:.2f}")

        env.close()

    # ── Summary stats ──────────────────────────────────────
    rewards = np.array(all_episode_rewards)
    balances = np.array(final_balances)
    total_trades = sum(len(t) for t in all_trades)

    print(f"\n Results ({episodes} evaluation episodes):")
    print(f"   Avg Reward:      {rewards.mean():+.4f} +/- {rewards.std():.4f}")
    print(f"   Avg Final Bal:   ${balances.mean():.4f}")
    print(f"   Total Trades:    {total_trades}")
    print(f"   Avg Trades/Ep:   {total_trades / episodes:.1f}")

    # ── Returns analysis ───────────────────────────────────
    all_pnls = [t["pnl"] for trades in all_trades for t in trades]
    if all_pnls:
        win_rate = sum(1 for p in all_pnls if p > 0) / len(all_pnls)
        avg_win = np.mean([p for p in all_pnls if p > 0]) if any(p > 0 for p in all_pnls) else 0
        avg_loss = np.mean([p for p in all_pnls if p < 0]) if any(p < 0 for p in all_pnls) else 0
        print(f"   Win Rate:        {win_rate:.1%}")
        print(f"   Avg Win:         ${avg_win:.4f}")
        print(f"   Avg Loss:        ${avg_loss:.4f}")
        if avg_loss != 0:
            print(f"   Profit Factor:   {abs(avg_win / avg_loss):.2f}")

    return all_episode_rewards, all_trades


# ── Plot Results ────────────────────────────────────────────

def plot_results(all_trades: list[list[dict]]):
    """Plot equity curve and trade markers."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        print(" matplotlib not installed — skipping plot")
        return

    # Flatten all trades sorted by step
    trades = [t for ep_trades in all_trades for t in ep_trades]
    if not trades:
        print(" No trades to plot")
        return

    trades = sorted(trades, key=lambda t: t["step"])
    steps = [t["step"] for t in trades]
    pnls = [t["pnl"] for t in trades]
    cumulative = np.cumsum(pnls)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [2, 1]})
    fig.suptitle("QUANTEX PPO Trading Agent — Evaluation Results", fontsize=14, fontweight="bold")

    # ── Equity Curve ───────────────────────────────────────
    ax1 = axes[0]
    ax1.plot(cumulative, color="#00ff88", linewidth=1.5, label="Cumulative PnL")
    ax1.fill_between(range(len(cumulative)), cumulative, alpha=0.1, color="#00ff88")
    ax1.axhline(y=0, color="#333", linewidth=0.5)
    ax1.set_ylabel("Cumulative PnL ($)")
    ax1.set_xlabel("Trade #")
    ax1.legend()
    ax1.grid(True, alpha=0.15)

    # Mark winners/losers
    colors = ["#00ff88" if p > 0 else "#ff0044" for p in pnls]
    ax1.scatter(range(len(pnls)), cumulative, c=colors, s=20, alpha=0.7, zorder=5)

    # ── Trade PnL Bars ─────────────────────────────────────
    ax2 = axes[1]
    bar_colors = ["#00ff88" if p > 0 else "#ff0044" for p in pnls]
    ax2.bar(range(len(pnls)), pnls, color=bar_colors, alpha=0.7, width=0.8)
    ax2.axhline(y=0, color="#333", linewidth=0.5)
    ax2.set_ylabel("Trade PnL ($)")
    ax2.set_xlabel("Trade #")
    ax2.grid(True, alpha=0.15)

    plt.tight_layout()

    plot_path = get_model_dir() / "evaluation_plot.png"
    plt.savefig(str(plot_path), dpi=150, bbox_inches="tight")
    print(f" Plot saved to: {plot_path}")

    # Also try to display on Windows
    try:
        plt.show(block=False)
    except Exception:
        pass


# ── CLI ─────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="QUANTEX RL Training — PPO on Trading Environment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Training
    parser.add_argument("--total-timesteps", type=int, default=20000,
                        help="Total training timesteps")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                        help="PPO learning rate")
    parser.add_argument("--n-steps", type=int, default=2048,
                        help="Steps per policy update")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Minibatch size")
    parser.add_argument("--n-epochs", type=int, default=10,
                        help="Epochs per update")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="GAE lambda")
    parser.add_argument("--clip-range", type=float, default=0.2,
                        help="PPO clip range")
    parser.add_argument("--ent-coef", type=float, default=0.01,
                        help="Entropy coefficient")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    # Environment
    parser.add_argument("--initial-balance", type=float, default=10.0,
                        help="Starting balance in USDT")
    parser.add_argument("--leverage", type=int, default=3,
                        help="Max leverage")
    parser.add_argument("--volatility", type=float, default=0.02,
                        help="Daily volatility for data generation")
    parser.add_argument("--real-data", action="store_true",
                        help="Fetch real Binance historical data instead of synthetic")
    parser.add_argument("--real-data-days", type=int, default=180,
                        help="Days of Binance data to fetch when --real-data is set")
    parser.add_argument("--mtf", action="store_true",
                        help="Enable Multi-Timeframe features (5m + 1H + 4H + 1D) in observation")
    parser.add_argument("--mtf-intervals", type=str, default="1h,4h,1d",
                        help="Comma-separated higher timeframes for MTF (default: 1h,4h,1d)")
    parser.add_argument("--gpu-encoder", action="store_true",
                        help="Enable GPU Deep Market Encoder (LSTM + Transformer) in observation[96:128]")
    parser.add_argument("--gpu-encoder-dim", type=int, default=32,
                        help="Encoding dimension for deep market encoder")

    # Evaluation
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training, load saved model and evaluate")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Model path for --eval-only")
    parser.add_argument("--eval-episodes", type=int, default=5,
                        help="Number of evaluation episodes")
    parser.add_argument("--reward-threshold", type=float, default=3.0,
                        help="Stop training when mean reward exceeds this")
    parser.add_argument("--render", action="store_true",
                        help="Render environment during evaluation")
    parser.add_argument("--tensorboard-log", type=str, default=None,
                        help="TensorBoard log directory")

    return parser.parse_args()


def main():
    args = parse_args()
    import time

    if args.eval_only:
        # ── Evaluation only ──────────────────────────────
        model_path = args.model_path
        if not model_path:
            latest = get_model_dir() / "ppo_trading_latest"
            zip_path = str(latest) + ".zip"
            if os.path.exists(zip_path):
                model_path = str(latest)
            else:
                print(f" No model found at {zip_path}")
                print(" Train a model first with: python train_ppo.py")
                return

        rewards, trades = evaluate_ppo(model_path, episodes=args.eval_episodes, render=args.render)
        plot_results(trades)
    else:
        # ── Train ────────────────────────────────────────
        model, model_path = train_ppo(args)

        # ── Evaluate after training (uses in-memory model — no file locking) ──
        try:
            rewards, trades = evaluate_ppo(
                model, episodes=args.eval_episodes, render=args.render
            )
        except Exception as e:
            print(f" Evaluation skipped: {e}")
            rewards, trades = [], []
        if trades:
            plot_results(trades)


if __name__ == "__main__":
    main()
