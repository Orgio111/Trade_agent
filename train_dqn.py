"""
QUANTEX RL Training Script — Stable-Baselines3 DQN on Trading Environment

Trains a DQN agent on Binance BTCUSDT data.

DQN (Deep Q-Network) is a value-based RL algorithm that:
  - Learns Q-values for each action (state-action value)
  - Uses experience replay for sample efficiency
  - Uses target network for stable learning
  - Built for discrete action spaces

Usage:
  python train_dqn.py                           # Quick test (20k steps)
  python train_dqn.py --total-timesteps 100000  # Full training
  python train_dqn.py --eval-only               # Load saved model & evaluate
"""
import argparse, asyncio, os, sys
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(_PROJECT_ROOT))

def fetch_binance_data(symbol="BTCUSDT", interval="5m", days=180):
    from datetime import timedelta
    try:
        from orchestrator.backtest import DataLoader
        end = datetime.now()
        start = end - timedelta(days=days)
        data = asyncio.run(DataLoader.from_binance_api(symbol=symbol, interval=interval, start_time=start, end_time=end, limit=1500))
        if not data.empty:
            print(f"   Source: REAL BINANCE DATA ({len(data)} candles)")
            return data
    except Exception as e:
        print(f"   Binance API unavailable: {e}")
    print("   Falling back to synthetic data")
    return generate_training_data(periods=days * 288, start_price=50000.0, volatility=0.02)

def generate_training_data(periods=5000, start_price=50000.0, volatility=0.02, seed=42):
    rng = np.random.RandomState(seed)
    dates = pd.date_range(end=datetime.now(), periods=periods, freq="5min")
    returns = rng.normal(0, volatility / np.sqrt(288), periods)
    for i in range(1, len(returns)):
        returns[i] += 0.05 * returns[i - 1]
    price = start_price * np.exp(np.cumsum(returns))
    price = np.maximum(price, start_price * 0.1)
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
    return pd.DataFrame(ohlc, columns=["open","high","low","close","volume"], index=dates)

def get_model_dir():
    path = Path(__file__).parent / "models" / "rl"
    path.mkdir(parents=True, exist_ok=True)
    return path

def train_dqn(args):
    print("=" * 60)
    print("  QUANTEX RL Training — DQN")
    print("=" * 60)

    # Data
    print(f"\n Gathering training data ({args.total_timesteps:,} steps needed)...")
    if args.real_data:
        needed_candles = args.total_timesteps + 200
        days_needed = max(needed_candles // 288 + 1, args.real_data_days)
        train_data = fetch_binance_data(days=days_needed, interval="5m")
    else:
        train_data = generate_training_data(periods=args.total_timesteps + 100, volatility=args.volatility)

    split = int(len(train_data) * 0.8)
    train_df = train_data.iloc[:split].copy()
    val_df = train_data.iloc[split:].copy()

    # Environment
    from orchestrator.rl.gym_env import GymTradingEnv
    from stable_baselines3 import DQN
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

    env = GymTradingEnv(data=train_df, initial_balance=args.initial_balance, leverage=args.leverage, lookback=50)
    eval_env = GymTradingEnv(data=val_df, initial_balance=args.initial_balance, leverage=args.leverage, lookback=50)

    # DQN uses different stopping criteria
    stop_callback = StopTrainingOnRewardThreshold(reward_threshold=args.reward_threshold, verbose=1)
    eval_callback = EvalCallback(eval_env, best_model_save_path=str(get_model_dir() / "best_dqn"),
                                  log_path=str(get_model_dir() / "eval_logs"), eval_freq=max(1000, args.total_timesteps // 20),
                                  n_eval_episodes=5, deterministic=True, callback_on_new_best=stop_callback, verbose=1)

    tensorboard_log = args.tensorboard_log or str(get_model_dir() / "tensorboard_dqn")

    vec_env = DummyVecEnv([lambda: env])

    # DQN Model
    model = DQN(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        tau=args.tau,
        gamma=args.gamma,
        train_freq=args.train_freq,
        target_update_interval=args.target_update_interval,
        exploration_fraction=args.exploration_fraction,
        exploration_initial_eps=args.exploration_initial_eps,
        exploration_final_eps=args.exploration_final_eps,
        verbose=1,
        tensorboard_log=tensorboard_log,
        seed=args.seed,
        device="auto",
    )

    print(f"\n DQN Network: {model.policy.__class__.__name__}")
    print(f"   Learning rate:     {args.learning_rate}")
    print(f"   Buffer size:       {args.buffer_size:,}")
    print(f"   Learning starts:   {args.learning_starts:,}")
    print(f"   Batch size:        {args.batch_size}")
    print(f"   Tau (soft update): {args.tau}")
    print(f"   Exploration:       {args.exploration_initial_eps} -> {args.exploration_final_eps}")
    print(f"   TensorBoard:       {tensorboard_log}")

    print(f"\n Training for {args.total_timesteps:,} timesteps...")
    model.learn(total_timesteps=args.total_timesteps, callback=eval_callback, progress_bar=True)

    # Save
    model_path = get_model_dir() / f"dqn_trading_{args.total_timesteps}steps_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    model.save(str(model_path))
    latest_path = get_model_dir() / "dqn_trading_latest"
    model.save(str(latest_path))
    print(f"\n Model saved to: {model_path}.zip")

    env.close()
    eval_env.close()
    return model, model_path

def evaluate_dqn(model_or_path, episodes=5, render=False):
    print("\n" + "=" * 60)
    print("  Evaluating DQN Agent")
    print("=" * 60)

    from orchestrator.rl.gym_env import GymTradingEnv
    from stable_baselines3 import DQN

    if isinstance(model_or_path, str):
        model = DQN.load(model_or_path)
        print(f" Model loaded from: {model_or_path}")
    else:
        model = model_or_path
        print(" Using in-memory model")

    # Fetch eval data
    try:
        from orchestrator.backtest import DataLoader
        from datetime import timedelta
        eval_end = datetime.now()
        eval_start = eval_end - timedelta(days=14)
        eval_data = asyncio.run(DataLoader.from_binance_api(symbol="BTCUSDT", interval="5m", start_time=eval_start, end_time=eval_end, limit=1500))
        if eval_data.empty:
            raise ValueError("No data")
        print(f" Eval data: REAL Binance ({len(eval_data)} candles)")
    except Exception:
        eval_data = generate_training_data(periods=2000, seed=99)
        print(f" Eval data: synthetic ({len(eval_data)} candles)")

    all_rewards, all_trades, all_balances = [], [], []
    action_names = {0: "HOLD", 1: "LONG", 2: "SHORT", 3: "CLOSE"}

    for ep in range(episodes):
        env = GymTradingEnv(data=eval_data, initial_balance=10.0, leverage=3, lookback=50)
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        action_hist = {i: 0 for i in range(4)}
        first_30 = []

        while not done:
            action_raw, _ = model.predict(obs, deterministic=True)
            if isinstance(action_raw, np.ndarray) and action_raw.ndim >= 1:
                action_val = int(action_raw[0])
            else:
                action_val = int(action_raw)
            obs, reward, terminated, truncated, info = env.step(action_val)
            total_reward += reward
            action_hist[action_val] = action_hist.get(action_val, 0) + 1
            if len(first_30) < 30:
                first_30.append(action_val)
            done = terminated or truncated

        all_rewards.append(total_reward)
        all_balances.append(env.balance)
        all_trades.append(env.trades)

        print(f"   Episode {ep+1}: Reward={total_reward:+.4f}  "
              f"Final Bal=${env.balance:.4f}  Trades={len(env.trades)}")
        if ep == 0:
            print(f"     Actions: H={action_hist.get(0,0)} L={action_hist.get(1,0)} "
                  f"S={action_hist.get(2,0)} C={action_hist.get(3,0)}")
            print(f"     First 30: {first_30}")
            print(f"     End state: pos={env._env.position} entry=${env._env.entry_price:.2f}")
        env.close()

    rewards = np.array(all_rewards)
    balances = np.array(all_balances)
    total_trades = sum(len(t) for t in all_trades)
    print(f"\n Results ({episodes} episodes):")
    print(f"   Avg Reward:      {rewards.mean():+.4f} +/- {rewards.std():.4f}")
    print(f"   Avg Final Bal:   ${balances.mean():.4f}")
    print(f"   Total Trades:    {total_trades}")

    all_pnls = [t["pnl"] for trades in all_trades for t in trades]
    if all_pnls:
        wins = sum(1 for p in all_pnls if p > 0)
        losses = sum(1 for p in all_pnls if p < 0)
        avg_win = np.mean([p for p in all_pnls if p > 0]) if wins else 0
        avg_loss = np.mean([p for p in all_pnls if p < 0]) if losses else 0
        pf = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")
        print(f"   Win Rate:        {wins/len(all_pnls):.1%} ({wins}W/{losses}L)")
        print(f"   Avg Win/Loss:    ${avg_win:.4f} / ${avg_loss:.4f}  (PF={pf:.2f})")

    return all_rewards, all_trades

def main():
    parser = argparse.ArgumentParser(description="QUANTEX RL Training — DQN", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Training
    parser.add_argument("--total-timesteps", type=int, default=30000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)  # DQN typically uses lower LR
    parser.add_argument("--buffer-size", type=int, default=50000)
    parser.add_argument("--learning-starts", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--train-freq", type=int, default=4)
    parser.add_argument("--target-update-interval", type=int, default=5000)
    parser.add_argument("--exploration-fraction", type=float, default=0.2)
    parser.add_argument("--exploration-initial-eps", type=float, default=1.0)
    parser.add_argument("--exploration-final-eps", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    # Environment
    parser.add_argument("--initial-balance", type=float, default=10.0)
    parser.add_argument("--leverage", type=int, default=3)
    parser.add_argument("--volatility", type=float, default=0.02)
    parser.add_argument("--real-data", action="store_true")
    parser.add_argument("--real-data-days", type=int, default=180)
    # Evaluation
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--reward-threshold", type=float, default=999999)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--tensorboard-log", type=str, default=None)

    args = parser.parse_args()

    if args.eval_only:
        model_path = args.model_path
        if not model_path:
            latest = get_model_dir() / "dqn_trading_latest"
            if (str(latest) + ".zip").exists():
                model_path = str(latest)
            else:
                print(f" No model found, train first: python train_dqn.py")
                return
        rewards, trades = evaluate_dqn(model_path, episodes=args.eval_episodes, render=args.render)
    else:
        model, model_path = train_dqn(args)
        try:
            rewards, trades = evaluate_dqn(model, episodes=args.eval_episodes, render=args.render)
        except Exception as e:
            print(f" Evaluation skipped: {e}")

if __name__ == "__main__":
    main()
