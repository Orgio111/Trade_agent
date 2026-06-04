"""Self-contained: train PPO, evaluate in-memory with step-by-step debug logging."""
import sys, os, asyncio
from pathlib import Path
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from datetime import datetime, timedelta

# ── Data ───────────────────────────────────────────────────
from orchestrator.backtest import DataLoader
async def fetch():
    end = datetime.now()
    start = end - timedelta(days=60)
    df = await DataLoader.from_binance_api(symbol="BTCUSDT", interval="5m", start_time=start, end_time=end, limit=1500)
    return df
data = asyncio.run(fetch())
print(f"Data: {len(data)} candles, price ${data.close.iloc[-1]:.2f}")

split = int(len(data) * 0.8)
train_df = data.iloc[:split].copy()
val_df = data.iloc[split:].copy()

# ── Environment ────────────────────────────────────────────
from orchestrator.rl.gym_env import GymTradingEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

env = GymTradingEnv(data=train_df, initial_balance=10.0, leverage=3, lookback=50)
vec_env = DummyVecEnv([lambda: env])

# ── Train (very short - just to test evaluation) ───────────
model = PPO("MlpPolicy", vec_env, learning_rate=3e-4, n_steps=2048, batch_size=64,
            n_epochs=10, ent_coef=0.01, verbose=0, seed=42)
print(f"\nTraining for 8192 steps (4 updates)...")
model.learn(total_timesteps=8192, progress_bar=True)
print("Training done.\n")

# ── Evaluate with full step-by-step logging ────────────────
print("=" * 70)
print("  EVALUATION: Step-by-step actions (first 50 steps)")
print("=" * 70)

action_names = {0: "HOLD", 1: "LONG", 2: "SHORT", 3: "CLOSE", 4: "SCL_IN", 5: "SCL_OUT"}

for ep in range(2):
    eval_env = GymTradingEnv(data=val_df, initial_balance=10.0, leverage=3, lookback=50)
    obs, _ = eval_env.reset()
    total_reward = 0.0
    step = 0
    action_counts = {k: 0 for k in range(6)}
    opened_position = False
    
    while True:
        action_arr, _ = model.predict(obs, deterministic=True)
        action_val = int(action_arr[0] if hasattr(action_arr, '__iter__') else action_arr)
        obs, reward, terminated, truncated, info = eval_env.step(action_val)
        total_reward += reward
        action_counts[action_val] = action_counts.get(action_val, 0) + 1
        
        if action_val in (1, 2):
            opened_position = True
            
        if step < 50:  # Print first 50 steps only
            print(f"    ep{ep+1} step{step:3d}: {action_names[action_val]:>7s} "
                  f"rw={reward:+.6f}  bal=${info['balance']:>8.4f}  "
                  f"pos={info['position']:>2}  dd={info['drawdown']:.4f}  "
                  f"trades={info['trades']}")
        
        if terminated or truncated:
            break
        step += 1
    
    print(f"\n  Episode {ep+1} results:")
    print(f"    Total steps: {step}")
    print(f"    Total reward: {total_reward:.4f}")
    print(f"    Final balance: ${eval_env.balance:.4f}")
    print(f"    Completed trades: {len(eval_env.trades)}")
    print(f"    Open position: {eval_env._env.position}")
    print(f"    Position qty: {eval_env._env.position_qty:.8f}")
    print(f"    Entry price: ${eval_env._env.entry_price:.2f}" if eval_env._env.entry_price > 0 else "    Entry price: N/A (flat)")
    print(f"    Action distribution: {action_counts}")
    print(f"    Opened position ever: {opened_position}")
    eval_env.close()
    print()
