"""Debug evaluation: trace reward source for PPO model step by step."""
import sys, os, asyncio
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from datetime import datetime, timedelta
from orchestrator.rl.gym_env import GymTradingEnv
from stable_baselines3 import PPO

# Load latest model
model_dir = os.path.join(os.path.dirname(__file__), "models", "rl")
latest = os.path.join(model_dir, "ppo_trading_latest")
model = PPO.load(latest)
print("Model loaded.")

# Fetch fresh Binance data like evaluate_ppo does
from orchestrator.backtest import DataLoader
async def get_data():
    end = datetime.now()
    start = end - timedelta(days=14)
    df = await DataLoader.from_binance_api(symbol="BTCUSDT", interval="5m", start_time=start, end_time=end, limit=500)
    return df
eval_data = asyncio.run(get_data())
print(f"Eval data: {len(eval_data)} candles, price ${eval_data.close.iloc[-1]:.2f}")

# Run 1 episode with detailed logging
env = GymTradingEnv(data=eval_data, initial_balance=10.0, leverage=3, lookback=50)
obs, _ = env.reset()

total_reward = 0.0
hold_count = 0
trade_actions = 0
step = 0

while step < 20:  # First 20 steps only
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    
    pos = info["position"]
    bal = info["balance"]
    
    action_names = ["HOLD", "LONG", "SHORT", "CLOSE", "SCALE_IN", "SCALE_OUT"]
    action_name = action_names[action] if action < 6 else f"UNK({action})"
    
    print(f"  Step {step:3d}: action={action_name:10s} reward={reward:+.6f}  "
          f"bal=${bal:>8.4f}  pos={pos:>2}  trades={info['trades']}")
    
    if action == 0:
        hold_count += 1
    else:
        trade_actions += 1
    
    if terminated or truncated:
        break
    step += 1

print(f"\nFirst 20 steps: total_reward={total_reward:.4f}, hold_count={hold_count}, trade_actions={trade_actions}")
env.close()

# Now run full episode
print(f"\n{'='*60}")
print("  FULL EPISODE (all steps)")
print(f"{'='*60}")
env = GymTradingEnv(data=eval_data, initial_balance=10.0, leverage=3, lookback=50)
obs, _ = env.reset()
total_reward = 0.0
hold_count = 0
trade_actions = 0
step = 0

while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    if action == 0:
        hold_count += 1
    else:
        trade_actions += 1
    if terminated or truncated:
        break
    step += 1

print(f"  Steps: {step}")
print(f"  Total reward: {total_reward:.4f}")
print(f"  Final balance: ${env.balance:.4f}")
print(f"  Trades: {len(env.trades)}")
print(f"  Hold count: {hold_count}")
print(f"  Trade actions: {trade_actions}")
print(f"  Position: {env._env.position}")
print(f"  Position qty: {env._env.position_qty}")
env.close()
