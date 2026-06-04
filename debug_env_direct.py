"""Direct env test: run fixed action sequences and see rewards."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Fetch real data like PPO training does
from orchestrator.backtest import DataLoader
import asyncio
async def fetch():
    end = datetime.now()
    start = end - timedelta(days=14)
    df = await DataLoader.from_binance_api(symbol="BTCUSDT", interval="5m", start_time=start, end_time=end, limit=500)
    return df
data = asyncio.run(fetch())
print(f"Data: {len(data)} candles, ${data.close.iloc[0]:.0f} -> ${data.close.iloc[-1]:.0f}")

from orchestrator.rl.gym_env import GymTradingEnv

# Create env
env = GymTradingEnv(data=data, initial_balance=10.0, leverage=3, lookback=50)
obs, _ = env.reset()

print("\n=== SCENARIO 1: ALWAYS HOLD (flat) ===")
env._env.reset()
total = 0.0
for i in range(50):
    obs, r, term, trunc, info = env.step(0)  # HOLD while flat
    total += r
print(f"  50 holds (flat): total reward = {total:.4f}")
assert total < 0, f"HOLD should be negative, got {total}"

print("\n=== SCENARIO 2: LONG -> HOLD x10 -> CLOSE ===")
env._env.reset()
total = 0.0
obs, r, term, trunc, info = env.step(1)  # LONG
print(f"  LONG: r={r:+.4f}, bal=${info['balance']:.4f}")
total += r
for i in range(10):
    obs, r, term, trunc, info = env.step(0)  # HOLD
    total += r
print(f"  HOLD x10: total={total:.4f}, bal=${info['balance']:.4f}")
obs, r, term, trunc, info = env.step(3)  # CLOSE
total += r
print(f"  CLOSE: r={r:+.4f}, bal=${info['balance']:.4f}")
print(f"  TOTAL: {total:.4f}")

print("\n=== SCENARIO 3: CYCLE LONG/CLOSE to avoid hold penalty ===")
env._env.reset()
total = 0.0
for cycle in range(25):
    obs, r, term, trunc, info = env.step(1)  # LONG
    total += r
    obs, r, term, trunc, info = env.step(3)  # CLOSE (next step)
    total += r
print(f"  25 LONG/CLOSE cycles: total = {total:.4f}")
print(f"  Avg per step: {total/50:.4f}")

print("\n=== SCENARIO 4: Full episode (all steps) with ALL HOLD ===")
env._env.reset()
total = 0.0
steps = 0
while True:
    obs, r, term, trunc, info = env.step(0)
    total += r
    steps += 1
    if term or trunc:
        break
print(f"  All HOLD ({steps} steps): total = {total:.4f}")
print(f"  Final balance: ${info['balance']:.4f}")

print("\n=== SCENARIO 5: LONG once, then HOLD forever ===")
env._env.reset()
total = 0.0
steps = 0
obs, r, term, trunc, info = env.step(1)  # LONG
total += r
while True:
    obs, r, term, trunc, info = env.step(0)  # HOLD
    total += r
    steps += 1
    if term or trunc:
        break
print(f"  LONG + HOLD ({steps+1} steps): total = {total:.4f}")
print(f"  Final balance: ${info['balance']:.4f}")
pos = env._env.position
qty = env._env.position_qty
print(f"  Position: {pos}, qty: {qty:.8f}")

print("\nDone.")
