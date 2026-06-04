"""Sanity test: verify the new reward shaping works correctly."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
from orchestrator.rl.trading_env import TradingEnvironment

dates = pd.date_range("2025-01-01", periods=100, freq="5min")
df = pd.DataFrame({
    "open": np.linspace(50000, 51000, 100),
    "high": np.linspace(50100, 51100, 100),
    "low": np.linspace(49900, 50900, 100),
    "close": np.linspace(50000, 51000, 100),
    "volume": np.ones(100) * 100,
}, index=dates)

print("=" * 60)
print("  TEST 1: Hold for 20 steps (should accumulate negative reward)")
print("=" * 60)
env = TradingEnvironment(data=df, initial_balance=10.0, lookback=10)
env.reset()
total = 0.0
for i in range(20):
    obs, reward, done, info = env.step(0)
    total += reward
print(f"  Total reward: {total:.4f}")
print(f"  Consecutive holds: {env._consecutive_hold_steps}")
print(f"  Balance: ${env.balance:.4f}")
assert total < 0, "Hold should produce negative reward!"
print("  [PASS] Hold penalty works\n")

print("=" * 60)
print("  TEST 2: LONG x3 cycles (should get trade bonuses)")
print("=" * 60)
env.reset()
total = 0.0
for cycle in range(3):
    obs, r, done, info = env.step(1)  # LONG
    total += r
    print(f"  Cycle {cycle+1}: LONG -> reward={r:+.4f}")
    for _ in range(5):
        obs, r, done, info = env.step(0)  # HOLD
        total += r
    obs, r, done, info = env.step(3)  # CLOSE
    total += r
    print(f"           CLOSE -> reward={r:+.4f}")
print(f"  Total reward: {total:.4f}")
print(f"  Trades: {info['trades']}")
assert info["trades"] > 0, "Should have completed trades!"
print("  [PASS] Trade bonuses work\n")

print("=" * 60)
print("  TEST 3: LONG -> HOLD many times -> penalty grows")
print("=" * 60)
env.reset()
env.step(1)  # LONG
penalties = []
for i in range(16):
    obs, r, done, info = env.step(0)  # HOLD
    penalties.append(r)
print(f"  Hold rewards (steps 1-16): {[f'{p:.4f}' for p in penalties]}")
assert penalties[0] > penalties[10], "Penalty should grow with consecutive holds!"
print("  [PASS] Escalating hold penalty works\n")

print("All tests passed!")
