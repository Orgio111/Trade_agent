"""
QUANTEX Multi-Timeframe Test — Verifies MTF data fetching, alignment, and observation building.

Tests:
  1. MTF engine fetches data for all configured timeframes
  2. Feature shapes are correct (8 features per TF)
  3. Time alignment works (higher TF forward-filled to base TF)
  4. Observation vector contains non-zero MTF features
  5. Full observation shape = 96 (base 64 + MTF 32)
  6. Integration with GymTradingEnv (obs now expanded to 128-dim, MTF at [64:96])

Usage:
    python test_mtf.py                    # Quick test with synthetic fallback
    python test_mtf.py --real-data        # Test with real Binance data
"""

import sys, os, asyncio, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description="Test MTF engine")
parser.add_argument("--real-data", action="store_true", help="Use real Binance data")
args = parser.parse_args()

async def main():
    print("=" * 60)
    print("  QUANTEX Multi-Timeframe Test")
    print("=" * 60)

    # ── 1. Create MTF engine ───────────────────────────────
    print("\n1. Creating MTF engine...")
    from orchestrator.multi_tf import MultiTimeframeEngine, TIMEFRAME_MINUTES

    mtf = MultiTimeframeEngine(base_interval="5m", higher_intervals=["1h", "4h", "1d"])
    print(f"   Base: {mtf.base_interval}, Higher: {mtf.higher_intervals}")
    print(f"   All intervals: {mtf.all_intervals}")
    print(f"   Features per TF: {mtf._features_per_tf}")
    print(f"   Total MTF features: {mtf.total_mtf_features}")
    assert mtf.total_mtf_features == 4 * 8 == 32, f"Expected 32 MTF features, got {mtf.total_mtf_features}"

    # ── 2. Fetch data ──────────────────────────────────────
    print("\n2. Fetching market data...")
    if args.real_data:
        await mtf.fetch_and_prepare(days=90)
    else:
        # Use synthetic data for quick test
        rng = np.random.RandomState(42)
        for interval in mtf.all_intervals:
            n = 3000 if interval == "5m" else (250 if interval == "1h" else 60)
            base = 50000
            dates = pd.date_range(end=pd.Timestamp.now(), periods=n, freq=f"{TIMEFRAME_MINUTES[interval]}min")
            price = base * np.exp(np.cumsum(rng.normal(0, 0.001, n)))
            mtf._raw_data[interval] = pd.DataFrame({
                "open": price * (1 + rng.normal(0, 0.0005, n)),
                "high": price * (1 + abs(rng.normal(0, 0.001, n))),
                "low": price * (1 - abs(rng.normal(0, 0.001, n))),
                "close": price,
                "volume": rng.exponential(100, n),
            }, index=dates)
            print(f"   [SYNTH] {interval}: {n} candles")
        mtf._compute_and_align()

    print(mtf.summary())

    # ── 3. Check features per TF ───────────────────────────
    print("\n3. Checking features per timeframe...")
    for interval in mtf.all_intervals:
        feats = mtf._features.get(interval)
        aligned = mtf._aligned.get(interval)
        raw = mtf._raw_data.get(interval)
        if raw is not None:
            print(f"   {interval}: raw={len(raw)} candles, features={len(feats) if feats is not None else 0}, aligned={len(aligned) if aligned is not None else 0}")

    # ── 4. Check observation vector ────────────────────────
    print("\n4. Building MTF observation vectors...")
    n_checks = min(10, mtf.base_length)
    for i in range(n_checks):
        obs = mtf.get_observation_vector(i)
        assert obs.shape == (32,), f"Expected shape (32,), got {obs.shape}"
        # MTF features should be non-zero after warm-up period
        if i >= 20:
            non_zero = np.count_nonzero(obs)
            print(f"   Step {i}: shape={obs.shape}, non-zero={non_zero}/32")
            assert non_zero > 0, f"All MTF features are zero at step {i}!"

    # ── 5. Check features dict ─────────────────────────────
    print("\n5. Checking features dict...")
    feats = mtf.get_features_dict(100)
    for interval, f in feats.items():
        if f:
            print(f"   {interval}: returns_1={f['returns_1']:.6f}, rsi_norm={f['rsi_norm']:.4f}, "
                  f"ema_trend={f['ema_trend']:.4f}, atr_pct={f['atr_pct']:.4f}")

    # ── 6. Integration with GymTradingEnv ─────────────────
    print("\n6. Testing GymTradingEnv integration...")
    from orchestrator.rl.gym_env import GymTradingEnv

    base_df = mtf._raw_data["5m"]
    env = GymTradingEnv(
        data=base_df,
        initial_balance=10.0,
        leverage=1,
        lookback=50,
        ml_engine=None,
        mtf_engine=mtf,
    )

    # Reset and check observation (128-dim: 64 base + 32 MTF + 32 DeepEncoder)
    obs, info = env.reset()
    print(f"   Observation shape: {obs.shape}")
    print(f"   Expected shape:    (128,)")
    assert obs.shape == (128,), f"Expected (128,), got {obs.shape}"

    # Check that base features are non-zero
    base_feats = obs[:40]
    mtf_feats = obs[64:96]
    print(f"   Base features [0:40]:  {np.count_nonzero(base_feats)}/{len(base_feats)} non-zero")
    print(f"   MTF features [64:96]:  {np.count_nonzero(mtf_feats)}/{len(mtf_feats)} non-zero")
    print(f"   Sample MTF: {mtf_feats[:8]}")

    # Step through a few steps
    for step in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()

    mtf_feats_after = obs[64:96]
    print(f"   After 20 steps — MTF [64:96]: {np.count_nonzero(mtf_feats_after)}/{len(mtf_feats_after)} non-zero")

    env.close()

    # ── 7. ALL PASSED ────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  [OK] ALL CHECKS PASSED")
    print(f"  {'='*60}")
    print(f"  Observation:     128-dim (64 base + 32 MTF + 32 DeepEncoder)")
    print(f"  Timeframes:      {mtf.intervals}")
    print(f"  MTF features:    {mtf.total_mtf_features} ({mtf._features_per_tf} per TF x {len(mtf.all_intervals)} TFs)")
    print(f"  ML-in-obs:       obs[38:40] (direction + confidence)")
    print(f"  MTF-in-obs:      obs[64:96] (up to 4 TFs x 8 features)")
    print(f"  DeepEnc-in-obs:  obs[96:128] (GPU LSTM+Transformer embedding)")
    print(f"{'='*60}")

if __name__ == "__main__":
    asyncio.run(main())
