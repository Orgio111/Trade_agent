"""Benchmark script for Custom NN Brain — measures inference latency."""

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from orchestrator.brains.custom_nn_brain import CustomNNBrain


async def benchmark():
    brain = CustomNNBrain()
    await brain.warmup()

    print("=" * 60)
    print("  Custom NN Brain — Performance Benchmark")
    print("=" * 60)

    # First call (cold — includes data fetch)
    t0 = time.perf_counter()
    s1 = await brain.compute_score("BTCUSDT")
    t1 = time.perf_counter()
    cold_ms = (t1 - t0) * 1000
    print(f"\n  Cold call (with data fetch): {cold_ms:.0f}ms")
    print(f"    Score: {s1.score:+.4f}  Confidence: {s1.confidence:.2%}")

    # Warm calls (data cached)
    latencies = []
    for i in range(5):
        t0 = time.perf_counter()
        s = await brain.compute_score("BTCUSDT")
        t1 = time.perf_counter()
        ms = (t1 - t0) * 1000
        latencies.append(ms)
        print(f"  Warm call #{i+1}: {ms:.1f}ms  score={s.score:+.4f}  conf={s.confidence:.2%}")

    avg = sum(latencies) / len(latencies)
    mn = min(latencies)
    mx = max(latencies)
    print(f"\n  Avg latency: {avg:.1f}ms")
    print(f"  Min: {mn:.1f}ms  Max: {mx:.1f}ms")
    print(f"  Cadence: 15s interval -> headroom: {15000 - avg:.0f}ms per tick")

    print("\n  -- What this brain does --")
    print("  1. Fetches 200 1h OHLCV bars from Binance (cold start only)")
    print("  2. Computes 12 technical features per bar:")
    print("     RSI, MACD/price, BB position, volume score, ATR/price,")
    print("     ROC(5h), ROC(10h), ROC(20h), HL spread, vol change,")
    print("     price/MA ratio, volatility")
    print("  3. Builds a 50-step feature sequence for LSTM input")
    print("  4. LSTM model predicts direction score [-1, +1]")
    print("  5. Blends: 60% model + 40% rule-based (RSI/MACD/BB)")
    print("  6. Returns BrainSignal(score, confidence) to orchestrator")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    asyncio.run(benchmark())
