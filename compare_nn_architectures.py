"""Compare LSTM vs Transformer architectures on the same OHLCV data.

Runs both architectures through identical market data, collects metrics:
- Prediction accuracy (direction correctness)
- Score distribution (mean, std, range)
- Inference latency
- Auto-train performance (loss convergence)
- Attention patterns (Transformer only)
- Simulated PnL on held-out data

Usage:
    python compare_nn_architectures.py
    python compare_nn_architectures.py --days 30 --symbol BTCUSDT
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np


def fetch_binance_ohlcv(symbol: str = "BTCUSDT", interval: str = "1h", days: int = 60) -> list[dict]:
    """Fetch historical OHLCV from Binance REST API."""
    import httpx

    base_url = "https://api.binance.com/api/v3/klines"
    end_ts = int(datetime.now().timestamp() * 1000)
    start_ts = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

    all_data = []
    current_ts = start_ts

    with httpx.Client(timeout=30) as client:
        while current_ts < end_ts:
            params = {
                "symbol": symbol,
                "interval": interval,
                "startTime": current_ts,
                "endTime": end_ts,
                "limit": 1000,
            }
            resp = client.get(base_url, params=params)
            if resp.status_code != 200:
                print(f"  Binance API error: {resp.status_code}")
                break
            data = resp.json()
            if not data:
                break
            for row in data:
                all_data.append({
                    "timestamp": row[0],
                    "open": float(row[1]),
                    "high": float(row[2]),
                    "low": float(row[3]),
                    "close": float(row[4]),
                    "volume": float(row[5]),
                })
            current_ts = data[-1][6] + 1

    print(f"  Fetched {len(all_data)} candles ({days}d {interval})")
    return all_data


def evaluate_brain(
    brain,
    ohlcv: list[dict],
    lookback: int = 50,
    train_bars: int = 100,
    eval_bars: int = 100,
) -> dict:
    """Evaluate a brain on OHLCV data, collecting metrics.

    Args:
        brain: CustomNNBrain instance (already has architecture set)
        ohlcv: List of OHLCV dicts
        lookback: Sequence length for features
        train_bars: Number of bars used for auto-training
        eval_bars: Number of bars used for evaluation
    """
    import asyncio

    # Push all bars into buffer
    for bar in ohlcv:
        brain.push_ohlcv(bar)

    # Auto-train
    t0 = time.time()
    trained = brain.auto_train()
    train_time = time.time() - t0

    # Evaluate on the last eval_bars
    eval_start = max(lookback, len(ohlcv) - eval_bars)
    scores = []
    latencies = []
    correct = 0
    total = 0

    for i in range(eval_start, len(ohlcv)):
        bar = ohlcv[i]
        closes = np.array([b["close"] for b in ohlcv[:i + 1]])
        highs = np.array([b["high"] for b in ohlcv[:i + 1]])
        lows = np.array([b["low"] for b in ohlcv[:i + 1]])
        volumes = np.array([b["volume"] for b in ohlcv[:i + 1]])

        if len(closes) < lookback:
            continue

        # Compute features
        features = brain._extract_features_sequence(closes, highs, lows, volumes)

        # Predict
        t0 = time.time()
        score = brain._model.predict(features)
        latency_ms = (time.time() - t0) * 1000

        scores.append(score)
        latencies.append(latency_ms)

        # Check direction correctness (next bar return)
        if i + 1 < len(ohlcv):
            next_return = (ohlcv[i + 1]["close"] - bar["close"]) / bar["close"]
            predicted_direction = 1 if score > 0 else (-1 if score < 0 else 0)
            actual_direction = 1 if next_return > 0 else (-1 if next_return < 0 else 0)
            if predicted_direction == actual_direction:
                correct += 1
            total += 1

    # Compute simulated PnL
    balance = 100.0
    peak = 100.0
    max_dd = 0.0
    equity_curve = [100.0]

    for i in range(eval_start, min(eval_start + len(scores) - 1, len(ohlcv) - 1)):
        idx = i - eval_start
        if idx >= len(scores):
            break
        score = scores[idx]
        bar = ohlcv[i]
        next_bar = ohlcv[i + 1]

        if abs(score) > 0.15:  # trade threshold
            direction = 1 if score > 0 else -1
            ret = (next_bar["close"] - bar["close"]) / bar["close"] * direction
            balance *= (1 + ret * 0.95)  # 5% fee impact

        equity_curve.append(balance)
        peak = max(peak, balance)
        dd = (peak - balance) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)

    scores_arr = np.array(scores) if scores else np.array([0.0])
    latencies_arr = np.array(latencies) if latencies else np.array([0.0])

    # Get attention data for Transformer
    attention_entropy = None
    attention_top_positions = None
    if hasattr(brain._model, "get_attention_weights") and len(ohlcv) >= lookback:
        try:
            last_closes = np.array([b["close"] for b in ohlcv[-lookback:]])
            last_highs = np.array([b["high"] for b in ohlcv[-lookback:]])
            last_lows = np.array([b["low"] for b in ohlcv[-lookback:]])
            last_volumes = np.array([b["volume"] for b in ohlcv[-lookback:]])
            features = brain._extract_features_sequence(last_closes, last_highs, last_lows, last_volumes)
            attn = brain._model.get_attention_weights(features)
            if attn is not None:
                avg_attn = np.mean(attn, axis=0) if attn.ndim == 3 else attn
                last_row = avg_attn[-1]
                last_row_norm = last_row / (last_row.sum() + 1e-10)
                attention_entropy = float(-np.sum(last_row_norm * np.log(last_row_norm + 1e-10)))
                attention_top_positions = np.argsort(last_row)[-5:][::-1].tolist()
        except Exception as e:
            print(f"  Attention extraction failed: {e}")

    return {
        "trained": trained,
        "train_time_s": round(train_time, 2),
        "eval_bars": len(scores),
        "direction_accuracy": round(correct / total, 4) if total > 0 else 0.0,
        "correct": correct,
        "total_predictions": total,
        "score_mean": round(float(scores_arr.mean()), 4),
        "score_std": round(float(scores_arr.std()), 4),
        "score_min": round(float(scores_arr.min()), 4),
        "score_max": round(float(scores_arr.max()), 4),
        "latency_mean_ms": round(float(latencies_arr.mean()), 2),
        "latency_p50_ms": round(float(np.percentile(latencies_arr, 50)), 2),
        "latency_p95_ms": round(float(np.percentile(latencies_arr, 95)), 2),
        "simulated_pnl": round(balance - 100.0, 2),
        "simulated_pnl_pct": round((balance - 100.0) / 100.0, 4),
        "max_drawdown_pct": round(max_dd, 4),
        "attention_entropy": round(attention_entropy, 4) if attention_entropy is not None else None,
        "attention_top_positions": attention_top_positions,
    }


def main():
    parser = argparse.ArgumentParser(description="LSTM vs Transformer Architecture Comparison")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--balance", type=float, default=100.0)
    args = parser.parse_args()

    print("=" * 70)
    print("  LSTM vs Transformer Architecture Comparison")
    print("=" * 70)
    print(f"  Symbol: {args.symbol} | Interval: {args.interval} | Days: {args.days}")
    print()

    # Fetch data
    print("[1/4] Fetching OHLCV data...")
    ohlcv = fetch_binance_ohlcv(args.symbol, args.interval, args.days)
    if len(ohlcv) < 150:
        print(f"  ERROR: Need at least 150 candles, got {len(ohlcv)}")
        return

    # Split data: 70% train+eval, 30% holdout test
    split = int(len(ohlcv) * 0.7)
    train_eval_data = ohlcv[:split]
    holdout_data = ohlcv[split:]
    print(f"  Train+Eval: {len(train_eval_data)} candles | Holdout: {len(holdout_data)} candles")

    results = {}

    for arch in ["lstm", "transformer"]:
        print(f"\n[{arch.upper()}] Testing {arch} architecture...")

        # Set architecture
        os.environ["CUSTOM_NN_ARCHITECTURE"] = arch

        # Import and create brain (fresh instance for each)
        import importlib
        import orchestrator.brains.custom_nn_brain as mod
        importlib.reload(mod)
        brain = mod.CustomNNBrain()

        print(f"  Architecture: {brain._architecture}")
        print(f"  Model type: {type(brain._model).__name__}")

        # Evaluate on train+eval split
        print(f"  Evaluating on {len(train_eval_data)} candles...")
        metrics = evaluate_brain(brain, train_eval_data, eval_bars=min(100, len(train_eval_data) - 60))

        # Also evaluate on holdout (without retraining)
        print(f"  Holdout test on {len(holdout_data)} candles...")
        holdout_metrics = evaluate_brain(brain, ohlcv, eval_bars=len(holdout_data))

        results[arch] = {
            "train_eval": metrics,
            "holdout": holdout_metrics,
        }

        print(f"  Train time: {metrics['train_time_s']}s")
        print(f"  Direction accuracy: {metrics['direction_accuracy']:.1%}")
        print(f"  Simulated PnL: {metrics['simulated_pnl_pct']:+.2%}")
        print(f"  Max drawdown: {metrics['max_drawdown_pct']:.2%}")
        print(f"  Latency p50: {metrics['latency_mean_ms']:.1f}ms")
        if metrics.get("attention_entropy") is not None:
            print(f"  Attention entropy: {metrics['attention_entropy']:.3f}")
            print(f"  Attention top positions: {metrics['attention_top_positions']}")

    # Print comparison table
    print()
    print("=" * 70)
    print("  COMPARISON RESULTS")
    print("=" * 70)

    header = f"{'Metric':<30} {'LSTM':>15} {'Transformer':>15} {'Winner':>8}"
    print(header)
    print("-" * 70)

    metrics_to_compare = [
        ("Direction Accuracy", "direction_accuracy", "{:.1%}", True),
        ("Simulated PnL %", "simulated_pnl_pct", "{:+.2%}", True),
        ("Max Drawdown", "max_drawdown_pct", "{:.2%}", False),
        ("Score Mean", "score_mean", "{:+.4f}", None),
        ("Score Std Dev", "score_std", "{:.4f}", None),
        ("Latency (ms)", "latency_mean_ms", "{:.1f}", False),
        ("Train Time (s)", "train_time_s", "{:.1f}", False),
        ("Attention Entropy", "attention_entropy", "{:.4f}", None),
    ]

    for label, key, fmt, higher_better in metrics_to_compare:
        lstm_val = results["lstm"]["train_eval"].get(key)
        trans_val = results["transformer"]["train_eval"].get(key)

        if lstm_val is None or trans_val is None:
            lstm_str = "N/A"
            trans_str = "N/A"
            winner = "-"
        else:
            lstm_str = fmt.format(lstm_val)
            trans_str = fmt.format(trans_val)
            if higher_better is True:
                winner = "LSTM" if lstm_val > trans_val else ("TF" if trans_val > lstm_val else "Tie")
            elif higher_better is False:
                winner = "LSTM" if lstm_val < trans_val else ("TF" if trans_val < lstm_val else "Tie")
            else:
                winner = "-"

        print(f"{label:<30} {lstm_str:>15} {trans_str:>15} {winner:>8}")

    # Holdout comparison
    print()
    print("-" * 70)
    print("  HOLDOUT (unseen data)")
    print("-" * 70)

    for label, key, fmt, higher_better in [
        ("Direction Accuracy", "direction_accuracy", "{:.1%}", True),
        ("Simulated PnL %", "simulated_pnl_pct", "{:+.2%}", True),
        ("Max Drawdown", "max_drawdown_pct", "{:.2%}", False),
    ]:
        lstm_val = results["lstm"]["holdout"].get(key)
        trans_val = results["transformer"]["holdout"].get(key)
        if lstm_val is None or trans_val is None:
            continue
        lstm_str = fmt.format(lstm_val)
        trans_str = fmt.format(trans_val)
        if higher_better:
            winner = "LSTM" if lstm_val > trans_val else ("TF" if trans_val > lstm_val else "Tie")
        else:
            winner = "LSTM" if lstm_val < trans_val else ("TF" if trans_val < lstm_val else "Tie")
        print(f"  {label:<28} LSTM: {lstm_str:>12}  TF: {trans_str:>12}  [{winner}]")

    print("=" * 70)

    # Save results
    results_dir = Path("backtest_results")
    results_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = results_dir / f"nn_arch_comparison_{ts}.json"
    out_file.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n  Results saved: {out_file}")


if __name__ == "__main__":
    main()
