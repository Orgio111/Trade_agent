"""
QUANTEX 4-Strategy Comparison: EMA vs ML vs Market Structure vs Delta
======================================================================
All strategies tested on identical REAL Binance BTCUSDT 1h data.

EMA Crossover:       StrategyEngine with 9/21 EMA + RSI filter + ATR SL/TP
ML Random Forest:    MLSignalEngine with pandas_ta features, trained on first 70%
Market Structure:    MarketStructureEngine with SMC (liquidity sweeps, BOS, OB, Wyckoff)
Delta Divergence:    DeltaCVDTracker with CVD divergence detection

For EMA and ML: full backtest via BacktestEngine (PnL, Sharpe, Win Rate, Drawdown)
For MS and Delta: signal agreement analysis (direction match rate, confidence profile)
"""

import asyncio
import sys
import time as time_module
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

from orchestrator.backtest import BacktestEngine, DataLoader
from orchestrator.strategy import StrategyEngine, Signal
from orchestrator.ml_signals import MLSignalEngine
from orchestrator.market_structure import MarketStructureEngine
from orchestrator.microstructure import DeltaCVDTracker


def compute_forward_return(df: pd.DataFrame, idx: int, periods: int = 6) -> float:
    """Compute future return from candle idx across `periods` candles."""
    if idx + periods >= len(df):
        return 0.0
    entry = df.iloc[idx]["close"]
    future = df.iloc[idx + periods]["close"]
    return (future - entry) / entry if entry > 0 else 0.0


def direction_accuracy(signals: list, future_returns: list, min_conf: float = 0.4) -> dict:
    """
    Evaluate how well directional signals predict future returns.
    """
    correct = 0
    total = 0
    tp = 0  # True positives (correct long)
    fp = 0  # False positives (wrong long)
    tn = 0  # True negatives (correct short)
    fn = 0  # False negatives (wrong short)

    for sig, ret in zip(signals, future_returns):
        direction = sig.get("direction", "hold")
        confidence = sig.get("confidence", 0.0)
        if direction == "hold" or confidence < min_conf:
            continue
        total += 1
        if direction == "long" and ret > 0:
            correct += 1
            tp += 1
        elif direction == "long" and ret <= 0:
            fp += 1
        elif direction == "short" and ret < 0:
            correct += 1
            tn += 1
        elif direction == "short" and ret >= 0:
            fn += 1

    return {
        "accuracy": correct / total if total > 0 else 0.0,
        "total_trades": total,
        "precision_long": tp / (tp + fp) if (tp + fp) > 0 else 0.0,
        "precision_short": tn / (tn + fn) if (tn + fn) > 0 else 0.0,
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }


async def main():
    print("=" * 72)
    print("  QUANTEX 4-Strategy Comparison: EMA vs ML vs Market Structure vs Delta")
    print("  Source: REAL Binance BTCUSDT 1h data")
    print("=" * 72)

    # -- 1. Fetch real data ---------------------------------------------------
    print("\n[1/5] Fetching real BTCUSDT 1h data from Binance...")
    end = datetime.now()
    start = end - timedelta(days=90)

    try:
        data = await DataLoader.from_binance_api(
            symbol="BTCUSDT", interval="1h", start_time=start, end_time=end, limit=1500,
        )
        using_mock = False
        print(f"       OK - {len(data)} real candles")
    except Exception as e:
        print(f"       Falling back to mock data: {e}")
        data = DataLoader.generate_mock_data(periods=2160, start_price=50000.0, seed=42)
        using_mock = True

    print(f"       Period:  {data.index[0].strftime('%Y-%m-%d')} -> {data.index[-1].strftime('%Y-%m-%d')}")
    print(f"       Range:   ${data['low'].min():.2f} -> ${data['high'].max():.2f}")
    print(f"       Source:  {'REAL BINANCE' if not using_mock else 'MOCK'}")

    # Split for ML: train on 70%, all strategies evaluated identically on 30%
    split_idx = int(len(data) * 0.70)
    train_data = data.iloc[:split_idx]
    test_data = data.iloc[split_idx:].copy()
    print(f"       Train:   {len(train_data)} candles")
    print(f"       Test:    {len(test_data)} candles")

    # Precompute future returns for all test candles
    future_returns = [compute_forward_return(test_data, i, periods=6) for i in range(len(test_data))]

    # -- 2. Train ML model ----------------------------------------------------
    print("\n[2/5] Training ML Random Forest model on training data...")
    ml_engine = MLSignalEngine(
        min_training_samples=100,
        lookahead_periods=6,
        threshold_buy=0.005,
        threshold_sell=-0.005,
    )
    t0 = time_module.time()
    train_result = ml_engine.train(train_data, force=True)
    elapsed = time_module.time() - t0
    if train_result.get("status") == "trained":
        print(f"       OK - Trained in {elapsed:.2f}s")
        print(f"       Train accuracy: {train_result.get('train_accuracy',0)*100:.1f}%")
        print(f"       Test accuracy:  {train_result.get('test_accuracy',0)*100:.1f}%")
        print(f"       Features:       {train_result.get('features',0)}")
    else:
        print(f"       WARNING - Not trained: {train_result.get('reason','unknown')}")

    # -- 3. Generate signals from ALL 4 sources --------------------------------
    print("\n[3/5] Generating signals from all 4 sources on test data...")

    ema_strategy = StrategyEngine(ema_fast=9, ema_slow=21)
    ms_engine = MarketStructureEngine()
    delta_tracker = DeltaCVDTracker()

    # Pre-populate delta tracker with test data ticks
    for i in range(len(test_data)):
        candle = test_data.iloc[i]
        # Simulate intra-candle ticks from OHLC
        direction = 1 if candle["close"] >= candle["open"] else -1
        for _ in range(5):
            tick_price = candle["open"] + (candle["close"] - candle["open"]) * np.random.random()
            tick_delta = direction * abs(candle["close"] - candle["open"]) / 5 * np.random.random()
            delta_tracker.record_tick(tick_price, tick_delta)

    # Walk through each candle and collect signals
    ema_signals = []
    ml_signals = []
    ms_signals = []
    delta_signals = []

    # Re-seed delta tracker for sequential evaluation
    delta_tracker2 = DeltaCVDTracker()

    for i in range(len(test_data)):
        chunk = test_data.iloc[: i + 1]
        candle = test_data.iloc[i]

        # Seed delta with this candle's data
        direction = 1 if candle["close"] >= candle["open"] else -1
        for _ in range(3):
            tick_price = candle["open"] + (candle["close"] - candle["open"]) * np.random.random()
            tick_delta = direction * abs(candle["close"] - candle["open"]) / 3 * np.random.random()
            delta_tracker2.record_tick(tick_price, tick_delta)

        # EMA signal
        if len(chunk) > 50:
            try:
                s = ema_strategy.generate_signal(chunk)
                ema_signals.append({
                    "direction": s.direction,
                    "confidence": s.confidence,
                    "price": float(candle["close"]),
                })
            except Exception:
                ema_signals.append({"direction": "hold", "confidence": 0.0, "price": float(candle["close"])})
        else:
            ema_signals.append({"direction": "hold", "confidence": 0.0, "price": float(candle["close"])})

        # ML signal
        if ml_engine._model and len(chunk) > 50:
            try:
                s = ml_engine.predict_signal(chunk)
                ml_signals.append({
                    "direction": s.direction,
                    "confidence": s.confidence,
                    "price": float(candle["close"]),
                })
            except Exception:
                ml_signals.append({"direction": "hold", "confidence": 0.0, "price": float(candle["close"])})
        else:
            ml_signals.append({"direction": "hold", "confidence": 0.0, "price": float(candle["close"])})

        # Market Structure signal
        if len(chunk) > 60:
            try:
                ms_signal = ms_engine.get_market_structure_signal(chunk, chunk.iloc[-1])
                ms_signals.append({
                    "direction": ms_signal["direction"],
                    "confidence": ms_signal["confidence"],
                    "wyckoff": ms_signal["wyckoff_phase"],
                    "price": float(candle["close"]),
                })
            except Exception:
                ms_signals.append({"direction": "hold", "confidence": 0.0, "price": float(candle["close"])})
        else:
            ms_signals.append({"direction": "hold", "confidence": 0.0, "price": float(candle["close"])})

        # Delta signal
        try:
            ds = delta_tracker2.get_delta_signal()
            delta_signals.append({
                "direction": ds["direction"],
                "confidence": ds["confidence"],
                "price": float(candle["close"]),
            })
        except Exception:
            delta_signals.append({"direction": "hold", "confidence": 0.0, "price": float(candle["close"])})

    print(f"       EMA signals:       {len(ema_signals)}")
    print(f"       ML signals:        {len(ml_signals)}")
    print(f"       Market Structure:  {len(ms_signals)}")
    print(f"       Delta signals:     {len(delta_signals)}")

    # -- 4. Run backtests for EMA and ML (tradeable strategies) ---------------
    print("\n[4/5] Running full backtests for EMA and ML...")

    engine = BacktestEngine(initial_balance=1000.0, maker_fee=0.0002, taker_fee=0.0004, slippage_bps=0.5)

    t0 = time_module.time()
    ema_result = await engine.run(ema_strategy, test_data, symbol="BTCUSDT", leverage=1)
    ema_time = time_module.time() - t0
    print(f"       EMA backtest: {ema_time:.2f}s - {ema_result.total_trades} trades")

    t0 = time_module.time()
    ml_result = await engine.run(ml_engine, test_data, symbol="BTCUSDT", leverage=1)
    ml_time = time_module.time() - t0
    print(f"       ML backtest:  {ml_time:.2f}s - {ml_result.total_trades} trades")

    # -- 5. Evaluate signal accuracy for all 4 sources ------------------------
    print("\n[5/5] EVALUATION RESULTS")
    print("=" * 72)

    # Align to future_returns length (skip first few candles where signals may be hold)
    min_len = min(len(ema_signals), len(ml_signals), len(ms_signals), len(delta_signals), len(future_returns))
    start_offset = 10  # Skip initial candles for warmup

    eval_results = {}
    for name, sigs in [
        ("EMA Crossover", ema_signals),
        ("ML Random Forest", ml_signals),
        ("Market Structure", ms_signals),
        ("Delta Divergence", delta_signals),
    ]:
        sliced = sigs[start_offset:min_len]
        returns = future_returns[start_offset:min_len]
        # Count directional signals
        longs = sum(1 for s in sliced if s["direction"] == "long")
        shorts = sum(1 for s in sliced if s["direction"] == "short")
        holds = sum(1 for s in sliced if s["direction"] == "hold")
        avg_conf = np.mean([s["confidence"] for s in sliced]) if sliced else 0

        # Direction accuracy vs forward returns
        acc = direction_accuracy(sliced, returns, min_conf=0.4)
        eval_results[name] = {
            "longs": longs,
            "shorts": shorts,
            "holds": holds,
            "trade_pct": (longs + shorts) / len(sliced) * 100 if sliced else 0,
            "avg_confidence": round(avg_conf, 4),
            "direction_accuracy": round(acc["accuracy"], 4),
            "total_directional_trades": acc["total_trades"],
            "precision_long": round(acc["precision_long"], 4),
            "precision_short": round(acc["precision_short"], 4),
        }

    # Print signal analysis table
    print(f"\n  --- Signal Analysis (directional accuracy vs 6h forward return) ---")
    header = f"  {'Strategy':<20} {'Trades':>8} {'Long/Short':>12} {'WinDir%':>8} {'PrecL':>8} {'PrecS':>8} {'AvgConf':>8}"
    print(header)
    print("  " + "-" * 72)
    for name, ev in eval_results.items():
        trades_str = f"{ev['total_directional_trades']}"
        ls_str = f"{ev['longs']}/{ev['shorts']}"
        print(f"  {name:<20} {trades_str:>8} {ls_str:>12} {ev['direction_accuracy']*100:>7.1f}% {ev['precision_long']*100:>7.1f}% {ev['precision_short']*100:>7.1f}% {ev['avg_confidence']:>7.2f}")

    # Print backtest results table (EMA + ML only)
    print(f"\n  --- Backtest Results (EMA vs ML on identical test data) ---")
    for name, result in [("EMA Crossover", ema_result), ("ML Random Forest", ml_result)]:
        print(f"\n  {name}:")
        print(f"    Trades: {result.total_trades} | Win Rate: {result.win_rate:.1%}")
        print(f"    PnL: ${result.total_pnl:+.2f} ({result.total_pnl_pct:+.2%})")
        print(f"    Sharpe: {result.sharpe_ratio:.2f} | Profit Factor: {result.profit_factor:.2f}")
        print(f"    Max DD: {result.max_drawdown_pct:.2%} | Expectancy: ${result.expectancy:.4f}")

    # Overall comparison table for ALL 4
    print(f"\n  --- Complete 4-Strategy Scoreboard ---")
    metric_names = [
        ("Directional Accuracy", "direction_accuracy", "{:.1%}"),
        ("Precision (Long)", "precision_long", "{:.1%}"),
        ("Precision (Short)", "precision_short", "{:.1%}"),
        ("Avg Confidence", "avg_confidence", "{:.2f}"),
        ("Trade Frequency", "trade_pct", "{:.1f}%"),
    ]

    header2 = f"  {'Metric':<22} {'EMA':>10} {'ML':>10} {'Struct':>10} {'Delta':>10}"
    print(header2)
    print("  " + "-" * 62)

    scores = {}
    for name, key, fmt in metric_names:
        vals = [eval_results[s][key] for s in ["EMA Crossover", "ML Random Forest", "Market Structure", "Delta Divergence"]]
        row = f"  {name:<22}"
        for v in vals:
            row += f" {fmt.format(v):>10}"
        print(row)
        best_idx = np.argmax(vals) if key != "avg_confidence" else np.argmax(vals)
        winner = ["EMA", "ML", "MS", "Delta"][best_idx]
        scores[winner] = scores.get(winner, 0) + 1

    print("  " + "-" * 62)
    print(f"  {'SCOREBOARD':<22}")
    for label in ["EMA Crossover", "ML Random Forest", "Market Structure", "Delta Divergence"]:
        short = {"EMA Crossover": "EMA", "ML Random Forest": "ML", "Market Structure": "MS", "Delta Divergence": "Delta"}[label]
        print(f"    {label:<20} {scores.get(short, 0):>2}")

    # Final verdict
    winner_name = max(scores, key=scores.get)
    winner_label = {"EMA": "EMA Crossover", "ML": "ML Random Forest", "MS": "Market Structure", "Delta": "Delta Divergence"}[winner_name]
    print(f"\n  {'=' * 62}")
    print(f"  WINNER: {winner_label} ({scores[winner_name]}/{len(metric_names)} metrics)")
    print(f"  {'=' * 62}")

    # Backtest comparison
    print(f"\n  --- PnL Comparison ---")
    print(f"  EMA PnL: ${ema_result.total_pnl:+.2f}  Sharpe: {ema_result.sharpe_ratio:.2f}")
    print(f"  ML PnL:  ${ml_result.total_pnl:+.2f}  Sharpe: {ml_result.sharpe_ratio:.2f}")

    if ema_result.total_pnl > ml_result.total_pnl:
        print(f"  EMA wins on PnL by ${ema_result.total_pnl - ml_result.total_pnl:+.2f}")
    else:
        print(f"  ML wins on PnL by ${ml_result.total_pnl - ema_result.total_pnl:+.2f}")


if __name__ == "__main__":
    asyncio.run(main())
