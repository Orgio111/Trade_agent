"""
QUANTEX Strategy Comparison: EMA Crossover vs ML Random Forest
Uses REAL Binance historical data instead of mock data.

Fetches 120 days of BTCUSDT hourly candles from Binance.
ML model is trained on first 70%, both strategies tested on last 30%.
"""

import asyncio
import sys
import time as time_module
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

from orchestrator.backtest import BacktestEngine, DataLoader, BacktestResult
from orchestrator.strategy import StrategyEngine
from orchestrator.ml_signals import MLSignalEngine


async def main():
    print("=" * 70)
    print("  QUANTEX Strategy Comparison: EMA Crossover vs ML Random Forest")
    print("  Source: REAL Binance historical data (BTCUSDT 1h)")
    print("=" * 70)

    # -- 1. Fetch real data from Binance --------------------------------
    print("\n[1/4] Fetching real BTCUSDT 1h data from Binance API...")
    end = datetime.now()
    start = end - timedelta(days=120)

    try:
        data = await DataLoader.from_binance_api(
            symbol="BTCUSDT",
            interval="1h",
            start_time=start,
            end_time=end,
            limit=1500,
        )
        using_mock = False
        print(f"       OK - Fetched {len(data)} real hourly candles")
    except Exception as e:
        print(f"       ERROR fetching real data: {e}")
        print("       Falling back to mock data...")
        data = DataLoader.generate_mock_data(
            periods=2880,
            start_price=50000.0,
            volatility=0.015,
            seed=42,
        )
        using_mock = True

    print(f"       Date range: {data.index[0]} -> {data.index[-1]}")
    print(f"       Price range: ${data['low'].min():.2f} -> ${data['high'].max():.2f}")
    print(f"       Source: {'REAL BINANCE DATA' if not using_mock else 'MOCK DATA (fallback)'}")

    # Split: train on first 70%, test on last 30%
    split_idx = int(len(data) * 0.70)
    train_data = data.iloc[:split_idx]
    test_data = data.iloc[split_idx:]
    print(f"       Training data:  {len(train_data)} candles ({data.index[0]} -> {data.index[split_idx-1]})")
    print(f"       Test data:      {len(test_data)} candles ({data.index[split_idx]} -> {data.index[-1]})")

    # -- 2. Train ML model -----------------------------------------------
    print("\n[2/4] Training ML Random Forest model on REAL data...")
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
        print(f"       OK - Model trained in {elapsed:.2f}s")
        print(f"       Train accuracy:  {train_result.get('train_accuracy', 0)*100:.1f}%")
        print(f"       Test accuracy:   {train_result.get('test_accuracy', 0)*100:.1f}%")
        print(f"       Features used:   {train_result.get('features', 0)}")
        print(f"       Training samples: {train_result.get('train_samples', 0)}")
        print(f"       Test samples:     {train_result.get('test_samples', 0)}")
    else:
        print(f"       WARNING - Model NOT trained: {train_result.get('reason', 'unknown')}")

    # -- 3. Run backtests ------------------------------------------------
    print("\n[3/4] Running backtests on REAL test data...")

    engine = BacktestEngine(
        initial_balance=1000.0,
        maker_fee=0.0002,
        taker_fee=0.0004,
        slippage_bps=0.5,
    )

    # EMA Crossover strategy
    ema_strategy = StrategyEngine(
        ema_fast=9,
        ema_slow=21,
        rsi_period=14,
        atr_multiplier_sl=1.5,
        atr_multiplier_tp1=2.0,
        atr_multiplier_tp2=3.5,
    )

    t0 = time_module.time()
    ema_result = await engine.run(ema_strategy, test_data, symbol="BTCUSDT", leverage=1)
    ema_time = time_module.time() - t0
    print(f"       OK - EMA strategy completed in {ema_time:.2f}s - {ema_result.total_trades} trades")

    # ML Random Forest strategy
    t0 = time_module.time()
    ml_result = await engine.run(ml_engine, test_data, symbol="BTCUSDT", leverage=1)
    ml_time = time_module.time() - t0
    print(f"       OK - ML strategy completed in {ml_time:.2f}s - {ml_result.total_trades} trades")

    # -- 4. Compare results ----------------------------------------------
    print("\n[4/4] COMPARISON RESULTS - REAL BINANCE DATA")
    print("=" * 70)

    metrics = [
        ("Total Trades",      "total_trades",      "{:.0f}"),
        ("Winning Trades",    "winning_trades",    "{:.0f}"),
        ("Losing Trades",     "losing_trades",     "{:.0f}"),
        ("Win Rate",           "win_rate",           "{:.1%}"),
        ("Total PnL",         "total_pnl",          "${:+.2f}"),
        ("Total PnL %",       "total_pnl_pct",      "{:+.2%}"),
        ("Max Drawdown",      "max_drawdown_pct",   "{:.2%}"),
        ("Sharpe Ratio",      "sharpe_ratio",       "{:.2f}"),
        ("Sortino Ratio",     "sortino_ratio",      "{:.2f}"),
        ("Calmar Ratio",      "calmar_ratio",       "{:.2f}"),
        ("Profit Factor",     "profit_factor",      "{:.2f}"),
        ("Avg Win",           "avg_win",            "${:+.2f}"),
        ("Avg Loss",          "avg_loss",           "${:+.2f}"),
        ("Expectancy",        "expectancy",         "${:+.2f}"),
    ]

    header_str = f"{'Metric':<22} {'EMA Crossover':>18} {'ML Random Forest':>18}  Winner"
    sep = "-" * 70
    print(f"\n{header_str}")
    print(sep)

    ema_wins = 0
    ml_wins = 0
    ties = 0

    for name, attr, fmt in metrics:
        e_val = getattr(ema_result, attr)
        m_val = getattr(ml_result, attr)
        e_str = fmt.format(e_val)
        m_str = fmt.format(m_val)

        if attr == "max_drawdown_pct":
            better = "ML" if m_val < e_val else "EMA" if e_val < m_val else "TIE"
        elif attr in ("total_fees",):
            better = "ML" if m_val < e_val else "EMA" if e_val < m_val else "TIE"
        else:
            better = "ML" if m_val > e_val else "EMA" if e_val > m_val else "TIE"

        if better == "ML":
            ml_wins += 1
        elif better == "EMA":
            ema_wins += 1
        else:
            ties += 1

        print(f"  {name:<20} {e_str:>18} {m_str:>18}  {better}")

    print(sep)
    print(f"{'SCOREBOARD':<22}")
    print(f"  {'EMA Crossover wins':<24} {ema_wins:>2}")
    print(f"  {'ML Random Forest wins':<24} {ml_wins:>2}")
    print(f"  {'Ties':<24} {ties:>2}")

    # Overall verdict
    print(f"\n{'=' * 70}")
    print("  VERDICT:", end=" ")
    if ema_wins > ml_wins * 1.5:
        print("EMA Crossover is the clear winner on REAL data")
    elif ml_wins > ema_wins * 1.5:
        print("ML Random Forest is the clear winner on REAL data")
    elif ema_wins > ml_wins:
        print("EMA Crossover edges out ML on REAL data - but it's close")
    elif ml_wins > ema_wins:
        print("ML Random Forest edges out EMA on REAL data - but it's close")
    else:
        print("Both strategies are evenly matched on REAL data")
    print(f"{'=' * 70}")

    # Key takeaways
    print(f"\n-- Key Takeaways (REAL Binance Data) --")
    print(f"  Data period: {data.index[0].strftime('%Y-%m-%d')} -> {data.index[-1].strftime('%Y-%m-%d')}")
    print(f"  BTC range:   ${data['low'].min():.2f} -> ${data['high'].max():.2f}")
    print(f"  Test period: {test_data.index[0].strftime('%Y-%m-%d')} -> {test_data.index[-1].strftime('%Y-%m-%d')}")
    print(f"")
    print(f"  EMA:")
    print(f"    Sharpe: {ema_result.sharpe_ratio:.2f} | Profit Factor: {ema_result.profit_factor:.2f} | Win Rate: {ema_result.win_rate:.1%}")
    print(f"    PnL: ${ema_result.total_pnl:+.2f} | Expectancy: ${ema_result.expectancy:.4f} | Max DD: {ema_result.max_drawdown_pct:.2%}")
    print(f"")
    print(f"  ML Random Forest:")
    print(f"    Sharpe: {ml_result.sharpe_ratio:.2f} | Profit Factor: {ml_result.profit_factor:.2f} | Win Rate: {ml_result.win_rate:.1%}")
    print(f"    PnL: ${ml_result.total_pnl:+.2f} | Expectancy: ${ml_result.expectancy:.4f} | Max DD: {ml_result.max_drawdown_pct:.2%}")
    print(f"")
    if ml_engine._model:
        print(f"  ML Model stats:")
        print(f"    Trained: {ml_engine._train_count}x, last: {ml_engine._last_train_time}")
        print(f"    Train accuracy: {train_result.get('train_accuracy', 0)*100:.1f}%")
        print(f"    Test accuracy:  {train_result.get('test_accuracy', 0)*100:.1f}%")
        print(f"    Features: {train_result.get('features', 0)}")

    # Recommendations
    print(f"\n-- Recommendation --")
    if ema_wins >= ml_wins:
        print(f"  Use EMA Crossover as the primary strategy.")
        print(f"  It won {ema_wins}/{ema_wins+ml_wins+ties} metrics on real data.")
        if ml_wins > 0:
            print(f"  Consider using ML signals as a secondary confirmation filter.")
    else:
        print(f"  Use ML Random Forest as the primary strategy.")
        print(f"  It won {ml_wins}/{ema_wins+ml_wins+ties} metrics on real data.")
        if ema_wins > 0:
            print(f"  Consider using EMA crossover as a fallback or confirmation.")

    return ema_result, ml_result, data


if __name__ == "__main__":
    asyncio.run(main())
