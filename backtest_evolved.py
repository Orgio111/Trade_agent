"""
QUANTEX Backtest with Evolved Strategy Parameters

Uses the best parameters from genetic evolution:
  ema_fast=11, ema_slow=92, rsi_period=14
  rsi_overbought=62, rsi_oversold=37
  atr_sl=1.63x, atr_tp=2.84x (tp1=1.90x, tp2=2.84x)
  min_confidence=0.41

Compares with default: ema_fast=9, ema_slow=21
"""
import sys, os, asyncio
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(__file__))
from orchestrator.backtest import DataLoader, BacktestEngine
from orchestrator.strategy import StrategyEngine

# -- Evolved params (from genetic evolution) ---------------
EVOLVED = {
    "ema_fast": 11,
    "ema_slow": 92,
    "rsi_period": 14,
    "rsi_overbought": 62,
    "rsi_oversold": 37,
    "atr_sl": 1.63,
    "atr_tp": 2.84,
    "min_confidence": 0.41,
}

async def run():
    # Fetch real Binance data
    print("=" * 60)
    print("  QUANTEX Backtest — Evolved vs Default Strategy")
    print("=" * 60)

    days = 60
    end = datetime.now()
    start = end - timedelta(days=days)

    print(f"\n Fetching {days}d BTCUSDT 1h from Binance...")
    df = await DataLoader.from_binance_api(
        symbol="BTCUSDT", interval="1h", start_time=start, end_time=end
    )
    if df.empty:
        print(" [ERR] No data!")
        return
    print(f" [OK] {len(df)} candles ({df.index[0].strftime('%Y-%m-%d')} -> {df.index[-1].strftime('%Y-%m-%d')})")
    print(f" Price range: ${df.close.min():.2f} -> ${df.close.max():.2f}")

    # 80/20 split
    split = int(len(df) * 0.8)
    train_df, val_df = df.iloc[:split].copy(), df.iloc[split:].copy()

    bt = BacktestEngine(initial_balance=100.0, taker_fee=0.0004, slippage_bps=0.5)

    # -- Evolved strategy --
    evolved_strat = StrategyEngine(
        ema_fast=EVOLVED["ema_fast"],
        ema_slow=EVOLVED["ema_slow"],
        rsi_period=EVOLVED["rsi_period"],
        rsi_overbought=EVOLVED["rsi_overbought"],
        rsi_oversold=EVOLVED["rsi_oversold"],
        atr_multiplier_sl=EVOLVED["atr_sl"],
        atr_multiplier_tp1=EVOLVED["atr_tp"] * 0.67,
        atr_multiplier_tp2=EVOLVED["atr_tp"],
    )
    # -- Default strategy --
    default_strat = StrategyEngine()

    def fmt_result(label, result):
        print(f"\n  [{label}]")
        print(f"  {'='*50}")
        print(f"  Sharpe:         {result.sharpe_ratio:.4f}")
        print(f"  Sortino:        {result.sortino_ratio:.4f}")
        print(f"  Profit Factor:  {result.profit_factor:.4f}")
        print(f"  Win Rate:       {result.win_rate:.2%} ({result.winning_trades}W/{result.losing_trades}L)")
        print(f"  Total PnL:      ${result.total_pnl:+.4f} ({result.total_pnl_pct:+.2%})")
        print(f"  Max Drawdown:   {result.max_drawdown_pct:.2%}")
        print(f"  Calmar Ratio:   {result.calmar_ratio:.4f}")
        print(f"  Expectancy:     ${result.expectancy:.4f}")
        print(f"  Total Trades:   {result.total_trades}")
        print(f"  Avg Win:        ${result.avg_win:.4f}")
        print(f"  Avg Loss:       ${result.avg_loss:.4f}")
        if result.total_trades > 0:
            print(f"  Avg Duration:   {result.avg_trade_duration}")

    print(f"\n{'='*60}")
    print(f"  TRAINING DATA ({len(train_df)} candles)")
    print(f"{'='*60}")

    for label, strat in [("Default  EMA 9/21", default_strat), ("Evolved  EMA 11/92", evolved_strat)]:
        r = await bt.run(strat, train_df)
        fmt_result(label, r)

    print(f"\n{'='*60}")
    print(f"  VALIDATION DATA ({len(val_df)} candles) — Out of Sample")
    print(f"{'='*60}")

    val_results = {}
    for label, strat in [("Default", default_strat), ("Evolved", evolved_strat)]:
        r = await bt.run(strat, val_df)
        fmt_result(label, r)
        val_results[label] = r

    # -- Summary table --
    print(f"\n{'='*60}")
    print(f"  COMPARISON TABLE (Validation Data)")
    print(f"{'='*60}")
    d = val_results["Default"]
    e = val_results["Evolved"]
    print(f"  {'Metric':<18} {'Default':>12} {'Evolved':>12} {'Change':>12}")
    print(f"  {'-'*18} {'-'*12} {'-'*12} {'-'*12}")
    print(f"  {'Sharpe':<18} {d.sharpe_ratio:>12.4f} {e.sharpe_ratio:>12.4f} {e.sharpe_ratio-d.sharpe_ratio:>+12.4f}")
    print(f"  {'Profit Factor':<18} {d.profit_factor:>12.4f} {e.profit_factor:>12.4f} {e.profit_factor-d.profit_factor:>+12.4f}")
    print(f"  {'Win Rate':<18} {d.win_rate:>12.2%} {e.win_rate:>12.2%} {e.win_rate-d.win_rate:>+12.2%}")
    print(f"  {'Total PnL':<18} {d.total_pnl:>12.4f} {e.total_pnl:>12.4f} {e.total_pnl-d.total_pnl:>+12.4f}")
    print(f"  {'Max DD':<18} {d.max_drawdown_pct:>12.2%} {e.max_drawdown_pct:>12.2%} {e.max_drawdown_pct-d.max_drawdown_pct:>+12.2%}")
    print(f"  {'Trades':<18} {d.total_trades:>12} {e.total_trades:>12} {e.total_trades-d.total_trades:>+12}")
    print(f"  {'Expectancy':<18} {d.expectancy:>12.4f} {e.expectancy:>12.4f} {e.expectancy-d.expectancy:>+12.4f}")

    # -- Equity curve (evolved, val) --
    if e.equity_curve:
        print(f"\n  Equity Curve (Evolved, Validation):")
        step = max(1, len(e.equity_curve) // 20)
        for i in range(0, len(e.equity_curve), step):
            bar = int((e.equity_curve[i] / max(e.equity_curve) - min(e.equity_curve)/max(e.equity_curve)) * 40) if max(e.equity_curve) > 0 else 0
            bar = max(1, min(40, bar))
            print(f"  {i:4d}: ${e.equity_curve[i]:>8.2f} {'#' * bar}")

    print(f"\n  Evolved parameters used:")
    for k, v in EVOLVED.items():
        print(f"    {k}: {v}")

if __name__ == "__main__":
    asyncio.run(run())
