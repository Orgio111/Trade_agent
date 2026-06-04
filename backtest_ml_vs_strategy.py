"""
QUANTEX Combined Backtest — ML vs Strategy vs Hybrid

Compares three approaches on real Binance data:
  1. ML Signal Engine (Random Forest)
  2. Rule-based Strategy (EMA crossover + RSI)
  3. Combined: ML filters strategy signals (ML confidence > 0.5)
"""
import sys, os, asyncio, time
from datetime import datetime, timedelta
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from orchestrator.backtest import DataLoader, BacktestEngine, BacktestResult
from orchestrator.strategy import StrategyEngine, Signal
from orchestrator.ml_signals import MLSignalEngine

DAYS = 60
INTERVAL = "1h"

def fmt_result(label: str, r: BacktestResult):
    print(f"\n  [{label}]")
    print(f"  {'='*50}")
    print(f"  Sharpe:         {r.sharpe_ratio:.4f}")
    print(f"  Sortino:        {r.sortino_ratio:.4f}")
    print(f"  Profit Factor:  {r.profit_factor:.4f}")
    print(f"  Win Rate:       {r.win_rate:.2%} ({r.winning_trades}W/{r.losing_trades}L)")
    pf_str = f"{r.profit_factor:.4f}" if r.profit_factor != float("inf") else "INF"
    print(f"  Total PnL:      ${r.total_pnl:+.2f}")
    print(f"  Max Drawdown:   {r.max_drawdown_pct:.2%}")
    print(f"  Calmar Ratio:   {r.calmar_ratio:.4f}")
    print(f"  Expectancy:     ${r.expectancy:.4f}")
    print(f"  Total Trades:   {r.total_trades}")
    print(f"  Avg Win:        ${r.avg_win:.4f}")
    print(f"  Avg Loss:       ${r.avg_loss:.4f}")

async def main():
    print("=" * 60)
    print("  QUANTEX Backtest — ML vs Strategy vs Hybrid")
    print("=" * 60)

    # 1. Fetch data
    end = datetime.now()
    start = end - timedelta(days=DAYS)
    print(f"\n  Fetching {DAYS}d BTCUSDT {INTERVAL} from Binance...")
    df = await DataLoader.from_binance_api(symbol="BTCUSDT", interval=INTERVAL, start_time=start, end_time=end)
    if df.empty:
        print("  [ERR] No data!")
        return
    print(f"  [OK] {len(df)} candles ({df.index[0].strftime('%Y-%m-%d')} -> {df.index[-1].strftime('%Y-%m-%d')})")
    print(f"  Price: ${df.close.iloc[-1]:.2f}")

    split = int(len(df) * 0.8)
    train_df, test_df = df.iloc[:split].copy(), df.iloc[split:].copy()

    # 2. Train/load ML model
    print(f"\n{'='*60}")
    print(f"  TRAINING ML MODEL")
    print(f"{'='*60}")
    ml = MLSignalEngine(min_training_samples=100, lookahead_periods=12)
    loaded = ml.load_model()
    if loaded:
        print(f"  [OK] Model loaded from disk (trained {ml._train_count}x)")
    else:
        print(f"  Training on {len(train_df)} candles...")
        t0 = time.time()
        result = ml.train(train_df, force=True)
        t1 = time.time()
        if result.get("status") == "trained":
            print(f"  [OK] Train acc={result['train_accuracy']:.2%} Test acc={result['test_accuracy']:.2%} ({t1-t0:.1f}s)")
        else:
            print(f"  [WARN] {result}")

    # 3. Run backtests
    print(f"\n{'='*60}")
    print(f"  BACKTEST ON TEST DATA ({len(test_df)} candles)")
    print(f"{'='*60}")

    bt = BacktestEngine(initial_balance=100.0, taker_fee=0.0004, slippage_bps=0.5)
    strat = StrategyEngine()

    # A) Strategy only (EMA 9/21 + RSI)
    print(f"\n  Running StrategyEngine backtest...")
    r_strat = await bt.run(strat, test_df)
    fmt_result("Strategy Only (EMA 9/21 + RSI)", r_strat)

    # B) ML only
    print(f"\n  Running ML Signal backtest...")
    r_ml = await bt.run(ml, test_df)
    fmt_result("ML Only (Random Forest)", r_ml)

    # C) Combined: ML confidence modifies strategy signal
    print(f"\n  Running Combined backtest...")

    class HybridStrategy:
        """Strategy + ML: use ML confidence to filter strategy signals."""
        def __init__(self, base_strategy, ml_engine):
            self.base = base_strategy
            self.ml = ml_engine

        def generate_signal(self, df):
            # Get base strategy signal
            base_sig = self.base.generate_signal(df)
            if base_sig.direction == "hold":
                return base_sig

            # Get ML prediction
            ml_pred = self.ml.predict(df)
            ml_direction = "long" if ml_pred["prob_buy"] > ml_pred["prob_sell"] else "short"
            ml_confidence = max(ml_pred["prob_buy"], ml_pred["prob_sell"])

            # If ML agrees with strategy, boost confidence
            if ml_direction == base_sig.direction and ml_confidence > 0.5:
                base_sig.confidence = min(1.0, base_sig.confidence + ml_confidence * 0.3)
                base_sig.reason += f" | ML confirms ({ml_confidence:.0%})"
                return base_sig

            # If ML disagrees or low confidence, reduce
            if ml_direction != base_sig.direction:
                base_sig.confidence = max(0.0, base_sig.confidence - ml_confidence * 0.3)
                base_sig.reason += f" | ML disagrees ({ml_confidence:.0%})"

            return base_sig

    hybrid = HybridStrategy(strat, ml)
    r_hybrid = await bt.run(hybrid, test_df)
    fmt_result("Combined (Strategy + ML filter)", r_hybrid)

    # 4. Comparison table
    print(f"\n{'='*60}")
    print(f"  COMPARISON TABLE")
    print(f"{'='*60}")
    print(f"  {'Metric':<20} {'Strategy':>10} {'ML':>10} {'Combined':>10}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10}")
    print(f"  {'Sharpe':<20} {r_strat.sharpe_ratio:>10.4f} {r_ml.sharpe_ratio:>10.4f} {r_hybrid.sharpe_ratio:>10.4f}")
    print(f"  {'Profit Factor':<20} {r_strat.profit_factor:>10.4f} {r_ml.profit_factor:>10.4f} {r_hybrid.profit_factor:>10.4f}")
    print(f"  {'Win Rate':<20} {r_strat.win_rate:>10.2%} {r_ml.win_rate:>10.2%} {r_hybrid.win_rate:>10.2%}")
    print(f"  {'Total PnL':<20} {r_strat.total_pnl:>10.2f} {r_ml.total_pnl:>10.2f} {r_hybrid.total_pnl:>10.2f}")
    print(f"  {'Max DD':<20} {r_strat.max_drawdown_pct:>10.2%} {r_ml.max_drawdown_pct:>10.2%} {r_hybrid.max_drawdown_pct:>10.2%}")
    print(f"  {'Calmar':<20} {r_strat.calmar_ratio:>10.4f} {r_ml.calmar_ratio:>10.4f} {r_hybrid.calmar_ratio:>10.4f}")
    print(f"  {'Expectancy':<20} {r_strat.expectancy:>10.4f} {r_ml.expectancy:>10.4f} {r_hybrid.expectancy:>10.4f}")
    print(f"  {'Trades':<20} {r_strat.total_trades:>10} {r_ml.total_trades:>10} {r_hybrid.total_trades:>10}")
    print(f"  {'Avg Win':<20} {r_strat.avg_win:>10.4f} {r_ml.avg_win:>10.4f} {r_hybrid.avg_win:>10.4f}")
    print(f"  {'Avg Loss':<20} {r_strat.avg_loss:>10.4f} {r_ml.avg_loss:>10.4f} {r_hybrid.avg_loss:>10.4f}")

    # Winner
    winners = {
        "Sharpe": max(r_strat.sharpe_ratio, r_ml.sharpe_ratio, r_hybrid.sharpe_ratio),
        "Profit Factor": max(r_strat.profit_factor, r_ml.profit_factor, r_hybrid.profit_factor),
        "Win Rate": max(r_strat.win_rate, r_ml.win_rate, r_hybrid.win_rate),
        "PnL": max(r_strat.total_pnl, r_ml.total_pnl, r_hybrid.total_pnl),
        "DD (min)": min(r_strat.max_drawdown_pct, r_ml.max_drawdown_pct, r_hybrid.max_drawdown_pct),
    }
    print(f"\n  {'='*60}")
    print(f"  BEST IN CATEGORY:")
    print(f"  {'='*60}")
    for metric, val in winners.items():
        best = max if metric != "DD (min)" else min
        if metric == "Sharpe":
            src = "Strategy" if val == r_strat.sharpe_ratio else ("ML" if val == r_ml.sharpe_ratio else "Combined")
        elif metric == "Profit Factor":
            src = "Strategy" if val == r_strat.profit_factor else ("ML" if val == r_ml.profit_factor else "Combined")
        elif metric == "Win Rate":
            src = "Strategy" if val == r_strat.win_rate else ("ML" if val == r_ml.win_rate else "Combined")
        elif metric == "PnL":
            src = "Strategy" if val == r_strat.total_pnl else ("ML" if val == r_ml.total_pnl else "Combined")
        else:
            src = "Strategy" if val == r_strat.max_drawdown_pct else ("ML" if val == r_ml.max_drawdown_pct else "Combined")
        print(f"  {metric:<20}: {val:>8.4f} ({src})")

    print(f"\n  [DONE] Backtest complete!")

if __name__ == "__main__":
    asyncio.run(main())
