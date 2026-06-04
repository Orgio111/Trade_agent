"""
QUANTEX Optuna Bayesian Optimization — Strategy Parameter Search

Uses Optuna's TPE sampler to find optimal strategy parameters
by backtesting on real Binance data. Runs 100+ trials efficiently.

Usage:
    python optuna_optimize.py
    python optuna_optimize.py --trials 200 --days 90
"""
import sys, os, asyncio, json, time, threading
import numpy as np
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(__file__))
from orchestrator.backtest import DataLoader, BacktestEngine
from orchestrator.strategy import StrategyEngine

# -- Config -------------------------------------------------
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--trials", type=int, default=100, help="Optuna trials")
parser.add_argument("--days", type=int, default=60, help="Days of Binance data")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
args = parser.parse_args()

# -- Fetch data ---------------------------------------------
async def fetch_data():
    print("=" * 60)
    print("  QUANTEX Optuna Bayesian Optimization")
    print(f"  {args.trials} trials on {args.days}d BTCUSDT 1h")
    print("=" * 60)

    end = datetime.now()
    start = end - timedelta(days=args.days)
    print(f"\n  Fetching data...")
    df = await DataLoader.from_binance_api(
        symbol="BTCUSDT", interval="1h", start_time=start, end_time=end
    )
    if df.empty:
        print("  [ERR] No data!")
        return None, None
    print(f"  [OK] {len(df)} candles ({df.index[0].strftime('%Y-%m-%d')} -> {df.index[-1].strftime('%Y-%m-%d')})")
    split = int(len(df) * 0.8)
    return df.iloc[:split].copy(), df.iloc[split:].copy()

# -- Objective (runs in thread, creates own event loop) -----
bt = BacktestEngine(initial_balance=100.0, taker_fee=0.0004, slippage_bps=0.5)

def _run_backtest(strat, data):
    """Run async backtest in a new event loop (thread-safe)."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(bt.run(strat, data))
    finally:
        loop.close()

def make_objective(train_data):
    """Create synchronous Optuna objective function (thread-safe).
    Returns just the objective function; benchmarks are computed inside thread."""
    def objective(trial):
        ema_fast = trial.suggest_int("ema_fast", 5, 20)
        ema_slow = trial.suggest_int("ema_slow", 20, 100)
        rsi_period = trial.suggest_int("rsi_period", 7, 21)
        rsi_overbought = trial.suggest_float("rsi_overbought", 60, 80)
        rsi_oversold = trial.suggest_float("rsi_oversold", 20, 40)
        atr_sl = trial.suggest_float("atr_sl", 1.0, 3.0)
        atr_tp = trial.suggest_float("atr_tp", 1.5, 5.0)

        strat = StrategyEngine(
            ema_fast=ema_fast, ema_slow=ema_slow,
            rsi_period=rsi_period,
            rsi_overbought=rsi_overbought, rsi_oversold=rsi_oversold,
            atr_multiplier_sl=atr_sl,
            atr_multiplier_tp1=atr_tp * 0.67,
            atr_multiplier_tp2=atr_tp,
        )

        result = _run_backtest(strat, train_data)

        if result.total_trades < 3:
            return -1.0

        pf = min(result.profit_factor if result.profit_factor != float("inf") else 3.0, 3.0)
        sharpe = max(result.sharpe_ratio, -2.0)
        score = sharpe * pf * (result.total_trades ** 0.3) / 10

        trial.set_user_attr("trades", result.total_trades)
        trial.set_user_attr("win_rate", round(result.win_rate, 4))
        trial.set_user_attr("pnl", round(result.total_pnl, 2))
        trial.set_user_attr("pf", round(result.profit_factor, 4))
        trial.set_user_attr("dd", round(result.max_drawdown_pct, 4))
        trial.set_user_attr("sharpe", round(result.sharpe_ratio, 4))

        return score

    return objective

# -- Run Optuna (in thread) ----------------------------------
def run_optuna_study(train_data, val_data, n_trials, seed):
    """Run Optuna optimization entirely in a thread (owns its event loops)."""
    import optuna
    from optuna.samplers import TPESampler

    # Compute defaults inside thread (clean event loop management)
    default = StrategyEngine()
    default_result = _run_backtest(default, train_data)
    default_val = _run_backtest(default, val_data)
    print(f"\n  Default (EMA 9/21): Train Sharpe={default_result.sharpe_ratio:.2f} "
          f"Val Sharpe={default_val.sharpe_ratio:.2f}")

    objective_fn = make_objective(train_data)

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=seed, n_startup_trials=10),
        study_name="quantex_strategy_opt",
    )
    study.optimize(objective_fn, n_trials=n_trials, show_progress_bar=False)
    return study, default_result, default_val

# -- Main ---------------------------------------------------
async def main():
    train_data, val_data = await fetch_data()
    if train_data is None:
        return

    print(f"\n  Running Optuna in thread ({args.trials} trials)...")
    print(f"  Each trial = 1 backtest on {len(train_data)} candles")
    print()

    study, default_result, default_val = await asyncio.to_thread(
        run_optuna_study, train_data, val_data, args.trials, args.seed
    )

    best = study.best_trial
    print(f"\n{'='*60}")
    print(f"  OPTUNA RESULTS — {args.trials} Trials")
    print(f"{'='*60}")
    print(f"  Best Score:      {best.value:.4f}")
    print(f"  Best Parameters:")
    for k, v in best.params.items():
        print(f"    {k}: {v}")
    print(f"\n  Trial Details:   {best.user_attrs['trades']} trades, "
          f"WR={best.user_attrs['win_rate']:.1%}, "
          f"PnL=${best.user_attrs['pnl']}, "
          f"PF={best.user_attrs['pf']}, "
          f"DD={best.user_attrs['dd']:.1%}")

    # Validate best params
    print(f"\n{'='*60}")
    print(f"  VALIDATION — Out of Sample ({len(val_data)} candles)")
    print(f"{'='*60}")

    best_strat = StrategyEngine(
        ema_fast=int(best.params["ema_fast"]),
        ema_slow=int(best.params["ema_slow"]),
        rsi_period=int(best.params["rsi_period"]),
        rsi_overbought=best.params["rsi_overbought"],
        rsi_oversold=best.params["rsi_oversold"],
        atr_multiplier_sl=best.params["atr_sl"],
        atr_multiplier_tp1=best.params["atr_tp"] * 0.67,
        atr_multiplier_tp2=best.params["atr_tp"],
    )
    bt = BacktestEngine(initial_balance=100.0)
    best_train = await bt.run(best_strat, train_data)
    best_val = await bt.run(best_strat, val_data)

    print(f"\n  {'Metric':<18} {'Default(train)':>15} {'Optuna(train)':>15} {'Optuna(val)':>15}")
    print(f"  {'-'*18} {'-'*15} {'-'*15} {'-'*15}")
    print(f"  {'Sharpe':<18} {default_result.sharpe_ratio:>15.4f} {best_train.sharpe_ratio:>15.4f} {best_val.sharpe_ratio:>15.4f}")
    print(f"  {'Profit Factor':<18} {default_result.profit_factor:>15.4f} {best_train.profit_factor:>15.4f} {best_val.profit_factor:>15.4f}")
    print(f"  {'Win Rate':<18} {default_result.win_rate:>15.2%} {best_train.win_rate:>15.2%} {best_val.win_rate:>15.2%}")
    print(f"  {'Total PnL':<18} {default_result.total_pnl:>15.2f} {best_train.total_pnl:>15.2f} {best_val.total_pnl:>15.2f}")
    print(f"  {'Max DD':<18} {default_result.max_drawdown_pct:>15.2%} {best_train.max_drawdown_pct:>15.2%} {best_val.max_drawdown_pct:>15.2%}")
    print(f"  {'Trades':<18} {default_result.total_trades:>15} {best_train.total_trades:>15} {best_val.total_trades:>15}")

    # Parameter importance
    print(f"\n{'='*60}")
    print(f"  PARAMETER IMPORTANCE")
    print(f"{'='*60}")
    try:
        importances = optuna.importance.get_param_importances(study)
        for k, v in sorted(importances.items(), key=lambda x: -x[1]):
            print(f"  {k:15s}: {v:.4f}")
    except Exception:
        pass

    # Top 5 trials
    print(f"\n{'='*60}")
    print(f"  TOP 5 TRIALS")
    print(f"{'='*60}")
    for i, t in enumerate(study.trials[:5]):
        if t.value is None:
            continue
        print(f"  #{i+1}: score={t.value:.4f} "
              f"ema=({t.params['ema_fast']},{t.params['ema_slow']}) "
              f"rsi={t.params['rsi_period']} "
              f"sl={t.params['atr_sl']:.2f} tp={t.params['atr_tp']:.2f} "
              f"trades={t.user_attrs.get('trades', '?')}")

    print(f"\n  [OK] Optimization complete!")
    print(f"  Best params: {json.dumps(best.params, indent=4)}")

if __name__ == "__main__":
    asyncio.run(main())
