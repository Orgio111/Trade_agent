"""
QUANTEX Backtest — ML vs PPO vs Combined (ML + PPO filter)

Compares four approaches on real Binance data:
  1. ML Signal Engine (Random Forest)
  2. PPO Agent (trained RL model)
  3. Combined: ML signal filters PPO actions (only trade when ML confident)
  4. Baseline: Buy & Hold
"""
import sys, os, asyncio, time, json
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from orchestrator.backtest import DataLoader, BacktestEngine, BacktestResult
from orchestrator.strategy import StrategyEngine, Signal
from orchestrator.ml_signals import MLSignalEngine
from orchestrator.rl.gym_env import GymTradingEnv
from stable_baselines3 import PPO

PROJECT_ROOT = Path(__file__).parent

def fmt_result(label: str, r: dict):
    print(f"  [{label}]")
    print(f"  {'='*50}")
    print(f"  Total PnL:      ${r['pnl']:+.2f}")
    print(f"  Total Trades:   {r['trades']}")
    print(f"  Win Rate:       {r['win_rate']:.2%} ({r['wins']}W/{r['losses']}L)")
    print(f"  Profit Factor:  {r.get('profit_factor', 0):.4f}")
    print(f"  Sharpe:         {r.get('sharpe', 0):.4f}")
    print(f"  Max Drawdown:   {r.get('max_dd', 0):.2%}")
    print(f"  Avg Trade PnL:  ${r.get('avg_trade', 0):.4f}")
    print()

def compute_metrics(trades: list, initial_balance: float = 100.0) -> dict:
    """Compute trading metrics from a list of trade dicts."""
    if not trades:
        return {"pnl": 0, "trades": 0, "win_rate": 0, "wins": 0, "losses": 0,
                "profit_factor": 0, "sharpe": 0, "max_dd": 0, "avg_trade": 0}

    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    total_pnl = sum(pnls)
    win_rate = len(wins) / len(pnls) if pnls else 0
    profit_factor = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else (float("inf") if wins else 0)

    # Cumulative PnL for Sharpe and drawdown
    cum_pnl = np.cumsum(pnls)
    equity = initial_balance + cum_pnl
    sharpe = (np.mean(pnls) / (np.std(pnls) + 1e-10)) * np.sqrt(252) if len(pnls) > 1 else 0
    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / peak
    max_dd = np.max(dd) if len(dd) > 0 else 0

    return {
        "pnl": round(total_pnl, 4),
        "trades": len(trades),
        "win_rate": round(win_rate, 4),
        "wins": len(wins),
        "losses": len(losses),
        "profit_factor": round(profit_factor, 4),
        "sharpe": round(sharpe, 4),
        "max_dd": round(max_dd, 4),
        "avg_trade": round(np.mean(pnls), 4),
    }

async def run_ppo_backtest(val_df: pd.DataFrame, model) -> list:
    """Run PPO agent on validation data and return list of trade dicts."""
    env = GymTradingEnv(data=val_df, initial_balance=100.0, leverage=1, lookback=50)
    obs, _ = env.reset()
    done = False

    while not done:
        action_raw, _ = model.predict(obs, deterministic=True)
        if isinstance(action_raw, np.ndarray) and action_raw.ndim >= 1:
            action_val = int(action_raw[0])
        else:
            action_val = int(action_raw)
        obs, reward, terminated, truncated, info = env.step(action_val)
        done = terminated or truncated

    # Clone trades to add exit_price from info
    trades = []
    for t in env.trades:
        trades.append({
            "entry_price": t["entry_price"],
            "exit_price": t["exit_price"],
            "pnl": t["pnl"],
            "side": t["side"],
        })
    env.close()
    return trades

async def run_ml_backtest(ml: MLSignalEngine, val_df: pd.DataFrame) -> list:
    """Run ML signal engine on validation data and return list of trade dicts."""
    bt = BacktestEngine(initial_balance=100.0, taker_fee=0.0004, slippage_bps=0.5)
    result = await bt.run(ml, val_df)
    trades = []
    for t in result.trades:
        if hasattr(t, "entry_price"):
            trades.append({
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "pnl": t.pnl,
                "side": t.direction,
            })
        elif isinstance(t, dict):
            trades.append({
                "entry_price": t.get("entry_price", 0),
                "exit_price": t.get("exit_price", 0),
                "pnl": t.get("pnl", 0),
                "side": t.get("side", "unknown"),
            })
    return trades

async def main():
    print("=" * 60)
    print("  QUANTEX Backtest — ML vs PPO vs Combined")
    print("=" * 60)

    # 1. Fetch data
    end = datetime.now()
    start = end - timedelta(days=90)
    print(f"\n  Fetching 90d BTCUSDT 1h from Binance...")
    df = await DataLoader.from_binance_api(symbol="BTCUSDT", interval="1h", start_time=start, end_time=end)
    if df.empty:
        print("  [ERR] No data!")
        return
    split = int(len(df) * 0.8)
    train_df = df.iloc[:split].copy()
    val_df = df.iloc[split:].copy()
    print(f"  [OK] {len(df)} candles (train={len(train_df)}, val={len(val_df)})")
    print(f"  Price: ${df.close.iloc[-1]:.2f}")

    # 2. Load ML model
    print(f"\n{'='*60}")
    print(f"  ML SIGNAL ENGINE")
    print(f"{'='*60}")
    ml = MLSignalEngine(min_training_samples=100, lookahead_periods=12)
    loaded = ml.load_model()
    if loaded:
        print(f"  [OK] Model loaded (trained {ml._train_count}x)")
    else:
        print(f"  Training on {len(train_df)} candles...")
        result = ml.train(train_df, force=True)
        status = result.get("status", "unknown")
        print(f"  [{status.upper()}] train_acc={result.get('train_accuracy', 0):.2%} test_acc={result.get('test_accuracy', 0):.2%}")

    # 3. Train / load PPO model
    print(f"\n{'='*60}")
    print(f"  PPO AGENT")
    print(f"{'='*60}")
    ppo_model = None
    # Try loading from disk first
    model_path = PROJECT_ROOT / "models" / "rl" / "ppo_trading_latest"
    zip_path = str(model_path) + ".zip"
    if os.path.exists(zip_path):
        try:
            ppo_model = PPO.load(str(model_path))
            print(f"  [OK] PPO model loaded from disk")
        except Exception as e:
            print(f"  [WARN] Disk model corrupt: {e}")
    # If not available, train a fresh one (quick, 10k steps)
    if ppo_model is None:
        print(f"  Training fresh PPO model on {len(train_df)} candles...")
        from stable_baselines3.common.vec_env import DummyVecEnv
        train_env = GymTradingEnv(data=train_df, initial_balance=100.0, leverage=1, lookback=50)
        vec_env = DummyVecEnv([lambda: train_env])
        ppo_model = PPO("MlpPolicy", vec_env, learning_rate=3e-4, n_steps=2048,
                        batch_size=64, n_epochs=10, ent_coef=0.05, verbose=0, seed=42)
        ppo_model.learn(total_timesteps=50000, progress_bar=True)
        train_env.close()
        print(f"  [OK] PPO trained on {len(train_df)} candles")

    # 4. Run backtests
    print(f"\n{'='*60}")
    print(f"  BACKTEST ON VALIDATION DATA ({len(val_df)} candles)")
    print(f"{'='*60}")

    results = {}

    # A) ML only
    print(f"\n  Running ML backtest...")
    ml_trades = await run_ml_backtest(ml, val_df)
    results["ML (Random Forest)"] = compute_metrics(ml_trades, initial_balance=100.0)
    fmt_result("ML (Random Forest)", results["ML (Random Forest)"])

    # B) PPO only
    if ppo_model:
        print(f"  Running PPO backtest...")
        ppo_trades = await run_ppo_backtest(val_df, ppo_model)
        results["PPO (trained RL)"] = compute_metrics(ppo_trades, initial_balance=100.0)
        fmt_result("PPO (trained RL)", results["PPO (trained RL)"])

        # C) Combined: ML + PPO (use ML signal to override PPO when confident)
        print(f"  Running Combined (ML + PPO) backtest...")
        combined_trades = []

        # Run PPO step by step, but use ML signal to override
        env = GymTradingEnv(data=val_df, initial_balance=100.0, leverage=1, lookback=50)
        obs, _ = env.reset()
        done = False
        step = 0

        while not done:
            # Get PPO action
            action_raw, _ = ppo_model.predict(obs, deterministic=True)
            if isinstance(action_raw, np.ndarray) and action_raw.ndim >= 1:
                ppo_action = int(action_raw[0])
            else:
                ppo_action = int(action_raw)

            # Get ML signal for current candle
            current_df = val_df.iloc[:min(step + 50 + 1, len(val_df))]
            if len(current_df) >= 50:
                ml_pred = ml.predict(current_df)
                ml_direction = ml_pred.get("direction", "hold")
                ml_conf = ml_pred.get("confidence", 0)

                # Override: if ML confident, force action to match ML direction
                if ml_conf > 0.6 and ml_direction != "hold":
                    # Map ML direction to PPO action
                    if ml_direction == "long":
                        ppo_action = 1  # LONG
                    elif ml_direction == "short":
                        ppo_action = 2  # SHORT
                # If ML says hold and PPO wants to trade, let PPO decide
                # (no override needed)

            obs, reward, terminated, truncated, info = env.step(ppo_action)
            done = terminated or truncated
            step += 1

        for t in env.trades:
            combined_trades.append({
                "entry_price": t["entry_price"],
                "exit_price": t["exit_price"],
                "pnl": t["pnl"],
                "side": t["side"],
            })
        env.close()
        results["Combined (ML+PPO)"] = compute_metrics(combined_trades, initial_balance=100.0)
        fmt_result("Combined (ML+PPO)", results["Combined (ML+PPO)"])

    # D) Buy & Hold baseline (in dollars, matching other metrics)
    buy_price = float(val_df["close"].iloc[0])
    sell_price = float(val_df["close"].iloc[-1])
    bh_return = (sell_price - buy_price) / buy_price
    bh_pnl_dollars = bh_return * 100.0
    results["Buy & Hold"] = {"pnl": round(bh_pnl_dollars, 2), "trades": 1,
                            "win_rate": 1.0 if bh_pnl_dollars > 0 else 0.0}

    # 5. Comparison table
    print(f"\n{'='*60}")
    print(f"  COMPARISON TABLE")
    print(f"{'='*60}")
    print(f"  {'Metric':<22} {'ML':>10} {'PPO':>10} {'Combined':>10} {'BH':>10}")
    print(f"  {'-'*22} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    metrics_to_show = [
        ("Total PnL ($)", "pnl"),
        ("Trades", "trades"),
        ("Win Rate", "win_rate"),
        ("Profit Factor", "profit_factor"),
        ("Sharpe", "sharpe"),
        ("Max DD", "max_dd"),
        ("Avg Trade ($)", "avg_trade"),
    ]

    for label, key in metrics_to_show:
        row = f"  {label:<22}"
        for name in ["ML (Random Forest)", "PPO (trained RL)", "Combined (ML+PPO)", "Buy & Hold"]:
            r = results.get(name, {})
            val = r.get(key, "-")
            if isinstance(val, float):
                if key in ("win_rate", "max_dd"):
                    row += f" {val:>9.2%}"
                elif key in ("sharpe", "profit_factor"):
                    row += f" {val:>9.4f}"
                elif key == "trades":
                    row += f" {val:>9}"
                else:
                    row += f" {val:>9.2f}"
            else:
                row += f" {str(val):>9}"
        print(row)

    # Winner
    print(f"\n  {'='*60}")
    print(f"  WINNER BY CATEGORY:")
    print(f"  {'='*60}")
    for metric, key in [("PnL", "pnl"), ("Sharpe", "sharpe"), ("PF", "profit_factor"),
                         ("Win Rate", "win_rate")]:
        best_val = -999999
        best_name = "-"
        for name in ["ML (Random Forest)", "PPO (trained RL)", "Combined (ML+PPO)"]:
            r = results.get(name, {})
            val = r.get(key, -999999)
            if isinstance(val, (int, float)) and val > best_val and val != float("inf"):
                best_val = val
                best_name = name
        if isinstance(best_val, float):
            fmtd = f"{best_val:.4f}" if key in ("sharpe", "profit_factor") else f"${best_val:.2f}"
            print(f"  {metric:<15}: {fmtd} ({best_name})")

    print(f"\n  [DONE] Backtest complete!")

if __name__ == "__main__":
    asyncio.run(main())
