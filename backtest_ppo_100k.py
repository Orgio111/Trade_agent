"""
QUANTEX Backtest — PPO (100k steps, in-memory) vs ML vs Combined vs BH

Trains a fresh PPO agent for 100k steps, then runs the comparison backtest
using the in-memory model (avoids Windows file corruption).
"""
import sys, os, asyncio, time, json
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from orchestrator.backtest import DataLoader, BacktestEngine, BacktestResult
from orchestrator.ml_signals import MLSignalEngine
from orchestrator.rl.gym_env import GymTradingEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

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
    if not trades:
        return {"pnl": 0, "trades": 0, "win_rate": 0, "wins": 0, "losses": 0,
                "profit_factor": 0, "sharpe": 0, "max_dd": 0, "avg_trade": 0}
    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    total_pnl = sum(pnls)
    win_rate = len(wins) / len(pnls) if pnls else 0
    profit_factor = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else (float("inf") if wins else 0)
    cum_pnl = np.cumsum(pnls)
    equity = initial_balance + cum_pnl
    sharpe = (np.mean(pnls) / (np.std(pnls) + 1e-10)) * np.sqrt(252) if len(pnls) > 1 else 0
    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / peak
    max_dd = np.max(dd) if len(dd) > 0 else 0
    return {
        "pnl": round(total_pnl, 4), "trades": len(trades),
        "win_rate": round(win_rate, 4), "wins": len(wins), "losses": len(losses),
        "profit_factor": round(profit_factor, 4), "sharpe": round(sharpe, 4),
        "max_dd": round(max_dd, 4), "avg_trade": round(np.mean(pnls), 4),
    }

def run_ppo_backtest(val_df: pd.DataFrame, model, ml_engine=None) -> list:
    env = GymTradingEnv(data=val_df, initial_balance=100.0, leverage=1, lookback=50, ml_engine=ml_engine)
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
    trades = [{"entry_price": t["entry_price"], "exit_price": t["exit_price"],
               "pnl": t["pnl"], "side": t["side"]} for t in env.trades]
    env.close()
    return trades

async def run_ml_backtest(ml: MLSignalEngine, val_df: pd.DataFrame) -> list:
    bt = BacktestEngine(initial_balance=100.0, taker_fee=0.0004, slippage_bps=0.5)
    result = await bt.run(ml, val_df)
    trades = []
    for t in result.trades:
        if hasattr(t, "entry_price"):
            trades.append({"entry_price": t.entry_price, "exit_price": t.exit_price,
                           "pnl": t.pnl, "side": t.direction})
        elif isinstance(t, dict):
            trades.append({"entry_price": t.get("entry_price", 0), "exit_price": t.get("exit_price", 0),
                           "pnl": t.get("pnl", 0), "side": t.get("side", "unknown")})
    return trades

async def main():
    print("=" * 60)
    print("  QUANTEX Backtest — PPO 100k steps (in-memory)")
    print("=" * 60)

    # 1. Fetch 180d data
    end = datetime.now()
    start = end - timedelta(days=180)
    print(f"\n  Fetching 180d BTCUSDT 1h from Binance...")
    df = await DataLoader.from_binance_api(symbol="BTCUSDT", interval="1h", start_time=start, end_time=end)
    if df.empty:
        print("  [ERR] No data!")
        return
    split = int(len(df) * 0.8)
    train_df = df.iloc[:split].copy()
    val_df = df.iloc[split:].copy()
    print(f"  [OK] {len(df)} candles (train={len(train_df)}, val={len(val_df)}), Price: ${df.close.iloc[-1]:.2f}")

    # 2. ML model
    print(f"\n{'='*60}\n  ML SIGNAL ENGINE\n{'='*60}")
    ml = MLSignalEngine(min_training_samples=100, lookahead_periods=12)
    loaded = ml.load_model()
    if loaded:
        print(f"  [OK] Model loaded (trained {ml._train_count}x)")
    else:
        print(f"  Training on {len(train_df)} candles...")
        result = ml.train(train_df, force=True)
        print(f"  [{result.get('status','unknown').upper()}] train_acc={result.get('train_accuracy',0):.2%} test_acc={result.get('test_accuracy',0):.2%}")

    # 3. Train PPO (100k steps) — WITH ML-in-obs, in-memory, never touch disk
    print(f"\n{'='*60}\n  PPO TRAINING (100k steps, ML-in-obs)\n{'='*60}")
    train_env = GymTradingEnv(data=train_df, initial_balance=100.0, leverage=1, lookback=50, ml_engine=ml)
    vec_env = DummyVecEnv([lambda: train_env])
    ppo_model = PPO("MlpPolicy", vec_env, learning_rate=3e-4, n_steps=2048,
                    batch_size=64, n_epochs=10, ent_coef=0.05, verbose=0, seed=42)
    print(f"  Training PPO for 100,000 steps...")
    start_t = time.time()
    ppo_model.learn(total_timesteps=100000, progress_bar=True)
    elapsed = time.time() - start_t
    train_env.close()
    print(f"  [OK] PPO trained in {elapsed:.0f}s")

    # — Quick eval of PPO on train set —
    print(f"\n{'='*60}\n  PPO QUICK EVAL (5 episodes on train set)\n{'='*60}")
    from orchestrator.rl.gym_env import GymTradingEnv as EvalEnv
    action_hist = {0:0,1:0,2:0,3:0}
    all_balances = []
    all_trade_counts = []
    for ep in range(5):
        e = EvalEnv(data=train_df, initial_balance=100.0, leverage=1, lookback=50, ml_engine=ml)
        obs, _ = e.reset()
        done = False
        while not done:
            a_raw, _ = ppo_model.predict(obs, deterministic=True)
            a = int(a_raw[0]) if isinstance(a_raw, np.ndarray) and a_raw.ndim >= 1 else int(a_raw)
            obs, _, terminated, truncated, _ = e.step(a)
            action_hist[a] = action_hist.get(a, 0) + 1
            done = terminated or truncated
        all_balances.append(e.balance)
        all_trade_counts.append(len(e.trades))
        e.close()
    print(f"  Avg Balance: ${np.mean(all_balances):.2f}  Avg Trades/Ep: {np.mean(all_trade_counts):.0f}")
    print(f"  Actions: H={action_hist.get(0,0)} L={action_hist.get(1,0)} S={action_hist.get(2,0)} C={action_hist.get(3,0)}")

    # 4. Backtests on validation set
    print(f"\n{'='*60}\n  BACKTEST ON VALIDATION DATA ({len(val_df)} candles)\n{'='*60}")
    results = {}

    print(f"\n  Running ML backtest...")
    ml_trades = await run_ml_backtest(ml, val_df)
    results["ML (Random Forest)"] = compute_metrics(ml_trades, initial_balance=100.0)
    fmt_result("ML (Random Forest)", results["ML (Random Forest)"])

    print(f"  Running PPO (100k) backtest WITHOUT ML-in-obs...")
    ppo_trades = run_ppo_backtest(val_df, ppo_model, ml_engine=None)
    results["PPO (RL 100k)"] = compute_metrics(ppo_trades, initial_balance=100.0)
    fmt_result("PPO (RL 100k)", results["PPO (RL 100k)"])

    print(f"  Running Combined (PPO + ML-in-obs) backtest...")
    combined_trades = run_ppo_backtest(val_df, ppo_model, ml_engine=ml)
    results["Combined (ML+PPO)"] = compute_metrics(combined_trades, initial_balance=100.0)
    fmt_result("Combined (ML+PPO)", results["Combined (ML+PPO)"])

    # BH baseline
    buy_price = float(val_df["close"].iloc[0])
    sell_price = float(val_df["close"].iloc[-1])
    bh_pnl = (sell_price - buy_price) / buy_price * 100.0
    results["Buy & Hold"] = {"pnl": round(bh_pnl, 2), "trades": 1, "win_rate": 1.0 if bh_pnl > 0 else 0.0}

    # 5. Comparison table
    print(f"\n{'='*60}")
    print(f"  COMPARISON TABLE (ML-in-Obs vs Override)")
    print(f"{'='*60}")
    print(f"  {'Metric':<22} {'ML':>10} {'PPO(100k)':>10} {'ML+PPO(obs)':>10} {'BH':>10}")
    print(f"  {'-'*22} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    for label, key in [("Total PnL ($)", "pnl"), ("Trades", "trades"), ("Win Rate", "win_rate"),
                        ("Profit Factor", "profit_factor"), ("Sharpe", "sharpe"),
                        ("Max DD", "max_dd"), ("Avg Trade ($)", "avg_trade")]:
        row = f"  {label:<22}"
        for name in ["ML (Random Forest)", "PPO (RL 100k)", "Combined (ML+PPO)", "Buy & Hold"]:
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

    print(f"\n  {'='*60}")
    print(f"  WINNER BY CATEGORY:")
    print(f"  {'='*60}")
    for metric, key in [("PnL", "pnl"), ("Sharpe", "sharpe"), ("PF", "profit_factor"), ("Win Rate", "win_rate")]:
        best_val = -999999
        best_name = "-"
        for name in ["ML (Random Forest)", "PPO (RL 100k)", "Combined (ML+PPO)"]:
            r = results.get(name, {})
            val = r.get(key, -999999)
            if isinstance(val, (int, float)) and val > best_val and val != float("inf"):
                best_val = val
                best_name = name
        if isinstance(best_val, float):
            fmtd = f"{best_val:.4f}" if key in ("sharpe", "profit_factor") else f"${best_val:.2f}"
            print(f"  {metric:<15}: {fmtd} ({best_name})")

    print(f"\n  [DONE] Backtest complete! (ML-in-obs replaces old override approach)")

if __name__ == "__main__":
    asyncio.run(main())
