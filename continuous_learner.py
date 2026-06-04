"""
QUANTEX Continuous Learner — Auto-retrain + Validate + Report

Pipeline:
  1. Fetch MAXIMUM data from Binance (up to 2 years of hourly candles)
  2. Split into train / validation
  3. Train/retrain ML model (Random Forest)
  4. Train/retrain PPO model (RL)
  5. Run full backtest (ML, PPO, Combined)
  6. Compare with previous results
  7. Report summary
  8. Save models for next cycle

Usage:
  python continuous_learner.py                    # Normal run
  python continuous_learner.py --days 365         # Max data
  python continuous_learner.py --skip-ppo         # Skip PPO (faster)
  python continuous_learner.py --cycles 3         # Retrain N times
"""
import sys, os, asyncio, json, time, pickle
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
PROJECT_ROOT = Path(__file__).parent

from orchestrator.backtest import DataLoader, BacktestEngine, BacktestResult
from orchestrator.ml_signals import MLSignalEngine
from orchestrator.rl.gym_env import GymTradingEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv


# ── Configuration ──────────────────────────────────────────
class Config:
    # Data
    symbol = "BTCUSDT"
    interval = "1h"
    max_days = 365  # Binance allows ~1500 candles per request, auto-paginated

    # ML model
    ml_retrain_hours = 24

    # PPO
    ppo_steps = 10000
    ppo_ent_coef = 0.05
    ppo_clip_range = 0.1

    # Backtest
    initial_balance = 10.0
    test_split = 0.2  # 20% for validation

    # History tracking
    history_file = PROJECT_ROOT / "models" / "training_history.json"


# ── Data Fetching ──────────────────────────────────────────
async def fetch_max_data(cfg: Config) -> pd.DataFrame:
    """Fetch the maximum available data from Binance with auto-pagination."""
    print(f"\n{'='*60}")
    print(f"  FETCHING DATA — {cfg.symbol} {cfg.interval} ({cfg.max_days} days)")
    print(f"{'='*60}")

    end = datetime.now()
    start = end - timedelta(days=cfg.max_days)

    # Binance pagination: fetch in chunks of 1500 candles
    all_data = []
    chunk_start = start
    total = 0

    while chunk_start < end:
        chunk_end = min(chunk_start + timedelta(days=60), end)
        df = await DataLoader.from_binance_api(
            symbol=cfg.symbol,
            interval=cfg.interval,
            start_time=chunk_start,
            end_time=chunk_end,
            limit=1500,
        )
        if df.empty:
            break
        all_data.append(df)
        total += len(df)
        chunk_start = chunk_end
        print(f"  Fetched {len(df):>5} candles  ({df.index[0].strftime('%Y-%m-%d')} -> {df.index[-1].strftime('%Y-%m-%d')})")

    if not all_data:
        print("  [ERR] No data from Binance!")
        return pd.DataFrame()

    combined = pd.concat(all_data)
    combined = combined[~combined.index.duplicated(keep="first")]
    combined = combined.sort_index()

    print(f"\n  [OK] Total: {len(combined)} candles")
    print(f"       Range: {combined.index[0].strftime('%Y-%m-%d')} -> {combined.index[-1].strftime('%Y-%m-%d')}")
    print(f"       Price: ${combined.close.iloc[-1]:.2f}")
    return combined


# ── ML Training ────────────────────────────────────────────
def train_ml_model(ml: MLSignalEngine, train_df: pd.DataFrame, val_df: pd.DataFrame) -> dict:
    """Train/retrain ML model and return metrics."""
    print(f"\n  Training ML model on {len(train_df)} candles...")
    t0 = time.time()
    result = ml.train(train_df, force=True)
    dt = time.time() - t0

    status = result.get("status", "error")
    if status == "trained":
        metrics = {
            "train_samples": result.get("train_samples", 0),
            "train_accuracy": result.get("train_accuracy", 0),
            "test_accuracy": result.get("test_accuracy", 0),
            "features": result.get("features", 0),
            "train_count": result.get("train_count", 0),
            "train_time_s": round(dt, 2),
        }
        print(f"  [OK] Train acc={metrics['train_accuracy']:.2%}  "
              f"Test acc={metrics['test_accuracy']:.2%}  "
              f"Features={metrics['features']}  ({dt:.1f}s)")
        return metrics
    else:
        print(f"  [WARN] ML training: {result}")
        return {"status": status, "reason": str(result)}


# ── PPO Training ───────────────────────────────────────────
def train_ppo_model(train_df: pd.DataFrame, cfg: Config) -> PPO:
    """Train PPO model and return it."""
    print(f"\n  Training PPO on {len(train_df)} candles ({cfg.ppo_steps} steps)...")
    t0 = time.time()

    train_env = GymTradingEnv(
        data=train_df, initial_balance=cfg.initial_balance,
        leverage=1, lookback=50,
    )
    vec_env = DummyVecEnv([lambda: train_env])
    model = PPO(
        "MlpPolicy", vec_env,
        learning_rate=3e-4, n_steps=2048,
        batch_size=64, n_epochs=10,
        ent_coef=cfg.ppo_ent_coef, clip_range=cfg.ppo_clip_range,
        verbose=0, seed=42,
    )
    model.learn(total_timesteps=cfg.ppo_steps, progress_bar=True)
    train_env.close()

    dt = time.time() - t0
    print(f"  [OK] PPO trained in {dt:.1f}s")
    return model


# ── PPO Evaluation ─────────────────────────────────────────
def eval_ppo(model: PPO, val_df: pd.DataFrame, cfg: Config) -> dict:
    """Evaluate PPO on validation data and return trade list + metrics."""
    env = GymTradingEnv(
        data=val_df, initial_balance=cfg.initial_balance,
        leverage=1, lookback=50,
    )
    obs, _ = env.reset()
    done = False
    trades = []

    while not done:
        action_raw, _ = model.predict(obs, deterministic=True)
        if isinstance(action_raw, np.ndarray) and action_raw.ndim >= 1:
            action_val = int(action_raw[0])
        else:
            action_val = int(action_raw)
        obs, reward, terminated, truncated, info = env.step(action_val)
        done = terminated or truncated

    for t in env.trades:
        trades.append({
            "entry_price": t["entry_price"],
            "exit_price": t["exit_price"],
            "pnl": t["pnl"],
            "side": t["side"],
        })
    env.close()

    return compute_metrics(trades, cfg.initial_balance)


# ── ML Backtest ───────────────────────────────────────────
async def eval_ml(ml: MLSignalEngine, val_df: pd.DataFrame, cfg: Config) -> dict:
    """Run ML backtest and return metrics."""
    bt = BacktestEngine(
        initial_balance=cfg.initial_balance,
        taker_fee=0.0004, slippage_bps=0.5,
    )
    result = await bt.run(ml, val_df)
    trades = []
    for t in result.trades:
        trades.append({
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "pnl": t.pnl,
            "side": t.direction,
        })
    return compute_metrics(trades, cfg.initial_balance)


# ── Combined ML+PPO Backtest ──────────────────────────────
def eval_combined(model: PPO, ml: MLSignalEngine, val_df: pd.DataFrame, cfg: Config) -> dict:
    """Run combined ML+PPO backtest on validation data."""
    env = GymTradingEnv(
        data=val_df, initial_balance=cfg.initial_balance,
        leverage=1, lookback=50,
    )
    obs, _ = env.reset()
    done = False
    step = 0

    while not done:
        action_raw, _ = model.predict(obs, deterministic=True)
        if isinstance(action_raw, np.ndarray) and action_raw.ndim >= 1:
            ppo_action = int(action_raw[0])
        else:
            ppo_action = int(action_raw)

        # ML override
        current_df = val_df.iloc[:min(step + 50 + 1, len(val_df))]
        if len(current_df) >= 50:
            ml_pred = ml.predict(current_df)
            ml_dir = ml_pred.get("direction", "hold")
            ml_conf = ml_pred.get("confidence", 0)
            if ml_conf > 0.6 and ml_dir != "hold":
                ppo_action = 1 if ml_dir == "long" else 2

        obs, reward, terminated, truncated, info = env.step(ppo_action)
        done = terminated or truncated
        step += 1

    trades = []
    for t in env.trades:
        trades.append({
            "entry_price": t["entry_price"],
            "exit_price": t["exit_price"],
            "pnl": t["pnl"],
            "side": t["side"],
        })
    env.close()
    return compute_metrics(trades, cfg.initial_balance)


# ── Metrics ────────────────────────────────────────────────
def compute_metrics(trades: list, initial_balance: float = 100.0) -> dict:
    """Compute trading metrics."""
    if not trades:
        return {"pnl": 0.0, "trades": 0, "win_rate": 0.0, "profit_factor": 0.0,
                "sharpe": 0.0, "max_dd": 0.0, "avg_trade": 0.0}

    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    total_pnl = sum(pnls)
    win_rate = len(wins) / len(pnls) if pnls else 0

    profit_factor = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 \
        else (float("inf") if wins else 0)

    cum_pnl = np.cumsum(pnls)
    equity = initial_balance + cum_pnl
    sharpe = (np.mean(pnls) / (np.std(pnls) + 1e-10)) * np.sqrt(252) if len(pnls) > 1 else 0
    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / peak
    max_dd = float(np.max(dd)) if len(dd) > 0 else 0

    return {
        "pnl": round(total_pnl, 4),
        "trades": len(trades),
        "win_rate": round(win_rate, 4),
        "profit_factor": round(profit_factor, 4),
        "sharpe": round(sharpe, 4),
        "max_dd": round(max_dd, 4),
        "avg_trade": round(np.mean(pnls), 4),
    }


# ── Reporting ──────────────────────────────────────────────
def print_comparison(results: dict, history: list):
    """Print comparison table."""
    labels = ["ML", "PPO", "Combined", "BH"]
    keys = ["pnl", "trades", "win_rate", "profit_factor", "sharpe", "max_dd"]

    print(f"\n{'='*60}")
    print(f"  BACKTEST RESULTS")
    print(f"{'='*60}")
    print(f"  {'Metric':<20} {'ML':>10} {'PPO':>10} {'Combined':>10} {'BH':>10}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    for key in keys:
        row = f"  {key:<20}"
        for lbl in labels:
            r = results.get(lbl, {})
            val = r.get(key, "-")
            if isinstance(val, float):
                if key in ("win_rate", "max_dd"):
                    row += f" {val:>9.2%}"
                elif key in ("sharpe", "profit_factor"):
                    row += f" {val:>9.4f}"
                elif key == "trades":
                    row += f" {int(val):>9}"
                else:
                    row += f" {val:>9.2f}"
            else:
                row += f" {str(val):>9}"
        print(row)

    # Winner
    print(f"\n  {'='*40}")
    print(f"  WINNERS:")
    print(f"  {'='*40}")
    for metric, key in [("PnL", "pnl"), ("Sharpe", "sharpe"),
                         ("Profit Factor", "profit_factor"), ("Win Rate", "win_rate")]:
        best_val, best_name = -999999, "-"
        for lbl in labels:
            r = results.get(lbl, {})
            val = r.get(key, -999999)
            if isinstance(val, (int, float)) and val > best_val and val != float("inf"):
                best_val, best_name = val, lbl
        if isinstance(best_val, float) and best_val > -999999:
            print(f"  {metric:<15}: {best_val:.4f} ({best_name})")

    # History trend
    if len(history) >= 2:
        print(f"\n  {'='*40}")
        print(f"  TRAINING HISTORY TREND:")
        print(f"  {'='*40}")
        prev = history[-2].get("results", {}).get("Combined", {}).get("pnl", 0)
        curr = results.get("Combined", {}).get("pnl", 0)
        diff = curr - prev
        arrow = "⬆" if diff > 0 else "⬇" if diff < 0 else "➡"
        print(f"  Combined PnL: ${prev:.4f} -> ${curr:.4f} ({arrow} ${diff:.4f})")


# ── Main Pipeline ──────────────────────────────────────────
async def main():
    cfg = Config()
    print(f"\n{'#'*60}")
    print(f"#  QUANTEX CONTINUOUS LEARNER")
    print(f"#  Started: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"#  Data: {cfg.max_days}d {cfg.symbol} {cfg.interval}")
    print(f"{'#'*60}")

    # 0. Load training history
    history = []
    if cfg.history_file.exists():
        with open(cfg.history_file) as f:
            history = json.load(f)
    run_number = len(history) + 1
    print(f"\n  Training run #{run_number}")

    # 1. Fetch data
    data = await fetch_max_data(cfg)
    if data.empty:
        return

    # 2. Split
    split = int(len(data) * (1 - cfg.test_split))
    train_df = data.iloc[:split].copy()
    val_df = data.iloc[split:].copy()
    print(f"\n  Split: {len(train_df)} train / {len(val_df)} validation")

    # 3. Load/create ML model
    print(f"\n{'='*60}")
    print(f"  ML SIGNAL ENGINE")
    print(f"{'='*60}")
    ml = MLSignalEngine(
        model_dir=str(PROJECT_ROOT / "models"),
        min_training_samples=500, lookahead_periods=12,
    )
    loaded = ml.load_model()
    print(f"  {'[OK] Loaded from disk' if loaded else '[INFO] Fresh model'}")
    ml_metrics = train_ml_model(ml, train_df, val_df)

    # 4. Train PPO
    print(f"\n{'='*60}")
    print(f"  PPO AGENT")
    print(f"{'='*60}")
    # Train or continue training PPO
    prev_zip = PROJECT_ROOT / "models" / "rl" / "ppo_trading_latest.zip"
    if prev_zip.exists():
        try:
            ppo_model = PPO.load(str(prev_zip.with_suffix("")))
            print(f"  [OK] Loaded previous PPO ({prev_zip.stat().st_size // 1024} KB) — continuing training...")
            train_env = GymTradingEnv(data=train_df, initial_balance=cfg.initial_balance, leverage=1, lookback=50)
            vec_env = DummyVecEnv([lambda: train_env])
            ppo_model.set_env(vec_env)
            t0 = time.time()
            ppo_model.learn(total_timesteps=cfg.ppo_steps, reset_num_timesteps=False, progress_bar=True)
            train_env.close()
            print(f"  [OK] Continued training ({cfg.ppo_steps} steps) in {time.time()-t0:.1f}s")
        except Exception as e:
            print(f"  [WARN] Previous PPO invalid: {e}")
            ppo_model = None
    
    if ppo_model is None:
        print(f"  Training FRESH PPO on {len(train_df)} candles ({cfg.ppo_steps} steps)...")
        train_env = GymTradingEnv(data=train_df, initial_balance=cfg.initial_balance, leverage=1, lookback=50)
        vec_env = DummyVecEnv([lambda: train_env])
        ppo_model = PPO("MlpPolicy", vec_env, learning_rate=3e-4, n_steps=2048,
                        batch_size=64, n_epochs=10, ent_coef=cfg.ppo_ent_coef,
                        clip_range=cfg.ppo_clip_range, verbose=0, seed=42)
        t0 = time.time()
        ppo_model.learn(total_timesteps=cfg.ppo_steps, progress_bar=True)
        train_env.close()
        print(f"  [OK] Fresh PPO trained in {time.time()-t0:.1f}s")

    # Save PPO for next cycle
    save_dir = PROJECT_ROOT / "models" / "rl"
    save_dir.mkdir(parents=True, exist_ok=True)
    ppo_model.save(str(save_dir / "ppo_trading_latest"))
    print(f"  [OK] PPO saved for next cycle")

    # 5. Run backtests
    print(f"\n{'='*60}")
    print(f"  BACKTEST ON VALIDATION DATA ({len(val_df)} candles)")
    print(f"{'='*60}")

    ml_result = await eval_ml(ml, val_df, cfg)
    ppo_result = eval_ppo(ppo_model, val_df, cfg)
    combined_result = eval_combined(ppo_model, ml, val_df, cfg)

    # Buy & Hold baseline
    bh_price_0 = float(val_df["close"].iloc[0])
    bh_price_1 = float(val_df["close"].iloc[-1])
    bh_pnl = ((bh_price_1 - bh_price_0) / bh_price_0) * cfg.initial_balance
    bh_result = {"pnl": round(bh_pnl, 2), "trades": 1}

    results = {
        "ML": ml_result,
        "PPO": ppo_result,
        "Combined": combined_result,
        "BH": bh_result,
    }

    # 6. Print comparison
    print_comparison(results, history)

    # 7. Save history
    record = {
        "run": run_number,
        "timestamp": datetime.now().isoformat(),
        "data_range": f"{data.index[0].strftime('%Y-%m-%d')} -> {data.index[-1].strftime('%Y-%m-%d')}",
        "data_candles": len(data),
        "ml": ml_metrics if isinstance(ml_metrics, dict) else {},
        "results": {k: {mk: mv for mk, mv in v.items() if isinstance(mv, (int, float))}
                    for k, v in results.items()},
    }
    history.append(record)

    # Keep last 20 records
    history = history[-20:]
    cfg.history_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.history_file, "w") as f:
        json.dump(history, f, indent=2, default=str)

    # 8. Save models
    ml._save_model()
    print(f"\n  [OK] Models saved")
    print(f"  [OK] History saved to {cfg.history_file}")

    # 9. Improvement recommendation
    print(f"\n{'='*60}")
    print(f"  RECOMMENDATION")
    print(f"{'='*60}")
    best_pnl = -999999
    best_name = ""
    for name in ["ML", "PPO", "Combined", "BH"]:
        p = results.get(name, {}).get("pnl", -999999)
        if isinstance(p, (int, float)) and p > best_pnl:
            best_pnl, best_name = p, name
    if best_pnl > 0:
        print(f"  ✅ RECOMMENDED: {best_name} ({best_pnl:+.2f}) — PROFITABLE!")
    elif best_pnl > -1:
        print(f"  ⚠️ NEAR BREAKEVEN: {best_name} ({best_pnl:+.2f})")
    else:
        print(f"  ❌ Best is {best_name} ({best_pnl:+.2f}) — needs improvement")

    print(f"\n{'#'*60}")
    print(f"#  DONE — Run #{run_number} complete")
    print(f"{'#'*60}")


if __name__ == "__main__":
    asyncio.run(main())
