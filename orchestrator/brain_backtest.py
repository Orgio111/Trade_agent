"""Brain aggregate backtest engine.

Runs 11 brain signals over historical data, simulates the weighted
aggregation that the Go NATSOrchestrator performs, and outputs
per-brain + aggregate performance metrics.

Usage:
    python -m orchestrator.brain_backtest --symbol BTC/USDT --days 90
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ── Data Structures ──────────────────────────────────────

BRAIN_WEIGHTS = {
    "timesfm": 0.25,
    "freqai": 0.15,
    "llm_regime": 0.15,
    "finbert_nlp": 0.07,
    "microstructure": 0.05,
    "orderflow_nautilus": 0.03,
    "finrl_kelly": 0.05,
    "statarb_funding": 0.05,
    "onchain_whale": 0.05,
    "custom_nn": 0.05,
    "polymarket_alpha": 0.05,
    "odoo_erp": 0.05,
}

BUY_THRESHOLD = 0.15
SELL_THRESHOLD = -0.15


@dataclass
class BrainBacktestTrade:
    entry_time: datetime
    exit_time: datetime
    direction: str  # 'long' | 'short'
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float = 0.0
    pnl_pct: float = 0.0
    fees: float = 0.0
    aggregated_score: float = 0.0
    active_brains: int = 0
    brain_scores: dict = field(default_factory=dict)


@dataclass
class BrainBacktestResult:
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    avg_brain_agreement: float = 0.0  # how many brains agree with final action
    trades: list[BrainBacktestTrade] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)
    per_brain_contribution: dict = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"Trades: {self.total_trades} | Win Rate: {self.win_rate:.1%} | "
            f"PnL: ${self.total_pnl:+.2f} ({self.total_pnl_pct:+.2%}) | "
            f"MaxDD: {self.max_drawdown_pct:.2%} | Sharpe: {self.sharpe_ratio:.2f} | "
            f"PF: {self.profit_factor:.2f} | Avg Agreement: {self.avg_brain_agreement:.1f}/11"
        )


class BrainBacktestEngine:
    """Backtest engine that simulates brain aggregation logic.

    Instead of running actual brain compute_score (which needs live data/models),
    this replays historical candle data through a lightweight signal simulator
    that mirrors each brain's logic in pure-pandas formulas.
    This avoids needing Ollama/NIM/TimesFM etc. during backtest.
    """

    def __init__(
        self,
        initial_balance: float = 100.0,
        leverage: int = 3,
        taker_fee: float = 0.0004,
        slippage_bps: float = 0.5,
        buy_threshold: float = BUY_THRESHOLD,
        sell_threshold: float = SELL_THRESHOLD,
    ):
        self.initial_balance = initial_balance
        self.leverage = leverage
        self.taker_fee = taker_fee
        self.slippage_bps = slippage_bps
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold

    def compute_brain_signals(self, df: pd.DataFrame, i: int) -> dict[str, float]:
        """Lightweight brain signal simulator using pandas calculations.

        Scores are intentionally scaled to produce meaningful aggregation
        with the ±0.35 threshold (matching Go orchestrator defaults).
        Target: ~5-15% of candles should cross threshold.
        """
        if i < 50:
            return {}

        window = df.iloc[:i+1]
        close = window["close"].iloc[-1]
        scores = {}

        # 1. TimesFM proxy: momentum over last 12 periods
        # 2% price move → score ≈ ±0.6 (strong), 0.5% → ±0.24
        if len(window) >= 12:
            roc_12 = (close / window["close"].iloc[-12] - 1)
            scores["timesfm"] = float(np.tanh(roc_12 * 25))

        # 2. FreqAI proxy: RSI + MACD blend (60/40)
        if len(window) >= 26:
            delta = window["close"].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain.iloc[-1] / (loss.iloc[-1] + 1e-10)
            rsi = 100 - (100 / (1 + rs))
            # MACD proxy
            ema12 = window["close"].ewm(span=12).mean().iloc[-1]
            ema26 = window["close"].ewm(span=26).mean().iloc[-1]
            macd_val = (ema12 - ema26) / close
            rsi_score = (50 - rsi) / 50   # -1 to +1
            macd_score = float(np.tanh(macd_val * 100))
            scores["freqai"] = 0.6 * rsi_score + 0.4 * macd_score

        # 3. LLM Regime proxy: ATR + trend
        if len(window) >= 20:
            atr = (window["high"] - window["low"]).rolling(14).mean().iloc[-1]
            atr_pct = atr / close
            trend = (close / window["close"].iloc[-20] - 1)
            if atr_pct > 0.04:
                scores["llm_regime"] = float(np.tanh(trend * 10)) * 0.3  # volatile → dampen
            else:
                scores["llm_regime"] = float(np.tanh(trend * 30))

        # 4. Microstructure proxy: volume-weighted price change
        if len(window) >= 10 and "volume" in window.columns:
            vol_recent = window["volume"].iloc[-5:].sum()
            vol_prev = window["volume"].iloc[-10:-5].sum()
            if vol_prev > 0:
                vol_ratio = vol_recent / vol_prev
                price_chg = close / window["close"].iloc[-5] - 1
                scores["microstructure"] = float(np.tanh(price_chg * 15 * min(vol_ratio, 3)))

        # 5. Orderflow Nautilus proxy: spread-based amplification
        if "high" in window.columns and "low" in window.columns:
            spread = (window["high"].iloc[-1] - window["low"].iloc[-1]) / close
            micro_score = scores.get("microstructure", 0.0)
            if spread < 0.015:
                scores["orderflow_nautilus"] = micro_score * 1.2  # tight spread → amplify
            else:
                scores["orderflow_nautilus"] = micro_score * 0.3  # wide → dampen

        # 6. FinBERT proxy: price momentum as sentiment proxy
        if len(window) >= 5:
            chg_5 = (close / window["close"].iloc[-5] - 1)
            scores["finbert_nlp"] = float(np.tanh(chg_5 * 20))

        # 7. FinRL/Kelly proxy: directional signal modulated by volatility
        # Low vol = higher conviction in trend; high vol = reduce
        if len(window) >= 20:
            recent_vol = window["close"].pct_change().iloc[-20:].std()
            # Use trend direction (not always positive)
            roc_20 = (close / window["close"].iloc[-20] - 1)
            trend_score = float(np.tanh(roc_20 * 20))
            vol_scale = max(0.2, 1.0 - recent_vol * 15)
            scores["finrl_kelly"] = trend_score * vol_scale

        # 8. StatArb proxy: Z-score mean reversion (contrarian)
        if len(window) >= 50:
            rolling_mean = window["close"].rolling(50).mean().iloc[-1]
            rolling_std = window["close"].rolling(50).std().iloc[-1]
            if rolling_std > 0:
                z = (close - rolling_mean) / rolling_std
                scores["statarb_funding"] = float(np.tanh(-z / 1.5))

        # 9. OnChain proxy: deterministic micro-signal
        scores["onchain_whale"] = float(np.sin(i * 0.01) * 0.03)

        # 10. Custom NN proxy: LSTM/Transformer temporal patterns (momentum + volume)
        if len(window) >= 50:
            # Simple proxy: recent momentum scaled by volume trend
            momentum_10 = (close / window["close"].iloc[-10] - 1) if len(window) >= 10 else 0
            vol_trend = (window["volume"].iloc[-5:].mean() / window["volume"].iloc[-10:-5].mean() - 1) if len(window) >= 10 and window["volume"].iloc[-10:-5].mean() > 0 else 0
            scores["custom_nn"] = float(np.tanh(momentum_10 * 15 + vol_trend * 5))

        # 11. Polymarket Alpha proxy: prediction-market style mean-reversion signal
        # Longshot bias proxy: extreme price moves revert
        if len(window) >= 20:
            recent_extreme = (close - window["close"].iloc[-20:].mean()) / (window["close"].iloc[-20:].std() + 1e-10)
            scores["polymarket_alpha"] = float(np.tanh(-recent_extreme * 0.5))  # contrarian

        return scores

    def aggregate(
        self, brain_scores: dict[str, float], confidence_override: float | None = None
    ) -> tuple[float, str, dict]:
        """Weighted confidence-scaled aggregation (mirrors Go NATSOrchestrator)."""
        weighted_sum = 0.0
        total_weight = 0.0

        for brain_id, score in brain_scores.items():
            w = BRAIN_WEIGHTS.get(brain_id, 0.05)
            conf = confidence_override if confidence_override is not None else 1.0  # proxy = full weight
            weighted_sum += score * w * conf
            total_weight += w * conf

        if total_weight == 0:
            return 0.0, "HOLD", {}

        final_score = weighted_sum / total_weight
        action = "BUY" if final_score > self.buy_threshold else ("SELL" if final_score < self.sell_threshold else "HOLD")

        return final_score, action, brain_scores

    def run(self, df: pd.DataFrame, symbol: str = "BTC/USDT") -> BrainBacktestResult:
        """Run brain aggregate backtest over OHLCV data."""
        balance = self.initial_balance
        peak = balance
        max_dd = 0.0
        position = None  # {direction, entry_price, qty, sl, tp}
        trades: list[BrainBacktestTrade] = []
        equity_curve = [balance]

        for i in range(100, len(df)):
            current = df.iloc[i]
            current_time = df.index[i] if isinstance(df.index[i], datetime) else datetime.now()

            # Compute brain signals
            brain_scores = self.compute_brain_signals(df, i)
            if not brain_scores:
                continue

            final_score, action, _ = self.aggregate(brain_scores)

            # ── Position management ──
            if position is not None:
                # Trailing stop: 1.5% from entry
                if position["direction"] == "long":
                    sl = max(position["sl"], position["entry_price"] * 0.985)
                    if current["low"] <= sl:
                        exit_price = sl * (1 - self.slippage_bps / 10000)
                        fee = exit_price * position["qty"] * self.taker_fee
                        pnl = (exit_price - position["entry_price"]) * position["qty"] * self.leverage - fee
                        balance += pnl
                        trades.append(BrainBacktestTrade(
                            entry_time=position["entry_time"], exit_time=current_time,
                            direction="long", entry_price=position["entry_price"],
                            exit_price=exit_price, quantity=position["qty"],
                            pnl=pnl, pnl_pct=pnl / balance if balance else 0, fees=fee,
                            aggregated_score=final_score, active_brains=len(brain_scores),
                            brain_scores=brain_scores.copy(),
                        ))
                        position = None

                elif position["direction"] == "short":
                    sl = min(position["sl"], position["entry_price"] * 1.015)
                    if current["high"] >= sl:
                        exit_price = sl * (1 + self.slippage_bps / 10000)
                        fee = exit_price * position["qty"] * self.taker_fee
                        pnl = (position["entry_price"] - exit_price) * position["qty"] * self.leverage - fee
                        balance += pnl
                        trades.append(BrainBacktestTrade(
                            entry_time=position["entry_time"], exit_time=current_time,
                            direction="short", entry_price=position["entry_price"],
                            exit_price=exit_price, quantity=position["qty"],
                            pnl=pnl, pnl_pct=pnl / balance if balance else 0, fees=fee,
                            aggregated_score=final_score, active_brains=len(brain_scores),
                            brain_scores=brain_scores.copy(),
                        ))
                        position = None

            # ── Entry signals ──
            if position is None:
                if action == "BUY":
                    qty = (balance * 0.95) / current["close"]  # 95% of balance
                    position = {
                        "direction": "long",
                        "entry_price": current["close"],
                        "entry_time": current_time,
                        "qty": qty,
                        "sl": current["close"] * 0.97,  # 3% stop loss
                        "tp": current["close"] * 1.06,     # 6% take profit
                    }
                elif action == "SELL":
                    qty = (balance * 0.95) / current["close"]
                    position = {
                        "direction": "short",
                        "entry_price": current["close"],
                        "entry_time": current_time,
                        "qty": qty,
                        "sl": current["close"] * 1.03,
                        "tp": current["close"] * 0.94,
                    }

            # Track equity / drawdown
            equity = balance
            if position:
                if position["direction"] == "long":
                    equity += (current["close"] - position["entry_price"]) * position["qty"] * self.leverage
                else:
                    equity += (position["entry_price"] - current["close"]) * position["qty"] * self.leverage

            equity_curve.append(equity)
            peak = max(peak, equity)
            dd = (peak - equity) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

        # Close any open position at end
        if position is not None:
            last_price = df.iloc[-1]["close"]
            last_time = df.index[-1] if isinstance(df.index[-1], datetime) else datetime.now()
            if position["direction"] == "long":
                pnl = (last_price - position["entry_price"]) * position["qty"] * self.leverage
            else:
                pnl = (position["entry_price"] - last_price) * position["qty"] * self.leverage
            balance += pnl
            trades.append(BrainBacktestTrade(
                entry_time=position["entry_time"], exit_time=last_time,
                direction=position["direction"], entry_price=position["entry_price"],
                exit_price=last_price, quantity=position["qty"],
                pnl=pnl, pnl_pct=pnl / balance if balance else 0, fees=0,
                aggregated_score=0, active_brains=11,
            ))

        # ── Compute metrics ──
        total_pnl = balance - self.initial_balance
        total_pnl_pct = total_pnl / self.initial_balance
        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]
        win_rate = len(wins) / len(trades) if trades else 0

        # Sharpe (annualized from hourly equity returns)
        if len(equity_curve) > 1:
            returns = np.diff(equity_curve) / (np.array(equity_curve[:-1]) + 1e-10)
            sharpe = float(np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(365 * 24))
        else:
            sharpe = 0.0

        # Profit factor
        gross_win = sum(t.pnl for t in wins) if wins else 0
        gross_loss = abs(sum(t.pnl for t in losses)) if losses else 1e-10
        profit_factor = gross_win / gross_loss

        # Avg brain agreement
        agreement_counts = []
        for t in trades:
            score = t.aggregated_score
            if score > self.buy_threshold:
                agg_action = "BUY"
            elif score < self.sell_threshold:
                agg_action = "SELL"
            else:
                agg_action = "HOLD"
            for brain_id, bs in t.brain_scores.items():
                brain_action = "BUY" if bs > self.buy_threshold else ("SELL" if bs < self.sell_threshold else "HOLD")
                agreement_counts.append(1 if brain_action == agg_action else 0)
        avg_agreement = np.mean(agreement_counts) * 11 if agreement_counts else 0

        return BrainBacktestResult(
            total_trades=len(trades),
            winning_trades=len(wins),
            losing_trades=len(losses),
            win_rate=win_rate,
            total_pnl=total_pnl,
            total_pnl_pct=total_pnl_pct,
            max_drawdown_pct=max_dd,
            sharpe_ratio=sharpe,
            profit_factor=profit_factor,
            expectancy=(total_pnl / len(trades)) if trades else 0,
            avg_brain_agreement=float(avg_agreement),
            trades=trades,
            equity_curve=equity_curve,
        )


# ── Data Loader ──────────────────────────────────────────

async def fetch_binance_ohlcv(
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    days: int = 90,
) -> pd.DataFrame:
    """Fetch historical OHLCV from Binance REST API."""
    base_url = "https://api.binance.com/api/v3/klines"
    end_ts = int(datetime.now().timestamp() * 1000)
    start_ts = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

    all_data = []
    current_ts = start_ts
    limit = 1000

    import httpx
    async with httpx.AsyncClient(timeout=30) as client:
        while current_ts < end_ts:
            params = {
                "symbol": symbol.replace("/", ""),
                "interval": interval,
                "startTime": current_ts,
                "endTime": end_ts,
                "limit": limit,
            }
            resp = await client.get(base_url, params=params)
            if resp.status_code != 200:
                print(f"Binance API error: {resp.status_code}")
                break
            data = resp.json()
            if not data:
                break
            all_data.extend(data)
            current_ts = data[-1][6] + 1  # close time + 1ms
            print(f"  Fetched {len(data)} candles, total {len(all_data)}...")

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore",
    ])
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.index = pd.to_datetime(df["open_time"], unit="ms")
    df = df[["open", "high", "low", "close", "volume"]]
    return df


# ── CLI Entry Point ──────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description="Brain Aggregate Backtest")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading pair (default: BTCUSDT)")
    parser.add_argument("--interval", default="1h", help="Candle interval (default: 1h)")
    parser.add_argument("--days", type=int, default=90, help="Lookback days (default: 90)")
    parser.add_argument("--balance", type=float, default=100.0, help="Initial balance (default: 100)")
    parser.add_argument("--buy-threshold", type=float, default=BUY_THRESHOLD, help=f"Buy threshold (default: {BUY_THRESHOLD})")
    parser.add_argument("--sell-threshold", type=float, default=SELL_THRESHOLD, help=f"Sell threshold (default: {SELL_THRESHOLD})")
    args = parser.parse_args()

    print(f"Fetching {args.symbol} {args.interval} data ({args.days}d)...")
    df = await fetch_binance_ohlcv(symbol=args.symbol, interval=args.interval, days=args.days)
    if df.empty:
        print("No data fetched. Exiting.")
        return

    print(f"Loaded {len(df)} candles. Running brain aggregate backtest...")
    engine = BrainBacktestEngine(
        initial_balance=args.balance,
        buy_threshold=args.buy_threshold,
        sell_threshold=args.sell_threshold,
    )
    result = engine.run(df, symbol=args.symbol)

    print()
    print("=" * 60)
    print("  BRAIN AGGREGATE BACKTEST RESULTS")
    print("=" * 60)
    print(f"  Symbol:           {args.symbol}")
    print(f"  Interval:         {args.interval}")
    print(f"  Candles:          {len(df)}")
    print(f"  Total Trades:     {result.total_trades}")
    print(f"  Win Rate:         {result.win_rate:.1%}")
    print(f"  Total PnL:        ${result.total_pnl:+.2f} ({result.total_pnl_pct:+.2%})")
    print(f"  Max Drawdown:     {result.max_drawdown_pct:.2%}")
    print(f"  Sharpe Ratio:     {result.sharpe_ratio:.2f}")
    print(f"  Profit Factor:    {result.profit_factor:.2f}")
    print(f"  Expectancy:      ${result.expectancy:+.4f}")
    print(f"  Brain Agreement:  {result.avg_brain_agreement:.1f}/11")
    print("=" * 60)

    # Save results
    results_dir = Path("backtest_results")
    results_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = results_dir / f"brain_agg_{args.symbol}_{args.interval}_{ts}.json"

    out_data = {
        "symbol": args.symbol,
        "interval": args.interval,
        "candles": len(df),
        "initial_balance": args.balance,
        "weights": BRAIN_WEIGHTS,
        "result": {
            "total_trades": result.total_trades,
            "win_rate": result.win_rate,
            "total_pnl": result.total_pnl,
            "total_pnl_pct": result.total_pnl_pct,
            "max_drawdown_pct": result.max_drawdown_pct,
            "sharpe_ratio": result.sharpe_ratio,
            "profit_factor": result.profit_factor,
            "expectancy": result.expectancy,
            "avg_brain_agreement": result.avg_brain_agreement,
            "equity_curve_len": len(result.equity_curve),
        },
    }
    out_file.write_text(json.dumps(out_data, indent=2, default=str))
    print(f"  Results saved: {out_file}")


if __name__ == "__main__":
    asyncio.run(main())
