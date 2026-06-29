---
title: "Brain Backtest Infrastructure"
type: concept
tags: [backtest, brain, evaluation, metrics, equity-curve]
created: 2026-06-30
updated: 2026-06-30
status: stable
---

# Brain Backtest Infrastructure

## Definition

A lightweight backtest engine that simulates the Go NATSOrchestrator's weighted brain aggregation using pure-pandas signal proxies. Runs 12 brain signals over historical OHLCV data, applies weighted voting with configurable buy/sell thresholds, and outputs per-brain + aggregate performance metrics (win rate, Sharpe, profit factor, max drawdown, brain agreement).

## Intuition

Running actual brain compute_score requires live data feeds, LLM APIs, and GPU inference — too slow for backtesting. Instead, this engine replaces each brain with a lightweight pandas formula that mirrors its logic (e.g., TimesFM → 12-period momentum, FreqAI → RSI+MACD blend). This lets us backtest the full 12-brain ensemble in seconds, not hours.

## Architecture

```
OHLCV DataFrame
    │
    ▼
Brain Signal Simulator (12 proxies)
  ├── timesfm: 12-period momentum → tanh
  ├── freqai: RSI + MACD blend (60/40)
  ├── llm_regime: ATR + trend
  ├── microstructure: volume-weighted price change
  ├── orderflow: spread-based amplification
  ├── finbert_nlp: 5-period momentum
  ├── finrl_kelly: trend × volatility scale
  ├── statarb_funding: Z-score mean reversion
  ├── onchain_whale: deterministic sine wave
  ├── custom_nn: momentum × volume trend
  ├── polymarket_alpha: extreme reversion
  └── odoo_erp: (not simulated — business signal)
    │
    ▼
Weighted Aggregation (matches Go NATSOrchestrator)
  consensus = Σ(score × weight × confidence) / Σ(weight × confidence)
    │
    ▼
Trade Simulation (3% SL, 6% TP, 1.5% trailing stop)
    │
    ▼
Metrics: win_rate, sharpe, profit_factor, max_drawdown, brain_agreement
```

## Key Metrics

| Metric | Description |
|--------|-------------|
| Win Rate | % of profitable trades |
| Sharpe Ratio | Annualized risk-adjusted return |
| Profit Factor | Gross wins / gross losses |
| Max Drawdown | Worst peak-to-trough decline |
| Brain Agreement | Average # of brains agreeing with final action (0-12) |
| Expectancy | Average PnL per trade |

## How we use it

- Source: `orchestrator/brain_backtest.py` — `BrainBacktestEngine` class
- CLI: `python -m orchestrator.brain_backtest --symbol BTCUSDT --days 90`
- Results saved to `backtest_results/brain_agg_*.json`
- Used to validate brain weight changes before deployment
- Data: fetched from Binance REST API (`fetch_binance_ohlcv`)

## Related

- [[brain-ecosystem]] — brain weights being backtested
- [[signal-aggregation-logic]] — aggregation formula this engine simulates
- [[ensemble-meta-model]] — meta-learner that could replace fixed weights

## Sources

- Internal code: `orchestrator/brain_backtest.py`
