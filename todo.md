# Trading AI TODO.md

## Goal
Build a practical trading AI system that can:
- ingest market data reliably,
- research and backtest strategies,
- paper trade before live execution,
- control risk hard,
- use open-source tools wherever possible,
- stay modular so each layer can be replaced later.

---

## 0) Decide the first market
- [ ] Start with **crypto** first if the goal is fastest open-source integration.
- [ ] Keep the architecture market-agnostic so stocks/forex can be added later.
- [ ] Define the exact exchange(s) you will support first.
- [ ] Define the timeframes you will trade: 1m / 5m / 15m / 1h / 4h / 1d.
- [ ] Define whether the system is:
  - [ ] signal-only
  - [ ] paper trading
  - [ ] semi-auto execution
  - [ ] fully autonomous execution

---

## 1) Core project architecture
- [ ] Separate the project into these modules:
  - [ ] `data/`
  - [ ] `features/`
  - [ ] `strategies/`
  - [ ] `backtest/`
  - [ ] `execution/`
  - [ ] `risk/`
  - [ ] `portfolio/`
  - [ ] `ai/`
  - [ ] `monitoring/`
  - [ ] `storage/`
  - [ ] `api/`
  - [ ] `ui/`
- [ ] Keep strategy logic isolated from exchange logic.
- [ ] Keep AI analysis isolated from order execution.
- [ ] Make every component testable independently.
- [ ] Add config management from day 1.
- [ ] Add environment-based secrets handling from day 1.
- [ ] Add logging, retries, and failure recovery from day 1.

---

## 2) Open-source stack to use first
### Market connectivity
- [ ] Use **CCXT** as the unified exchange connector layer.
- [ ] Use CCXT REST or direct exchange SDKs only when CCXT is missing something critical.
- [ ] Create one connector interface:
  - [ ] fetch OHLCV
  - [ ] fetch trades
  - [ ] fetch order book
  - [ ] fetch balances
  - [ ] place/cancel orders
  - [ ] fetch open positions
  - [ ] fetch funding / fees where relevant

### Backtesting
- [ ] Use **vectorbt** for fast research and large strategy sweeps.
- [ ] Use **backtrader** when you need event-driven logic or broker-style simulation.
- [ ] Build a repeatable backtest pipeline:
  - [ ] same data format
  - [ ] same fees/slippage assumptions
  - [ ] same risk rules
  - [ ] same symbol mapping
  - [ ] same timezone handling

### Bot / live trading base
- [ ] Evaluate **Freqtrade** as the fastest open-source crypto bot base.
- [ ] Reuse its ideas even if you do not adopt it fully:
  - [ ] strategy interface
  - [ ] backtesting
  - [ ] plotting
  - [ ] money management
  - [ ] hyperparameter optimization
  - [ ] Telegram/web UI style control

### AI / agent layer
- [ ] Use AI for:
  - [ ] market summarization
  - [ ] strategy brainstorming
  - [ ] parameter search assistance
  - [ ] news/sentiment digestion
  - [ ] regime detection assistance
  - [ ] post-trade analysis
- [ ] Do **not** let the AI directly fire orders without hard risk checks.
- [ ] Add a human approval mode before autonomous live trading.
- [ ] Use a small, deterministic decision layer for live execution.
- [ ] Treat the AI as an analyst and controller, not as the sole trader.

---

## 3) Data pipeline TODO
- [ ] Collect historical OHLCV.
- [ ] Collect trades and order book snapshots where available.
- [ ] Collect fees, spread, slippage, funding, and latency assumptions.
- [ ] Normalize all symbols and timestamps.
- [ ] Store raw data and cleaned data separately.
- [ ] Save data in a format that is fast to read:
  - [ ] Parquet
  - [ ] DuckDB
  - [ ] Postgres for metadata
- [ ] Add data quality checks:
  - [ ] missing candles
  - [ ] duplicate rows
  - [ ] impossible prices
  - [ ] bad timestamps
  - [ ] zero-volume anomalies
- [ ] Build a data refresh job.
- [ ] Build a dataset versioning rule.
- [ ] Build a fallback plan if an exchange data source fails.

---

## 4) Feature engineering TODO
- [ ] Build a feature library for:
  - [ ] trend
  - [ ] momentum
  - [ ] volatility
  - [ ] volume
  - [ ] liquidity
  - [ ] market structure
  - [ ] correlation
  - [ ] drawdown state
  - [ ] regime detection
- [ ] Start with simple features before deep learning.
- [ ] Track feature leakage aggressively.
- [ ] Normalize feature generation across backtest and live mode.
- [ ] Make features versioned and reproducible.

---

## 5) Strategy research TODO
- [ ] Define 3–5 simple baseline strategies first.
- [ ] Test mean reversion, trend following, breakout, and volatility expansion ideas.
- [ ] Add regime filters.
- [ ] Add multi-timeframe confirmation.
- [ ] Add long/short logic if the market supports it.
- [ ] Add entry, exit, stop loss, take profit, and trailing logic.
- [ ] Measure each strategy on:
  - [ ] CAGR
  - [ ] Sharpe
  - [ ] Sortino
  - [ ] max drawdown
  - [ ] win rate
  - [ ] profit factor
  - [ ] exposure
  - [ ] trade frequency
  - [ ] fee sensitivity
  - [ ] slippage sensitivity
- [ ] Keep a strategy scorecard.
- [ ] Kill weak strategies fast.

---

## 6) Backtesting rules
- [ ] Use realistic fees.
- [ ] Use realistic slippage.
- [ ] Use no-lookahead data only.
- [ ] Use walk-forward testing.
- [ ] Use out-of-sample splits.
- [ ] Use multiple market regimes.
- [ ] Test bull, bear, sideways, and high-volatility periods.
- [ ] Test multiple assets, not just one.
- [ ] Test parameter stability.
- [ ] Reject strategies that only work on one tiny slice of history.
- [ ] Compare vectorized vs event-driven results for sanity.

---

## 7) Risk management TODO
- [ ] Hard cap risk per trade.
- [ ] Hard cap daily loss.
- [ ] Hard cap max drawdown.
- [ ] Hard cap position size.
- [ ] Add exposure limits per asset and sector.
- [ ] Add correlation-aware portfolio sizing.
- [ ] Add kill-switch logic.
- [ ] Add circuit breaker for exchange/API failures.
- [ ] Add stale-data protection.
- [ ] Add duplicate-order protection.
- [ ] Add max open positions limit.
- [ ] Add max consecutive-loss protection.

---

## 8) Execution layer TODO
- [ ] Build an order manager.
- [ ] Support market, limit, stop, and reduce-only orders if the venue supports them.
- [ ] Add order state tracking.
- [ ] Add retry logic with idempotency.
- [ ] Add partial fill handling.
- [ ] Add cancel/replace logic.
- [ ] Track latency from signal to order submission.
- [ ] Track fill quality vs expected price.
- [ ] Record every live decision.
- [ ] Paper trade before any real capital deployment.
- [ ] Add emergency flat-position action.

---

## 9) AI/agent layer TODO
- [ ] Create a market analyst agent.
- [ ] Create a strategy researcher agent.
- [ ] Create a risk auditor agent.
- [ ] Create a trade review agent.
- [ ] Create a post-trade journal agent.
- [ ] Create a data-quality watchdog agent.
- [ ] Make each agent produce structured outputs only.
- [ ] Add confidence scoring.
- [ ] Add evidence linking for every recommendation.
- [ ] Prevent agents from overriding risk rules.
- [ ] Keep the final execution decision deterministic.

---

## 10) Model development TODO
- [ ] Start with small models and simple tasks.
- [ ] Use models for classification, ranking, and summarization first.
- [ ] Avoid training a giant model before the pipeline is stable.
- [ ] Build datasets for:
  - [ ] regime classification
  - [ ] trade outcome prediction
  - [ ] volatility forecasting
  - [ ] news impact classification
  - [ ] strategy selection ranking
- [ ] Validate against leakage.
- [ ] Track live drift.
- [ ] Retrain only when there is evidence of degradation.

---

## 11) Monitoring and observability TODO
- [ ] Add structured logs.
- [ ] Add metrics dashboard.
- [ ] Add alerting for:
  - [ ] API failure
  - [ ] order rejection
  - [ ] drawdown breach
  - [ ] abnormal spread
  - [ ] stale data
  - [ ] inventory mismatch
- [ ] Save full audit trails.
- [ ] Save every signal, order, fill, and cancellation.
- [ ] Add daily PnL and risk reports.
- [ ] Add a trade journal.

---

## 12) Security and operational safety TODO
- [ ] Never hardcode API keys.
- [ ] Use least-privilege API permissions.
- [ ] Use testnet/paper endpoints first.
- [ ] Restrict withdrawal permissions on trading keys.
- [ ] Rotate secrets when needed.
- [ ] Keep environment variables out of git.
- [ ] Back up configs and models.
- [ ] Protect logs from leaking secrets.

---

## 13) UI / control panel TODO
- [ ] Build a simple dashboard for:
  - [ ] balances
  - [ ] open positions
  - [ ] active strategies
  - [ ] recent signals
  - [ ] recent orders
  - [ ] live PnL
  - [ ] drawdown
  - [ ] risk status
- [ ] Add manual override buttons.
- [ ] Add paper/live toggle.
- [ ] Add a kill switch.
- [ ] Add strategy enable/disable controls.

---

## 14) Suggested build order
### Phase 1
- [ ] Pick market and exchange
- [ ] Build data ingestion
- [ ] Build storage
- [ ] Build backtest pipeline
- [ ] Build baseline strategies

### Phase 2
- [ ] Add risk engine
- [ ] Add paper trading
- [ ] Add monitoring
- [ ] Add dashboard
- [ ] Add AI analysis layer

### Phase 3
- [ ] Add live trading
- [ ] Add walk-forward optimization
- [ ] Add portfolio allocation
- [ ] Add regime detection
- [ ] Add post-trade learning

### Phase 4
- [ ] Add multi-agent coordination
- [ ] Add automated research loops
- [ ] Add model retraining workflow
- [ ] Add performance attribution
- [ ] Add long-term strategy lifecycle management

---

## 15) Best-practice rule set
- [ ] Use open source for speed, but verify everything yourself.
- [ ] Keep execution logic boring and reliable.
- [ ] Keep AI helpful, not reckless.
- [ ] Prefer fewer strong strategies over many weak ones.
- [ ] Measure everything.
- [ ] Assume data can be wrong.
- [ ] Assume APIs can fail.
- [ ] Assume a strategy can die without warning.

---

## 16) Minimum viable version
- [ ] One exchange connector
- [ ] One clean historical dataset
- [ ] One backtesting engine
- [ ] One baseline strategy
- [ ] One risk module
- [ ] One paper trading loop
- [ ] One dashboard
- [ ] One AI assistant for analysis only

---

## 17) Future upgrades
- [ ] Multi-exchange routing
- [ ] Cross-asset arbitrage research
- [ ] Portfolio optimization
- [ ] News/sentiment integration
- [ ] On-chain signals if crypto
- [ ] Alternative data
- [ ] Reinforcement learning experiments
- [ ] Ensemble strategy selection
- [ ] Adaptive regime switching

---

## 18) Definition of done
- [ ] Backtests are reproducible.
- [ ] Paper trading behaves like backtesting within reason.
- [ ] Risk controls block bad trades.
- [ ] Live execution is stable.
- [ ] Monitoring catches failures early.
- [ ] AI improves research speed without breaking safety.
- [ ] The system can run, pause, recover, and continue cleanly.
