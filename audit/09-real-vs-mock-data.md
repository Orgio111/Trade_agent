# Real Data vs Mock Data

`REAL DATA VERIFIED: PARTIAL`. Real public-feed adapters exist in source and real
infrastructure is running, but the observed market feed and downstream trading
flow were broken. No subsystem is classified “Real” solely because its container
is healthy.

| Subsystem | Claimed behavior | Actual implementation | Data source | Runtime verified | Classification | Evidence |
|---|---|---|---|---:|---|---|
| Canonical market producer | Closed Binance 1m candles | Public WebSocket adapter with validation | Binance public feed | No | Partially real | Source exists; worker absent |
| Legacy market service | Live market data | WebSocket reconnect loop | Binance public feed | Yes, failed | Broken | HTTP 404, status 500 |
| Feature worker | Incremental indicators | Deterministic feature engine + DB state | Validated events | Source/tests only | Partially real | Canonical tests pass; not deployed |
| Candidate worker | AI candidate signal | Ollama JSON/digest checks plus explicit fallback | Features + local model | No | Unverified | Canonical worker absent |
| Risk worker | Ten-gate deterministic approval | Policy, snapshot, constraints, kill switch | PostgreSQL state | Source/tests only | Partially real | No runtime risk rows |
| Paper execution | Simulated fills/PnL | Deterministic paper service | Order intents and prices | Source/tests only | Simulated | No order intents/fills |
| Binance execution | Broker integration | Legacy/testnet integration | Binance testnet | Failed | Broken | User-stream errors; canonical live disabled |
| FIX broker | Institutional-style protocol | Simulated FIX 4.4 fills | Local simulator | Not needed | Simulated | Explicit simulator |
| Backtesting | Historical simulation | Multiple engines/generators | Mixed real files and synthetic data | Partial | Partially real | No provenance-unified dataset |
| RL training | Adaptive policy | Model restore/training code | Historical/simulated transitions | Failed | Broken | Restore incompatibility, zero steps |
| Qdrant trade memory | Pattern retrieval | Real vector service | Runtime documents | Yes | Hardcoded/empty | Zero points |
| Chroma project knowledge | Project RAG | Real Chroma collection | Repository knowledge | Yes | Real | 1,066 records |
| Chroma runtime memory | Agent operational memory | Real collection | Runtime events | Yes | Empty | Zero records |
| Frontend portfolio | Live portfolio dashboard | API/WebSocket views | Backend state | Yes | Hardcoded/empty | Empty assets; inconsistent paper balances |
| Monitoring | Production telemetry | Prometheus/Grafana | Four targets | Partial | Partially real | Scrapes up; config invalid |

## [AUD-014] Acceptance evidence does not test market-to-fill behavior

- Severity: Medium
- Category: Validation / real-data integrity
- Component: Canonical acceptance harness
- File: `scripts/acceptance.py`, `artifacts/acceptance-latest.json`
- Line: Acceptance steps and recorded database counts
- Runtime service: Isolated canonical Compose project
- Status: Confirmed
- Evidence: The artifact passes migrations/readiness/NATS outage/recovery but records zero risk decisions, zero order intents, and zero fills.
- Impact: A release can pass acceptance while the product's central paper-trading loop is disconnected.
- Root cause: Acceptance was scoped to infrastructure/recovery and not to a deterministic exchange-event fixture or supervised public-feed event.
- Reproduction: Inspect the artifact's database-count section and acceptance script.
- Recommended fix: Add two gates: deterministic replay to approved/rejected decisions and a supervised public closed-candle paper flow to a reconciled fill; assert lineage/trace IDs and no bypass.
- Validation after fix: Artifact contains nonzero validated events/features/candidates/risk decisions/order intents/fills plus exact rejection cases and reconciliation equality.
- Estimated effort: M
- Priority: P1

## Trading intelligence assessment

### Market structure

The canonical one-minute BTC/ETH public-feed scope is explicit, but exchange
session continuity, maintenance windows, symbol filters, clock skew, and order-book
microstructure are not runtime-proven.

### Risk factors and failure scenarios

- Stale/duplicated/out-of-order candles can create repeated entries.
- Empty feature/model output can silently invoke fallback behavior.
- Inconsistent paper balances (`10,000` vs `100,000`) invalidate sizing comparisons.
- Reconciliation auto-clears a system-owned paper kill switch after a clean cycle;
  operator-owned/manual activation is protected, but policy should require explicit
  supervised enablement before first session.
- No market event should be tradable when freshness, lease, policy, snapshot, or
  broker constraints are missing.

### Entry improvements

Require closed-candle identity, exchange timestamp drift, sequence continuity,
spread/slippage bounds, volatility regime, liquidity, and a stable feature/model
digest before risk evaluation.

### Exit improvements

Prove stop/target/reduce-only semantics, gap handling, partial fills, stale-order
cancellation, restart reconciliation, and maximum time-in-trade in deterministic
replay before continuous paper operation.

### Alternative strategies

Keep a no-model deterministic baseline and compare it against the candidate model
under walk-forward, fee/slippage, regime, and latency assumptions. Promote only
when incremental out-of-sample value survives risk-adjusted and operational costs.

