---
title: Trade Project Full Integration and Build Plan
type: comparison
tags:
  - architecture
  - trading-system
  - market-data
  - risk
  - execution
  - backtesting
  - agents
  - local-ai
  - roadmap
created: 2026-07-14
updated: 2026-07-15
sources:
  - "[[infrastructure-overview]]"
  - "[[multi-agent-pipeline]]"
  - "[[nats-event-system]]"
  - "[[broker-abstraction-layer]]"
  - "[[brain-backtest-infrastructure]]"
  - "[[inference-router]]"
  - "[[local-trading-ai-architecture]]"
  - "[[brain-ecosystem]]"
status: stable
---

# Trade Project Full Integration and Build Plan

| Architecture | Build speed | Runtime safety | Modularity | RTX 4050 / 16 GB fit | Decision |
|---|---:|---:|---:|---:|---|
| Simple Python bot | 5/5 | 2/5 | 2/5 | 5/5 | Use only for throwaway research |
| FastAPI modular monolith | 4/5 | 4/5 | 4/5 | 5/5 | Use for the control plane and shared packages |
| Multi-agent runtime | 2/5 | 2/5 | 4/5 | 2/5 | Keep off the order hot path |
| Event-driven workers | 3/5 | 5/5 | 5/5 | 4/5 | Use as the production-shaped core |
| Rust low-latency execution | 1/5 | 4/5 | 4/5 | 3/5 | Defer until profiling proves a need |
| Hybrid local/cloud LLM | 4/5 | 4/5 | 5/5 | 4/5 | Use only as an asynchronous advisory lane |

**Decision:** build an event-driven modular monolith deployed as three workers—market data, decision, and execution—plus a FastAPI control plane and the existing Next.js dashboard. Use NATS JetStream for durable inter-worker events, PostgreSQL as the authoritative ledger, Redis for disposable hot state, and Parquet for immutable research data. AI models may enrich or critique a candidate, but deterministic code alone owns risk approval and order placement.

This is the best version of the Trade project for the stated hardware and budget, and the phases below are the exact order in which to build it.

## 1. Executive summary

### What exists

The repository contains many valuable parts, but it is not yet one coherent trading system:

- The current Docker Compose file defines 21 services, which exceeds the useful local operating envelope of an RTX 4050 laptop with 16 GB RAM. [[infrastructure-overview]] still describes an older 13-service topology.
- The event vocabulary in [[nats-event-system]] is a useful design direction, but Python, Go, and Rust currently do not share one wire contract.
- The current [[multi-agent-pipeline]] documentation describes production execution and parallel VLM/RAG work; the code currently sequences VLM then RAG, creates synthetic risk state, and returns simulated execution.
- The repository has several risk engines, execution implementations, backtest engines, and vector-memory systems. None is a single authoritative runtime.
- The test baseline on 2026-07-14 is 144 passing Python tests when the broken MOSS integration test is excluded. Full collection fails because tests import a missing MossCompositeEngine. The Next.js production build and Go compilation tests pass; Go has no test files. Rust could not be verified because Cargo is unavailable in the current environment.

### What to do

1. Freeze live execution and make paper mode an invariant.
2. Define one versioned event contract and one authoritative portfolio/order ledger.
3. Build point-in-time market-data capture, quality checks, and deterministic replay.
4. Replace the fragmented backtests with a two-stage validation stack.
5. Consolidate risk before connecting any execution adapter.
6. Implement idempotent paper execution plus exchange reconciliation.
7. Add AI, RAG, news, and vision only after the deterministic pipeline passes replay and fault-injection tests.
8. Promote to shadow and tiny canary modes only through explicit, auditable gates.

### Implementation checkpoint — 2026-07-15

The first deterministic vertical slice is now implemented as [[deterministic-paper-core-v1]]. It is intentionally a paper/replay reference core, not a live-trading release.

Implemented and verified:

- A strict `MarketEvent` v1 contract with UTC timestamps, `Decimal` serialization, deterministic event IDs, payload checksums, and provenance-tamper rejection.
- Binance and Bybit closed-kline normalizers, reason-coded candle/timestamp/sequence checks, and a fail-closed order-book builder.
- One deterministic risk engine binding every verdict to account, venue, market type, candidate content hash, policy version, and portfolio snapshot. Historical data cannot execute; live mode remains blocked by default.
- Paper execution with deterministic intent/client IDs, exact issued-decision lookup, persist-before-submit semantics, explicit `AMBIGUOUS` handling for lost acknowledgements, execution TTL and price-deviation gates, append-only fills, and expanded reconciliation.
- A deterministic replay harness and CLI. The golden six-candle fixture produces one candidate, one approval, one order, one fill, a clean reconciliation result, and a stable digest across ambient `Decimal` precision 6/10/28/50.
- PostgreSQL migration `002_execution_ledger.sql`, a checksum-locked migration runner for both fresh and existing volumes, and a minimal Compose core containing PostgreSQL, Redis, NATS, and the migration job.
- The pre-existing MOSS integration collection defects were repaired; the full Python suite now reports **294 passed**. Go tests, the Next.js production build, Compose validation, migration image build, and a fresh PostgreSQL apply/no-op reapply test also pass. Cargo remains unavailable in this environment.

Deliberately incomplete and blocked from live promotion:

- Runtime event bus, decision authority, and execution ledger are still in-memory reference implementations; PostgreSQL repositories and durable NATS consumers are not connected to the workers yet.
- The replay portfolio projection is not a tax-lot/cash/PnL ledger. Daily and weekly loss gates therefore need a durable ledger projection before promotion evidence is meaningful.
- Entry intents do not yet create or reconcile protective stop/OCO legs. The risk engine's approved risk amount is an estimate until a persisted protective-order lifecycle exists.
- Raw immutable Parquet capture, full order-book venue feeds, promotion-grade backtesting, control-plane kill-switch APIs, dashboard integration, AI advisory lanes, shadow mode, and live adapters remain future phases.

The next implementation gate is durable PostgreSQL decision/order/cash/position state plus crash-recovery reconciliation. No live key or live order path should be enabled before that gate and protective exits are complete.

### Project-knowledge storage supersession — 2026-07-15

[[local-knowledge-pipeline-v1]] supersedes this plan's pgvector/Qdrant recommendation **only for repository project knowledge**. Code, architecture, documentation, research synthesis, and lessons learned now use `project.manifest.toml`, a deterministic lock, ChromaDB 1.5.9 in local Docker server mode, and local Ollama `nomic-embed-text`; the operational workflow is [[local-knowledge-pipeline-operations]]. PostgreSQL remains authoritative for execution/portfolio state, while the legacy trading-pattern memory in [[rag-agent]] remains quarantined and unchanged. This addendum changes no strategy, risk, broker, or execution logic.

### MVP versus advanced system

| Scope | MVP | Advanced, only after promotion evidence |
|---|---|---|
| Markets | BTC/USDT and ETH/USDT spot, 1m/5m | More symbols, derivatives, cross-venue |
| Runtime | Python workers + FastAPI + Next.js | Extract measured hotspots to Go/Rust |
| Data | Binance + Bybit, candles/trades/L2/funding/OI | Full-depth archives, on-chain, richer news |
| Models | One 3–4B Ollama model, one embedding model | Hosted NIM critic, optional VLM and specialist models |
| Storage | PostgreSQL + pgvector, Redis, Parquet | Qdrant only after a retrieval benchmark justifies it |
| Validation | Replay, walk-forward, paper | Shadow, canary, portfolio strategies |
| Deployment | Docker Compose core profile | Kubernetes only when a real availability requirement exists |

## 2. Current-state audit

| Area | Evidence in the current repository | Consequence | Required action |
|---|---|---|---|
| Compose | 21 services, several open ports, development passwords, and a GPU vLLM service | Resource contention and a false sense of distribution | Create core, observability, research, and experimental profiles |
| Market data | Services instantiate callback-based Binance clients independently; environment variables for NATS/Redis/PostgreSQL are not wired into those paths | Duplicate WebSocket connections and no durable provenance | Rewrite one market-data worker with venue adapters |
| Feature/orderbook/MoE | Feature and orderbook services each create market-data clients; MoE creates both again | The Compose graph is not the runtime graph | Merge into the decision worker as packages |
| Events | Go publishes fields that the Rust subscriber does not accept; exact and wildcard NATS subjects differ | Aggregated messages cannot reliably reach execution | One schema package plus cross-language contract fixtures |
| Risk | Several Python risk engines plus service and Rust variants; some state is hard-coded or disconnected | A trade can be approved against fictitious portfolio state | Consolidate into one deterministic engine reading ledger state |
| Execution | Multiple implementations; the service exposes status reads but no complete intent-to-fill flow; LangGraph execution is simulated | No authoritative order lifecycle | Build intent, adapter, idempotency, and reconciliation layers |
| Backtesting | Multiple engines have inconsistent fee, sizing, partial-exit, intrabar, and annualization semantics | Reported performance is not promotion-grade | Replace with a point-in-time event model and cost tests |
| Memory | Chroma, Qdrant, pgvector, and cache paths coexist; one Qdrant path uses process-random pseudo-embeddings and recreates collections | Retrieval is irreproducible and can erase data | Use one versioned embedding model and pgvector in MVP |
| AI brains | Fixed ensemble weights mix direction, sizing, sentiment, ERP, prediction-market, VLM, and RL concepts | Scores are not calibrated probabilities | Treat each as a research hypothesis, not a production voter |
| Realtime UI | The Go path can inject simulated market activity | Synthetic and live state can be confused | Make simulation an explicit environment and label every event |
| Documentation | Several latency and production claims describe intent rather than measured behavior | Operators may trust functionality that is not present | Mark contradictions and publish verified readiness gates |

### Immediate safety classification

- **Safe to keep running:** read-only research, historical data processing, unit tests, dashboard build, paper simulation.
- **Quarantined:** all live keys, live order placement, Rust execution, Go aggregation-to-execution, automatic liquidation, RL promotion, and any AI-controlled order path.
- **Never acceptable:** silent synthetic price fallback in live mode, widening stops after entry, retrying an unknown order state as a new order, or bypassing the risk engine.

## 3. Recommended final architecture

### Runtime modules

1. **Market-data worker**
   - Owns exchange REST/WebSocket connections.
   - Produces normalized, sequenced events.
   - Repairs order books from snapshot plus delta.
   - Writes immutable raw partitions and data-gap metadata.

2. **Decision worker**
   - Consumes validated market events.
   - Maintains incremental features.
   - Runs deterministic strategies and calibrated statistical models.
   - Retrieves advisory memory asynchronously.
   - Produces a candidate signal, never an order.

3. **Risk engine inside the decision boundary**
   - Loads current ledger-derived account state.
   - Applies exposure, loss, liquidity, spread, staleness, volatility, and news gates.
   - Emits an immutable approve/reject decision.
   - Has no LLM dependency.

4. **Execution worker**
   - Converts an approved decision to an order intent.
   - Enforces idempotency and paper-only mode.
   - Uses one broker port with Paper, Binance, Bybit, and later MT5 adapters.
   - Reconciles acknowledgements, fills, open orders, balances, and positions.

5. **Control plane**
   - FastAPI provides status, configuration, manual kill, promotion workflows, and read APIs.
   - It does not sit in the market-to-order data path.

6. **Dashboard**
   - Reuse the existing Next.js application described by [[real-time-trading-dashboard]].
   - Show provenance, staleness, mode, risk state, order reconciliation, and model availability.

7. **Research/advisory lane**
   - Runs backtests, RAG, journal synthesis, news classification, VLM experiments, and hosted critics.
   - Cannot publish an executable order intent.

### Architecture diagram

~~~text
Binance / Bybit WebSocket + REST
               |
               v
       Market-data worker
   snapshot/delta | validation
        |          +-------> immutable Parquet
        v
 NATS JetStream: market.validated.v1
        |
        v
       Decision worker <--------- PostgreSQL + pgvector memory
  features -> strategy -> candidate       ^
        |                                 |
        +---- async context request -------+
        |
        v
 Deterministic risk engine <------ ledger state / kill state / limits
        |
        | risk.approved.v1 or risk.rejected.v1
        v
       Execution worker
 intent -> idempotency -> broker adapter -> reconciliation
        |                                    |
        +-----------> PostgreSQL ledger <-----+
                           |
              NATS portfolio/order events
                           |
             FastAPI control plane + Next.js

 Optional advisory lane:
 GDELT / FRED / RAG / Ollama / hosted NIM -> context only -> Decision worker
~~~

### Hard invariants

- Paper mode is the compile-time/configuration default and the database default.
- No AI model receives exchange secrets or can call a broker adapter.
- Every event carries schema version, event ID, trace ID, venue, instrument, exchange timestamp, receive timestamp, source mode, and payload checksum.
- Every approved signal references the exact feature snapshot, strategy version, risk policy version, and data-quality verdict used.
- Every order intent has a deterministic client order ID and can be applied at most once.
- Ambiguous exchange outcomes enter reconciliation; they are never blindly retried.
- Stale, gapped, crossed, non-monotonic, or synthetic data cannot open a live position.
- The kill switch is durable and fail-closed.

### Practical latency budgets

This is a 1m/5m research-and-execution system, not co-located HFT:

| Stage | Internal p95 target | Failure behavior |
|---|---:|---|
| Event decode + quality gate | <10 ms | Drop/quarantine invalid event |
| Incremental features | <20 ms | Use last valid snapshot only if explicitly permitted |
| Strategy scoring | <20 ms | No signal |
| Deterministic risk | <10 ms | Reject |
| Intent creation + durable write | <20 ms | No send |
| Exchange network acknowledgement | Measure, do not promise | Reconcile ambiguous state |
| Local AI enrichment | 2–5 s asynchronous | Continue without it or neutral context |
| Hosted critic/research | 8–30 s asynchronous | Timeout/circuit-break; never delay an order |

The previous sub-5 ms or sub-200 ms claims in [[multi-agent-pipeline]] and [[local-trading-ai-architecture]] are not end-to-end execution guarantees.

## 4. Tech stack decision

| Layer | Recommended tool | Reason |
|---|---|---|
| Runtime language | Python 3.12 | Existing codebase, research ecosystem, sufficient for candle/L2 decisions |
| API/control plane | FastAPI + Pydantic | Typed boundaries and existing project fit |
| Event backbone | NATS JetStream | Durable replay, consumer state, compact operational footprint |
| Authoritative database | PostgreSQL | Orders, fills, positions, policies, experiments, and audit transactions |
| Vector search | pgvector in MVP | One operational database; exact and approximate search available |
| Hot state | Redis | Books, latest features, idempotency cache, rate buckets, locks |
| Raw research store | Partitioned Parquet + Polars/DuckDB | Cheap, immutable, reproducible point-in-time analysis |
| Fast research | Polars/DuckDB; optional vectorbt after license review | Fast point-in-time screening without making a fair-code dependency foundational |
| Event validation | NautilusTrader or a small explicit simulator | Realistic event/order semantics; see [[nautilustrader]] |
| Exchange adapters | Official Binance/Bybit APIs behind a local broker port | Precise venue behavior; keep CCXT out of the critical execution path |
| Local inference | Ollama on the Windows host | Fits 6 GB VRAM better than local vLLM/NIM |
| Cloud inference | Hosted NVIDIA NIM, then pinned OpenRouter fallback | Optional deeper critique, outside the hot path |
| Embeddings | nomic-embed-text, pinned version | Small local model and deterministic versioning |
| Frontend | Existing Next.js/React/TypeScript | Already builds; Tauri adds little to MVP |
| Metrics | Prometheus + Grafana optional profile | Useful after canonical business metrics exist |
| Packaging | Docker Compose profiles | Fits local development and separates core from experiments |
| Secrets | Environment/secret files outside Git; later OS/key vault | Prevent keys in images, events, logs, or model prompts |
| CI | Unit + contract + replay + integration + E2E stages | Trading correctness needs more than unit coverage |

### Keep, merge, rewrite, and remove from default deployment

| Action | Components | Rationale |
|---|---|---|
| Keep | Next.js frontend, NATS, PostgreSQL, Redis, broker interface ideas, pure feature functions, metrics definitions, wiki | These provide reusable foundations |
| Merge | All Python risk modules into packages/risk; execution modules into one worker; memory systems into pgvector adapter; feature/orderbook/MoE services into decision packages | One owner per invariant |
| Rewrite | Canonical events, market-data ingestion, order-book repair, execution ledger/reconciler, promotion-grade backtest, embedding pipeline | Current implementations are inconsistent or incomplete |
| Quarantine | Rust execution, Go aggregation/simulator, LangGraph hot-path execution, RL promotion, Odoo and Polymarket production weights | Valuable experiments, unsafe as current core |
| Remove from core profile | vLLM, InfluxDB, Qdrant, Ray, Kubernetes, Terraform, duplicate microservices | Resource cost exceeds current MVP value |
| Add later only on evidence | Rust/Go extraction, Qdrant, K8s, Tauri, more agents | Triggered by profiling, retrieval, availability, or UX requirements |

“Remove” here means remove from the default runtime and archive behind an experimental profile—not destroy potentially useful work before migration tests exist.

## 5. Data pipeline

### Sources and operational caveats

| Data | MVP source | Backup/research source | Limits and policy |
|---|---|---|---|
| Historical candles/trades | [Binance public data](https://github.com/binance/binance-public-data) | Bybit public history/API | Verify published checksums; retain raw files immutably |
| Live candles/trades | Binance Spot WebSocket | Bybit public WebSocket | Reconnect before 24h on Binance; enforce sequence and clock checks |
| L2 order book | Binance snapshot + diff stream | Bybit snapshot + delta | Rebuild on any sequence gap; never patch across a gap |
| Funding | Binance/Bybit derivatives REST and streams | Locally retained history | Backfill before venue retention windows expire |
| Open interest | Binance/Bybit REST | Locally retained history | Treat as lagged regime context, not a direct signal |
| Liquidations | Bybit all-liquidation stream | Binance liquidation stream | Binance reports the latest liquidation per symbol in each 1s window, not a complete tape |
| News | GDELT | Manually curated feeds | Research/advisory only; absence or staleness becomes neutral |
| Macro | FRED | Official release calendars | Store release timestamp and vintage/revision metadata |
| On-chain | Coin Metrics Community | Etherscan, mempool.space, DefiLlama | Low-frequency regime context; cache and label provider coverage |
| TradingView | Signed webhook candidate | None | Validate, deduplicate, enqueue, and acknowledge within three seconds |
| XM/MetaTrader | Host-side MT5 bridge later | None | Stateful terminal IPC; accepted order request is not proof of a fill |

Venue limits change and must be discovered from official responses, not hard-coded forever. Binance exposes current request weights and usage headers; repeated HTTP 429 violations can become 418 bans. Its Spot WebSocket currently limits control messages and streams per connection and requires ping/pong handling. Bybit applies separate IP and UID limits. Use a shared rate governor and persist limit telemetry:

- [Binance Spot REST documentation](https://developers.binance.com/en/docs/products/spot/rest-api)
- [Binance Spot WebSocket documentation](https://github.com/binance/binance-spot-api-docs/blob/master/web-socket-streams.md)
- [Bybit V5 rate limits](https://bybit-exchange.github.io/docs/v5/rate-limit)
- [Bybit all-liquidation stream](https://bybit-exchange.github.io/docs/v5/websocket/public/all-liquidation)

### Rate-limit starting configuration

These values are a 2026-07-14 discovery baseline; adapters must still learn current endpoint metadata and response headers at runtime.

| Provider/interface | Published constraint used by the design | Implementation rule |
|---|---|---|
| Binance Spot REST | Endpoint weights are dynamic; exchangeInfo publishes limits and X-MBX-USED-WEIGHT headers expose usage | Shared weighted token bucket; honor Retry-After; 429 pauses requests; repeated violation/418 opens a long circuit |
| Binance Spot WebSocket | Connection lifetime 24h; server ping every 20s; pong required within 1m; 5 incoming control messages/s; 1,024 streams/connection; 300 connection attempts/5m/IP | Planned rotation before 24h; connection-level control queue; bounded multiplexing and reconnect budget |
| Binance funding/OI REST | Funding history: maximum 1,000 rows and a shared 500 requests/5m/IP pool; historical OI is limited in range/rows | Scheduled incremental capture with a durable high-water mark |
| Bybit HTTP/IP | 600 requests/5s/IP default protection; a 403 IP ban requires at least a 10-minute wait | Venue-wide circuit breaker plus UID endpoint buckets |
| Bybit WebSocket | 500 connection attempts/5m; no more than 1,000 market-data connections/IP | Multiplex subscriptions and cap reconnect concurrency |
| Coin Metrics Community | 10 requests/6s/IP and at most 10 parallel requests | Low-frequency cached batch ingestion only |
| Etherscan free | 3 calls/s and 100,000 calls/day | Daily-budgeted batch jobs; never part of the decision deadline |
| GDELT | Public data updates approximately every 15 minutes; no public production SLA | Poll at low frequency, deduplicate, and mark stale/missing as neutral |
| FRED | API key required; release revisions/vintages matter more than a fixed public quota | Scheduled release/vintage capture, cache, and publication-time joins |
| TradingView webhook | Only ports 80/443; requests taking over 3 seconds are cancelled | Validate/enqueue/deduplicate quickly; never execute synchronously |

### Canonical market event envelope

~~~text
schema_version
event_id
trace_id
event_type
venue
market_type
instrument_id
exchange_ts
received_ts
sequence_start / sequence_end
source_mode          # historical | replay | paper-live | live
ingest_run_id
payload
payload_checksum
quality_flags[]
~~~

### Streaming and persistence flow

1. WebSocket adapter receives bytes and records receive time.
2. Decoder validates schema, numeric bounds, symbol mapping, and clock skew.
3. Sequencer rejects duplicates and detects gaps.
4. Order-book builder applies snapshot plus deltas and verifies checksum/continuity where the venue supports it.
5. Raw event is appended to date/venue/type/symbol Parquet partitions.
6. Curated event is published to NATS only after a quality verdict.
7. PostgreSQL records ingest run, gap, repair, and current canonical metadata.
8. Replay reads the same envelope and publishes to the same subjects under replay mode.

NATS JetStream is at-least-once; consumers must be idempotent and acknowledge only after durable side effects. See [JetStream consumer semantics](https://docs.nats.io/nats-concepts/jetstream/consumers).

### Data cleaning rules

- Normalize instrument, quote/base currency, tick size, lot size, timezone, and decimal precision from venue metadata.
- Reject impossible OHLC, negative volume, non-finite numbers, crossed books, decreasing sequence numbers, and excessive timestamp skew.
- Preserve raw data; corrections create a curated version, never overwrite the source.
- Record missing intervals explicitly. Never forward-fill trades, funding, open interest, or order-book deltas.
- Candle features use only information available at or before the decision timestamp.
- Corporate/macro/news features use publication time, not event date.
- Mark simulated, backfilled, delayed, and revised values in the envelope.
- Run cross-venue sanity checks without treating one venue as automatically correct.

### Feature engineering and statistical hypotheses

| Feature/hypothesis | Testable claim | Guardrail |
|---|---|---|
| Order-flow imbalance | OFI has short-horizon predictive value conditional on spread and volatility | Test by venue/regime; OFI is pressure, not certainty; see [[order-flow-imbalance]] |
| Book imbalance | Top-N depth imbalance improves entry timing | Require reconstructed books and test sensitivity to depth |
| CVD/trade sign | Aggressor flow confirms or rejects a breakout | Validate tick-rule error against quote data; see [[microstructure]] |
| Funding + OI | Extreme funding/OI identifies crowded regimes | Use as regime context, not automatic contrarian direction |
| Liquidity sweep | Post-sweep return differs between continuation and reversion regimes | Define levels without future leakage |
| Breakout probability | A calibrated classifier beats base rate after costs | Evaluate Brier score and calibration, not accuracy alone |
| Volatility regime | ATR/realized-volatility state changes optimal stops and participation | Fit thresholds in training folds only |
| Entry quality | Structure + flow + liquidity + cost + staleness score predicts net expectancy | Calibrate probability and abstain below confidence |

## 6. Strategy engine

### Decision flow

~~~text
Validated snapshot
  -> deterministic features
  -> regime classifier
  -> one or more versioned strategy candidates
  -> calibrated probability and expected net value
  -> trade invalidation / stop / target proposal
  -> deterministic risk decision
  -> order intent or explicit abstention
~~~

### Required strategy interface

Every strategy returns:

- strategy ID and immutable version/hash;
- data cut-off timestamp and feature snapshot ID;
- direction: long, short, or abstain;
- calibrated probability and uncertainty;
- entry constraints, invalidation level, time stop, stop loss, and optional target;
- expected fees, spread, slippage, and funding;
- maximum acceptable price deviation;
- reasoning codes, not unconstrained prose;
- supported instruments, timeframe, and regimes.

### Entry and exit design

- **Market structure:** define swing points with past-only windows; classify higher-high/lower-low transitions.
- **Support/resistance:** use volume/profile or swing clusters fitted only from prior data; record level age and touches.
- **Breakouts:** require close/hold or flow confirmation; compare limit, market, and no-trade alternatives after cost.
- **Liquidity sweeps:** require a prior visible level, excursion, and response window; test continuation and reversion separately.
- **Trend:** combine slope, directional movement, and persistence; avoid voting correlated indicators as independent evidence.
- **Position sizing:** derive quantity from stop distance, account risk, venue lot/min-notional rules, and liquidity cap.
- **Stops:** based on invalidation and volatility, never widened after entry.
- **Targets:** optional; compare fixed R, trailing, time exit, and structure exit without cherry-picking.
- **Partial exits:** model quantity and fees at every fill, not a boolean milestone.
- **Abstention:** is the default when data quality, cost, calibration, or risk confidence is insufficient.

LLMs may summarize why a deterministic decision occurred, but they must not invent a target, override an invalidation, or predict an exact price.

## 7. AI agent system

| Agent | Input -> output | Model/tools | Main failure | Safety check |
|---|---|---|---|---|
| Market Data Agent | Venue bytes -> validated market events | Deterministic Python, venue adapters, NATS | Gaps, stale books, clock skew | Sequence repair; fail closed; no LLM |
| Quant Research Agent | Point-in-time dataset -> hypothesis/backtest artifact | Polars, DuckDB, vectorbt, NautilusTrader; local qwen3.5:4b or hosted NIM for critique | Leakage and multiple testing | Reproducible run ID, OOS and walk-forward gates |
| Signal Agent | Feature snapshot -> candidate signal | Deterministic rules/calibrated ML | Correlated features and uncalibrated confidence | Schema, calibration, cost, and abstention checks |
| Risk Manager Agent | Candidate + ledger state + policy -> approve/reject | Deterministic risk package | Stale or split state | Single ledger snapshot, fail closed, no LLM |
| Execution Agent | Approved decision -> intent/order/fills | Broker port, idempotency, reconciler | Duplicate or ambiguous orders | Deterministic client ID; reconcile before retry |
| Journal Agent | Events and outcomes -> structured journal | Templates; optional phi4-mini summary | Hallucinated rationale | Store raw IDs; prose is non-authoritative |
| Memory/RAG Agent | Query + metadata -> cited historical cases | nomic-embed-text + pgvector | Look-ahead, wrong embedding version | Event-time filter, namespace/version filter, advisory only |
| Code Agent | Issue/test context -> proposed patch | qwen2.5-coder:3b or hosted coding model | Unsafe runtime edits | Offline review, tests, no production credentials |
| Critic Agent | Strategy report -> objections and experiments | qwen3.5:4b or hosted NIM | Persuasive but unsupported claims | Must reference metrics/artifacts; no trading permission |
| News/Macro Agent | GDELT/FRED releases -> timestamped context | Deterministic parsers + small classifier | Late, revised, duplicate, manipulative news | Publication-time semantics; stale/missing means neutral |

The “agent” label is organizational. Market data, risk, and execution are deterministic services, not conversational LLM personas. This is a necessary correction to the mixed-brain design described in [[brain-ecosystem]].

## 8. Model routing plan

### Hardware decision

Local NVIDIA NIM is not viable on a 6 GB laptop GPU: NVIDIA documents 24 GB as the minimum GPU memory for common 8B NIM profiles. vLLM is Linux-oriented and a 7B/8B 4-bit weight file leaves too little headroom for KV cache and runtime overhead. Use Ollama on Windows with one small model loaded at a time. See [NVIDIA NIM prerequisites](https://docs.nvidia.com/nim/large-language-models/latest/get-started/prerequisites.html).

| Task | Model | Local/API | Target | Fallback |
|---|---|---|---:|---|
| Risk, execution, sizing | No LLM | Deterministic | <10 ms per gate | Reject |
| Fast text enrichment | phi4-mini:3.8b | Ollama local | 2–5 s async | No enrichment |
| General research/critic | qwen3.5:4b | Ollama local | 5–15 s async | Hosted NIM |
| Code assistance | qwen2.5-coder:3b-instruct | Ollama local/offline | Interactive | Hosted coding model |
| Embeddings | nomic-embed-text, pinned digest | Ollama/local | Batch; <1 s/item goal | Queue for retry |
| Chart vision research | qwen3.5:4b vision or gemma3:4b | Ollama local | 5–20 s | Skip |
| Deep strategy review | Pinned NVIDIA NIM model | Hosted API | 8–30 s | Pinned OpenRouter model |
| Emergency offline | Small local model only | Ollama | Best effort | Deterministic pipeline continues |

Requested qwen3:8b and deepseek-r1:8b are approximately 5.2 GB just for quantized weights and are unsuitable as always-on services on 6 GB VRAM. Starcoder2:3b and phi3 remain experiments; qwen2.5-coder:3b and phi4-mini provide the cleaner current role split.

### Router policy

1. If the task can change portfolio state, call no model.
2. Classify the task into embedding, fast enrichment, vision, code, or deep critique.
3. Require a JSON schema, temperature zero for structured classification, and prompt/model version.
4. Try local Ollama when the role is installed, healthy, and within its concurrency budget.
5. Use hosted NIM only for asynchronous high-value critique with scrubbed context.
6. Use one pinned OpenRouter fallback, never a random free-model route. Configure privacy/ZDR policy where available.
7. On timeout, malformed output, rate limit, or circuit-open state, return neutral/missing context rather than improvising.
8. Store provider, model digest, prompt hash, latency, token counts, cache status, validation result, and estimated cost.

Recommended local settings: 4K context, concurrency 1, one generation model resident, explicit keep-alive, and queued embedding batches. Do not keep a 4B generator, vision model, and large embedding model resident simultaneously.

- Keep the default text generator resident for roughly five idle minutes, then unload it.
- Serialize vision and generator jobs; unload the current generator before a vision model if free VRAM is below the configured margin.
- Run embeddings in queued batches, preferably on CPU or in a separate serialized GPU window.
- If estimated load time exceeds the request deadline, return neutral context instead of swapping models.
- Apply per-role daily token/cost budgets, response-size caps, semantic caching, and a global hosted-provider circuit breaker.

OpenRouter’s free-tier limits and model availability are not a production SLA; current published limits are described in [OpenRouter rate limits](https://openrouter.ai/docs/api/reference/limits).

## 9. Database and memory design

### PostgreSQL tables

| Group | Tables | Purpose |
|---|---|---|
| Reference | venues, instruments, trading_calendars | Canonical venue/symbol/tick/lot metadata |
| Ingestion | ingest_runs, data_files, data_gaps, data_quality_events | Provenance, checksum, coverage, repairs |
| Curated market | candles, funding_rates, open_interest | Queryable lower-volume time series |
| Features | feature_definitions, feature_snapshots | Versioned point-in-time features |
| Research | strategy_versions, datasets, backtest_runs, walk_forward_folds, experiment_metrics | Reproducible validation |
| Decisions | signals, risk_decisions | Candidate, evidence, policy verdict |
| Ledger | accounts, order_intents, orders, order_events, fills, position_lots, cash_ledger | Authoritative execution history |
| Portfolio | position_snapshots, portfolio_snapshots, limits, kill_switch_state | Risk state and audit |
| Journal | trade_journal, incident_reviews, promotion_gates | Outcomes, mistakes, lifecycle |
| Models | model_registry, prompt_versions, model_calls | Routing, versions, latency, cost |
| Memory | memory_items, memory_embeddings | Cited lessons and pattern retrieval |
| Operations | system_events, service_heartbeats, reconciliation_runs | Monitoring and incident evidence |

Orders and fills are append-oriented. Positions and PnL are projections rebuilt from the ledger, not independent mutable truths.

### Redis keys

~~~text
md:{venue}:{market}:{symbol}:book
candle:{venue}:{symbol}:{timeframe}:latest
feature:{strategy}:{symbol}:{timeframe}
risk:state:{account}
idem:{client_order_id}
rate:{venue}:{bucket}
lock:execution:{account}
health:{service}
kill:{account}
~~~

Redis is never the only copy of an order, fill, limit, or kill state. Idempotency keys have a TTL for speed, while PostgreSQL enforces permanent uniqueness.

### Vector namespaces

Use pgvector in MVP; pgvector supports exact search plus HNSW/IVFFlat indexes when scale requires them: [pgvector](https://github.com/pgvector/pgvector).

| Namespace | Content | Required metadata |
|---|---|---|
| trade-patterns-v1 | Feature/outcome summaries | instrument, timeframe, regime, event cutoff, strategy version |
| research-chunks-v1 | Research notes and experiment conclusions | source, date, authority, tags |
| decision-lessons-v1 | Reviewed mistakes and risk blocks | trade ID, incident ID, review status |

Every vector stores embedding model ID, model digest, dimension, text hash, created time, and validity interval. A model change creates a new namespace. Qdrant becomes an optional rebuildable index only when corpus size or measured query latency exceeds PostgreSQL targets.

### Obsidian memory

The wiki remains the durable human/project brain governed by this file’s parent constitution. High-volume trades stay in PostgreSQL; only reviewed syntheses enter the wiki:

~~~text
wiki/
  content/
    overviews/       system and research synthesis
    concepts/        canonical trading/engineering concepts
    strategies/      approved strategy definitions
    indicators/      feature semantics
    entities/        venues, tools, models
    comparisons/     audits and architecture decisions
    playbooks/       runbooks and promotion procedures
    decisions/       explicit architectural/risk decisions
    sources/         ingested source summaries
  raw/               immutable human-provided sources only
~~~

Proposed automated journal syntheses should be reviewed weekly and linked to underlying run/trade IDs. Do not copy tick data, raw prompts, or every trade into Obsidian.

### RAG retrieval flow

1. Build a query only from data available at the event cutoff.
2. Apply instrument, timeframe, regime, strategy, model-version, and timestamp filters.
3. Retrieve candidates by vector similarity.
4. Re-rank using structured outcome similarity.
5. Return cited memory IDs and historical outcome distributions.
6. The signal path may use the result only as a bounded advisory feature.
7. Log the retrieved IDs with the decision to detect hindsight leakage.
8. Write new lessons only after outcome closure and human/automated quality review.

### CAG policy

Cache-augmented generation uses a small, versioned context bundle—strategy card, reviewed research summary, current regime label, and non-secret operating constraints—without a vector lookup. Store the bundle hash, event cutoff, and source IDs with every model call. CAG is useful for journal/critic consistency, but the deterministic risk engine reads its own policy and ledger directly; a cached prompt is never a safety authority.

## 10. Production folder tree

This is a migration target, not a big-bang rename:

~~~text
trade-agent/
  apps/
    api/
      main.py
      routes/
      schemas/
    dashboard/                 # migrate existing frontend incrementally
  workers/
    market_data/
      main.py
      adapters/
    decision/
      main.py
    execution/
      main.py
  packages/
    domain/
      events.py
      identifiers.py
      money.py
    event_contracts/
      schemas/
      fixtures/
    data_quality/
      envelope.py
      orderbook.py
      clock.py
    features/
    strategies/
    risk/
      policy.py
      engine.py
    brokers/
      base.py
      paper.py
      binance.py
      bybit.py
      mt5_bridge.py
    execution/
      intent.py
      state_machine.py
      reconciler.py
    backtest/
      dataset.py
      costs.py
      simulator.py
      walk_forward.py
    memory/
      repository.py
      embeddings.py
    model_router/
    journal/
    notifications/
      telegram.py              # alerts and stop requests only; no live enable/order command
  infra/
    compose/
      compose.yml
      profiles/
    migrations/
    monitoring/
  data/
    raw/                       # trading Parquet; distinct from wiki/raw
    curated/
    fixtures/
  tests/
    unit/
    contract/
    integration/
    replay/
    e2e/
  scripts/
    record_market.py
    replay_market.py
    reconcile_account.py
    verify_promotion.py
  wiki/
~~~

## 11. Backtesting and validation

### Two-stage stack

1. **Screening:** Polars/DuckDB evaluates broad hypotheses quickly with conservative close-to-next-open execution assumptions. vectorbt may be used as an isolated research tool only after accepting its current Commons Clause license terms.
2. **Promotion validation:** an event-driven simulator—preferably NautilusTrader where its semantics fit—replays trades/books, order types, partial fills, fees, funding, latency, rejections, and venue constraints.

Do not fork a large framework before a thin adapter proves the event/data contract. [[brain-backtest-infrastructure]] remains useful as an experiment catalog, not a promotion authority.

### Validation sequence

~~~text
Data-quality pass
  -> train/validation split by time
  -> purged walk-forward folds
  -> untouched out-of-sample period
  -> fee/slippage/latency stress
  -> parameter-neighborhood stability
  -> bootstrap confidence
  -> deterministic replay
  -> paper trading
  -> live shadow
  -> tiny canary
  -> bounded scale-up
~~~

### Starting promotion criteria

These are conservative initial policy values, not universal guarantees:

| Gate | Initial pass criterion |
|---|---|
| Sample | At least 300 OOS trades; 500 preferred; slower strategies need a full regime cycle |
| Walk-forward | At least 5 folds; no catastrophic fold; positive median net expectancy |
| Profit factor | >=1.20 net of normal costs |
| Sharpe | >=1.0 net; use only with consistent sampling |
| Sortino | >=1.5 net |
| Max drawdown | <=10% in research; live policy is tighter |
| Cost stress | Still positive at 2x fees and 3x modeled slippage |
| Stability | Neighboring parameters remain viable; no single parameter spike |
| Calibration | Probability bins approximately match outcomes; acceptable Brier score versus baseline |
| Leakage | Zero known point-in-time violations |
| Paper | At least 30 days and 200 reconciled trades, or a full regime cycle |
| Shadow | At least 30 days with zero risk bypasses and zero unexplained ledger differences |
| Canary | At least 50 reconciled fills before any size increase |

Win rate alone is not a gate. Report expectancy, payoff distribution, tail loss, exposure, turnover, capacity, fees, funding, and drawdown duration.

### Required tests

- Unit tests for feature boundaries, money/decimal arithmetic, sizing, fees, funding, and risk gates.
- Property tests: quantity never exceeds risk or venue limits; missing state never increases permission; stops never widen.
- Golden contract tests shared by Python/Go/Rust if those languages remain.
- Replay tests for sequence gaps, reconnects, duplicate delivery, stale data, and out-of-order events.
- Execution tests for partial fills, rejects, cancels, ambiguous timeout, and duplicate acknowledgements.
- Database tests for unique client order IDs and ledger reconstruction.
- Walk-forward tests that fail when future data is injected.
- End-to-end paper test using a recorded market fixture, not a live network.
- UI tests for clear paper/live/synthetic/stale labels and kill-switch state.

## 12. Conservative risk engine

### Initial policy

| Limit | Paper default | Earliest live canary |
|---|---:|---:|
| Risk per trade | 0.50% | 0.25%; hard cap 0.50% initially |
| Daily realized + open loss stop | 2.0% | 1.5% |
| Weekly loss stop | 4.0% | 3.0–4.0% |
| Consecutive losses | 3 -> cooldown | 3 -> manual review |
| Open positions | 3 paper | 1 live initially |
| Leverage | 1x | 1x |
| Data staleness | Strategy-specific hard reject | Hard reject |
| Spread/slippage | Above tested bound -> reject | Hard reject |
| News blackout | Configured by instrument/strategy | Hard reject when enabled |

### Gate order

1. Mode and promotion gate.
2. Durable kill switch/manual lock.
3. Data quality and staleness.
4. Account reconciliation freshness.
5. Instrument/venue permission.
6. Daily/weekly/consecutive-loss limits.
7. Existing exposure and correlated exposure.
8. Volatility, liquidity, spread, and expected slippage.
9. Stop validity and risk-based quantity.
10. Venue lot, precision, minimum notional, and maximum order limits.
11. Final price-deviation and order-expiry constraints.
12. Immutable decision record before order intent.

Any unavailable required input rejects the trade. Manual override may reduce or stop risk; it must not bypass a hard safety gate.

### $10 reality

- Risk at 0.5% is $0.05, often below practical minimum notional or fee-efficient sizing.
- One position consumes most usable capital and prevents diversification.
- Futures/leverage makes liquidation, precision, and fee risk dominate any statistical edge.
- A valid risk-sized order below venue minimum means **no trade**, not rounded-up risk.
- The useful outcome is a paper-trading, data-engineering, and validation platform. Treat $10 as an exchange-integration canary only after the full promotion process, not an income strategy.

## 13. Execution layer

### Order lifecycle

~~~text
risk decision
  -> durable order intent
  -> deterministic client_order_id
  -> idempotency check
  -> pre-send reconciliation
  -> broker submit
  -> acknowledgement or ambiguous state
  -> order/fill stream + REST reconciliation
  -> ledger projection
  -> portfolio/risk update
~~~

### Error and retry rules

- Retriable market-data reads use capped exponential backoff with jitter and venue rate-budget awareness.
- An order timeout is **unknown**, not failed. Query by client order ID before any resend.
- Retry only operations documented as idempotent or protected by the same client order ID.
- Reconcile user-data streams against REST snapshots on startup, reconnect, schedule, and ambiguity.
- Persist the intent before network transmission and every subsequent order event afterward.
- Reject stale limit prices and market orders whose expected impact exceeds policy.
- Synchronize exchange clock offset and reject signed requests outside a safe window.
- Separate read, trade, and withdrawal credentials. Withdrawal permission is never enabled.
- Redact API keys, signatures, account IDs, raw headers, and sensitive payloads from logs and prompts.
- Telegram is an alert/incident adapter only. It may request a fail-safe stop through the audited control plane, but it cannot enable live mode or submit an order.

TradingView webhooks are candidate inputs only. The platform accepts only ports 80/443 and cancels slow requests; it also warns that delivery can fail and sensitive information should not be included. See [TradingView webhook guidance](https://www.tradingview.com/support/solutions/43000529348-how-to-configure-webhook-alerts/).

MT5 is a later Windows-host bridge, not a normal stateless REST adapter. A successful order_send response must still be reconciled to order/deal state: [MetaTrader 5 Python order_send](https://www.mql5.com/en/docs/python_metatrader5/mt5ordersend_py).

## 14. Dashboard

Reuse the existing web dashboard. Tauri is deferred until offline desktop packaging is a demonstrated requirement.

Required views:

- global banner: replay, paper, shadow, canary, or live;
- data health: venue connection, last event age, sequence gaps, clock skew, synthetic flag;
- market state: candles, book, spread, flow, funding/OI, regime;
- candidates: feature snapshot, calibrated probability, expected cost, invalidation;
- risk: current limits, utilization, rejects, kill/cooldown, policy version;
- execution: intent/order/fill timeline and reconciliation status;
- portfolio: ledger-derived positions, realized/unrealized PnL, drawdown;
- research: dataset/run/version, OOS folds, stress results, promotion status;
- model router: provider/model version, queue, latency, errors, cost, but no secrets;
- memory search: cited cases with event cutoff and outcome;
- incidents and journal: linked trace, order, fill, strategy, and review IDs.

The UI must never use color alone to distinguish paper and live. A stale or synthetic feed must be visible on every trading screen.

## 15. Open-source project research

Snapshot checked 2026-07-14 against the official GitHub repositories and default-branch activity. Star counts are rounded discovery signals, not quality gates. The rule is to adopt interfaces and proven semantics selectively, not combine complete bots.

| Repository | Purpose / stack | Stars / latest checked activity | License | Strengths | Weaknesses | Decision and project fit |
|---|---|---:|---|---|---|---|
| [NautilusTrader](https://github.com/nautechsystems/nautilus_trader) | Deterministic research/backtest/live engine; Rust + Python/PyO3 | 24.7k / [2026-07-14](https://github.com/nautechsystems/nautilus_trader/commit/47b11ff10f27ca27f0e0574400b24d04209a5e5b) | LGPL-3.0 | Strong event time, matching, execution, and research/live parity | Large API; v2 changes; Windows wheel uses 64-bit precision | **Pilot/use through one adapter.** Benchmark its simulator before adopting broader abstractions; do not copy LGPL code into the core casually. See [[nautilustrader]]. |
| [Freqtrade](https://github.com/freqtrade/freqtrade) | Crypto bot, backtest, hyperopt, dry-run, Telegram; Python/pandas/CCXT | 52.3k / [2026-07-14](https://github.com/freqtrade/freqtrade/commit/2acf03f0340992b5723e2f88b9f1b6bce831b5cc) | GPL-3.0 | Mature dry-run, protections, strategy lifecycle, and look-ahead tests | Opinionated bot; not an L2/low-latency engine; GPL obligations | **Copy ideas or use as an isolated tool.** Do not embed/fork its code into the permissive core without an explicit license decision. |
| [QuantConnect LEAN](https://github.com/QuantConnect/Lean) | Multi-asset event-driven backtest/live engine; C# + Python algorithms | 20.5k / [2026-07-13](https://github.com/QuantConnect/Lean/commit/c22774e49ee80ecef5ca84f57616f6b66fad8bc5) | Apache-2.0 | Mature brokerage, fill, slippage, margin, and data models | Large C# platform duplicates this runtime | **Copy semantics and conformance cases.** Do not embed the whole engine. |
| [Hummingbot](https://github.com/hummingbot/hummingbot) | CEX/DEX connectors, market making, arbitrage; Python/Cython/Docker | 19.1k / [2026-06-16](https://github.com/hummingbot/hummingbot/commit/816b8ab539360557cee7d9248c2f24473b10b16f) | Apache-2.0 | Connector lifecycle, clock, order tracking, and gateway patterns | Large market-making-oriented codebase | **Copy/fork only small reviewed patterns.** Use its failure cases to strengthen broker adapters. |
| [FinRL](https://github.com/AI4Finance-Foundation/FinRL) | Financial RL research; Python, PyTorch, Gym/SB3 | 15.7k / [2026-07-12](https://github.com/AI4Finance-Foundation/FinRL/commit/2334a5fe6d30629157f13c3b0319e1637e15e123) | MIT | Useful environment and reward baselines | Research framework; README points production work elsewhere; RL is easy to overfit | **Ideas/benchmark only.** Never use as live risk or execution authority. |
| [cryptofeed](https://github.com/bmoscon/cryptofeed) | L1/L2/L3, trade, funding, OI, liquidation feeds; Python/Cython/asyncio | 2.9k / [2026-02-01](https://github.com/bmoscon/cryptofeed/commit/fe5993b017df8d57df64b2ad4d8f45f45884b231) | Custom BSD-like 4-clause | Good normalization and book-resynchronization patterns | Latest release/activity cadence is slower; license is not a standard SPDX choice | **Conditional use/ideas.** Review the license and benchmark coverage before distribution. |
| [CCXT](https://github.com/ccxt/ccxt) | Unified exchange APIs; TypeScript-generated JS/Python/C#/PHP/Go/Java | 43.3k / [2026-07-14](https://github.com/ccxt/ccxt/commit/0e7fb01638cbbeaeda7401fd2d2ed3b607626156) | MIT | Broad exchange and historical-data coverage | Lowest-common-denominator semantics can hide venue-specific idempotency, rate, and order behavior | **Use for research/backfill and secondary venues.** Keep official venue adapters on the critical path. |
| [Binance Python Connector](https://github.com/binance/binance-connector-python) | Official generated modular REST/WebSocket SDK; Python/OpenAPI | 2.9k / [2026-07-14](https://github.com/binance/binance-connector-python/commit/4d0e9e7292c141136e3ad8b0049ec4327192e772) | MIT | Official transport coverage and current API generation | Generated API churn; supplies no strategy, risk, ledger, or reconciliation policy | **Use as pinned transport.** Wrap it behind the local adapter and test against venue fixtures/testnet. |
| [LangGraph](https://github.com/langchain-ai/langgraph) | Durable stateful agent graphs and checkpoints; Python/Pydantic | 37.2k / [2026-07-10](https://github.com/langchain-ai/langgraph/commit/55ec2f21939ce7755e6398c11b541de8926245ee) | MIT | Useful human-in-loop and slow research workflows | Nondeterministic agent orchestration is inappropriate for order/risk hot paths | **Keep only in the advisory/research lane.** Validate all outputs as untrusted input. See [[multi-agent-pipeline]]. |
| [Qdrant](https://github.com/qdrant/qdrant) | Filtered HNSW vector database; Rust, REST/gRPC | 33.3k / [2026-06-03](https://github.com/qdrant/qdrant/commit/44ad62f8cd69642be5afa6441612525e24a0d063) | Apache-2.0 | Strong filtered retrieval and mature vector operations | Adds a service and risks becoming a second memory truth | **Keep the adapter; defer the service in MVP.** Adopt only if a pgvector recall/latency benchmark loses. |
| [Haystack](https://github.com/deepset-ai/haystack) | Modular RAG/agent/retrieval evaluation; Python | 25.9k / [2026-07-14](https://github.com/deepset-ai/haystack/commit/bcb28673507c2d40f5608a0ef9b83f97605fabd5) | Apache-2.0 | Hybrid retrieval, reranking, and evaluation patterns | Duplicates LangGraph and the custom RAG runtime | **Copy evaluation ideas; do not add another orchestration framework.** |
| [TradingView Lightweight Charts](https://github.com/tradingview/lightweight-charts) | Financial Canvas charts; TypeScript | 16.6k / [2026-07-08](https://github.com/tradingview/lightweight-charts/commit/871d2dd42d989f41aeecbfb05f19b14e0d825fce) | Apache-2.0 + NOTICE attribution | Efficient price/order/marker rendering | It is not a webhook or execution system; attribution is required | **Keep/use in the dashboard.** Preserve the required notice. See [[price-chart]]. |
| [Tauri](https://github.com/tauri-apps/tauri) | Desktop/mobile WebView shell; Rust + TypeScript/JavaScript | 109k / [2026-07-14](https://github.com/tauri-apps/tauri/commit/459fc315eb790d9aa2d2cea693c20c8978f4b1b0) | MIT / Apache-2.0 | Small native wrapper around web UI | Adds packaging, signing, updater, and IPC security surfaces | **Defer.** Revisit only after the existing web dashboard is stable. |
| [FastAPI](https://github.com/fastapi/fastapi) | Typed async API/WebSocket; Python, Starlette, Pydantic | 100k / [2026-07-13](https://github.com/fastapi/fastapi/commit/b1346bb142c8154950953255eea535e5a83d62eb) | MIT | Excellent control-plane fit and existing project alignment | HTTP is not the durable market/order backbone | **Keep/use for control, status, configuration, and research APIs.** |
| [pgvector](https://github.com/pgvector/pgvector) | Vector similarity in PostgreSQL; C, HNSW/IVFFlat | 22.2k / [2026-07-11](https://github.com/pgvector/pgvector/commit/a6420355c5d1c08f4c5fbd5112fc17e4cf3b5eb5) | PostgreSQL License | One ACID store for journal metadata and embeddings | Requires explicit chunk/version/evaluation logic; not a complete RAG system | **Use as the MVP vector store.** Benchmark against Qdrant before adding another source of truth. |

### Screened out or constrained

- [vectorbt](https://github.com/polakowo/vectorbt) was active with about 7.9k stars when checked, but its current terms combine Apache-2.0 with Commons Clause fair-code restrictions. It is not treated as an OSI-open-source foundational dependency here. Use only as an isolated research tool after license acceptance.
- Public TradingView webhook bots are generally too small, stale, weak on key handling, or ambiguously licensed to justify inheriting their risk. Build the narrow webhook-to-candidate adapter locally.
- No “AI trading agent” repository bypasses the need for point-in-time data, execution semantics, and promotion evidence. Agent frameworks are advisory infrastructure, not alpha.

Before adoption, record repository, pinned version/SHA, license, owner module, allowed usage, SBOM, update cadence, and rollback test in an OSS dependency manifest. CI should block unreviewed GPL/custom-license code from entering the core.

## 16. Development roadmap

### Phase 0 — Cleanup and audit

**Tasks**

- Freeze live execution and revoke/disable withdrawal-capable keys.
- Record the current test/build baseline and dirty-worktree scope.
- Inventory all event subjects, schemas, risk engines, execution paths, backtests, stores, and synthetic fallbacks.
- Define canonical IDs, Decimal/money rules, clock policy, source-mode labels, and safety invariants.
- Split Compose into core, observability, research, and experimental profiles.
- Fix the MOSS test collection failure or explicitly quarantine that optional suite.

**Files**

- packages/domain/identifiers.py
- packages/domain/money.py
- packages/event_contracts/schemas/*.json
- docs/architecture/runtime-inventory.md
- infra/compose/compose.yml

**Tests**

- Baseline unit/build commands.
- Contract fixture decode in every retained runtime language.
- Secret scan and configuration tests.

**Complete when**

- No live adapter can start without an explicit promotion record.
- One reviewed map covers every current producer, consumer, risk check, and execution call.
- Core Compose starts without GPU, Qdrant, InfluxDB, Ray, Go, or Rust services.

### Phase 1 — Data pipeline

**Tasks**

- Implement Binance and Bybit adapters with a shared rate governor.
- Implement canonical envelope, instrument metadata, timestamp/sequence checks, order-book rebuild, and gap events.
- Write immutable Parquet and ingestion metadata.
- Build recorder and deterministic replay CLI.

**Files**

- packages/data_quality/envelope.py
- packages/data_quality/orderbook.py
- workers/market_data/adapters/binance.py
- workers/market_data/adapters/bybit.py
- workers/market_data/main.py
- scripts/record_market.py
- scripts/replay_market.py

**Tests**

- Golden venue payloads.
- Snapshot/delta sequence-gap and reconnect tests.
- Rate-limit/backoff tests.
- Raw-to-curated determinism and checksum test.

**Complete when**

- A 24-hour recording replays to the same event hashes and feature inputs.
- A forced sequence gap prevents book publication until a verified rebuild.

### Phase 2 — Backtesting

**Tasks**

- Define point-in-time datasets and cost/latency models.
- Implement next-event execution, partial fills, fees, funding, precision, and intrabar policy.
- Add vectorized screening and event-driven promotion validation.
- Add purged walk-forward, OOS, bootstrap, calibration, and parameter-stability reports.

**Files**

- packages/backtest/dataset.py
- packages/backtest/costs.py
- packages/backtest/simulator.py
- packages/backtest/walk_forward.py
- scripts/run_experiment.py

**Tests**

- Hand-calculated fill/PnL fixtures.
- Look-ahead injection failure.
- Fee/slippage/funding and partial-fill accounting.
- Deterministic run artifact hash.

**Complete when**

- One simple baseline strategy produces an entirely reproducible report and fails when leakage is introduced.

### Phase 3 — Risk engine

**Tasks**

- Consolidate one policy schema and engine.
- Derive state from the ledger.
- Implement loss, exposure, correlation, volatility, liquidity, spread, staleness, news, min-notional, and kill gates.
- Add durable manual stop and reason-coded decisions.

**Files**

- packages/risk/policy.py
- packages/risk/engine.py
- packages/risk/state.py
- infra/migrations/002_risk_and_ledger.sql

**Tests**

- Table-driven gate tests.
- Decimal boundary and property tests.
- Stale/missing-state fail-closed tests.
- Daily/weekly timezone and restart tests.

**Complete when**

- Every candidate receives one immutable, reproducible verdict against a ledger snapshot.
- No code path can directly produce an order without an approved decision ID.

### Phase 4 — Paper trading

**Tasks**

- Implement broker port, paper adapter, durable intent, state machine, and reconciler.
- Model venue precision, min notional, latency, partial fills, rejects, fees, and slippage.
- Publish order/portfolio events and alerts.
- Run recorded and live-paper sessions.

**Files**

- packages/brokers/base.py
- packages/brokers/paper.py
- packages/execution/intent.py
- packages/execution/state_machine.py
- packages/execution/reconciler.py
- workers/execution/main.py

**Tests**

- Duplicate intent, ambiguous timeout, partial fill, cancel race, reconnect, and restart recovery.
- Ledger-to-position reconstruction.
- End-to-end replay-to-paper fixture.

**Complete when**

- Replaying the same input yields the same decisions and ledger.
- Restart/reconnect produces zero duplicate orders and zero unexplained position differences.

### Phase 5 — AI agents

**Tasks**

- Implement model registry/router and scrubbed prompts.
- Add versioned embeddings and event-time RAG.
- Add research critic, news/macro context, and journal synthesis.
- Calibrate any AI-derived feature before strategy use.

**Files**

- packages/model_router/router.py
- packages/model_router/registry.py
- packages/memory/repository.py
- packages/memory/embeddings.py
- packages/journal/summarizer.py

**Tests**

- Timeout, malformed JSON, provider outage, privacy scrub, cache, and model-version tests.
- RAG cutoff/leakage and citation tests.
- Assert model clients are absent from risk/execution dependency graphs.

**Complete when**

- The deterministic pipeline operates unchanged with every model offline.
- Every model output is versioned, validated, cited, and non-authoritative.

### Phase 6 — Dashboard

**Tasks**

- Replace simulated status with canonical API/events.
- Add data-quality, risk, reconciliation, promotion, and model-router views.
- Add durable kill/cooldown controls with confirmation and audit.

**Files**

- existing frontend components and routes, migrated incrementally
- apps/api/routes/status.py
- apps/api/routes/risk.py
- apps/api/routes/research.py

**Tests**

- Playwright paper/live/stale/synthetic label tests.
- Kill-switch authorization/audit tests.
- WebSocket reconnect and snapshot recovery tests.

**Complete when**

- An operator can explain every position from market event through fill and can see any stale or unreconciled state immediately.

### Phase 7 — Live trading safety gate

**Tasks**

- Add Binance testnet/demo adapter validation, then shadow mode.
- Add read-only account reconciliation before order permission.
- Run paper and shadow promotion criteria.
- Require a signed manual canary approval with expiry and tiny limits.

**Files**

- packages/brokers/binance.py
- packages/brokers/bybit.py
- scripts/reconcile_account.py
- scripts/verify_promotion.py
- wiki/content/playbooks/live-promotion-runbook.md

**Tests**

- Venue sandbox contract tests.
- Key-scope, clock-skew, rate-limit, disconnect, and ambiguous-order drills.
- Kill-switch and incident-response game day.

**Complete when**

- Paper/shadow criteria pass, there are no unexplained reconciliations, and the human explicitly approves a bounded canary.

### Phase 8 — Optimization

**Tasks**

- Profile end-to-end latency, CPU, memory, queue lag, database, and vector retrieval.
- Batch/partition/cache based on evidence.
- Extract only proven bottlenecks to Go/Rust.
- Add Qdrant or Kubernetes only if measured thresholds are exceeded.

**Files**

- benchmarks/
- architecture decision records for every extraction
- load/fault test scenarios

**Tests**

- Regression benchmarks and soak tests.
- Cross-language golden contracts.
- Chaos tests for NATS, Redis, PostgreSQL, venue disconnects, and process restarts.

**Complete when**

- Each added service has a measured target, an owner, an SLO, and a rollback path.

## 17. First seven days

| Day | Exact work | End-of-day artifact |
|---|---|---|
| 1 | Freeze live mode; inventory subjects/schemas/risk/execution; record tests/builds; define safety invariants | Runtime inventory + baseline + quarantine list |
| 2 | Write canonical IDs, Decimal rules, event envelope, JSON schemas, and golden fixtures | event-contracts v1 decoded by retained components |
| 3 | Implement Binance/Bybit recorder skeleton, rate governor, clock/sequence checks, and raw Parquet layout | One recorded session plus provenance manifest |
| 4 | Add PostgreSQL ledger/ingestion migration and repositories; define Redis keys | Rebuildable empty ledger and schema tests |
| 5 | Consolidate policy and risk engine; add table/property tests | Candidate -> immutable reject/approve against ledger state |
| 6 | Implement PaperBroker, deterministic intent ID, order state machine, and reconciler | Restart-safe paper order lifecycle |
| 7 | Run recorded market replay through feature -> simple baseline -> risk -> paper ledger; document every failure | First deterministic vertical slice and Phase 1–4 issue list |

Suggested commands after the target structure exists:

~~~powershell
.venv\Scripts\python.exe -m pytest tests\unit tests\contract -q
.venv\Scripts\python.exe -m pytest tests\replay tests\integration -q
docker compose --profile core config --quiet
docker compose --profile core up -d
npm --prefix frontend run build
~~~

Do not run a live-key smoke test during these seven days.

## 18. First code files to create

| Order | File | Must contain |
|---:|---|---|
| 1 | packages/domain/events.py | Canonical enums, envelope, schema versioning, Decimal/time serialization |
| 2 | packages/event_contracts/schemas/market-event-v1.json | Language-neutral market wire contract |
| 3 | packages/event_contracts/schemas/order-event-v1.json | Intent/order/fill state contract |
| 4 | packages/data_quality/envelope.py | Range, timestamp, source-mode, checksum, and provenance validation |
| 5 | packages/data_quality/orderbook.py | Snapshot/delta state machine and sequence-gap recovery |
| 6 | packages/risk/policy.py | Versioned conservative limits and hard/soft gate definitions |
| 7 | packages/risk/engine.py | Pure candidate + state + policy -> decision function |
| 8 | packages/execution/intent.py | Deterministic client ID and durable intent schema |
| 9 | packages/execution/reconciler.py | Stream/REST/ledger comparison and ambiguity resolution |
| 10 | packages/brokers/base.py | Minimal broker port with explicit venue semantics |
| 11 | packages/brokers/paper.py | Venue-aware deterministic paper fills and costs |
| 12 | workers/market_data/main.py | One owner for exchange connections and publications |
| 13 | workers/decision/main.py | Feature, strategy, memory-context, and risk orchestration |
| 14 | workers/execution/main.py | Approved-intent consumer and reconciliation loop |
| 15 | infra/migrations/002_execution_ledger.sql | Unique intent, append-only order/fill/cash ledger |
| 16 | tests/contract/test_event_fixtures.py | Golden schema and compatibility assertions |
| 17 | tests/replay/test_vertical_slice.py | Recorded event -> deterministic paper ledger |

Create these incrementally and import/adapt proven existing code; do not duplicate them beside the old modules without a migration owner and deletion trigger.

## 19. Required decisions and current assumptions

No answer is required to start Phases 0–4. This plan assumes:

- BTC/USDT and ETH/USDT spot;
- 1-minute and 5-minute decision horizons;
- Binance primary and Bybit secondary;
- paper trading only;
- local Windows/Ollama with Docker for infrastructure;
- zero strategy revenue expectation until OOS, replay, paper, and shadow gates pass.

Before any live canary, the human must explicitly specify:

1. legal jurisdiction, permitted venue/account, spot versus derivatives, and key scope;
2. maximum acceptable daily, weekly, and total capital loss;
3. instruments, timeframes, and whether overnight/weekend exposure is allowed;
4. notification/incident channel and who can enable/disable live mode;
5. minimum paper/shadow observation period if stricter than this plan.

## 20. New ideas / missing components / system upgrades

### New ideas

- Make “abstain quality” a first-class metric: a good system should avoid poor liquidity, stale context, and uncalibrated regimes.
- Use cross-venue disagreement as a data-quality and regime feature before treating it as alpha.
- Maintain a model-free baseline strategy; every agent/model must demonstrate incremental OOS value over it.
- Convert every production incident into a replay fixture and a permanent regression test.

### Missing components

- Canonical cross-language contract and compatibility CI.
- Market-data provenance, sequence repair, and replay.
- Authoritative double-entry-like execution/cash ledger.
- Idempotent order intent and ambiguity reconciliation.
- Point-in-time research dataset and promotion registry.
- Durable kill/promotion state and incident runbooks.
- Calibration, capacity, and cost-stress reporting.

### System upgrades

1. **Highest-value upgrade:** MarketDataQualityGate + ReplayHarness with fault injection for gaps, stale clocks, 429/418/403 rate limits, duplicate deliveries, ambiguous orders, and partial fills.
2. Replace fixed “brain weights” with individually validated, calibrated features; a component receives production weight only after incremental OOS evidence.
3. Add Compose profiles and a resource budget so the core runs within laptop limits before optional services start.
4. Generate Python/TypeScript/Go/Rust event types from one schema if multiple runtime languages survive Phase 0.
5. Add a promotion registry that ties dataset, code hash, strategy, policy, model, and approval to every deployed mode.

## 21. Prediction engine

### Next bottleneck

Point-in-time data quality and replay coverage will constrain research before model quality does.

### Next technical failure

A WebSocket sequence gap will silently corrupt an order book, or an ambiguous order timeout will be retried into a duplicate position, unless the quality gate and reconciler are built first.

### Next scaling limit

The 16 GB machine will hit memory and Docker I/O pressure long before Python calculation latency. Starting vLLM, Qdrant, InfluxDB, Ray, observability, multiple market clients, and the dashboard together is the likely cause.

### Next required feature

An authoritative intent/order/fill ledger with deterministic replay and startup reconciliation.

### Next optimization opportunity

Incremental feature state and Parquet/Polars batch research will produce more value than a premature Rust/Go rewrite.

## 22. What would make this 10× stronger?

Not ten more agents. The 10× improvement is a **canonical point-in-time dataset plus deterministic replay, fault injection, paper/live parity, and evidence-based promotion**. That combination turns every bug, strategy result, risk rejection, and fill into a reproducible artifact. Once that foundation exists, models and strategies can be compared honestly; without it, more AI only increases the speed and confidence of unverified decisions.

## Related

- [[infrastructure-overview]]
- [[multi-agent-pipeline]]
- [[nats-event-system]]
- [[broker-abstraction-layer]]
- [[brain-backtest-infrastructure]]
- [[inference-router]]
- [[local-trading-ai-architecture]]
- [[brain-ecosystem]]
- [[local-knowledge-pipeline-v1]]
- [[local-knowledge-pipeline-operations]]
- [[order-flow-imbalance]]
- [[microstructure]]
- [[real-time-trading-dashboard]]
- [[nautilustrader]]
