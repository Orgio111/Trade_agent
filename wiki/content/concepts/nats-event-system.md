---
title: "NATS JetStream Event System"
type: concept
tags: [nats, event-system, jetstream, pub-sub, messaging, real-time]
created: 2026-06-30
updated: 2026-07-14
status: draft
---

# NATS JetStream Event System

## Definition

The standardized event backbone of the Trade_agent system. All brain signals, market data, portfolio updates, and system events flow through NATS JetStream as typed, versioned messages. Every component publishes and subscribes via subject-based routing — no direct HTTP calls between services.

## Intuition

NATS JetStream is the nervous system. When a brain computes a score, it publishes a `TradeSignalEvent` to `signals.raw.timesfm.BTCUSDT`. The Go aggregator subscribes to `signals.raw.>`, aggregates, and publishes `AggregatedSignalEvent` to `signals.aggregated`. The frontend WebSocket handler subscribes and pushes to the browser. Zero polling. Zero HTTP between services.

## Event Categories & NATS Subjects

| Category | Subject Pattern | Events | Stream |
|----------|----------------|--------|--------|
| `signals` | `signals.raw.<source>.<symbol>` | TradeSignal, AggregatedSignal, ExecutedSignal | file, 7d |
| `market` | `market.<type>.<symbol>` | MarketCandle, Orderbook, Ticker, Trade | file, 3d |
| `portfolio` | `portfolio.<type>` | PnL, Balance, Position, Order, Risk | file, 30d |
| `rl` | `rl.<type>.<agent>` | Reward, Weight, Training, Evaluation | file, 14d |
| `ws` | `ws.<type>` | Update, Signal, Portfolio | memory |
| `system` | `system.<type>` | Health, Error, Warning, Info, Risk check/approve/reject | memory, 7d |
| `agent` | `agent.<type>.<symbol>` | ChartSnapshot, VLMAnalysis, RAGQuery, RAGResult | memory, 3d |

## Event Types

| Class | Fields | Usage |
|-------|--------|-------|
| `TradeSignalEvent` | symbol, signal, confidence, price, entry/stop/tp | Brain → NATS (signals.raw) |
| `AggregatedSignalEvent` | symbol, consensus, weights, brain_signals, regime | Go orchestrator → NATS (signals.aggregated) |
| `ExecutedSignalEvent` | symbol, order_id, filled_qty, avg_price, status | Execution → NATS (signals.executed) |
| `MarketDataEvent` | symbol, event_type, data (kline/ticker) | Market feed → NATS |
| `OrderbookEvent` | bids, asks, imbalance, spread, mid_price | L2 → NATS → Frontend |
| `PortfolioEvent` | balance, equity, total_pnl, drawdown | Portfolio → NATS → Frontend |
| `RLEvent` | agent_id, reward, weights, metrics | RL engine → NATS |
| `WSEvent` | event_type, payload (flexible data) | NATS → WebSocket → Frontend |
| `SystemEvent` | level, message, component | Any component → NATS |
| `ChartSnapshotEvent` | symbol, image_url, timeframe, indicators | VLM input → agent.vlm.chart_snapshot |
| `VLMAnalysisEvent` | symbol, trend, pattern, support, resistance, confidence | VLM output → agent.vlm.analysis |
| `RAGQueryEvent` | symbol, query, top_k, filters | RAG input → agent.rag.query |
| `RAGResultEvent` | symbol, documents, query_time_ms | RAG output → agent.rag.result |
| `RiskCheckEvent` | symbol, signal, confidence, drawdown, volatility | Risk input → system.risk.check |
| `RiskApprovedEvent` | symbol, signal, approved_size, max_leverage | Risk pass → system.risk.approved |
| `RiskRejectedEvent` | symbol, signal, reason, severity, cooldown | Risk block → system.risk.rejected |

## Base Event Fields (ACP v2)

All events inherit from `QuantexEvent` with these base fields:

| Field | Type | Description |
|-------|------|-------------|
| `id` | str | UUID (8 chars) |
| `timestamp` | float | Unix timestamp |
| `source` | str | Component name |
| `category` | str | Event category |
| `subject` | str | Full NATS subject |
| `priority` | str | `low`, `medium`, `high`, `critical` |
| `trace_id` | str | Session/correlation ID for end-to-end tracing |
| `context` | dict | Shared state (session_state, market_regime, last_signal) |
| `metadata` | dict | Arbitrary key-value metadata |

## Data Flow

```
12 Brains (Layer A)          Go Aggregator (Layer B)      Frontend (Layer C)
┌──────────────┐             ┌──────────────┐             ┌──────────────┐
│ brain.compute│             │ subscribe     │             │ WebSocket    │
│ → publish    │──signals.raw──▶│ → aggregate  │──signals.aggregated──▶│ → React UI │
│   signal     │             │ → publish     │             │              │
└──────────────┘             └──────────────┘             └──────────────┘
       │                            │
       ▼                            ▼
  NATS JetStream              NATS JetStream
  (4222)                      (4222)
```

## NATS Stream Configuration

```python
NATS_STREAMS = {
    "signals": {
        "subjects": ["signals.raw.>", "signals.aggregated", "signals.executed"],
        "storage": "file", "max_age_days": 7, "max_size_gb": 10,
    },
    "market": {"subjects": ["market.>"], "storage": "file", ...},
    "portfolio": {"subjects": ["portfolio.>"], "storage": "file", ...},
    "rl": {"subjects": ["rl.>"], "storage": "file", ...},
    "ws": {"subjects": ["ws.>"], "storage": "memory", ...},
    "system": {"subjects": ["system.>"], "storage": "memory", ...},
    "agent": {"subjects": ["agent.>"], "storage": "memory", "max_age_days": 3},
}
```

## Quick Usage

```python
from orchestrator.events import (
    TradeSignalEvent, OrderbookEvent, PortfolioEvent,
    quantex_event, raw_signal_subject
)

# Create and publish a signal event
event = TradeSignalEvent(
    source="custom_nn",
    symbol="BTCUSDT",
    signal="long",
    confidence=0.85,
    price=50000.0,
)
await nc.publish(event.subject, event.to_json().encode())

# Deserialize from any source
data = json.loads(msg.data)
event = quantex_event(data)
```

## Deserialization Factory

`quantex_event(data: dict)` — automatically creates the correct typed event from a dict by inspecting `category` and `event_type` fields. Handles all 9 event types.

## How we use it

- Source file: `orchestrator/events.py` — all 9 event types in one file
- Go aggregator: `realtime/nats_orchestrator.go` — subscribes, aggregates, publishes
- Frontend: `frontend/src/app/page.tsx` — WebSocket handler routes NATS events to React state
- Dashboard routing: see [[real-time-trading-dashboard]] for NATS subject → panel mapping

## Strengths & weaknesses

**Strengths:**
- Subject-based routing — decoupled publishers/subscribers
- JetStream persistence — messages survive restarts (file storage)
- Exactly-once delivery possible with deduplication
- Sub-millisecond latency for local NATS

**Weaknesses:**
- No built-in schema evolution (manual versioning)
- Stream retention limits (7d signals, 3d market data)
- Single NATS node = single point of failure (no clustering configured)

## Related

- [[trade-project-full-integration-build-plan]] — canonical-contract, idempotency, and replay migration
- [[real-time-trading-dashboard]] — WebSocket routing of NATS events to UI
- [[brain-ecosystem]] — the 12 brains that produce signal events
- [[multi-agent-pipeline]] — LangGraph pipeline that publishes ACP v2 events
- [[nats-langgraph-bridge]] — bridge that subscribes candle events and triggers pipeline
- [[vlm-agent]] — VLM chart analysis agent (publishes to agent.vlm.*)
- [[rag-agent]] — RAG pattern retrieval agent (publishes to agent.rag.*)
- [[odoo-erp-trading-integration]] — Odoo ERP events flow through this system
- [[signal-aggregation-logic]] — how aggregated signals are computed

## Contradictions / updates

**2026-07-14 repository audit:** Python, Go, and Rust do not yet share one compatible wire schema. Subject use also mixes exact signals.raw with hierarchical signals.raw.source.symbol forms. JetStream provides at-least-once delivery; “exactly once” requires producer deduplication, consumer idempotency, and durable side-effect handling that are not yet verified. Treat this page as target vocabulary until the golden contract and replay tests in [[trade-project-full-integration-build-plan]] pass.

## Sources

- Internal code: `orchestrator/events.py`
- Internal design: AGENTS.md §EVENT SYSTEM
