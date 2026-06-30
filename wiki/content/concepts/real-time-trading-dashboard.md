---
title: "Real-Time AI Trading Dashboard"
type: concept
tags: [dashboard, real-time, websocket, frontend, nats, architecture]
created: 2026-06-30
updated: 2026-06-30
sources: []
status: stable
---

# Real-Time AI Trading Dashboard

## Definition

A live trading command center built with Next.js + WebSocket, consuming events from NATS JetStream via a Go aggregation layer. Displays market data, AI brain signals, portfolio state, risk metrics, inference routing, and Odoo ERP business intelligence in a single cockpit view.

## Intuition

A mini Bloomberg terminal for autonomous AI trading. Every brain signal, market tick, risk alert, and ERP update appears in real-time — no polling, only push. The dashboard is the human's window into what the 12-brain swarm is thinking and doing.

## Architecture

```
Live Market Feed
        ↓
Inference Server (vLLM / Ollama)
        ↓
MoE Router (12 brains)
        ↓
NATS JetStream (event backbone)
        ↓
┌──────────────┼──────────────┐
▼              ▼              ▼
Trades Stream   AI Decisions   Risk Engine
        ▼              ▼              ▼
Go Realtime Aggregator (Layer B)
        ↓
WebSocket API
        ↓
NEXT.JS DASHBOARD (Layer C)
```

## Tech Stack

| Layer | Technology | Role |
|-------|-----------|------|
| Backend | FastAPI (Python) | Orchestrator API, brain runners |
| Event Bus | NATS JetStream | All signal/data flow |
| Aggregator | Go (realtime/) | Event aggregation, WebSocket push |
| Frontend | Next.js + React | Dashboard UI |
| Charts | lightweight-charts (TradingView OSS) | Candlestick + volume |
| 3D Viz | Three.js | Agent swarm visualization |
| WebSocket | Native browser WS | Real-time data streaming |

## NATS Event Types → Dashboard Panels

| NATS Subject | Event Type | Dashboard Panel |
|-------------|-----------|----------------|
| `signals.raw.<source>.<symbol>` | Brain signals | Agent Swarm Visor, Latest Signal |
| `signals.aggregated` | Consensus signal | Event Log |
| `market.candle.<symbol>` | OHLCV data | Price Chart |
| `market.orderbook.<symbol>` | L2 depth | Orderbook Heatmap |
| `portfolio.pnl` | PnL updates | Account Balance, Win Rate |
| `portfolio.risk` | Risk metrics | Risk Dashboard |
| `portfolio.position` | Position state | Positions count |
| `odoo.erp` | ERP business signal | Odoo ERP Panel |

## Dashboard Layout

```
┌──────────────────────────────────────────────┐
│ Header: RISK score | LIVE indicator | PAPER  │
├──────────────────────────────────────────────┤
│ Balance │ Win Rate │ Drawdown │ Positions    │
├─────── Tabs: overview│trading│risk│micro│inf│odoo ────┤
│                                              │
│ [OVERVIEW] Agent Swarm Viz │ Latest Signal   │
│           Price Chart (2col) │ Risk Gauge    │
│                                              │
│ [TRADING] Market Structure │ Event Log       │
│                                              │
│ [RISK] Risk Parameters │ Kill Switch         │
│                                              │
│ [MICRO] Microstructure │ Structure │ Signals │
│                                              │
│ [INFERENCE] Provider Health │ Agent Chains    │
│           Adaptive Routing │ Latency Heatmap │
│           Agent Mappings │ Cost/Cache        │
│                                              │
│ [ODOO] Connection │ Inventory │ Sales │ Rev  │
│        ERP Signal │ Signal Mapping Reference │
├──────────────────────────────────────────────┤
│ Footer                                       │
└──────────────────────────────────────────────┘
```

## WebSocket Event Routing

The frontend WebSocket handler routes NATS events by subject prefix:

```javascript
// Subject-based routing
if (subject.startsWith("signals.raw.")) → update brain agents
if (subject.startsWith("portfolio.")) → update portfolio state
if (subject.startsWith("market.")) → update market data
if (subject === "signals.aggregated") → log consensus
if (subject.startsWith("odoo.")) → update Odoo ERP panel
```

## Performance Rules

- **No polling** — only WebSocket push
- **Batch UI updates** — 50-100ms debounce
- **Redis/NATS pub/sub** — not HTTP
- **GPU inference isolated** — separate server
- **CPU only** — dashboard logic runs on CPU

## How we use it

- Frontend components: `frontend/src/components/` — AgentSwarmVisor, PriceChart, MicrostructurePanel, MarketStructurePanel, InferenceRoutingPanel, OrderbookHeatmap, OdooErpPanel
- Go aggregator: `realtime/nats_orchestrator.go` — event routing and WebSocket push
- Event types: `orchestrator/events.py` — all NATS event definitions
- Brain weights: `realtime/config.go` — DefaultBrainWeights for aggregation

## Strengths & weaknesses

**Strengths:**
- Real-time: sub-second latency from brain signal to dashboard
- Comprehensive: 12 brains + market data + ERP + risk in one view
- Modern stack: Next.js + Three.js + lightweight-charts
- Event-driven: no polling overhead, scales horizontally

**Weaknesses:**
- WebSocket reconnection adds 3s latency on disconnect
- Three.js 3D swarm may be GPU-intensive on low-end devices
- No historical data persistence in dashboard (current state only)
- No authentication/authorization on WebSocket endpoint

## Related

- [[odoo-erp-trading-integration]] — Odoo ERP panel in dashboard
- [[signal-aggregation-logic]] — how brain signals flow to dashboard
- [[odoo-erp-brain]] — Brain #12 feeding Odoo panel
- [[local-trading-ai-architecture]] — overall system design
- [[agent-swarm-visor]] — 3D brain swarm visualization component
- [[price-chart]] — TradingView candlestick chart component
- [[microstructure-panel]] — orderbook/delta/spoofing/cascade panel
- [[inference-routing-panel]] — multi-provider inference observability
- [[orderbook-heatmap]] — L2 depth visualization with whale detection
- [[infrastructure-overview]] — Docker Compose + K8s + Terraform deployment
- [[odoo-erp-panel]] — planned Odoo ERP business intelligence panel

## Sources

- Internal design: AGENTS.md §REAL-TIME AI TRADING DASHBOARD
- Implementation: `frontend/src/app/page.tsx`, `frontend/src/components/*.tsx`
