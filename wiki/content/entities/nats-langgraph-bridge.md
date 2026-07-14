---
title: "NATS↔LangGraph Bridge"
type: entity
tags: [nats, langgraph, bridge, event-driven, pipeline-trigger, candle-subscriber]
created: 2026-07-01
updated: 2026-07-14
status: draft
---

# NATS↔LangGraph Bridge

Subscribes to NATS `market.candle.>` events and triggers the multi-agent LangGraph pipeline. Publishes pipeline results (signals, risk decisions, metrics) back to NATS as typed events.

## Architecture

```
NATS market.candle.<symbol>
        ↓
CandleSubscriber (_on_candle_close)
        ↓
CandleBuffer.push() (hot state, last 200 candles)
        ↓
MultiAgentPipeline.process_candle()
        ↓
Publish results:
  ├── TradeSignalEvent → signals.raw.langgraph_pipeline.<symbol>
  ├── RiskCheckEvent → system.risk.check
  ├── RiskApprovedEvent → system.risk.approved
  ├── RiskRejectedEvent → system.risk.rejected
  ├── SystemEvent → system.info
  └── WSEvent → ws.update
```

## Event Flow

| Input | NATS Subject | Trigger |
|-------|-------------|---------|
| Candle close | `market.candle.>` | Pipeline execution |

| Output | NATS Subject | Content |
|--------|-------------|---------|
| Strategy signal | `signals.raw.langgraph_pipeline.<symbol>` | TradeSignalEvent |
| Risk check | `system.risk.check` | RiskCheckEvent |
| Risk approved | `system.risk.approved` | RiskApprovedEvent |
| Risk rejected | `system.risk.rejected` | RiskRejectedEvent |
| Pipeline metrics | `system.info` | SystemEvent (latency, stage) |
| WS update | `ws.update` | WSEvent (full result for frontend) |

## Usage

```python
from orchestrator.nats_langgraph_bridge import NATSLangGraphBridge

bridge = NATSLangGraphBridge(
    nats_url="nats://localhost:4222",
    symbols=["BTCUSDT", "ETHUSDT"],
)
await bridge.start()
# Bridge now listens for candle events and runs pipeline automatically

# Manual trigger (for testing)
result = await bridge.process_single("BTCUSDT")
```

## Statistics

```python
bridge.get_stats()
# → {"processed": 42, "errors": 0, "running": True, "nats_connected": True}
```

## How we use it

- Source file: `orchestrator/nats_langgraph_bridge.py`
- Integrates: `orchestrator/langgraph_pipeline.py`, `orchestrator/candle_buffer.py`, `orchestrator/events.py`
- Production trigger: NATS `market.candle.>` events from Binance WebSocket feed
- Testing: `process_single()` method for manual pipeline invocation

## Related

- [[multi-agent-pipeline]] — the LangGraph pipeline that the bridge triggers
- [[nats-event-system]] — NATS event types used by the bridge
- candle-buffer — missing wiki page; hot-state contract still needs documentation
- [[brain-ecosystem]] — 12 brains that also publish to NATS (separate from LangGraph pipeline)

## Sources

- Internal code: `orchestrator/nats_langgraph_bridge.py`
- Internal design: Multi-Agent Orchestration sketch §8 (Event-Driven Orchestration)
