---
title: "Multi-Agent Trading Pipeline"
type: concept
tags: [pipeline, langgraph, multi-agent, orchestration, parallel, vlm, rag, swarm, risk, scalping, fast-path, dual-path]
created: 2026-07-01
updated: 2026-07-01
status: stable
---

# Multi-Agent Trading Pipeline

Production LangGraph state machine with **dual-path routing**: a CPU-only fast path (<5ms) for high-confidence scalping decisions, and a heavy GPU path (~500ms) for full multi-agent analysis.

## Definition

A deterministic, event-driven pipeline that replaces the previous disconnected systems with a single coordinated flow. The pipeline dynamically routes between two execution paths based on scalping confidence:

- **Fast Path** (<5ms): ScalpingEngine → risk gate → execution (skips VLM/RAG/swarm)
- **Heavy Path** (~500ms): VLM → RAG → swarm debate → risk gate → execution

Every agent is a specialist; the orchestrator is the hedge fund desk manager.

## Intuition

When a candle closes, the pipeline:
1. **Validates** the candle data
2. **Computes** technical indicators (RSI, MACD, ATR, regime)
3. **Runs scalping engine** (CPU-only, <5ms) — checks momentum, rejection, trend alignment
4. **Routes dynamically**:
   - If scalping confidence > threshold (default 80%) → **fast path** (skip VLM/RAG/swarm)
   - Otherwise → **heavy path** (full multi-agent analysis)
5. **Checks** risk gates (drawdown, position size, volatility, leverage)
6. **Executes** if approved (simulated in paper mode)
7. **Logs** and publishes all results to NATS (including ScalpDecisionEvent for fast-path trades)

## Dual-Path Architecture

```
                    ┌─────────────────────────────────────────────────────┐
                    │              LangGraph State Machine v3             │
                    │                                                     │
                    │  candle_close ─→ data_ingest ─→ feature_engine      │
                    │                                    │                │
                    │                                    ▼                │
                    │                              scalping_node          │
                    │                              (CPU <5ms)            │
                    │                                    │                │
                    │                    ┌───────────────┴───────────┐    │
                    │                    │ conf > threshold?         │    │
                    │                    ▼ YES                       ▼ NO  │
                    │              ┌──────────┐    ┌──→ vlm_agent ──┐    │
                    │              │ risk_gate│    └──→ rag_agent ──┤    │
                    │              │ (FAST)   │            ↓       │    │
                    │              └────┬─────┘     swarm_strategy │    │
                    │                   │              ↓           │    │
                    │                   │         risk_gate (HEAVY)│    │
                    │                   │              ↓           │    │
                    │                   └──→ execution → log_and_publish │
                    └─────────────────────────────────────────────────────┘
```

### Fast Path (Scalping)
- **Trigger:** ScalpingEngine confidence > configurable threshold (default 80%)
- **Latency:** <5ms (CPU-only, no I/O)
- **Skips:** VLM analysis, RAG retrieval, swarm debate
- **Use case:** High-confidence momentum breakouts, sharp rejections at S/R
- **Events:** Publishes `ScalpDecisionEvent` to `signals.scalp.<symbol>`

### Heavy Path (Full Analysis)
- **Trigger:** Scalping confidence ≤ threshold, or HOLD signal
- **Latency:** ~500ms (VLM + RAG + swarm)
- **Includes:** VLM chart analysis, RAG pattern search, 7-agent swarm debate
- **Use case:** Ambiguous signals, low-confidence setups, regime transitions

### Configurable Threshold

```bash
# Environment variable (default: 80)
export SCALPING_CONFIDENCE_THRESHOLD=70

# Or per-pipeline override
state = TradingState(scalping_confidence_threshold=60)
```

When threshold is lowered, more trades take the fast path (lower latency but less analysis). When raised, more trades go through the full heavy path (higher latency but more conviction).

## Graph Topology

```
data_ingest → feature_engine → scalping_node
                                    │
                            ┌───────┴───────┐
                            │ conf > 80%?   │
                            ▼ YES           ▼ NO
                      risk_gate       vlm_agent → rag_agent → swarm → risk_gate
                            │               │
                            └───────┬───────┘
                                    ▼
                            execution → log_and_publish
```

### Conditional Routing

The scalping node uses LangGraph's `add_conditional_edges` for dynamic routing:

```python
graph.add_conditional_edges(
    "scalping_node",
    _route_after_scalping,  # Pure function: checks confidence vs threshold
    {
        "risk_gate": "risk_gate",    # Fast path
        "vlm_agent": "vlm_agent",    # Heavy path
    },
)
```

The routing function is **pure** — it only reads state and returns the next node name. Strategy signal assignment happens in `scalping_node`, not in the router.

## Node Details

| Node | Function | Latency | Path | Blocking | Fallback |
|------|----------|:-------:|:----:|:--------:|----------|
| `data_ingest` | Validate candle OHLCV | <1ms | Both | Yes | Error → pipeline stops |
| `feature_engine` | Compute RSI, MACD, ATR, regime | 5-20ms | Both | No | Minimal features |
| `scalping_node` | CPU-only scalping decision | <5ms | Both | No | HOLD |
| `vlm_agent` | Chart pattern recognition (GPU) | 1-3s | Heavy | No | trend="unknown" |
| `rag_agent` | Vector search for similar patterns | 50-200ms | Heavy | No | Empty context |
| `swarm_strategy` | 7-agent debate + voting | 5-30s | Heavy | Yes | Feature-based fallback |
| `risk_gate` | 10-gate risk check | <1ms | Both | Yes | Reject all |
| `execution` | Order placement | 50-500ms | Both | No | Skip |
| `log_and_publish` | NATS event publishing | <1ms | Both | No | Log error |

### Latency Comparison

| Path | Total Latency | Use Case |
|------|:-------------:|----------|
| **Fast** (scalping) | **<10ms** | High-confidence momentum/rejection |
| **Heavy** (full) | **~500ms** | Ambiguous, low-confidence, regime change |

## Strategy Agent (Swarm Debate)

The strategy node uses the existing `AgentSwarm` with 7 specialized agents:

| Agent | Focus | Latency |
|-------|-------|:-------:|
| Scalping | 1-5m microstructure | Fast |
| Swing | 4h-1d setups | Medium |
| Sentiment | Fear & Greed index | Fast |
| Regime | Market regime classification | Fast |
| Anomaly | Volume spikes, manipulation | Fast |
| DeepSeek | Deep reasoning analysis | Slow |
| Market | Technical analysis | Medium |

**Debate rounds:**
1. Parallel independent analysis
2. Cross-examination (agents challenge each other)
3. Weighted vote aggregation with credibility scores
4. Risk veto check (blocking)

## Risk Gate

10 cascading risk checks before execution:

1. Kill switch (25% DD = halt)
2. Daily loss limit (5%)
3. Consecutive losses (4 = cooldown)
4. Position count (max 3)
5. Portfolio exposure (15% max)
6. Leverage limit (10x)
7. Volatility filter (95th percentile = kill)
8. Correlation check (70% max)
9. Time decay (inactivity penalty)
10. Anti-overtrading (20 trades/hour)

## Trace Propagation

Every pipeline run generates a unique `trace_id` (UUID) that flows through:
- Pipeline state → NATS events → Frontend WebSocket
- Enables end-to-end debugging, replay, and audit

## State Fields

The `TradingState` carries scalping-specific fields for dual-path routing:

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `scalping_confidence` | int | 0 | 0-100, from ScalpingEngine |
| `scalping_action` | str | "HOLD" | BUY, SELL, or HOLD |
| `scalping_decision` | dict | {} | Full decision dict |
| `use_scalping_fast_path` | bool | True | Enable/disable fast path |
| `scalping_confidence_threshold` | int | 80 | Per-pipeline threshold override |

## Pipeline Context

The `log_and_publish_node` includes scalping metadata in `pipeline_context`:

```json
{
  "scalping_threshold": 80,
  "scalping_confidence": 92,
  "scalping_action": "BUY",
  "fast_path": true,
  "strategy_source": "scalping_fast_path"
}
```

This allows the NATS bridge to publish accurate `ScalpDecisionEvent` metadata.

## How we use it

- Source file: `orchestrator/langgraph_pipeline.py` (v3, dual-path)
- Scalping engine: `orchestrator/scalping_engine.py` (CPU-only, <5ms)
- Triggered by: `orchestrator/nats_langgraph_bridge.py` (NATS candle events)
- Uses: `orchestrator/candle_buffer.py` (hot state), `orchestrator/vlm_agent.py`, `orchestrator/rag_agent.py`
- Integrates: `orchestrator/swarm/debate_system.py` (existing swarm debate)
- Risk: `orchestrator/risk/risk_engine.py` (existing 10-gate risk)
- GPU: `orchestrator/gpu_optimizer.py` (RTX 4050 tuning)
- Inference: `orchestrator/local_inference_server.py` (vLLM + FastAPI)

## Related

- [[scalping-engine]] — CPU-only fast-path decision engine
- [[nats-langgraph-bridge]] — NATS event trigger, publishes ScalpDecisionEvent
- [[vlm-agent]] — Vision analysis node (heavy path only)
- [[rag-agent]] — Pattern retrieval node (heavy path only)
- [[nats-event-system]] — Event types including ScalpDecisionEvent, ScalpExecutedEvent
- [[brain-ecosystem]] — 12 brains (separate from LangGraph pipeline)
- [[ensemble-meta-model]] — Alternative signal fusion approach
- [[signal-aggregation-logic]] — Go aggregator (separate aggregation path)
- [[chart-segmentation]] — OpenCV chart feature extraction
- [[local-inference-server]] — vLLM GPU inference server

## Sources

- Internal code: `orchestrator/langgraph_pipeline.py`
- Internal design: Multi-Agent Orchestration sketch (pasted text)
- ACP protocol: `orchestrator/events.py` (v2 event types)
