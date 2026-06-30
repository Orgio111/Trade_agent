---
title: "RAG Agent"
type: entity
tags: [rag, retrieval, vector-search, qdrant, embeddings, pattern-memory, nim]
created: 2026-07-01
updated: 2026-07-01
status: active
---

# RAG Agent

Retrieval Augmented Generation agent for trade pattern memory. Queries Qdrant vector store for historically similar trade setups and provides context to the strategy agent for informed decisions.

## Architecture

```
Current Market State (symbol, regime, RSI, MACD, price, VLM trend)
        ↓
Query text construction
        ↓
NVIDIA NIM embedding (nv-embedqa-e5-v5, 1536-dim)
        ↓
Qdrant vector search (cosine similarity, top-k=5)
        ↓
Pattern statistics computation (win rate, avg PnL, confidence adjustment)
        ↓
RAGContext → Strategy Agent
```

## Pipeline Integration

The RAG Agent runs as a **parallel node** alongside the VLM Agent:

```
feature_engine → [vlm_agent ‖ rag_agent] → swarm_strategy
```

## Output Format

```python
@dataclass
class RAGContext:
    documents: list[dict]      # Retrieved document payloads
    similar_trades: list[dict] # Trades with similarity scores
    pattern_stats: dict        # win_rate, avg_pnl, confidence_adjustment
    query_time_ms: float       # Retrieval latency
    source: str                # "qdrant", "local", "fallback"
```

## Retrieval Pipeline

1. **Query construction** — builds text from current market state (symbol, regime, RSI, MACD, volume ratio, price, VLM trend)
2. **Embedding** — NVIDIA NIM `nv-embedqa-e5-v5` (1536-dim), falls back to deterministic hash embedding if NIM unavailable
3. **Search** — Qdrant `cosine` search with `score_threshold=0.3`, top-k results
4. **Statistics** — computes win rate, avg PnL, confidence adjustment from retrieved patterns

## Trade Storage

```python
await rag.store_trade(
    symbol="BTCUSDT", direction="long", entry_price=50000,
    regime="trending", rsi=65.0, pnl=12.5, outcome="win",
    lesson="RSI oversold bounce in uptrend"
)
```

Stored embeddings enable future semantic retrieval of similar setups.

## Confidence Adjustment

The RAG agent computes a `confidence_adjustment` based on historical win rate of similar patterns:

```
confidence_adjustment = (win_rate - 0.5) × 0.2
```

This adjusts the strategy agent's confidence by ±0.1 based on historical evidence.

## Fallback Chain

1. **Qdrant** (primary) — real vector search via HTTP API
2. **Local memory** (fallback) — in-memory cosine similarity on stored embeddings
3. **Empty result** (degraded) — pipeline continues without RAG context

## How we use it

- Source file: `orchestrator/rag_agent.py`
- Pipeline node: `orchestrator/langgraph_pipeline.py` → `rag_agent_node()`
- Consumed by: `swarm_strategy_node()` via `state.rag_context`
- Also used by: `_fallback_signal()` for pattern stats confidence adjustment
- Memory storage: `orchestrator/memory/vector_memory.py` (Qdrant wrapper)

## Strengths & weaknesses

**Strengths:**
- Provides historical context for current market conditions
- Semantic search finds similar patterns even with different exact parameters
- Non-blocking — pipeline continues if RAG fails
- Confidence adjustment improves signal quality over time

**Weaknesses:**
- Cold start: first queries have no stored patterns
- Embedding quality depends on NVIDIA NIM availability
- Hash-based fallback embedding has low semantic resolution
- Small sample sizes (<5 similar trades) produce unreliable statistics

## Related

- [[multi-agent-pipeline]] — pipeline architecture showing RAG parallel execution
- [[nats-event-system]] — RAG query/result events flow through NATS
- [[nats-langgraph-bridge]] — triggers pipeline that includes RAG retrieval
- [[ensemble-meta-model]] — pattern statistics used in ensemble decisions
- [[continual-learning-pipeline]] — trade outcomes feed back into RAG memory

## Sources

- Internal code: `orchestrator/rag_agent.py`
- Internal design: Multi-Agent Orchestration sketch §3 (RAG Agent role)
- ACP protocol: `RAG_QUERY` and `RAG_RESULT` event types in `orchestrator/events.py`
