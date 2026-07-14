---
title: "VLM Agent"
type: entity
tags: [vlm, vision, chart-analysis, ollama, moondream, pattern-recognition, gpu]
created: 2026-07-01
updated: 2026-07-14
status: draft
---

# VLM Agent

Vision Language Model agent for chart pattern recognition. Renders candlestick charts as PNG images and sends them to Ollama VLM (moondream/llava) for structured technical analysis.

## Architecture

```
CandleBuffer (last 60 candles)
        ↓
Matplotlib chart rendering (dark theme, candlesticks + EMA + volume)
        ↓
Base64 encode PNG
        ↓
Ollama VLM endpoint (/api/generate)
        ↓
Structured JSON parse (trend, pattern, support/resistance, confidence)
        ↓
VLMResult → Strategy Agent
```

## Pipeline Integration

The VLM Agent runs as a **parallel node** in the LangGraph pipeline — alongside the RAG Agent:

```
feature_engine → [vlm_agent ‖ rag_agent] → swarm_strategy
```

This parallel fan-out reduces total pipeline latency by ~30-50% compared to sequential execution.

## Output Format

```python
@dataclass
class VLMResult:
    trend: str       # "bullish", "bearish", "neutral"
    pattern: str     # "ascending_triangle", "head_shoulders", etc.
    support: float   # Key support level
    resistance: float # Key resistance level
    confidence: float # 0.0–1.0
    reasoning: str   # Human-readable analysis
    model: str       # "moondream", "llava"
    latency_ms: float
```

## Chart Rendering

- Dark theme (#1a1a2e background, #16213e plot area)
- Last 60 candles with candlestick bodies + wicks
- EMA 20 (gold) and EMA 50 (red) overlays
- Volume bars (green/red) in separate subplot
- Output: 12×8 inch, 100 DPI PNG

## Model Configuration

| Model | Latency | Quality | Use Case |
|-------|:-------:|:-------:|----------|
| moondream | ~1-2s | Good | Fast pattern detection |
| llava | ~3-5s | Better | Detailed analysis |

Configurable via `VLM_MODEL` env var (default: `moondream`).

## Fallback Behavior

If VLM is unavailable (Ollama not running, model not loaded, timeout), the agent returns `VLMResult(error=...)` and the pipeline continues. VLM is an **enhancement**, not a blocker.

## How we use it

- Source file: `orchestrator/vlm_agent.py`
- Pipeline node: `orchestrator/langgraph_pipeline.py` → `vlm_agent_node()`
- Consumed by: `swarm_strategy_node()` via `state.vlm_output`
- Also used by: `_fallback_signal()` when swarm debate is unavailable

## Strengths & weaknesses

**Strengths:**
- Visual pattern recognition that complements numerical indicators
- Non-blocking — pipeline continues if VLM fails
- Dark-theme chart optimized for VLM consumption

**Weaknesses:**
- Depends on Ollama server availability
- ~1-2s latency per inference (GPU-bound)
- Pattern recognition quality depends on VLM model capability
- No image caching — re-renders chart each call

## Related

- [[multi-agent-pipeline]] — pipeline architecture showing VLM parallel execution
- candle-buffer — missing wiki page; chart-history input contract still needs documentation
- [[nats-langgraph-bridge]] — triggers pipeline that includes VLM analysis
- [[local-trading-ai-architecture]] — Ollama model deployment for RTX 4050
- [[agent-swarm-visor]] — frontend 3D visualization of brain swarm

## Sources

- Internal code: `orchestrator/vlm_agent.py`
- Internal design: Multi-Agent Orchestration sketch §3 (VLM Agent role)
