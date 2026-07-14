---
title: Local Trading AI Architecture
type: concept
tags: [local-ai, ollama, rtx4050, real-time-trading, incremental-learning]
created: 2026-06-30
updated: 2026-07-14
sources: []
status: draft
---

# Local Trading AI Architecture

## Definition

A fully autonomous, real-time trading intelligence system optimized for local execution on consumer hardware (RTX 4050 6GB VRAM, 16GB RAM) using Ollama as the LLM runtime. The system processes 1-minute candle increments incrementally — never recomputing full history — and maintains compressed memory state across all timescales.

## Intuition

Traditional trading AI systems require cloud GPUs, full-history retraining, and separate services for vision/reasoning/execution. This architecture proves that a **single-machine, sequential-model, incremental-state** design can achieve sub-200ms latency for scalping decisions while maintaining deep reasoning capability for macro decisions — all without internet dependency after model download.

## Mechanics / math

### VRAM Budget (6GB RTX 4050)

| Model | 4-bit VRAM | Role | Load Strategy |
|-------|------------|------|---------------|
| phi3:3.8b | ~2.5GB | Scalping (always warm) | Persistent |
| qwen3:8b | ~5.2GB | Normal reasoning | On-demand |
| deepseek-r1:8b | ~5.2GB | Macro (async) | On-demand |
| mistral:7b | ~4.5GB | Tool execution | On-demand |
| moondream | ~1.8GB | Vision (optional) | On-demand |
| nomic-embed-text | ~0.5GB | Embeddings | Persistent |

**Constraint**: Only ONE 8B-class model fits in VRAM at a time. Solution: **sequential loading with phi3 warm**.

### Incremental State (No Recompute)

```
Level 0: Active Candle (updated every 60s)
  - OHLCV + 50 features + regime hint

Level 1: Rolling Buffer (200 candles, fixed)
  - Compressed feature vectors (64-dim each)
  - Streaming HMM regime state (5 states)

Level 2: Key Levels (persistent, event-driven)
  - Support/Resistance zones (volume-weighted)
  - Volume profile nodes (high-volume areas)

Level 3: Session Context (daily reset)
  - VWAP, session high/low, daily bias

Level 4: Strategy Performance (per-trade update)
  - Win/loss, avg R, regime-conditioned stats
```

### Model Routing Logic

```python
def select_model(context: dict) -> str:
    if context.get("task") == "vision":
        return "moondream"
    if context.get("task") == "execution":
        return "mistral"
    if context.get("mode") == "scalp" or context.get("urgency") == "high":
        return "phi3:3.8b"      # <100ms, always warm
    if context.get("mode") == "deep":
        return "deepseek-r1:8b" # async, >2s
    return "qwen3:8b"           # <300ms, load on demand
```

## How we use it

- **Scalping path**: phi3:3.8b (warm) → features → decision → risk → execute (~185ms)
- **Normal path**: qwen3:8b (load 500ms + infer 250ms) → decision
- **Macro path**: deepseek-r1:8b (background) → writes to memory
- **Vision path**: moondream (on screenshot) → chart analysis → memory
- **Execution**: mistral:7b (load → tool calls → unload)

## Strengths & weaknesses

| Strength | Weakness |
|----------|----------|
| Zero cloud cost / privacy | 6GB VRAM limits concurrent models |
| Sub-200ms scalping latency | Sequential loading adds latency spikes |
| Incremental state = no retrain | Phi3 reasoning ceiling lower than 8B |
| Fully offline after download | No multi-GPU scaling |
| Deterministic execution | Model swap requires Ollama API call |

## Related

- [[trade-project-full-integration-build-plan]] — current 6 GB VRAM model and runtime decision
- [[rtx4050-trading-system]] — hardware-specific deployment entity
- [[local-ai-deployment-guide]] — installation playbook
- [[incremental-candle-state]] — memory architecture concept
- [[model-sequential-loading]] — VRAM management technique
- [[signal-aggregation-logic]] — brain weight aggregation (existing)

## Contradictions / updates

**2026-07-14 hardware audit:** 8B 4-bit model weights consume nearly all 6 GB VRAM before KV cache and runtime overhead, so qwen3:8b/deepseek-r1:8b are not safe always-on choices. Model load and sub-200 ms generation figures are unverified. Execution must not use an LLM. The current target in [[trade-project-full-integration-build-plan]] is one 3–4B Ollama model at 4K context and concurrency one, used asynchronously; deterministic code owns signals, risk, and orders.

## Sources

- Internal design session 2026-06-30
