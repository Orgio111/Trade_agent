---
title: "Incremental Candle State"
type: concept
tags: [candle, state, incremental, streaming, memory-efficient]
created: 2026-06-30
updated: 2026-06-30
status: stable
---

# Incremental Candle State

## Definition

A 4-level incremental state management system for candle data that avoids reprocessing the full history on each tick. Maintains: active candle (current), rolling buffer (last N), key levels (support/resistance), and session context (session high/low/volume). Each new tick updates only the relevant level, enabling O(1) per-tick processing instead of O(N) full reprocessing.

## Intuition

Traditional candle processing re-reads the entire history on each tick. For a 1-minute scalping system on RTX 4050, this wastes precious milliseconds. Incremental state means each new price tick only touches the data structures that need updating — the active candle, the rolling window, and the session statistics. This is critical for maintaining <100ms latency on consumer hardware.

## 4-Level State

```
Level 1: Active Candle
  └── Current open/high/low/close/volume (updated every tick)

Level 2: Rolling Buffer
  └── Last 50-200 candles (deque, O(1) append/drop)

Level 3: Key Levels
  └── Support/resistance, order blocks, FVG gaps (updated on candle close)

Level 4: Session Context
  └── Session high/low, total volume, VWAP (updated every tick)
```

## How we use it

- Referenced by: `orchestrator/brains/` — all brains consume incremental candle state
- Part of: `orchestrator/data_pipeline.py` — data ingestion and state management
- Critical for: RTX 4050 latency budget (100ms per tick)
- Memory efficient: deque-based rolling buffer with max length

## Related

- [[model-sequential-loading]] — VRAM management for model loading alongside candle state
- [[local-trading-ai-architecture]] — overall local AI system design
- [[brain-ecosystem]] — brains that consume candle state

## Sources

- Internal design: AGENTS.md §Real-Time AI Trading Dashboard
