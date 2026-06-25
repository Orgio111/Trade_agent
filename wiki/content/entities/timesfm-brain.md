---
title: TimesFM Brain
type: entity
tags: [brain, forecasting, timesfm, time-series]
created: 2026-06-26
updated: 2026-06-26
weight: 0.25
brain_id: timesfm
source_file: timesfm_brain.py
status: active
---

# TimesFM Brain

## Overview

**Brain #1** — Primary forecaster. Highest weight (0.25) in the orchestrator. Uses Google's TimesFM 2.5 200M foundation model for zero-shot time-series forecasting. Direction output: long (+1), short (-1), or hold (0).

## Architecture

```
push_price → price_buffer (max 512) → TimesFM forecast(horizon=12) → direction * confidence
                                     ↘ fallback: momentum heuristic (20-period ROC, tanh-scaled)
```

## Parameters

| Parameter | Default | Env var |
|-----------|---------|---------|
| direction_threshold | 0.002 (0.2%) | — |
| horizon | 12 bars | — |
| max_context | 512 | `TIMESFM_MAX_CONTEXT` |
| checkpoint | `google/timesfm-2.5-200m-pytorch` | `TIMESFM_CHECKPOINT` |

## Outputs

- `score`: direction × confidence ∈ [-1, +1]
- `confidence`: model confidence ∈ [0, 1]
- `metadata`: model, expected_return, horizon

## Fallback chain

1. **TimesFM** (if checkpoint loaded + ≥20 prices)
2. **Momentum heuristic** (20-period ROC, `tanh(roc * 50)`, confidence ≤ 0.5)

## Fail modes

- TimesFM package not installed → `ImportError`, momentum fallback
- Checkpoint missing/corrupt → momentum fallback
- Insufficient prices (< 20) → score=0, confidence=0.1

## Edge & known weaknesses

- **Edge:** Zero-shot forecast — no retraining needed, adapts to any market regime
- **Weakness:** 200M model is small; forecasts degrade in high-volatility regime shifts
- **Weakness:** No volume/order-flow input — pure price series
- **10× opportunity:** Upgrade to TimesFM 2.5 500M when available; add multi-variate inputs (volume, OFI)

## Related

- [[freqai-brain]] — technical signals complement
- [[llm-regime-brain]] — regime context can weight this brain's output
- [[order-flow-imbalance]] — OFI could augment TimesFM inputs

## Sources

- Internal code: `orchestrator/brains/timesfm_brain.py`
