---
title: FreqAI Brain
type: entity
tags: [brain, xgboost, technical-indicators, rsi, macd]
created: 2026-06-26
updated: 2026-06-26
weight: 0.15
brain_id: freqai
source_file: freqai_brain.py
status: active
---

# FreqAI Brain

## Overview

**Brain #2** — Technical indicator brain. Computes RSI, MACD, Bollinger Band position, and volume profile from OHLCV data. Uses pre-trained XGBoost model (60/40 blend with rule-based scoring) or falls back to pure rule-based scoring.

## Architecture

```
push_ohlcv → buffer (max 200 bars) → [XGBClassifier] ─── 60% ──┐
                                    → [RSI/MACD/BB rules] ── 40% ──┤→ blend → score
                                    ↘ (no model): 100% rule-based
```

## Parameters

| Parameter | Default | Env var |
|-----------|---------|---------|
| max_bars | 200 | — |
| model_dir | `models/freqai/` | `FREQAI_MODEL_DIR` |
| model_path | auto-discover | `FREQAI_MODEL_PATH` |
| rsi_period | 14 | — |
| macd (fast/slow/signal) | 12/26/9 | — |
| bb_period | 20 | — |
| model_blend | 60/40 model/rules | — |

## Outputs

- `score`: ∈ [-1, +1]
- `confidence`: ∈ [0, 1]
- `direction`: long / short / hold

## Fallback chain

1. **XGBoost + rules blend** (model loaded + ≥30 bars)
2. **Rule-based only** (RSI/MACD/BB scoring, no model)
3. **Insufficient data** (< 30 bars) → score=0, confidence=0.1

## Fail modes

- `joblib` not installed → skip model, rule-based only
- Model file corrupt → warning, rule-based only
- XGBoost not installed → rule-based only
- CCXT/Binance fetch fails → stale buffer used

## Edge & known weaknesses

- **Edge:** Hybrid blend is robust — model enhances, rules provide floor
- **Edge:** Auto-train from historical data (100+ bars triggers `_auto_train()`)
- **Weakness:** XGBoost overfits on short training windows
- **Weakness:** Rule-based fallback is lagging (RSI/MACD are reactive, not predictive)
- **10× opportunity:** Add frequency-domain features (FFT coefficients); integrate [[order-flow-imbalance]] as input feature

## Related

- [[timesfm-brain]] — primary forecaster complement
- [[llm-regime-brain]] — regime context can modulate RSI thresholds
- [[finrl-brain]] — position sizing uses this brain's directional signal

## Sources

- Internal code: `orchestrator/brains/freqai_brain.py`
