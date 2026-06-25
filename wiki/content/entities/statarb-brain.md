---
title: StatArb Brain
type: entity
tags: [brain, stat-arb, mean-reversion, funding-rate, basis-spread]
created: 2026-06-26
updated: 2026-06-26
weight: 0.10
brain_id: statarb_funding
source_file: statarb_brain.py
status: active
---

# StatArb Funding Brain

## Overview

**Brain #8** — Statistical arbitrage brain. Purely contrarian. Three sub-signals blended: z-score mean reversion (45%), funding rate contrarian (35%), basis spread contrarian (20%). Zero output when no data available.

## Architecture

```
push_price ──────────→ Z-score mean reversion ──── 0.45 ──┐
push_funding_rate ──→ Funding rate contrarian ─── 0.35 ──┤→ clip(-1,+1) → score
push_basis_spread ──→ Basis spread contrarian ──── 0.20 ──┘
                                         ↘ (no data): 0.0 on each component
```

## Sub-signals

### 1. Z-score mean reversion (45%)
- 30-period lookback window
- `z = (price - mean) / std`
- Score = `-tanh(z * 0.5)` (INVERTED: high z → sell, low z → buy)
- Threshold: `z_threshold = 2.0` (below → no edge, return 0)

### 2. Funding rate contrarian (35%)
- Last 8 funding periods (~1 day at 8h intervals)
- `score = -tanh(avg_funding * 500)` (INVERTED: positive funding → sell)
- Rationale: crowded longs → overleveraged → bearish risk

### 3. Basis spread contrarian (20%)
- Last 10 basis observations
- `score = -tanh(avg_basis * 100)` (INVERTED: extreme positive → bearish)
- Ratio: `(perp_price - spot_price) / spot_price`

## Parameters

| Parameter | Default | Env var |
|-----------|---------|---------|
| lookback | 100 bars | `STATARB_LOOKBACK` |
| z_threshold | 2.0 | `STATARB_Z_THRESHOLD` |
| funding_deque | 72 (3 days at 8h) | — |
| basis_deque | 50 | — |

## Confidence model

Base 0.25 + 0.20 (≥50 prices) + 0.15 (≥8 funding) + 0.10 (≥10 basis) → max 0.85

## Outputs

- `score`: ∈ [-1, +1] (contrarian direction)
- `confidence`: ∈ [0, 1]
- `metadata`: z_score, funding_score, basis_score

## Fail modes

- No price data → z=0, confidence 0.1
- No funding rates → funding=0
- No basis data → basis=0
- All empty → score=0, confidence=0.1

## Edge & known weaknesses

- **Edge:** Contrarian signal is orthogonal to trend-following brains — decorrelates portfolio
- **Edge:** Funding rate is a unique crypto signal (no traditional finance equivalent)
- **Weakness:** Mean reversion fails in trending markets (gets stepped on repeatedly)
- **Weakness:** Small model — 3 simple tanh signals, no learning
- **Weakness:** Funding rate is 8h lagging — fast crowding changes missed
- **10× opportunity:** Add Kalman filter for dynamic z-score threshold; add open interest delta as 4th sub-signal

## Related

- [[timesfm-brain]] — trend signal (orthogonal to StatArb's contrarian nature)
- [[finbert-brain]] — sentiment can explain WHY funding is extreme
- [[onchain-brain]] — whale flows often precede funding rate shifts
- [[order-flow-imbalance]] — OFI divergence from mean reversion = warning

## Sources

- Internal code: `orchestrator/brains/statarb_brain.py`
