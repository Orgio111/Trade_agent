---
title: Polymarket Brain
type: entity
tags: [brain, polymarket, prediction-market, bayesian, kelly, nash, #11]
created: 2026-06-30
updated: 2026-06-30
weight: 0.05
brain_id: polymarket_alpha
source_file: polymarket_brain.py
status: active
---

# Polymarket Brain

## Overview

**Brain #11** — Polymarket Alpha Engine. Weight 0.05 (supplementary). A 5-formula mathematical pipeline derived from 72M trade analysis of prediction markets. Processes Polymarket contracts through Bayesian evidence tracking, longshot bias correction, EV/ROI filtering, Quarter-Kelly position sizing, and Nash-equilibrium order routing.

## Architecture

```
Evidence (news, order flow, signals)
    │
    ▼
F1: Bayesian Evidence Tracking
  LR = Π reliability_i × (1 + |impact_i|)
  posterior = prior × LR / (prior × LR + (1 − prior))
    │
    ▼
F2: Longshot Bias Correction
  price < 0.10 → ×0.57 (penalty)
  price > 0.80 → +0.02 (boost)
    │
    ▼
F3: EV & ROI Filter
  EV = adjusted_prob × payout − cost
  Trade only if EV > 0
    │
    ▼
F4: Quarter-Kelly Sizing
  f_kelly = (b·p − q) / b
  f_safe = 0.25 × f_kelly, capped at 5%
    │
    ▼
F5: Nash Router
  70% MAKER default
  TAKER if edge > 20%
  MAKER edge = +1.12%
    │
    ▼
AlphaResult → BrainSignal
```

## The 5 Formulas

### Formula 1: Bayesian Evidence Tracking
- Likelihood Ratio shortcut instead of full Bayesian update
- Each evidence piece contributes: `LR *= reliability × (1 + |impact|)`
- Sudden LR jump > 0.20 triggers re-evaluation
- Sources: news headlines, order flow signals, whale activity

### Formula 2: Longshot Bias Correction
- Empirical finding from 72M trades: low-probability events are systematically overpriced
- `price < 0.10` → actual win rate = 57% of quoted price (penalty)
- `price > 0.80` → actual win rate = quoted price + 2% (boost)

### Formula 3: Expected Value & ROI Filter
- `EV = adjusted_prob × payout − cost`
- `ROI = EV / cost`
- Hard filter: only trade if EV > 0 (positive expected value)

### Formula 4: Quarter-Kelly Position Sizing
- Full Kelly: `f = (b·p − q) / b` where b = profit/loss ratio
- Safe sizing: `f_safe = 0.25 × f_kelly` (conservative)
- Hard cap: 5% of bankroll per position

### Formula 5: Nash Equilibrium Order Routing
- Default: 70% MAKER orders (capture maker rebate)
- Switch to TAKER when edge > 20% (need fill certainty)
- MAKER edge bonus: +1.12% from empirical analysis

## Data Injection

```python
brain = PolymarketBrain()

# Push evidence (news, signals)
brain.push_evidence(
    source="Reuters",
    headline="Emergency BJ meeting scheduled",
    impact=0.90,      # -1.0 (strong NO) to +1.0 (strong YES)
    reliability=0.92,  # 0.0 to 1.0
)

# Push market state
brain.push_market(
    question="Fed Rate Cut July 2026",
    yes_price=0.35,
    volume_24h=500000,
    liquidity=100000,
)
```

## Confidence Model

- Base: 0.3
- +0.4 if `should_trade` AND evidence_count >= 2
- +0.35 if `should_trade` with 1 evidence
- Max: 0.85

## How it fits in the brain ecosystem

See [[brain-ecosystem]] for the full weight table. This brain sits at the supplementary tier (0.05) alongside onchain_whale, custom_nn, and odoo_erp. Total ensemble: 12 brains, weight sum = 1.00.

## Registration

| File | Status |
|------|--------|
| `orchestrator/brains/polymarket_brain.py` | ✅ Implemented |
| `orchestrator/brains/__init__.py` | ✅ Import + BRAIN_REGISTRY |
| `orchestrator/main.py` | ✅ BRAIN_INTERVALS |
| `orchestrator/brain_backtest.py` | ✅ BRAIN_WEIGHTS (0.05) |
| `orchestrator/brains/brain_registry.json` | ✅ Metadata entry |
| `realtime/config.go` | ✅ DefaultBrainWeights (0.05) |

## Config

All empirical constants loaded from `config/polymarket.yaml` with safe defaults:

| Parameter | Default | Source |
|-----------|---------|--------|
| longshot_low_threshold | 0.10 | 72M trade analysis |
| longshot_high_threshold | 0.80 | 72M trade analysis |
| longshot_low_penalty | 0.57 | 72M trade analysis |
| longshot_high_boost | 0.02 | 72M trade analysis |
| kelly_quarter_factor | 0.25 | Standard quarter-Kelly |
| kelly_hard_cap | 0.05 | Risk management |
| maker_default_probability | 0.70 | Nash equilibrium |
| maker_edge_pct | 0.0112 | Empirical analysis |
| taker_threshold_pct | 0.20 | Edge threshold |

## Related

- [[brain-ecosystem]] — full brain weight table
- [[ensemble-meta-model]] — could consume Polymarket signals as additional source
- [[signal-aggregation-logic]] — how this brain's signal is weighted
- [[onchain-brain]] — similar supplementary brain pattern

## Sources

- Internal code: `orchestrator/brains/polymarket_brain.py`
- Config: `config/polymarket.yaml`
