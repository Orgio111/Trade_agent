---
title: "Ensemble Meta-Model"
type: concept
tags: [ensemble, meta-model, signal-fusion, adaptive-weights, attention-entropy]
created: 2026-06-30
updated: 2026-06-30
status: stable
---

# Ensemble Meta-Model

## Definition

A meta-model that fuses ML (Random Forest), PPO RL, and rule-based (HMM regime + Smart Money Concepts) signals into a single trading decision. Uses adaptive weights that adjust based on recent accuracy (rolling 20-trade window) and Transformer attention entropy for regime-dependent scaling.

## Intuition

No single signal source works well in all market conditions. The ensemble collects signals from 4 sources, weights them by recent accuracy, and uses attention entropy from the Transformer brain to dynamically adjust weights based on whether the market is trending, ranging, or volatile. The result is a weighted vote that's more robust than any individual source.

## Architecture

```
ML/RF (signal)    PPO RL (action)    Rule-based (regime+SMC)
       │                │                     │
       └────────────────┼─────────────────────┘
                        ▼
              Signal Fusion Engine
              (adaptive weighted voting)
                        │
                        ▼
                  Risk Gate
                        │
                        ▼
                Final Decision
```

## Signal Sources

| Source | Weight | Description |
|--------|:------:|-------------|
| `ml_rf` | 1.0 | Random Forest technical predictions |
| `ppo_rl` | 1.5 | PPO RL action (highest base weight) |
| `regime` | 0.8 | HMM/ADX regime detection |
| `rule_smc` | 0.7 | Smart Money Concepts (BOS, FVG) |

## Attention Entropy Regime Adjuster

Transformer attention entropy classifies market regime and scales source weights:

| Regime | Entropy | ml_rf | ppo_rl | regime | rule_smc |
|--------|:-------:|:-----:|:------:|:------:|:--------:|
| Trending | < 2.0 | 1.2× | 1.3× | 0.7× | 1.1× |
| Ranging | 2.0–3.5 | 0.8× | 0.7× | 1.4× | 1.3× |
| Volatile | > 3.5 | 0.5× | 0.5× | 1.2× | 0.6× |

## How we use it

- Source: `orchestrator/ensemble_meta.py` — `EnsembleMetaModel` class
- Consumes outputs from: `MLSignalEngine`, PPO model, `HMMRegimeDetector`, SMC analysis
- `decide(market_data)` → weighted vote → direction + confidence + reasoning
- `record_trade_outcome(decision, pnl)` → adapts weights based on trade results
- Feeds into execution layer via [[broker-abstraction-layer]]

## Strengths & weaknesses

**Strengths:**
- Adaptive: weights improve over time as trade outcomes are recorded
- Multi-source: 4 diverse signal types reduce single-source risk
- Regime-aware: attention entropy adjusts weights dynamically
- Transparent: reasoning string shows each source's contribution

**Weaknesses:**
- Complex: 4 sources + entropy adjuster = many moving parts
- Cold start: weights are arbitrary until enough trades are recorded
- Circular: Transformer attention feeds back into weight adjustment

## Related

- [[hmm-regime]] — HMM regime detector used as signal source
- [[brain-ecosystem]] — brains that produce signals for the ensemble
- [[signal-aggregation-logic]] — how brain signals are aggregated (separate from ensemble)
- [[dynamic-brain-weight-adjustment]] — the specific implementation of entropy-based weight scaling in this model
- [[attention-based-regime-detection]] — attention entropy for regime classification

## Sources

- Internal code: `orchestrator/ensemble_meta.py`
