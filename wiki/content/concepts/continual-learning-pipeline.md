---
title: "Continual Learning Pipeline"
type: concept
tags: [continual-learning, retraining, auto-retrain, performance-monitoring, model-versioning]
created: 2026-06-30
updated: 2026-06-30
status: stable
---

# Continual Learning Pipeline

## Definition

An automatic retraining pipeline that monitors strategy performance, detects degradation, and retrains ML + RL models on fresh data when needed. Uses a 5-stage flow: Monitor → Trigger → Train → Validate → Promote. Prevents model staleness by combining time-based (daily) and performance-based (win rate drop) retrain triggers.

## Intuition

Models degrade as market regimes shift. A model trained on bull-market data fails in a bear market. The continual learning pipeline detects this degradation automatically and retrains on the most recent data — like an immune system that adapts to new threats.

## Architecture

```
┌─────────────────────┐
│ PerformanceTracker   │ ← rolling windows: win rate (20/50/100), Sharpe, PnL
├─────────────────────┤
│ Retrain Trigger     │ ← time-based (24h) + perf-based (WR < 45%)
├─────────────────────┤
│ Model Training      │ ← ML (Random Forest) + PPO RL on fresh data
├─────────────────────┤
│ Model Validation    │ ← walk-forward backtest on held-out 20%
├─────────────────────┤
│ Model Promotion     │ ← promote via ModelRegistry if better than current
└─────────────────────┘
```

## Retrain Triggers

| Trigger | Condition | Priority |
|---------|-----------|:--------:|
| Time-based | ≥24 hours since last retrain | High |
| Performance | Win rate(20) < 45% AND Win rate(50) < 45% | High |
| First run | No previous retrain | Medium |
| Manual | Explicit call to `retrain()` | Override |

## How we use it

- Source: `orchestrator/continual_learning.py` — `ContinualLearningPipeline` class
- Monitors: `PerformanceTracker` with rolling windows (20, 50, 100 trades)
- Retrains: `MLSignalEngine` (Random Forest) + PPO via callback function
- Validates: Walk-forward backtest on 20% held-out data
- Versions: Uses `ModelRegistry` for model versioning and rollback
- CLI: `python -m orchestrator.continual_learning` for manual trigger

## Strengths & weaknesses

**Strengths:**
- Automatic: no human intervention needed for routine retraining
- Safe: validates on held-out data before promoting
- Versioned: model registry enables rollback
- Multi-trigger: time + performance + manual triggers

**Weaknesses:**
- Limited to ML + PPO — doesn't retrain all 12 brains
- Walk-forward validation may overfit to recent regime
- No A/B testing between old and new models
- Training data window is not bounded (may use too much history)

## Related

- [[brain-ecosystem]] — brains that could benefit from continual learning
- [[ensemble-meta-model]] — ensemble weights adapt alongside model retraining
- [[ppo-portfolio-manager]] — PPO models retrained by this pipeline

## Sources

- Internal code: `orchestrator/continual_learning.py`
