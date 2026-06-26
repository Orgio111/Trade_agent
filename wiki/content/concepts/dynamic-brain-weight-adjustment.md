---
title: "Dynamic Brain Weight Adjustment"
type: concept
tags: [ensemble, attention, regime, dynamic-weights, brain-orchestration]
created: 2026-06-26
updated: 2026-06-26
sources: [[attention-based-regime-detection]]
status: stable
---

# Dynamic Brain Weight Adjustment

## Definition

Mechanism to dynamically adjust brain weights in the ensemble meta-model based on Transformer attention entropy, which serves as a real-time regime indicator. When the Transformer's attention pattern shifts (detected via entropy changes), brain weights are recalibrated to match the current market regime.

## Integration Point

Implemented in `orchestrator/ensemble_meta.py` via the `AttentionEntropyAdjuster` class.

## Regime-Weight Mapping

| Regime | Entropy | ml_rf | ppo_rl | regime | rule_smc | Rationale |
|--------|---------|-------|--------|--------|----------|-----------|
| **Trending** | < 2.0 | 1.2x | 1.3x | 0.7x | 1.1x | RL exploits trends, regime detection redundant |
| **Ranging** | 2.0-3.5 | 0.8x | 0.7x | 1.4x | 1.3x | ML struggles in noise, regime detection critical |
| **Volatile** | > 3.5 | 0.5x | 0.5x | 1.2x | 0.6x | All risky sources dampened defensively |

## How It Works

```
1. custom_nn_brain (Transformer) computes attention weights
   ↓
2. Attention entropy extracted: H = -Σ p(i) * log(p(i))
   ↓
3. Regime classified: trending / ranging / volatile
   ↓
4. EnsembleMetaModel.decide() receives attention_entropy
   ↓
5. base_weights × regime_multiplier = adjusted_weights
   ↓
6. Weighted voting uses adjusted weights
   ↓
7. Regime info included in decision metadata
```

## Key Properties

- **No compounding**: Weights always adjust from `base_weight`, not from previous adjusted value
- **Backward compatible**: When no `attention_entropy` is provided, behavior is unchanged
- **Regime stability**: Tracks regime persistence across calls (high stability = regime is trustworthy)
- **Clamped**: Adjusted weights stay in [0.1, 3.0] range

## Validation Results

| Test | Result |
|------|--------|
| Regime classification (trending/ranging/volatile) | ✅ Correct |
| Weight adjustment per regime | ✅ Matches multipliers |
| No weight compounding across 5 calls | ✅ Range < 0.001 |
| Backward compatibility (no entropy) | ✅ Works unchanged |
| Regime stability tracking | ✅ > 0 after repeated calls |

## Related

- [[attention-based-regime-detection]] — research behind this implementation
- [[neural-network-brain]] — Transformer that produces attention weights
- [[ensemble-meta-model]] — ensemble where weights are applied

## Open Questions

- Should the entropy thresholds (2.0, 3.5) be adaptive rather than fixed?
- How does regime stability affect position sizing? (stable regime → larger positions?)
- Should we add a "regime transition" detection that proactively adjusts before regime fully shifts?
