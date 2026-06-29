---
title: MicrostructureBrain (orchestrator brain)
type: entity
tags: [trading, microstructure, order-flow, ofi, system-component, brain]
created: 2026-06-25
updated: 2026-06-25
sources: [[cont-order-flow-imbalance-2014]]
status: stable
---

# MicrostructureBrain

## What it is

`MicrostructureBrain` is orchestrator **brain #4** — the trade-tick Order Flow Imbalance analyzer. It is the system's noisiest-but-cheapest OFI path: it classifies executed trades as buyer- or seller-initiated (tick rule), accumulates signed volume, normalizes, saturates, and emits a directional score. Go-orchestrator ensemble weight **0.05**.

- **Code**: `orchestrator/brains/microstructure_brain.py`
- **brain_id**: `microstructure`
- **Weight**: 0.10

## How it works

1. **Ingest trades** into a 100-tick lookback deque (`OFI_LOOKBACK` env, default 100).
2. **Classify** each trade via a simplified Lee–Ready tick test (`> prev` → buy, `< prev` → sell, carry on equal). See [[tick-rule-classification]].
3. **Accumulate** signed volume $\text{OFI}^{\text{tick}}_L$.
4. **Normalize**: z-score against a 50-tick rolling window ($\sigma$ floor at $10^{-8}$ to avoid blowups).
5. **Saturate**: $\text{score} = \tanh(0.5 \cdot z)$, mapping to $[-1, +1]$.
6. **Confidence**: $\min(0.9,\ 0.3 + |z|\cdot 0.1)$.
7. **Absorption override**: if absorption detected, invert & dampen — `score *= -0.5`, `confidence *= 0.7`.
8. **Cold-start guard**: < 20 trades → score 0.0, confidence 0.1.

## Theoretical basis

Implements the **tick-rule proxy** of [[order-flow-imbalance]] as described in [[cont-order-flow-imbalance-2014]]. Every design choice maps to a claim in that source:
- tick-rule classification = the trade-only proxy when no LOB event feed exists;
- tanh saturation = the non-linearity-at-extremes failure mode;
- absorption inversion = the "high OFI but no follow-through" failure mode.

## Strengths & limits

- **Pro**: works on any trade feed; deterministic; cheap.
- **Con**: tick-test misclassification noise; no queue-shape info; strictly dominated by [[orderflow-nautilus-brain]] when depth is available.

## Related

- [[order-flow-imbalance]] · [[tick-rule-classification]] · [[orderflow-nautilus-brain]]

## Sources

- [[cont-order-flow-imbalance-2014]]
