---
title: Signal Aggregation Logic
type: playbook
tags: [orchestrator, aggregation, weights, signal, go]
created: 2026-06-26
updated: 2026-06-26
---

# Signal Aggregation Logic

## Overview

Layer B — Go NATS Orchestrator aggregates raw brain signals into a final **BUY / SELL / HOLD** decision. All 9 Python brains publish to `signals.raw`; Go computes a confidence-weighted average and publishes to `signals.aggregated`.

## Architecture

```
9× Python Brains ──publish──▶ signals.raw (NATS JetStream)
                                    │
                              Go NATSOrchestrator
                              (15s aggregation window)
                                    │
                              signals.aggregated
                                    │
                              NautilusTrader (execution)
                                    │
                              signals.executed
```

## Aggregation formula

```
weightedSum = Σ (brain_score × brain_weight × brain_confidence)
totalWeight = Σ (brain_weight × brain_confidence)
finalScore = weightedSum / totalWeight
```

Key insight: **confidence acts as a dynamic weight modulator.** A brain with weight=0.15 but confidence=0.3 contributes only 0.15×0.3=0.045 effective weight — less than a brain at weight=0.10 confidence=0.8=0.08.

## Action thresholds

| Threshold | Default | Env var |
|-----------|---------|---------|
| BUY | > 0.35 | `BUY_THRESHOLD` |
| SELL | < -0.35 | `SELL_THRESHOLD` |
| HOLD | between | — |

## Brain weights (Go default)

| Brain | Weight (Go) | Weight (Python) | Status |
|-------|------------|-----------------|--------|
| timesfm | 0.25 | 0.25 | MATCH |
| freqai | 0.15 | 0.15 | MATCH |
| llm_regime | **0.10** | **0.15** | **MISMATCH** |
| microstructure | 0.10 | 0.10 | MATCH |
| finbert | 0.10 | 0.10 | MATCH |
| finrl | 0.10 | 0.10 | MATCH |
| onchain | 0.10 | **0.05** | **MISMATCH** |
| statarb | 0.10 | 0.10 | MATCH |
| orderflow_nautilus | **MISSING** | 0.12 | **MISSING IN GO** |

**Total Go:** 1.00 / **Total Python:** 1.02 (overflow — needs normalization)

## Weight override mechanism

Every brain weight can be overridden via env var: `BRAIN_WEIGHT_<ID>` (uppercase, underscores).
Example: `BRAIN_WEIGHT_TIMESFM=0.30`

## Known issues & fixes needed

1. **llm_regime weight mismatch** — Python: 0.15, Go: 0.10. Recommend Go update to 0.15.
2. **onchain weight mismatch** — Python: 0.05, Go: 0.10. Recommend Go update to 0.05 (on-chain is supplementary).
3. **orderflow_nautilus missing from Go** — Brain #4b (weight 0.12) not in Go config. Must add: `{ID: "orderflow_nautilus", Weight: 0.12}`.
4. **Total weight ≠ 1.0** — After fixes: 0.25+0.15+0.15+0.10+0.10+0.10+0.05+0.10+0.12 = **1.12**. Needs rebalancing.

## Proposed rebalanced weights (total = 1.00)

| Brain | New Weight | Rationale |
|-------|-----------|-----------|
| timesfm | 0.22 | Still primary; slightly reduced for 9th brain |
| freqai | 0.13 | Technical complement |
| llm_regime | 0.13 | LLM regime is high-value context |
| microstructure | 0.09 | OFI is clean signal |
| orderflow_nautilus | 0.09 | Depth-based OFI |
| finbert | 0.08 | NLP is noisy; true edge is lower |
| finrl | 0.08 | Sizing brain — not directional |
| statarb | 0.10 | Contrarian — decorrelation value |
| onchain | 0.04 | Lowest information; supplementary |
| **TOTAL** | **1.00** | ✓ |

This is a PROPOSAL — must be backtested before deployment (see [[backtesting-pipeline]]).

## Aggregation interval

Default: **15 seconds** (`AGGREGATION_INTERVAL_SECS=15`). Can be lowered for faster trading or raised for stability.

## Confidence dynamics

Each brain returns confidence ∈ [0, 1]:
- High confidence (→ 0.8-0.9): model is sure — full weight applied
- Low confidence (→ 0.1-0.3): fallback/marginal — weight scaled down
- This means the *effective* weight distribution changes every tick

## Related

- [[timesfm-brain]] — primary forecaster at highest weight
- [[microstructure-brain]] — brain #4 (trade-tick OFI)
- [[orderflow-nautilus-brain]] — brain #4b (depth OFI, missing from Go)
- [[finrl-brain]] — sizing brain (score is sizing, not direction)
- [[backtesting-pipeline]] — validate proposed weights before deployment

## Sources

- `realtime/nats_orchestrator.go` — aggregation logic (L184-271)
- `realtime/config.go` — weight config, env overrides (L1-113)
- `orchestrator/brains/brain_registry.json` — Python-side weight registry
