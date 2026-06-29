---
title: "Brain Ecosystem"
type: overview
tags: [brain, ensemble, weights, architecture, orchestration]
created: 2026-06-30
updated: 2026-06-30
status: stable
---

# Brain Ecosystem

> **Single source of truth** for all 12 orchestrator brains, their weights, classes, and registration status. Entity pages should link here instead of duplicating this table.

## Weight Table

| # | Brain ID | Class | Weight | Tier | Source File |
|---|----------|-------|:------:|------|-------------|
| 1 | `timesfm` | TimesFMBrain | 0.25 | Primary | timesfm_brain.py |
| 2 | `freqai` | FreqAIBrain | 0.15 | Core | freqai_brain.py |
| 3 | `llm_regime` | LLMRegimeBrain | 0.15 | Core | llm_regime_brain.py |
| 4 | `finbert_nlp` | FinBERTBrain | 0.07 | Secondary | finbert_brain.py |
| 5 | `microstructure` | MicrostructureBrain | 0.05 | Supplementary | microstructure_brain.py |
| 6 | `orderflow_nautilus` | OrderFlowNautilusBrain | 0.03 | Supplementary | orderflow_nautilus_brain.py |
| 7 | `finrl_kelly` | FinRLBrain | 0.05 | Supplementary | finrl_brain.py |
| 8 | `statarb_funding` | StatArbBrain | 0.05 | Supplementary | statarb_brain.py |
| 9 | `onchain_whale` | OnChainBrain | 0.05 | Supplementary | onchain_brain.py |
| 10 | `custom_nn` | CustomNNBrain | 0.05 | Supplementary | custom_nn_brain.py |
| 11 | `polymarket_alpha` | PolymarketBrain | 0.05 | Supplementary | polymarket_brain.py |
| 12 | `odoo_erp` | OdooBrain | 0.05 | Supplementary | odoo_brain.py |
| | **TOTAL** | | **1.00** | | |

## Tier Distribution

```
Primary (0.25):   timesfm                    ████████████████████████████
Core (0.15×2):    freqai, llm_regime          ████████████████ ×2
Secondary (0.07): finbert_nlp                 ████████
Supplementary:    7 brains × 0.03–0.05 each   █████ ×7
```

## Weight Tiers

| Tier | Weight Range | Brains | Purpose |
|------|-------------|--------|---------|
| **Primary** | 0.25 | timesfm | Foundation model — highest single-brain influence |
| **Core** | 0.15 | freqai, llm_regime | Technical ML + regime detection |
| **Secondary** | 0.07 | finbert_nlp | NLP sentiment — moderate influence |
| **Supplementary** | 0.03–0.05 | microstructure, orderflow, finrl, statarb, onchain, custom_nn, polymarket, odoo | Specialized signals — low individual, high aggregate diversity |

## Aggregation Flow

```
12 brains each compute_score(symbol) → BrainSignal(score, confidence)
        │
        ▼
    Go Aggregator (realtime/nats_orchestrator.go)
        │
        consensus = Σ(weight_i × score_i) / Σ(weight_i)
        │
        ▼
    AggregatedSignal → Execution / Dashboard
```

## Registration Files

Every brain must be registered in all of these:

| File | What it contains |
|------|-----------------|
| `orchestrator/brains/__init__.py` | Import + `BRAIN_REGISTRY` dict |
| `orchestrator/brains/brain_registry.json` | Structured metadata (id, class, weight, description, tiers) |
| `orchestrator/brain_backtest.py` | `BRAIN_WEIGHTS` dict for backtesting |
| `orchestrator/main.py` | `BRAIN_INTERVALS` dict (polling cadence) |
| `realtime/config.go` | `DefaultBrainWeights` for Go aggregator |
| `wiki/content/entities/<brain>.md` | Knowledge base entity page |

## Key Relationships

- [[signal-aggregation-logic]] — how weights are applied in the Go aggregator
- [[ensemble-meta-model]] — proposed meta-learner for dynamic weight adjustment
- [[brain-backtest-infrastructure]] — backtesting framework using these weights

## Individual Brain Pages

| Brain | Wiki Page |
|-------|-----------|
| TimesFM | [[timesfm-brain]] |
| FreqAI | [[freqai-brain]] |
| LLM Regime | [[llm-regime-brain]] |
| FinBERT | [[finbert-brain]] |
| Microstructure | [[microstructure-brain]] |
| OrderFlow Nautilus | [[orderflow-nautilus-brain]] |
| FinRL | [[finrl-brain]] |
| StatArb | [[statarb-brain]] |
| OnChain | [[onchain-brain]] |
| Custom NN | [[neural-network-brain]] |
| Polymarket | — (no entity page yet) |
| Odoo ERP | [[odoo-erp-brain]] |
