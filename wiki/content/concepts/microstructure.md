---
title: "Microstructure"
type: concept
tags: [microstructure, order-flow, ofi, tick-rule, market-microstructure]
created: 2026-06-30
updated: 2026-06-30
sources: [[cont-order-flow-imbalance-2014]]
status: stable
---

# Microstructure

## Definition

Market microstructure — the study of how orders, trades, and information flow interact to determine price formation. In our system, microstructure analysis is implemented via Order Flow Imbalance (OFI), tick-rule trade classification, and cumulative volume delta, forming the foundation for brains #4 (microstructure) and #4b (orderflow nautilus).

## Key Concepts

- **Order Flow Imbalance (OFI)**: Net signed order volume — see [[order-flow-imbalance]]
- **Tick Rule**: Trade direction proxy comparing price to previous trade — see [[tick-rule-classification]]
- **Cumulative Volume Delta**: Running sum of signed trade volume — see [[cumulative-volume-delta]]

## How we use it

- Brain #4: `orchestrator/brains/microstructure_brain.py` — tick-rule OFI
- Brain #4b: `orchestrator/brains/orderflow_nautilus_brain.py` — depth-based OFI via NautilusTrader
- Both brains produce directional scores that feed into [[brain-ecosystem]] weighted aggregation

## Related

- [[order-flow-imbalance]] — the core OFI concept
- [[tick-rule-classification]] — trade direction proxy
- [[cumulative-volume-delta]] — signed volume delta
- [[microstructure-brain]] — Brain #4 entity page
- [[orderflow-nautilus-brain]] — Brain #4b entity page
- [[nautilustrader]] — Rust-core trading platform for depth-based OFI

## Sources

- [[cont-order-flow-imbalance-2014]] — foundational OFI paper
- Internal code: `orchestrator/brains/microstructure_brain.py`, `orchestrator/brains/orderflow_nautilus_brain.py`
