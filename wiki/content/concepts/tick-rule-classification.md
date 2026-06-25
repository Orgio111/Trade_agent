---
title: Tick Rule Classification
type: concept
tags: [microstructure, tick-rule, trade-classification, ofi]
created: 2026-06-26
updated: 2026-06-26
sources: [[cont-order-flow-imbalance-2014]]
status: stable
---

# Tick Rule Classification

## Definition

The **tick rule** classifies each trade as buyer- or seller-initiated by comparing its price to the previous trade price. It is the standard proxy for trade direction when bid/ask quotes are unavailable — a common situation in crypto venue data.

## Mechanics / math

- **Uptick** (trade price > previous) → **buyer-initiated** (bid)
- **Downtick** (trade price < previous) → **seller-initiated** (ask)
- **Zero-tick** (trade price = previous) → inherit direction from the *last non-zero-tick trade* ("reversal" variant)

This is the **Lee–Ready tick test** in its simplest form. More sophisticated variants (Lee–Ready 1991) combine quote midpoint comparison with the tick rule for ambiguous cases.

## How we use it

Our [[microstructure-brain]] and [[orderflow-nautilus-brain]] use a tick-rule variant to classify trades from Binance into buy/sell pressure for order flow imbalance (OFI) calculation. Every trade from the OHLCV-derived tick proxy goes through this classification.

## Strengths & weaknesses

- **Strength:** No quote data required — works with trade-only feeds (most crypto venues).
- **Weakness:** Misclassification rate ~15–25% for zero-tick trades in fast markets (Lee–Ready 1991). In crypto, where makers post limit orders at multiple ticks, the error can be higher.
- **Mitigation:** Our OFI uses z-score normalization which smooths out individual misclassifications at the aggregate level.

## Related

- [[order-flow-imbalance]] · [[microstructure-brain]] · [[orderflow-nautilus-brain]] · [[cumulative-volume-delta]]

## Sources

- [[cont-order-flow-imbalance-2014]] — introduced tick-rule proxy for OFI calculation when depth data unavailable
