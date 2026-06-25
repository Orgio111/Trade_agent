---
title: Cumulative Volume Delta
type: concept
tags: [microstructure, cvd, order-flow, volume]
created: 2026-06-26
updated: 2026-06-26
sources: [[cont-order-flow-imbalance-2014]]
status: stable
---

# Cumulative Volume Delta (CVD)

## Definition

**CVD** is the running sum of signed trade volume: buyer-initiated volume minus seller-initiated volume over a given window. It is a real-time measure of net buying/selling pressure — a coarser but more intuitive cousin of [[order-flow-imbalance]].

## Mechanics / math

$$
\text{CVD}_t = \sum_{i=1}^{t} v_i \cdot d_i
$$

where $v_i$ is trade size and $d_i \in \{+1, -1\}$ is trade direction from [[tick-rule-classification]].

- Positive CVD → net buying pressure
- Negative CVD → net selling pressure
- CVD divergence from price → potential reversal signal

## How we use it

Referenced in our [[microstructure-brain]] as a complement to OFI. Where OFI measures *order book level pressure*, CVD measures *executed volume pressure*. In our system, OFI is the primary signal; CVD is a potential future enhancement for divergence detection.

## Strengths & weaknesses

- **Strength:** Simple, interpretable, real-time. Captures "who is actually hitting the book."
- **Weakness:** Cumulative — needs reset or decay to remain meaningful. Vulnerable to tick-rule misclassification errors compounding over time.
- **Comparison to OFI:** OFI includes *uncanceled* depth changes (passive pressure). CVD only captures *aggressive* executed flow. They are complementary, not substitutes.

## Related

- [[order-flow-imbalance]] · [[tick-rule-classification]] · [[microstructure-brain]] · [[orderflow-nautilus-brain]]

## Sources

- [[cont-order-flow-imbalance-2014]] — OFI foundational framework; CVD as the simpler alternative discussed in context
