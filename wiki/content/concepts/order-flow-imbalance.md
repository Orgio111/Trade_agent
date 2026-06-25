---
title: Order Flow Imbalance (OFI)
type: concept
tags: [trading, microstructure, order-flow, ofi, signal, market]
created: 2026-06-25
updated: 2026-06-25
sources: [[cont-order-flow-imbalance-2014]]
status: stable
---

# Order Flow Imbalance (OFI)

## Definition

**Order Flow Imbalance (OFI)** is a signed aggregation of order-book activity that measures net directional pressure on price. Positive OFI = net buying pressure (bids being added/aggressed); negative = net selling pressure. It is the cleanest known mechanical link between order flow and contemporaneous price formation — a *now-pressure* signal, not a directional forecast.

## Intuition

At every instant, liquidity is being added, cancelled, and crossed on both sides of the book. If buy-side events dominate (bids added, asks lifted), the book is being pushed up; if sell-side dominates, down. OFI just counts that imbalance. The non-obvious part: the relationship is strong *concurrently* but weak *predictively* — it tells you what pressure exists right now, not where price goes next.

## Mechanics / math

**Event-based OFI (Cont, Kukanov & Stoikov 2014)** — the gold standard, requires full LOB event feed:

$$
\text{OFI}_\Delta = \sum_{t \in \Delta} \mathbf{1}_{\text{bid event}}\,\Delta V^B_t \;-\; \sum_{t \in \Delta} \mathbf{1}_{\text{ask event}}\,\Delta V^A_t
$$

Empirical regularity: $\Delta P \approx \lambda \cdot \text{OFI}_\Delta$, with event OFI typically explaining **40–65%** of concurrent price-change variance on liquid books. $\lambda$ is instrument/regime-dependent.

**Tick-rule proxy** — what we actually compute on trade-only feeds, via the [[tick-rule-classification]] (Lee–Ready):

$$
\text{OFI}^{\text{tick}}_L = \sum_{i=1}^{L} \text{sign}_i \cdot q_i, \qquad \text{sign}_i \in \{+1, -1\}
$$

The proxy is materially noisier (single-to-low-double-digit % variance explained) because the tick test misclassifies trades and the queue-shape information is lost.

**Our implementation** (`MicrostructureBrain`): the tick proxy is then z-scored against a rolling 50-tick window and saturated:

$$
\text{score} = \tanh\!\big(0.5 \cdot z\big), \quad z = \frac{\text{OFI}^{\text{tick}}_L - \mu_{50}}{\sigma_{50}}
$$

The `tanh` saturation is deliberate — see Failure modes (non-linearity at extremes).

## How we use it

Two orchestrator brains, both already in the repo:

- [[microstructure-brain]] — pure tick-rule OFI, Go-orchestrator weight **0.10**. Lookback 100 (`OFI_LOOKBACK`), 50-tick normalization, tanh saturation, **absorption inversion** (score `*= -0.5`, confidence `*= 0.7` on absorption).
- [[orderflow-nautilus-brain]] — preferred depth-based OFI via NautilusTrader L2/L3, weight **0.12**; falls back to the tick rule when NautilusTrader is unavailable. The weight premium encodes the core hierarchy: **depth > tick**.

Both feed into the orchestrator's weighted brain ensemble, where their OFI-derived score is combined with other brain signals (regime, sentiment, ML/RL, risk) before execution.

## Strengths & weaknesses

**Strengths**
- Mechanically grounded in price formation — not a statistical artifact.
- Cheap, real-time computable; the tick proxy works on any trade feed.
- Combines well with CVD, spread dynamics, and absorption detectors.

**Failure modes**
- **Tick-test misclassification** — injects noise; prefer depth/event OFI where the feed allows (we do, via the Nautilus brain).
- **Absorption** — aggressive flow into a large resting order yields high OFI but no price follow-through. Naïve OFI is *fadeable* here. Our brains invert/dampen on absorption; this handling is load-bearing.
- **Spoofing / flickering liquidity** — distorts event OFI; trade-based proxies are partly insulated (another reason the tick path is robust-if-noisy).
- **Regime dependence** — $\lambda$ is not constant; far less informative around news or in illiquid windows than in continuous-liquidity regimes.
- **Non-linearity at extremes** — marginal impact per unit flow flattens (partly absorbed). A linear score over-reacts; we saturate with `tanh` to encode this.
- **Horizon misuse** — OFI's edge is in *the present*. Treating it as a multi-minute directional bet is the most common misapplication.

## Related

- [[tick-rule-classification]] · [[microstructure-brain]] · [[orderflow-nautilus-brain]] · [[cumulative-volume-delta]] (not yet a page)

## Sources

- [[cont-order-flow-imbalance-2014]] — foundational; provides event-OFI definition, the 40–65% variance figure, the linear-with-attenuation claim, and the conceptual basis for the depth-over-tick hierarchy our system encodes.
