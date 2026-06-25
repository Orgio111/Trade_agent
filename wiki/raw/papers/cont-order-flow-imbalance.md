# Order Flow Imbalance — canonical reference note

> This file is a **reference note** (the human-authored source) placed in `raw/`.
> It is the immutable source of truth. ZCode synthesizes from it into the wiki; it never edits this file.
> Topic: theoretical foundation of Order Flow Imbalance (OFI) as a price-formation signal.

## Origin / attribution

The concept of Order Flow Imbalance as a formal predictor of price changes is primarily associated with **Rama Cont, Arseniy Kukanov, and Sasha Stoikov**, *"The price impact of order book events"*, *Journal of Financial Econometrics*, Vol. 12, No. 1, pp. 47–88 (2014). The core empirical finding — that a linear combination of order-book event flows explains a large fraction of concurrent price changes — is the backbone of modern microstructure-based signal design.

A widely used operational simplification is the **trade-imbalance / tick-rule OFI**: classify each executed trade as buyer- or seller-initiated (via the Lee–Ready tick rule when bid/ask labels are unavailable), accumulate the signed volume, and treat the running imbalance as a directional pressure proxy.

## Core definitions

**Order Flow Imbalance (event-based, Cont et al. 2014).** On a limit-order-book (LOB) with best bid $B_t$, best ask $A_t$, and quantities $V^B_t, V^A_t$ at the top of each side, define the signed event flow over an interval $\Delta t$:

$$
\text{OFI}_\Delta \;=\; \sum_{t \in \Delta} \mathbf{1}_{\text{bid event}}\, \Delta V^B_t \;-\; \sum_{t \in \Delta} \mathbf{1}_{\text{ask event}}\, \Delta V^A_t
$$

where bid events that *increase* $V^B$ or *lift* $B$ contribute $+\Delta V^B$, and symmetrically for the ask side. The empirical regularity: contemporaneous price change $\Delta P \approx \lambda \cdot \text{OFI}_\Delta$, with $\lambda$ stable but instrument/time-dependent.

**Tick-rule (trade-based) proxy.** When only executed trades are observed (no full LOB), classify trade $i$ by the **Lee–Ready tick test**:

- $\text{sign}_i = +1$ if $p_i > p_{i-1}$ (uptick → buyer-initiated)
- $\text{sign}_i = -1$ if $p_i < p_{i-1}$ (downtick → seller-initiated)
- $\text{sign}_i = \text{sign}_{i-1}$ if $p_i = p_{i-1}$ (zero-uptime rule)

Then accumulated signed volume over a lookback $L$:

$$
\text{OFI}^{\text{tick}}_L \;=\; \sum_{i=1}^{L} \text{sign}_i \cdot q_i
$$

This is a noisier proxy than the true event OFI but is computable from trade-only feeds and is the de-facto fallback when full depth is unavailable.

## Key empirical claims

1. **Explanatory power.** Event-based OFI typically explains **40–65% of the variance** of concurrent price changes at the high-frequency scale on liquid equity/ futures books (Cont et al. and follow-ups). The tick-rule proxy explains substantially less — single-digit to low-double-digit % — because the tick test misclassifies a meaningful fraction of trades.
2. **Decay / horizon.** OFI's predictive power is overwhelmingly **concurrent**. Lead (out-of-sample, forward) predictability decays within seconds to low-tens-of-seconds on liquid venues. It is a *now*-pressure signal, not a directional forecast.
3. **Saturation / non-linearity.** The OFI–price relationship is approximately linear at moderate magnitudes but **flattens at extremes** — large imbalances are partially absorbed by resting liquidity and by liquidity providers, so the marginal price impact per unit of flow decreases.
4. **Microprice / depth-weighted mid.** Using depth-weighted quantities (microprice) rather than top-of-book quantities improves signal quality because it incorporates the shape of the queue, not just its tip.

## Strengths and weaknesses

**Strengths**
- Mechanically grounded in price formation, not a statistical artifact.
- Cheap to compute; works in real time.
- Combines well with other microstructure features (CVD, absorption, spread dynamics).

**Weaknesses / failure modes**
- **Tick-test misclassification** injects noise; prefer true LOB event OFI or depth-based imbalance where possible.
- **Absorption**: aggressive flow absorbed by a passive, large resting order produces a large OFI but *no* price follow-through — naive OFI signals are fadeable in this regime.
- **Spoofing / flickering liquidity**: ghost orders distort event OFI; trade-based proxies are partly insulated.
- **Regime dependence**: $\lambda$ is not constant; OFI is far more informative in continuous-liquidity regimes than around news/illiquid windows.

## Why it matters for this system

The `Trade_agent` orchestrator ships two OFI-flavored brains that this reference directly informs:
- `MicrostructureBrain` (weight 0.10): pure trade-tick OFI with Lee–Ready-style classification, z-score normalization, tanh saturation, and explicit absorption inversion.
- `OrderFlowNautilusBrain` (weight 0.12): L2/L3 depth-based OFI via NautilusTrader when available, falling back to the same tick rule.

See the synthesized wiki page `content/concepts/order-flow-imbalance.md` for the system-specific interpretation.

## Suggested further reading (not yet ingested)

- Lee & Ready (1991) — tick-test classification.
- Cartea, Jaimungal & Penalva — *Algorithmic and High-Frequency Trading* (microstructure textbook).
- Wei & Ren (2019) — absorption / informed-flow detection.
