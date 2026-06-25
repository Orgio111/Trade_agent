---
title: "Cont, Kukanov & Stoikov (2014) — The price impact of order book events"
type: source
tags: [microstructure, order-flow, ofi, price-impact, market-mechanics]
created: 2026-06-25
updated: 2026-06-25
source_path: raw/papers/cont-order-flow-imbalance.md
source_type: paper
authors: ["Rama Cont", "Arseniy Kukanov", "Sasha Stoikov"]
date: 2014-01-01
status: stable
---

# Cont, Kukanov & Stoikov (2014) — Order book events & price impact

> **TL;DR** — A linear combination of signed limit-order-book event flows ("Order Flow Imbalance") explains a large share (~40–65%) of the *concurrent* variance of price changes on liquid books. It is a *now-pressure* signal that decays in seconds, not a directional forecast. The trade-only tick-rule proxy we actually compute is far noisier and explains far less.

## Key takeaways

- **Event OFI is the gold standard** — it uses full LOB events (adds/cancels/quotes at best bid/ask), and its linear relationship to price is the strongest microstructure price-impact regularity on record.
- **The tick-rule proxy is what we have on trade-only feeds.** Lee–Ready misclassifies a non-trivial share of trades, so trade-based OFI explains only single-to-low-double-digit % of price variance. Treat it as a *directional pressure proxy*, not a tight predictor.
- **OFI is concurrent, not predictive.** Predictive lead decays in seconds on liquid venues. Using OFI as a multi-minute directional bet is a category error.
- **Non-linearity matters at extremes.** Marginal impact per unit flow flattens — large imbalances are partly absorbed. A linear score will over-react to extremes unless saturated.
- **Absorption breaks the naïve signal.** Large aggressive flow into a big resting order yields high OFI but no follow-through — the naïve signal is fadeable here, so absorption detection (which our `MicrostructureBrain` has) is load-bearing, not optional.
- **Depth-weighting > top-of-book.** Microprice / depth-weighted mid is more informative than raw top-of-book quantities, which justifies the `OrderFlowNautilusBrain`'s L2/L3 depth path.

## Summary

Cont, Kukanov & Stoikov formalize Order Flow Imbalance as a signed aggregation of limit-order-book events at the top of the book: events that improve or add liquidity on the bid side contribute positively; the symmetric ask-side events contribute negatively. Empirically, over short intervals this signed flow is approximately linearly related to the concurrent price change, with a coefficient $\lambda$ that is stable within an instrument/regime but varies across them. This is the cleanest known mechanical link between order flow and price formation.

Because full LOB events are not always available, practitioners substitute the trade-tick OFI: classify each executed trade via the Lee–Ready tick test (uptick → buy, downtick → sell, carry forward on zero), accumulate signed volume. This recovers the *sign* of pressure but loses queue-shape information and introduces classification noise — it is a downgrade, not an equivalent.

Two operational consequences dominate: (1) OFI's edge is in *sizing and timing the present*, not forecasting the next minute; (2) extreme readings and absorption events must be handled non-linearly, or the signal systematically over-triggers into resting liquidity.

## Concepts introduced / updated

- [[order-flow-imbalance]] — this is the foundational source for the OFI concept page; the page's math, horizon claims, and failure-mode list trace here.
- [[tick-rule-classification]] — (not yet a page; flagged for creation during lint) the Lee–Ready tick test that makes trade-only OFI computable.

## Relevance to our trading

This is the theoretical anchor for two orchestrator brains that already ship in this repo:

- **`MicrostructureBrain`** (`orchestrator/brains/microstructure_brain.py`, Go-orchestrator weight 0.10) — implements the *tick-rule proxy* path: signed trade volume over a 100-tick lookback (`OFI_LOOKBACK`), z-score normalized against a 50-tick rolling window, mapped to $[-1, +1]$ via `tanh(z·0.5)`, with an explicit absorption-inversion step (`score *= -0.5`, `confidence *= 0.7`). The absorption handling is *exactly* the failure mode this source warns about — that is not accidental.
- **`OrderFlowNautilusBrain`** (`orchestrator/brains/orderflow_nautilus_brain.py`, weight 0.12) — implements the *preferred depth-based* path via NautilusTrader's L2/L3 orderbook reconstruction (microprice, depth-weighted mid, spread compression, liquidity-gap detection), falling back to the tick rule when NautilusTrader is unavailable. The weight premium over the pure-micro brain reflects this source's central point: true depth OFI strictly dominates the tick proxy.

Net: our system already encodes this source's main hierarchy (depth > tick, saturate extremes, invert on absorption). The wiki should keep that hierarchy explicit so future changes are auditable against the theory.

## Contradictions / updates

None — first source ingested on this topic. Future sources may challenge the 40–65% variance figure (venue/instrument-dependent) or the decay-horizon numbers; flag any such update on [[order-flow-imbalance]] rather than overwriting here.

## Open questions

- What is OFI's *lead* (forward) horizon specifically on the crypto venues (Binance) we trade, versus the equity-futures venues the original results were measured on? Crypto microstructure differs (maker/taker fees, no true uptick rule, spoofing prevalence). Worth ingesting a crypto-specific microstructure source.
- Our tick-rule path uses a simplified sign rule (`>` / `<` on consecutive prices, carry on equal) — how does its misclassification rate compare to Lee–Ready proper? Could be quantified on our own trade tape.
- The absorption detector in `MicrostructureBrain` — what is its false-positive rate, and does it fire in the wrong regime?
