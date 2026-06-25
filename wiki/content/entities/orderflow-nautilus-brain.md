---
title: OrderFlowNautilusBrain (orchestrator brain)
type: entity
tags: [trading, microstructure, order-flow, ofi, depth, system-component, brain, nautilus]
created: 2026-06-25
updated: 2026-06-25
sources: [[cont-order-flow-imbalance-2014]]
status: stable
---

# OrderFlowNautilusBrain

## What it is

`OrderFlowNautilusBrain` is orchestrator **brain #4b** — the depth-based Order Flow Imbalance analyzer. It is the system's *preferred* OFI path: when NautilusTrader's Rust-core L2/L3 orderbook engine is available, it uses real bid/ask depth imbalance (microprice, depth-weighted mid, spread compression, liquidity gaps); otherwise it falls back to the same tick rule as [[microstructure-brain]]. Go-orchestrator ensemble weight **0.12** — the premium over the tick-only brain encodes the **depth > tick** hierarchy.

- **Code**: `orchestrator/brains/orderflow_nautilus_brain.py`
- **brain_id**: `orderflow_nautilus`
- **Weight**: 0.12

## How it works

- **Preferred path** (`compute_from_depth`): consumes NautilusTrader `OrderbookMetrics` — `bid_ask_imbalance`, `buy_pressure`, `sell_pressure`, `spread_bps`, `liquidity_gap`, `depth_weighted_mid`.
- **Fallback path** (`compute_from_trades`): identical tick-rule OFI logic to [[microstructure-brain]] — signed volume accumulation over a 100-tick lookback.
- **Manual feed**: `push_orderbook_snapshot(bids, asks)` for testing/non-Nautilus paths.
- **Lifecycle**: `warmup()` starts the `OrderbookEngine`; `cooldown()` stops it. Engine failure is non-fatal — degrades to tick OFI with a warning log.

## Theoretical basis

Implements the **depth/event-based** variant of [[order-flow-imbalance]] that [[cont-order-flow-imbalance-2014]] identifies as strictly more informative than the tick proxy: depth-weighted quantities incorporate queue *shape*, not just its tip, and avoid tick-test misclassification noise. The two-path design (depth, with tick fallback) is the operational answer to "depth is better when you can get it."

## Strengths & limits

- **Pro**: uses true depth when available — microprice, spread compression, gap detection; strictly dominates the tick brain in signal quality.
- **Con**: depends on NautilusTrader being importable and the `OrderbookEngine` starting cleanly; without it, collapses to the noisy tick path (but does not fail).
- **Coupling**: ties the brain to NautilusTrader's availability and the `nautilus_bridge.orderbook_engine` module — a dependency/install surface worth tracking.

## Related

- [[order-flow-imbalance]] · [[microstructure-brain]] · [[tick-rule-classification]] · [[nautilustrader]] (not yet a page)

## Sources

- [[cont-order-flow-imbalance-2014]]
