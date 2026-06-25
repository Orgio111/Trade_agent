---
title: OnChain Whale Brain
type: entity
tags: [brain, on-chain, whale, exchange-flow, mempool]
created: 2026-06-26
updated: 2026-06-26
weight: 0.05
brain_id: onchain_whale
source_file: onchain_brain.py
status: active
---

# OnChain Whale Brain

## Overview

**Brain #7** — On-chain analytics brain. Lowest weight (0.05) — supplementary signal. Monitors exchange flows (50%), whale transactions (35%), and mempool fees (15%). Falls back to zero on all components when no data source configured.

## Architecture

```
push_exchange_flow → net_outflow ──── 0.50 ──┐
push_whale_txn ─────→ net whale flow ─ 0.35 ──┤→ clip(-1,+1) → score
push_mempool_fee ───→ fee activity ─── 0.15 ──┘
                                      ↘ (no data): 0.0 on each
```

## Sub-signals

### 1. Exchange flow (50%)
- Net outflow = bullish (coins leaving exchanges)
- Net inflow = bearish (coins entering for selling)
- Last 24 data points
- `score = tanh(net_flow / 100)` (100 BTC = strong signal)

### 2. Whale transactions (35%)
- Large outflows from exchanges → bullish
- Large inflows to exchanges → bearish
- Last 10 transactions
- `score = tanh((outflow - inflow) / whale_threshold)`

### 3. Mempool fees (15%)
- Higher fees → more on-chain activity → slight bullish bias
- Last 10 readings
- `score = tanh(avg_fee / 100) * 0.3`

## Parameters

| Parameter | Default | Env var |
|-----------|---------|---------|
| whale_threshold_btc | 10.0 BTC | `WHALE_THRESHOLD_BTC` |
| exchange_flow_deque | 200 | — |
| whale_txn_deque | 100 | — |
| mempool_fee_deque | 50 | — |

## Confidence model

Base 0.3 + 0.20 (>10 flows) + 0.15 (>5 whale txns) + 0.10 (>10 fees) → max 0.85

## Outputs

- `score`: ∈ [-1, +1]
- `confidence`: ∈ [0, 1]
- `metadata`: exchange_flow, whale_signal, mempool

## Fail modes

- No on-chain data source → all zeros, confidence 0.1
- Mempool data not available → mempool=0 (weight shifts to flow+whale implicitly)
- Whale threshold too low → noise from small transfers

## Edge & known weaknesses

- **Edge:** Exchange flow is a leading indicator — coins move before selling
- **Edge:** Whales are informed — their flow predicts short-term price
- **Weakness:** Currently depends on external data feeds (no auto-fetch for on-chain)
- **Weakness:** Mempool signal is very weak (0.15 weight, ×0.3 dampening)
- **Weakness:** Crypto exchange flows are noisy — OTC not captured
- **10× opportunity:** Integrate Glassnode/CryptoQuant API for auto-fetching; add stablecoin supply ratio as 4th sub-signal; track whale wallet clusters over time

## Related

- [[statarb-brain]] — funding rate + exchange flow = complementary on-chain signals
- [[finbert-brain]] — NLP + on-chain = multi-modal conviction
- [[tick-rule-classification]] — trade classification vs on-chain flow (micro vs macro)

## Sources

- Internal code: `orchestrator/brains/onchain_brain.py`
