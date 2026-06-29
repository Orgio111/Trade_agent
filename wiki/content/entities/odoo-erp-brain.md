---
title: Odoo ERP Brain
type: entity
tags: [brain, odoo, erp, business-intelligence, xml-rpc, #12]
created: 2026-06-30
updated: 2026-06-30
weight: 0.05
brain_id: odoo_erp
source_file: odoo_brain.py
status: active
---

# Odoo ERP Brain

## Overview

**Brain #12** — Odoo ERP business intelligence brain. Weight 0.05 (supplementary). Polls Odoo ERP via XML-RPC for inventory changes, sales velocity, purchase orders, revenue trends, and CRM pipeline data. Converts business signals to trading scores using weighted combination of 5 sub-signals.

## Architecture

```
Odoo XML-RPC API (http://odoo-host:8069)
    │
    ├── stock.move → inventory_changes ──── 0.25 ──┐
    ├── sale.order → sales_velocity ─────── 0.30 ──┤
    ├── purchase.order → purchase_volume ── 0.15 ──┤→ weighted sum → score
    ├── account.move → revenue_trend ────── 0.20 ──┤
    └── crm.lead → pipeline_health ──────── 0.10 ──┘
                                                    ↓
                                              clip(-1, +1)
                                                    ↓
                                              BrainSignal
```

## Sub-signals

### 1. Inventory Changes (25%)
- Net delta: negative = demand building (bullish), positive = oversupply (bearish)
- Score: `tanh(-net_delta / 10)` — negative delta → positive score

### 2. Sales Velocity (30%)
- Growth rate of recent sales orders
- Score: `tanh(growth_rate * 5)` — positive growth → bullish

### 3. Purchase Orders (15%)
- Volume of recent purchases as demand proxy
- Score: `tanh(purchase_amount / 10000) * 0.5` — moderate signal

### 4. Revenue Trend (20%)
- Direction of revenue change (up/down/stable)
- Score: `0.3 if up, -0.3 if down, 0.0 if stable`

### 5. CRM Pipeline (10%)
- Conversion rate and pipeline value
- Score: `tanh(conversion_rate - 0.3) * 0.5` — above 30% conversion is bullish

## Parameters

| Parameter | Default | Env var |
|-----------|---------|---------|
| odoo_url | http://localhost:8069 | `ODOO_URL` |
| odoo_db | (required) | `ODOO_DB` |
| odoo_user | (required) | `ODOO_USER` |
| odoo_password | (required) | `ODOO_PASSWORD` |
| poll_interval | 60s | — |

## Confidence model

Base 0.3 + 0.15 (>3 data sources) + 0.10 (>50 data points) + 0.10 (all 5 sources) → max 0.65

## Outputs

- `score`: ∈ [-1, +1] — combined business intelligence signal
- `confidence`: ∈ [0, 1] — based on data availability
- `metadata`: inventory_signal, sales_signal, purchase_signal, revenue_signal, crm_signal, data_points, active_sources

## Fail modes

- Odoo not configured (no env vars) → returns neutral score=0.0, confidence=0.1
- Odoo server unreachable → graceful fallback, logs warning
- Invalid XML-RPC domain syntax → caught and logged, returns neutral
- All 5 sub-signals return 0.0 when data unavailable → implicit zero-weight

## Edge & known weaknesses

- **Edge:** Only trading brain that considers real-world business state
- **Edge:** Leading indicator — ERP changes precede market moves
- **Weakness:** ERP data granularity is daily/hourly, not tick-level
- **Weakness:** Requires running Odoo instance (additional infra dependency)
- **Weakness:** Correlation between ERP signals and crypto prices unvalidated
- **Weakness:** Polling interval (60s) may miss fast-moving events
- **10× opportunity:** Add WebSocket listener for real-time Odoo bus events; integrate Odoo IoT for warehouse sensors; add cross-company ERP aggregation for sector-level signals

## How it fits in the brain ecosystem

See [[signal-aggregation-logic]] for the full brain weight table. This brain sits at the supplementary tier (0.05) alongside onchain_whale, custom_nn, and polymarket_alpha. Total ensemble: 12 brains, weight sum = 1.00.

## Registration

| File | Status |
|------|--------|
| `orchestrator/brains/odoo_brain.py` | ✅ Implemented |
| `orchestrator/brains/__init__.py` | ✅ Import + BRAIN_REGISTRY |
| `orchestrator/main.py` | ✅ BRAIN_INTERVALS (60s) |
| `orchestrator/brain_backtest.py` | ✅ BRAIN_WEIGHTS (0.05) |
| `orchestrator/brains/brain_registry.json` | ✅ Metadata entry |
| `realtime/config.go` | ✅ DefaultBrainWeights (0.05) |
| `frontend/src/components/OdooErpPanel.tsx` | ✅ Dashboard panel |

## Related

- [[odoo-erp-trading-integration]] — full integration architecture concept
- [[real-time-trading-dashboard]] — dashboard displaying Odoo signals
- [[signal-aggregation-logic]] — how this brain's signal is weighted
- [[onchain-brain]] — similar supplementary brain pattern
- [[local-trading-ai-architecture]] — Ollama model routing for brain

## Sources

- Internal code: `orchestrator/brains/odoo_brain.py`
- Internal design: AGENTS.md §ODOO ERP → TRADING AI INTEGRATION
