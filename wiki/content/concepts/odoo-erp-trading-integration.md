---
title: "Odoo ERP → Trading AI Integration"
type: concept
tags: [erp, odoo, business-intelligence, trading-integration, signal-fusion]
created: 2026-06-30
updated: 2026-06-30
sources: []
status: stable
---

# Odoo ERP → Trading AI Integration

## Definition

A system architecture that treats Odoo ERP data (inventory, sales, revenue, CRM) as real-world business intelligence signals, fused with market data to produce trading decisions. Odoo becomes a "business state engine" — ERP changes map directly to directional trading signals.

## Intuition

Traditional trading systems react only to market data (price, volume, order flow). This concept extends the signal universe to include **real-world business state**: if a company's inventory is building (weak demand), that's bearish context; if sales are spiking (strong demand), that's bullish context. ERP data is a leading indicator that market price hasn't yet absorbed.

## Signal Mapping

| Odoo Data | Trading Signal | Direction |
|-----------|---------------|-----------|
| Inventory ↑ sharply | Demand weakening | Bearish |
| Sales ↑ | Demand strength | Bullish |
| Revenue ↓ | Risk-off mode | Reduce exposure |
| Revenue ↑ | Risk-on mode | Increase exposure |
| CRM pipeline ↑ | Pipeline strength | Bullish macro |
| Purchase orders ↑ | Supply building | Bearish (oversupply) |

## Architecture

```
Odoo ERP (XML-RPC API)
    │
    ├── Inventory Module → supply_pressure
    ├── Sales Module → demand_index
    ├── Accounting Module → cash_flow, revenue_trend
    └── CRM Module → business_health, pipeline_value
            │
            ▼
    Feature Fusion Layer (OdooFeatureEngine)
            │
            ▼
    Odoo Brain #12 (compute_score)
            │
            ▼
    NATS JetStream → signals.raw.odoo_erp.BTCUSDT
            │
            ▼
    Go Aggregator → weighted consensus
            │
            ▼
    Dashboard + Execution
```

## Feature Engine

```python
def odoo_features(data):
    return {
        "demand_index": data["sales_growth"],
        "supply_pressure": data["inventory_change"],
        "cash_flow": data["revenue_trend"],
        "business_health": data["crm_conversion"],
    }
```

## Decision Engine Flow

```
Market Data (candles)  ─┐
                        ├── Feature Fusion → AI Router → Trade Decision
Odoo ERP Data ─────────┘                         ↓
                                            Execution Layer
                                                 ↓
                                          Performance Log
                                                 ↓
                                           Memory + Learning
```

## Risk Rules (Odoo-Aware)

| Condition | Action |
|-----------|--------|
| Odoo revenue ↓ AND market volatility ↑ | Reduce position size |
| Odoo demand ↑ AND market trend ↑ | Increase exposure |
| Odoo signals conflict with market | Stay neutral |

## How we use it

- [[odoo-erp-brain]] implements this as Brain #12 in the orchestrator, polling Odoo via XML-RPC every 60 seconds
- Business signals are weighted 0.05 in the ensemble (low weight — supplementary signal)
- Dashboard displays Odoo ERP panel with real-time business intelligence overlay

## Strengths & weaknesses

**Strengths:**
- Leading indicator — ERP changes precede market price movements
- Unique signal source — most crypto trading systems ignore real-world business data
- Closed-loop: trading decisions can feed back into ERP (position sizing → inventory management)

**Weaknesses:**
- Depends on Odoo instance availability (XML-RPC must be running)
- Latency: ERP data updates on business hours, not tick-level
- Correlation between ERP signals and crypto prices is unvalidated (needs backtesting)
- Single-company ERP data may not generalize to broader market

## Related

- [[odoo-erp-brain]] — Brain #12 implementation
- [[local-trading-ai-architecture]] — overall local AI system design
- [[signal-aggregation-logic]] — how brain signals are weighted
- [[custom-trading-brain-architecture]] — brain design patterns

## Sources

- Internal design: AGENTS.md §ODOO ERP → TRADING AI INTEGRATION
