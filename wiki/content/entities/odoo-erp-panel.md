---
title: OdooErpPanel
type: entity
tags:
  - frontend
  - odoo
  - erp
created: 2026-06-30
updated: 2026-06-30
source_file: frontend/src/components/OdooErpPanel.tsx
---

# OdooErpPanel

**File:** `frontend/src/components/OdooErpPanel.tsx`
**Type:** React component (pure CSS grid, no external charting)
**Purpose:** Dashboard panel for real-time Odoo ERP business intelligence signals — demand index, supply pressure, cash flow, CRM pipeline, inventory activity, sales performance.
**Status:** ✅ Implemented (2026-06-30)

## Current State

The Odoo ERP integration is defined at the architecture level in [[odoo-erp-trading-integration]] and documented in AGENTS.md, but no dedicated `OdooErpPanel.tsx` component exists yet. The planned panel would display:

## Planned Layout

```
┌──────────────────────────────────────────┐
│ 🏢 Odoo ERP Signals                      │
├──────────┬───────────┬───────────────────┤
│ Demand   │ Supply    │ Cash Flow         │
│ Index    │ Pressure  │ Trend             │
├──────────┴───────────┴───────────────────┤
│ Business Health (CRM conversion rate)    │
├──────────────────────────────────────────┤
│ Recent ERP Events (NATS market.odoo.*)  │
└──────────────────────────────────────────┘
```

## Planned Data Interface

```typescript
interface OdooErpData {
  demand_index: number;       // sales_growth derived
  supply_pressure: number;    // inventory_change derived
  cash_flow: number;          // revenue_trend derived
  business_health: number;    // crm_conversion derived
  recent_events: OdooEvent[];
}

interface OdooEvent {
  event_type: string;         // "inventory_change", "sales_spike", etc.
  timestamp: number;
  data: Record<string, number>;
  impact: "bullish" | "bearish" | "neutral";
}
```

## NATS Integration

Data would stream from: `market.odoo.<event_type>` subjects (defined in [[nats-event-system]])

## Related

- [[odoo-erp-trading-integration]] — Full Odoo ERP → trading signal architecture
- [[odoo-erp-brain]] — Brain #12 that generates these signals
- [[brain-ecosystem]] — Brain weight context (odoo_erp: 0.05)
- [[real-time-trading-dashboard]] — Dashboard layout where this panel would be added
