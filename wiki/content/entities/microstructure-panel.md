---
title: MicrostructurePanel
type: entity
tags:
  - frontend
  - microstructure
  - orderbook
  - spoofing
  - liquidation
created: 2026-06-30
updated: 2026-06-30
source_file: frontend/src/components/MicrostructurePanel.tsx
---

# MicrostructurePanel

**File:** `frontend/src/components/MicrostructurePanel.tsx`
**Type:** React component (pure CSS grid, no external charting)
**Purpose:** 2×2 grid panel displaying real-time market microstructure signals — orderbook imbalance, delta/CVD divergence, spoofing detection, and liquidation cascade risk.

## Architecture

```
MicroData (props)
    ├── orderbook  → Imbalance bar + bid/ask volumes + signal label
    ├── delta      → CVD value + divergence direction + strength %
    ├── spoofing   → Alert indicator + cancel-to-order ratio + confidence
    └── cascade    → Risk score bar + severity label + reasoning text
```

## Data Interface

```typescript
interface MicroData {
  orderbook: {
    imbalance: number;       // -1.0 to 1.0 (negative = bearish, positive = bullish)
    bid_volume: number;
    ask_volume: number;
    signal: string;          // "bullish" | "bearish" | "neutral"
  };
  delta: {
    divergence: string;      // "bullish" | "bearish" | "none"
    strength: number;        // 0.0–1.0
    cvd: number;             // cumulative volume delta
    delta_trend: string;     // "rising" | "falling" | "flat"
  };
  spoofing: {
    spoofing_detected: boolean;
    confidence: number;      // 0.0–1.0
    cancel_to_order_ratio: number;  // >5x = suspicious
  };
  cascade: {
    cascade_risk: number;    // 0.0–1.0
    severity: string;        // "critical" | "high" | "elevated" | "low" | "minimal"
    reasoning: string;       // LLM-generated explanation
  };
}
```

## Panel Sections

### 1. Orderbook Imbalance

| Visual | Meaning |
|--------|---------|
| Bar position | Left of center = bearish (more asks), right = bullish (more bids) |
| Bar color | Green (`#00ff88`) if imbalance > 0.3, Red (`#ff0044`) if < -0.3 |
| Bid/Ask volumes | Numeric display below the bar |
| Signal text | `BULLISH` / `BEARISH` / `NEUTRAL` |

### 2. Delta / CVD

| Visual | Meaning |
|--------|---------|
| Divergence label | `BULLISH` / `BEARISH` / `NEUTRAL` (18px bold) |
| CVD value | Green if positive, red if negative |
| Strength | Percentage display |
| Delta trend | `RISING` / `FALLING` / `FLAT` |

### 3. Spoofing Detection

| Visual | Meaning |
|--------|---------|
| Status dot | Green (clean) or Red with pulse animation (spoofing detected) |
| Cancel/Order ratio | Numeric (e.g., `5.2x`) |
| Confidence | Percentage |

### 4. Liquidation Cascade Risk

| Visual | Meaning |
|--------|---------|
| Severity label | `CRITICAL` (red) / `HIGH` (orange-red) / `ELEVATED` (orange) / `LOW` (blue) |
| Risk bar | Width = `cascade_risk × 100%`, color-coded by threshold |
| Score | Numeric percentage |

## Design Patterns

- **Pure CSS grid** — no external charting library (lightweight, fast)
- **Animated indicators** — CSS `pulse` animation for spoofing alerts
- **Color-coded severity** — consistent palette across all 4 sub-panels
- **Monospace typography** — `var(--font-mono)` for numeric values

## Related

- [[microstructure]] — OFI, tick-rule, and CVD concepts
- [[orderflow-nautilus-brain]] — Backend brain that generates these signals
- [[orderbook-heatmap]] — Complementary L2 depth visualization
- [[brain-ecosystem]] — Brain weight context
