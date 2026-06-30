---
title: OrderbookHeatmap
type: entity
tags:
  - frontend
  - orderbook
  - heatmap
  - l2
  - whale-detection
created: 2026-06-30
updated: 2026-06-30
source_file: frontend/src/components/OrderbookHeatmap.tsx
---

# OrderbookHeatmap

**File:** `frontend/src/components/OrderbookHeatmap.tsx`
**Type:** React component (pure CSS + useMemo, no external charting)
**Purpose:** Real-time L2 orderbook depth visualization — bid/ask bars, cumulative depth, mid-price marker, imbalance indicator, and whale liquidity cluster detection.

## Architecture

```
OrderbookData (props)
    ├── computeDepthBars()  → bid/ask bars with intensity normalization
    ├── whale detection     → statistical outlier detection (μ + 2σ)
    └── rendering
        ├── Summary bar     (mid price, spread, imbalance %)
        ├── Bid depth bars  (green, left side)
        ├── Mid-price line  (purple, center)
        ├── Ask depth bars  (red, right side)
        └── Whale clusters  (glowing radial gradient badges)
```

## Data Interface

```typescript
interface OrderbookData {
  symbol: string;
  bids: [number, number][];  // [price, quantity] sorted descending
  asks: [number, number][];  // [price, quantity] sorted ascending
  spread: number;
  mid_price: number;
  imbalance: number;         // -1.0 to 1.0
  timestamp: number;
}
```

## Visual Mapping

### Depth Bars

| Property | Visual |
|----------|--------|
| Bar height | `min(4px, intensity × 40px)` — proportional to quantity |
| Bar color | Green (`rgba(0,255,136,α)`) for bids, Red (`rgba(255,0,68,α)`) for asks |
| Alpha | `max(0.1, min(1, intensity))` — brighter = larger quantity |
| Cumulative | Thin gradient line behind bar (right-aligned) |
| Quantity label | White text overlaid on bar (`formatQty`: K suffix for ≥1000) |
| Price label | Right-aligned, monospace, white for highest bid / lowest ask |

### Configuration

```typescript
const HEATMAP_LEVELS = 20;        // Price levels to display
const MIN_BAR_HEIGHT = 4;         // Minimum px
const MAX_BAR_HEIGHT = 40;        // Maximum px
const MAX_CUMULATIVE_WIDTH = 120; // Cumulative depth bar width
```

### Whale Detection

Statistical outlier detection:
1. Compute mean and standard deviation of all level quantities
2. Flag levels where `quantity > mean + 2σ` as whale clusters
3. Render as pulsing radial gradient badges with price + size labels

```typescript
const mean = allBars.reduce((s, b) => s + b.qty, 0) / allBars.length;
const std = Math.sqrt(allBars.reduce((s, b) => s + (b.qty - mean) ** 2, 0) / allBars.length);
whaleClusters = allBars.filter((b) => b.qty > mean + 2 * std && std > 0);
```

### Mid-Price Marker

- Thin purple line (`#444488`) separating bids and asks
- Small circular dot at center (8×8px, purple fill, dark border)

### Imbalance Display

| Imbalance | Color | Meaning |
|-----------|-------|---------|
| > 0.3 | Green `#00ff88` | Bid-heavy (bullish pressure) |
| < -0.3 | Red `#ff0044` | Ask-heavy (bearish pressure) |
| -0.3 to 0.3 | Gray `#888` | Balanced |

## Price Formatting

```typescript
formatPrice(p): $XX.XX (≥$1), $X.XXXXXX (<$1), $X,XXX.XX (≥$1000)
formatQty(q):   XXK (≥1000), XX.XX (≥1), X.XXXX (<1)
```

## Performance

- **useMemo** for depth bar computation and whale detection — recalculates only on `data` change
- **Pure CSS** transitions (0.3s ease) on all bars — no JavaScript animation loop
- **Compact mode** prop available for smaller layouts

## Related

- [[microstructure-panel]] — Complementary orderbook imbalance / spoofing / cascade signals
- [[microstructure]] — OFI, tick-rule, and CVD concepts
- [[orderflow-nautilus-brain]] — Backend brain that generates L2 signals
- [[real-time-trading-dashboard]] — Dashboard layout context
