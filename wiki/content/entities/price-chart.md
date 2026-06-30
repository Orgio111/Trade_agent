---
title: PriceChart
type: entity
tags:
  - frontend
  - charting
  - tradingview
  - candlestick
created: 2026-06-30
updated: 2026-06-30
source_file: frontend/src/components/PriceChart.tsx
---

# PriceChart

**File:** `frontend/src/components/PriceChart.tsx`
**Type:** React component (TradingView lightweight-charts)
**Purpose:** Real-time candlestick chart with volume histogram and EMA overlays (EMA 9, EMA 21). Primary price visualization for the trading dashboard.

## Architecture

```
Candle[] (props: time, OHLCV)
    ↓
createChart() ─────────────────── Chart creation (one-time)
    ├── CandlestickSeries        (green up, red down)
    ├── HistogramSeries          (volume, 85% top scale)
    ├── LineSeries (EMA 9)       (blue, solid)
    └── LineSeries (EMA 21)      (orange, dashed)
    ↓
setData() ─────────────────────── Data updates (every tick)
    ├── toCandleData()           (NaN/Infinity filtering)
    ├── toVolumeData()           (green/red based on close ≥ open)
    └── computeEMA() + toEmaData() (EMA alignment)
```

## Data Filtering

The component aggressively validates all numeric values to prevent `lightweight-charts` "Value is null" errors:

```typescript
function isValidNum(v: unknown): v is number {
  return v != null && typeof v === 'number' && isFinite(v);
}
```

Every data point passes through `toCandleData()`, `toVolumeData()`, and `toEmaData()` — all of which filter out NaN, Infinity, and null values before feeding to the chart.

## EMA Computation

Client-side EMA calculation with NaN-safe filtering:

| EMA | Period | Color | Style |
|-----|--------|-------|-------|
| EMA 9 | 9 candles | `#00aaff` (blue) | Solid, width 1 |
| EMA 21 | 21 candles | `#ffaa00` (orange) | Dashed, width 1 |

```typescript
function computeEMA(closes: number[], period: number): number[] {
  const valid = closes.filter((c) => isValidNum(c));
  if (valid.length < period) return [];
  const k = 2 / (period + 1);
  let ema = valid.slice(0, period).reduce((a, b) => a + b, 0) / period;
  // ... EMA smoothing
}
```

## Chart Configuration

| Setting | Value |
|---------|-------|
| Background | `#0a0a0f` (dark) |
| Font | Courier New, monospace, 10px |
| Grid lines | `#1a1a2e` (subtle) |
| Crosshair | Mode 0 (normal), purple labels |
| Time scale | `timeVisible: true`, no seconds |
| Volume scale | Top 15% (`scaleMargins: { top: 0.85, bottom: 0 }`) |
| Fit content | One-time on first data load (prevents snap-back on every tick) |

## Performance

- **Chart created once** (empty deps `useEffect`) — avoids re-creation on data changes
- **Data updates** run on every `data` prop change via separate `useEffect`
- **Debounced resize** (100ms) — prevents layout thrashing on window resize
- **try/catch** around all `setData()` calls — gracefully handles chart errors

## Related

- [[real-time-trading-dashboard]] — Dashboard layout where this component is the primary chart
- [[microstructure-panel]] — Complementary orderbook analysis below the chart
- [[incremental-candle-state]] — Candle state management for streaming data
