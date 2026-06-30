---
title: "Chart Segmentation Pipeline"
type: entity
tags: [chart, opencv, computer-vision, segmentation, candle-detection, trend, support-resistance, volume]
created: 2026-07-01
updated: 2026-07-01
status: active
---

# Chart Segmentation Pipeline (RT-CSP v1)

Real-time chart image → structured market features using OpenCV. CPU-only, deterministic, no VLM needed.

## Definition

A computer vision pipeline that converts chart images (screenshots, rendered candles) into structured JSON market data. Instead of forcing a full VLM to "understand" charts, this pipeline uses OpenCV for pixel-level analysis: candle detection, trend lines, support/resistance levels, and volume extraction. The output is fed to a small LLM for reasoning.

## Intuition

The core insight: **trading charts are structured time-series visuals, not natural images**. So instead of using a heavy VLM (which hallucinates on charts), we:

1. **Segment** the chart region from the full frame
2. **Detect** individual candles by color (green/red HSV masks)
3. **Measure** trend slope via polyfit on candle positions
4. **Cluster** support/resistance from percentile analysis of candle extremes
5. **Extract** volume from bottom-region brightness analysis

Output is a clean JSON that a 3B LLM can reason over in <200ms.

## Pipeline Architecture

```
Chart Image (numpy array or file path)
      ↓
1. Denoise (Gaussian blur 3×3)
      ↓
2. Chart Region Extraction (Canny edge → largest contour, ≥30% frame)
      ↓
3. Candle Detection (HSV color masks → morphological close → contour detection)
      ↓
4. Trend Detection (polyfit slope on candle positions)
      ↓
5. Support/Resistance (percentile clustering of candle extremes)
      ↓
6. Volume Detection (bottom 20% brightness analysis)
      ↓
7. Market Structure (higher highs / lower lows from half-comparison)
      ↓
8. Volatility Classification (candle height coefficient of variation)
      ↓
9. Wick Analysis (rejection detection from wick/body ratios)
      ↓
Structured JSON Output
```

## Output Format

```json
{
  "trend": "uptrend",
  "candles_detected": 42,
  "support": 0.35,
  "resistance": 0.78,
  "last_candle": "bullish",
  "volume_spike": true,
  "structure": "higher_highs",
  "volatility": "medium",
  "trend_slope": -0.452,
  "avg_body_ratio": 0.62,
  "wick_analysis": "neutral"
}
```

## Detection Algorithms

### Candle Detection (HSV Color Segmentation)
- Green mask: HSV (35,50,50) → (85,255,255)
- Red mask: HSV (0,50,50)→(10,255,255) ∪ HSV (170,50,50)→(180,255,255)
- Morphological close (3×3 kernel) to connect candle body + wicks
- Contour detection → bounding box → filter by min height (5px)
- Center pixel HSV determines bullish/bearish
- Sorted left-to-right (chronological), limited to max 100 candles

### Trend Detection (Polyfit)
- Linear regression on candle x,y positions
- **Image coordinate note:** y increases downward, so negative slope = uptrend
- Thresholds: slope < -0.3 = uptrend, > 0.3 = downtrend, else sideways

### Support/Resistance (Percentile Clustering)
- Convert pixel positions to normalized 0-1 values (inverted y-axis)
- Support = 20th percentile of candle lows
- Resistance = 80th percentile of candle highs

### Volume Detection (Bottom Region Brightness)
- Extract bottom 20% of chart (where volume bars typically appear)
- Convert to grayscale, compute mean + std
- Bright pixels (above mean+std) as ratio of total
- Volume spike if bright_ratio > 15%

### Market Structure (Half Comparison)
- Split candles into first half vs second half
- Compare average y positions (lower y = higher price)
- Second half higher → "higher_highs" (bullish)
- Second half lower → "lower_lows" (bearish)

### Volatility Classification (Coefficient of Variation)
- Compute mean and std of candle heights
- CV = std / mean
- CV > 0.8 = high, CV < 0.3 = low, else medium

### Wick Analysis (Rejection Detection)
- If last candle body ratio < 20% (small body = potential pin bar)
- Lower wick > 2× upper wick → "rejection_down" (bullish)
- Upper wick > 2× lower wick → "rejection_up" (bearish)

## Latency Profile

| Stage | Time |
|-------|:----:|
| Image load / numpy pass | <1ms |
| Gaussian denoise | 1-3ms |
| Chart region extraction | 2-5ms |
| Candle detection (HSV + contours) | 3-8ms |
| Trend (polyfit) | <1ms |
| S/R (percentile) | <1ms |
| Volume (brightness) | <1ms |
| Structure + volatility + wicks | <1ms |
| **Total** | **10-25ms** |

## Graceful Degradation

- OpenCV optional — returns empty `ChartFeatures` if not installed
- Empty/small images → empty features (no crash)
- Fewer than 3 candles → trend defaults to "sideways"
- Chart region detection fails → uses full frame as fallback

## Pipeline Integration

Chart segmentation feeds into the LangGraph pipeline as a potential pre-processing step:

```
Chart Image → ChartSegmenter → Structured JSON → LLM Reasoning → Trading Decision
```

This replaces the full VLM path for speed-critical scenarios:
- **Chart segmentation** (10-25ms, CPU) + **small LLM** (80-200ms) = ~100-225ms total
- **Full VLM** (moondream, GPU) = ~100-300ms total

The segmentation approach is **10× faster** than full VLM for trading charts because charts are structured, not natural images.

## Source File

`orchestrator/chart_segmentation.py`

## Related

- [[scalping-engine]] — CPU-only decision engine that uses similar structured features
- [[multi-agent-pipeline]] — pipeline that could use chart segmentation as VLM alternative
- [[vlm-agent]] — alternative vision approach (full VLM, slower but more flexible)
- [[local-trading-ai-architecture]] — "NO FULL VLM" design principle for speed
- [[rtx4050-trading-system]] — hardware constraints that favor segmentation over VLM
