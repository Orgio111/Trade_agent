"""
QUANTEX Chart Segmentation Pipeline (RT-CSP v1) — Real-time chart image → structured features.

Converts chart images into structured market data using OpenCV:
  Image → Denoise → Chart Region → Candle Detection → Trend Lines → S/R Levels → Volume → JSON

Architecture:
  ┌─────────────────────────────────────────────────────────────┐
  │              Chart Segmentation Engine                      │
  │                                                              │
  │  Input:  Chart image (numpy array or file path)             │
  │                                                              │
  │  Pipeline:                                                   │
  │    1. Denoise frame (Gaussian blur)                         │
  │    2. Detect chart region (largest contour)                 │
  │    3. Extract candles (color-based segmentation)            │
  │    4. Detect trend line (polyfit slope)                     │
  │    5. Detect support/resistance (percentile clustering)     │
  │    6. Extract volume bars (bottom region analysis)          │
  │    7. Build structured JSON                                 │
  │                                                              │
  │  Output: Structured market state JSON                       │
  └─────────────────────────────────────────────────────────────┘

Usage:
    from orchestrator.chart_segmentation import ChartSegmenter

    segmenter = ChartSegmenter()
    features = segmenter.analyze(chart_image)
    # features = {
    #     "trend": "uptrend",
    #     "candles_detected": 42,
    #     "support": 41980,
    #     "resistance": 42350,
    #     "last_candle": "bullish",
    #     "volume_spike": true,
    #     "structure": "higher_highs",
    #     "volatility": "medium"
    # }
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Optional, Union

import numpy as np

# OpenCV is optional — graceful degradation if not installed
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


# ── Data Classes ────────────────────────────────────────────

@dataclass
class DetectedCandle:
    """A candle detected from chart image."""
    x: int
    y: int
    width: int
    height: int
    is_bullish: bool = True
    body_top: int = 0
    body_bottom: int = 0
    wick_top: int = 0
    wick_bottom: int = 0

    @property
    def body_height(self) -> int:
        return abs(self.body_top - self.body_bottom)

    @property
    def total_height(self) -> int:
        return self.height

    @property
    def body_ratio(self) -> float:
        if self.total_height == 0:
            return 0.0
        return self.body_height / self.total_height


@dataclass
class ChartFeatures:
    """Structured features extracted from chart image."""
    trend: str = "sideways"           # "uptrend", "downtrend", "sideways"
    candles_detected: int = 0
    support: float = 0.0
    resistance: float = 0.0
    last_candle: str = "neutral"      # "bullish", "bearish", "neutral"
    volume_spike: bool = False
    structure: str = "neutral"        # "higher_highs", "lower_lows", "neutral"
    volatility: str = "medium"        # "low", "medium", "high"
    trend_slope: float = 0.0
    avg_body_ratio: float = 0.0
    wick_analysis: str = "neutral"    # "rejection_up", "rejection_down", "neutral"

    def to_dict(self) -> dict:
        return asdict(self)


# ── Chart Segmenter ─────────────────────────────────────────

class ChartSegmenter:
    """
    OpenCV-based chart feature extraction engine.

    Converts chart images into structured market data for LLM reasoning.
    All operations are CPU-only and deterministic.
    """

    def __init__(self, min_candle_height: int = 5, max_candles: int = 100):
        self.min_candle_height = min_candle_height
        self.max_candles = max_candles
        self._analysis_count = 0

    def analyze(self, image: Union[np.ndarray, str]) -> ChartFeatures:
        """
        Analyze a chart image and extract structured features.

        Args:
            image: Chart image as numpy array (BGR) or file path string

        Returns:
            ChartFeatures with extracted market structure
        """
        t0 = time.perf_counter()
        self._analysis_count += 1

        if not CV2_AVAILABLE:
            return ChartFeatures()

        # Load image if path
        if isinstance(image, str):
            frame = cv2.imread(image)
            if frame is None:
                return ChartFeatures()
        else:
            frame = image

        if frame is None or frame.size == 0:
            return ChartFeatures()

        try:
            # Step 1: Denoise
            denoised = self._denoise(frame)

            # Step 2: Extract chart region
            chart_region = self._extract_chart_region(denoised)
            if chart_region is None:
                chart_region = denoised  # Use full frame as fallback

            # Step 3: Detect candles
            candles = self._detect_candles(chart_region)

            # Step 4: Detect trend
            trend, slope = self._detect_trend(candles)

            # Step 5: Detect support/resistance
            support, resistance = self._detect_levels(candles, chart_region.shape[0])

            # Step 6: Detect volume
            volume_spike = self._detect_volume(chart_region)

            # Step 7: Detect market structure
            structure = self._detect_structure(candles)

            # Step 8: Classify volatility
            volatility = self._classify_volatility(candles)

            # Step 9: Analyze last candle
            last_candle = self._analyze_last_candle(candles)

            # Step 10: Wick analysis
            wick_analysis = self._analyze_wicks(candles)

            # Step 11: Body ratio
            avg_body_ratio = self._avg_body_ratio(candles)

            elapsed_ms = (time.perf_counter() - t0) * 1000

            return ChartFeatures(
                trend=trend,
                candles_detected=len(candles),
                support=support,
                resistance=resistance,
                last_candle=last_candle,
                volume_spike=volume_spike,
                structure=structure,
                volatility=volatility,
                trend_slope=round(slope, 6),
                avg_body_ratio=round(avg_body_ratio, 4),
                wick_analysis=wick_analysis,
            )

        except Exception:
            return ChartFeatures()

    def _denoise(self, frame: np.ndarray) -> np.ndarray:
        """Apply Gaussian blur for denoising."""
        return cv2.GaussianBlur(frame, (3, 3), 0)

    def _extract_chart_region(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Isolate the chart region from the full frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # Find largest rectangular contour (the chart area)
        largest = max(contours, key=cv2.contourArea)

        # Only use if it covers at least 30% of the frame
        frame_area = frame.shape[0] * frame.shape[1]
        if cv2.contourArea(largest) < frame_area * 0.3:
            return None

        x, y, w, h = cv2.boundingRect(largest)

        # Add small padding
        pad = 5
        y_start = max(0, y - pad)
        y_end = min(frame.shape[0], y + h + pad)
        x_start = max(0, x - pad)
        x_end = min(frame.shape[1], x + w + pad)

        return frame[y_start:y_end, x_start:x_end]

    def _detect_candles(self, chart: np.ndarray) -> list[DetectedCandle]:
        """Detect candlestick patterns from chart image using color segmentation."""
        hsv = cv2.cvtColor(chart, cv2.COLOR_BGR2HSV)

        # Green candle mask (bullish)
        green_mask = cv2.inRange(hsv, (35, 50, 50), (85, 255, 255))

        # Red candle mask (bearish)
        red_mask_lower = cv2.inRange(hsv, (0, 50, 50), (10, 255, 255))
        red_mask_upper = cv2.inRange(hsv, (170, 50, 50), (180, 255, 255))
        red_mask = cv2.bitwise_or(red_mask_lower, red_mask_upper)

        # Combine masks
        mask = cv2.bitwise_or(green_mask, red_mask)

        # Morphological close to connect candle parts
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        candles = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)

            # Filter by minimum height
            if h < self.min_candle_height:
                continue

            # Determine if bullish or green
            # Check center pixel color
            center_x = x + w // 2
            center_y = y + h // 2
            if 0 <= center_x < chart.shape[1] and 0 <= center_y < chart.shape[0]:
                pixel_hsv = hsv[center_y, center_x]
                is_bullish = 35 <= pixel_hsv[0] <= 85
            else:
                is_bullish = True

            candle = DetectedCandle(
                x=x, y=y, width=w, height=h,
                is_bullish=is_bullish,
                body_top=y + h // 4,
                body_bottom=y + 3 * h // 4,
                wick_top=y,
                wick_bottom=y + h,
            )
            candles.append(candle)

        # Sort by x position (left to right = chronological)
        candles.sort(key=lambda c: c.x)

        # Limit to max_candles
        return candles[-self.max_candles:]

    def _detect_trend(self, candles: list[DetectedCandle]) -> tuple[str, float]:
        """Detect trend direction from candle positions using polyfit."""
        if len(candles) < 3:
            return "sideways", 0.0

        xs = np.array([c.x for c in candles], dtype=np.float64)
        ys = np.array([c.y for c in candles], dtype=np.float64)

        # Polyfit (linear regression)
        coeffs = np.polyfit(xs, ys, 1)
        slope = coeffs[0]

        # Note: in image coordinates, y increases downward
        # So negative slope = price going up (uptrend)
        # Positive slope = price going down (downtrend)

        if slope < -0.3:
            return "uptrend", slope
        elif slope > 0.3:
            return "downtrend", slope
        else:
            return "sideways", slope

    def _detect_levels(
        self, candles: list[DetectedCandle], chart_height: int
    ) -> tuple[float, float]:
        """Detect support and resistance levels from candle extremes."""
        if not candles:
            return 0.0, 0.0

        # Convert pixel positions to price-like values (normalized 0-1)
        # Lower y = higher price (image coordinates inverted)
        lows = [1.0 - (c.y + c.height) / chart_height for c in candles]
        highs = [1.0 - c.y / chart_height for c in candles]

        # Support = cluster of lows (80th percentile of lows)
        support = float(np.percentile(lows, 20)) if lows else 0.0

        # Resistance = cluster of highs (20th percentile of highs)
        resistance = float(np.percentile(highs, 80)) if highs else 0.0

        return round(support, 4), round(resistance, 4)

    def _detect_volume(self, chart: np.ndarray) -> bool:
        """Detect volume spike from bottom region of chart."""
        h = chart.shape[0]

        # Volume bars typically in bottom 20% of chart
        volume_region = chart[int(h * 0.8):h, :]

        if volume_region.size == 0:
            return False

        gray = cv2.cvtColor(volume_region, cv2.COLOR_BGR2GRAY)

        # Bright pixels = volume bars
        bright_threshold = np.mean(gray) + np.std(gray)
        bright_ratio = np.sum(gray > bright_threshold) / gray.size

        # High volume = more bright pixels than typical
        return bright_ratio > 0.15

    def _detect_structure(self, candles: list[DetectedCandle]) -> str:
        """Detect market structure (higher highs, lower lows)."""
        if len(candles) < 6:
            return "neutral"

        # Compare first half vs second half
        mid = len(candles) // 2
        first_half = candles[:mid]
        second_half = candles[mid:]

        avg_high_first = np.mean([c.y for c in first_half])
        avg_high_second = np.mean([c.y for c in second_half])

        # In image coords: lower y = higher price
        if avg_high_second < avg_high_first - 2:
            return "higher_highs"
        elif avg_high_second > avg_high_first + 2:
            return "lower_lows"
        return "neutral"

    def _classify_volatility(self, candles: list[DetectedCandle]) -> str:
        """Classify volatility from candle height variation."""
        if len(candles) < 3:
            return "medium"

        heights = [c.height for c in candles]
        mean_h = np.mean(heights)
        std_h = np.std(heights)

        if mean_h <= 0:
            return "medium"

        cv = std_h / mean_h  # Coefficient of variation

        if cv > 0.8:
            return "high"
        elif cv < 0.3:
            return "low"
        return "medium"

    def _analyze_last_candle(self, candles: list[DetectedCandle]) -> str:
        """Analyze the last detected candle."""
        if not candles:
            return "neutral"

        last = candles[-1]
        if last.is_bullish:
            return "bullish"
        return "bearish"

    def _analyze_wicks(self, candles: list[DetectedCandle]) -> str:
        """Analyze wick patterns for rejection signals."""
        if not candles:
            return "neutral"

        last = candles[-1]

        if last.body_ratio < 0.2:
            # Small body = potential rejection/pin bar
            upper_wick = last.body_top - last.wick_top
            lower_wick = last.wick_bottom - last.body_bottom

            if lower_wick > upper_wick * 2:
                return "rejection_down"  # Bullish rejection
            elif upper_wick > lower_wick * 2:
                return "rejection_up"    # Bearish rejection

        return "neutral"

    def _avg_body_ratio(self, candles: list[DetectedCandle]) -> float:
        """Average body ratio across all candles."""
        if not candles:
            return 0.0

        ratios = [c.body_ratio for c in candles]
        return float(np.mean(ratios))

    def get_stats(self) -> dict:
        """Get segmenter statistics."""
        return {
            "total_analyses": self._analysis_count,
            "cv2_available": CV2_AVAILABLE,
        }


# ── Convenience Function ────────────────────────────────────

def segment_chart(image: Union[np.ndarray, str]) -> dict:
    """
    One-shot chart segmentation. Returns structured features dict.

    Usage:
        features = segment_chart("chart.png")
        # {"trend": "uptrend", "candles_detected": 42, ...}
    """
    segmenter = ChartSegmenter()
    result = segmenter.analyze(image)
    return result.to_dict()
