"""
QUANTEX VLM Agent — Vision Language Model chart analysis.

Renders recent candles as a matplotlib chart image, sends to Ollama
moondream/llava for pattern recognition, and returns structured analysis.

Architecture:
  CandleBuffer → Matplotlib chart → Ollama VLM → StructuredOutput

Usage:
    agent = VLMAgent(ollama_url="http://localhost:11434")
    result = await agent.analyze(symbol="BTCUSDT", timeframe="1m")
    # result = {"trend": "bullish", "pattern": "ascending_triangle", ...}
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx

logger = logging.getLogger("quantex.vlm_agent")

# Default model — moondream for fast chart analysis, llava as fallback
DEFAULT_VLM_MODEL = os.getenv("VLM_MODEL", "moondream")
DEFAULT_OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


@dataclass
class VLMResult:
    """Structured output from VLM chart analysis."""
    trend: str = "neutral"          # bullish, bearish, neutral
    pattern: str = ""               # ascending_triangle, head_shoulders, etc.
    support: float = 0.0
    resistance: float = 0.0
    confidence: float = 0.0
    reasoning: str = ""
    indicators: dict[str, Any] = field(default_factory=dict)
    model: str = ""
    latency_ms: float = 0.0
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "trend": self.trend,
            "pattern": self.pattern,
            "support": self.support,
            "resistance": self.resistance,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "indicators": self.indicators,
            "model": self.model,
            "latency_ms": self.latency_ms,
            "error": self.error,
        }


class VLMAgent:
    """
    Vision Language Model agent for chart pattern recognition.

    Pipeline:
      1. Render candlestick chart as PNG (matplotlib)
      2. Send base64 image to Ollama VLM endpoint
      3. Parse structured JSON response
      4. Return VLMResult with trend, pattern, levels, confidence
    """

    def __init__(
        self,
        ollama_url: str = DEFAULT_OLLAMA_URL,
        model: str = DEFAULT_VLM_MODEL,
        timeout: float = 30.0,
    ):
        self.ollama_url = ollama_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    async def analyze(
        self,
        symbol: str,
        candles: list[dict],
        indicators: dict[str, Any] | None = None,
        timeframe: str = "1m",
    ) -> VLMResult:
        """
        Analyze a chart from recent candle data.

        Args:
            symbol: Trading pair
            candles: List of candle dicts (oldest first)
            indicators: Optional computed indicators (RSI, MACD, etc.)
            timeframe: Candle interval label

        Returns:
            VLMResult with structured analysis
        """
        start = time.perf_counter()

        if len(candles) < 10:
            return VLMResult(
                error=f"Insufficient candles ({len(candles)} < 10)",
                latency_ms=(time.perf_counter() - start) * 1000,
            )

        # Step 1: Render chart as PNG
        try:
            chart_bytes = self._render_chart(candles, symbol, timeframe, indicators)
        except Exception as e:
            logger.error(f"Chart render failed: {e}")
            return VLMResult(
                error=f"Chart render failed: {e}",
                latency_ms=(time.perf_counter() - start) * 1000,
            )

        # Step 2: Send to Ollama VLM
        try:
            result = await self._query_vlm(chart_bytes, symbol, timeframe, indicators)
        except Exception as e:
            logger.error(f"VLM query failed: {e}")
            result = VLMResult(error=f"VLM query failed: {e}")

        result.latency_ms = (time.perf_counter() - start) * 1000
        result.model = self.model
        return result

    def _render_chart(
        self,
        candles: list[dict],
        symbol: str,
        timeframe: str,
        indicators: dict | None = None,
    ) -> bytes:
        """
        Render candlestick chart as PNG bytes using matplotlib.

        Returns base64-encoded PNG for Ollama vision API.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            from matplotlib.patches import Rectangle
        except ImportError:
            raise RuntimeError("matplotlib required for VLM chart rendering")

        # Take last 60 candles for cleaner chart
        recent = candles[-60:]
        n = len(recent)

        # Parse OHLC
        opens = [c.get("open", 0) for c in recent]
        highs = [c.get("high", 0) for c in recent]
        lows = [c.get("low", 0) for c in recent]
        closes = [c.get("close", 0) for c in recent]
        volumes = [c.get("volume", 0) for c in recent]

        # X positions
        x = list(range(n))

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(12, 8), height_ratios=[3, 1],
            gridspec_kw={"hspace": 0.05},
        )
        fig.patch.set_facecolor("#1a1a2e")

        for ax in (ax1, ax2):
            ax.set_facecolor("#16213e")
            ax.tick_params(colors="#a0a0a0", labelsize=8)
            for spine in ax.spines.values():
                spine.set_color("#333")

        # Candlesticks
        for i in range(n):
            color = "#26a69a" if closes[i] >= opens[i] else "#ef5350"
            # Wick
            ax1.plot([i, i], [lows[i], highs[i]], color=color, linewidth=0.8)
            # Body
            body_low = min(opens[i], closes[i])
            body_high = max(opens[i], closes[i])
            body_height = max(body_high - body_low, highs[i] * 0.0001)
            rect = Rectangle(
                (i - 0.35, body_low), 0.7, body_height,
                facecolor=color, edgecolor=color, linewidth=0.5,
            )
            ax1.add_patch(rect)

        # EMA lines (if close prices available)
        if n >= 20:
            ema20 = self._ema(closes, 20)
            ax1.plot(x, ema20, color="#ffd700", linewidth=1.0, alpha=0.8, label="EMA20")
        if n >= 50:
            ema50 = self._ema(closes, 50)
            ax1.plot(x, ema50, color="#ff6b6b", linewidth=1.0, alpha=0.8, label="EMA50")

        # Volume bars
        for i in range(n):
            color = "#26a69a" if closes[i] >= opens[i] else "#ef5350"
            ax2.bar(i, volumes[i], width=0.7, color=color, alpha=0.6)

        # Title and labels
        ax1.set_title(
            f"{symbol} — {timeframe} — Last {n} candles",
            color="white", fontsize=12, fontweight="bold",
        )
        ax1.legend(loc="upper left", fontsize=8, facecolor="#16213e", edgecolor="#333")
        ax1.set_ylabel("Price", color="#a0a0a0", fontsize=9)
        ax2.set_ylabel("Volume", color="#a0a0a0", fontsize=9)

        plt.tight_layout()

        # Render to PNG bytes
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        return buf.read()

    @staticmethod
    def _ema(data: list[float], period: int) -> list[float]:
        """Compute EMA over a list."""
        if len(data) < period:
            return data[:]
        alpha = 2.0 / (period + 1)
        ema = [sum(data[:period]) / period]
        for val in data[period:]:
            ema.append(alpha * val + (1 - alpha) * ema[-1])
        # Pad start with NaN
        return [float("nan")] * (period - 1) + ema

    async def _query_vlm(
        self,
        chart_png: bytes,
        symbol: str,
        timeframe: str,
        indicators: dict | None,
    ) -> VLMResult:
        """
        Send chart image to Ollama VLM endpoint and parse response.
        """
        b64_image = base64.b64encode(chart_png).decode("utf-8")

        # Build indicator context
        ind_text = ""
        if indicators:
            parts = []
            for k, v in indicators.items():
                if isinstance(v, (int, float)):
                    parts.append(f"{k}={v:.2f}")
                else:
                    parts.append(f"{k}={v}")
            ind_text = "\nIndicators: " + ", ".join(parts[:10])

        prompt = f"""You are a professional crypto market analyst. Analyze this {symbol} {timeframe} candlestick chart.

{ind_text}

Provide a structured JSON analysis:
{{
  "trend": "bullish|bearish|neutral",
  "pattern": "description of detected pattern or 'none'",
  "support": price_level,
  "resistance": price_level,
  "confidence": 0.0-1.0,
  "reasoning": "brief technical analysis reasoning"
}}

Output ONLY valid JSON. No markdown, no explanation outside JSON."""

        payload = {
            "model": self.model,
            "prompt": prompt,
            "images": [b64_image],
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 512,
            },
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
            )
            resp.raise_for_status()

        raw_text = resp.json().get("response", "")

        # Parse JSON from response (handle markdown code blocks)
        text = raw_text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        text = text.strip()

        try:
            data = json.loads(text)
            return VLMResult(
                trend=data.get("trend", "neutral"),
                pattern=data.get("pattern", ""),
                support=float(data.get("support", 0)),
                resistance=float(data.get("resistance", 0)),
                confidence=float(data.get("confidence", 0.5)),
                reasoning=data.get("reasoning", ""),
                indicators=indicators or {},
            )
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"VLM JSON parse failed: {e}\nRaw: {text[:200]}")
            return VLMResult(
                error=f"JSON parse failed: {e}",
                reasoning=text[:200],
                confidence=0.0,
            )

    async def health_check(self) -> bool:
        """Check if Ollama VLM server is available."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.ollama_url}/api/tags")
                if resp.status_code == 200:
                    models = [m.get("name", "") for m in resp.json().get("models", [])]
                    available = any(self.model in m for m in models)
                    if not available:
                        logger.warning(
                            f"VLM model '{self.model}' not found. "
                            f"Available: {models}"
                        )
                    return available
        except Exception:
            return False
        return False
