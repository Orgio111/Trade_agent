"""Sentiment Agent: news headlines + social media scored via NVIDIA NIM."""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime

import httpx

from core.config import get_settings
from core.messaging import MsgType, get_bus
from core.models import SentimentSignal, Side
from core.nim_client import nim_json
from core.observability import AGENT_LATENCY, SIGNAL_COUNTER

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a financial sentiment analyst. Given news headlines and social mentions
about a crypto/equity asset, produce:
{
  "news_score": <float -1 to 1>,
  "social_score": <float -1 to 1>,
  "fear_greed_index": <float 0 to 100>,
  "trending_topics": ["<topic>", ...],
  "trend": "BUY" | "SELL" | "HOLD",
  "confidence": <float 0 to 1>
}
1.0 = extremely bullish, -1.0 = extremely bearish. Be unbiased and data-driven."""


class SentimentAgent:
    """Scrapes news + social and runs NIM sentiment scoring."""

    async def analyze(self, symbol: str) -> SentimentSignal:
        with AGENT_LATENCY.labels(agent="sentiment").time():
            headlines, social = await asyncio.gather(
                self._fetch_news(symbol),
                self._fetch_social(symbol),
            )
            all_content = headlines + social
            result = await nim_json(
                [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Asset: {symbol}\n"
                            f"Headlines:\n" + "\n".join(f"- {h}" for h in headlines[:10]) + "\n"
                            f"Social mentions:\n" + "\n".join(f"- {s}" for s in social[:10])
                        ),
                    },
                ]
            )

        trend_str = result.get("trend", "HOLD")
        signal = SentimentSignal(
            symbol=symbol,
            timestamp=datetime.utcnow(),
            news_score=float(result.get("news_score", 0.0)),
            social_score=float(result.get("social_score", 0.0)),
            fear_greed_index=float(result.get("fear_greed_index", 50.0)),
            trending_topics=result.get("trending_topics", []),
            trend=Side(trend_str),
            confidence=float(result.get("confidence", 0.5)),
            raw_headlines=headlines[:5],
        )

        cfg = get_settings()
        bus = await get_bus()
        await bus.publish(
            cfg.stream_signals, MsgType.SENTIMENT_SIGNAL, signal.model_dump(mode="json")
        )
        SIGNAL_COUNTER.labels(agent="sentiment", symbol=symbol, side=trend_str).inc()
        return signal

    async def _fetch_news(self, symbol: str) -> list[str]:
        base = symbol.split("/")[0].upper()
        headlines: list[str] = []
        try:
            async with httpx.AsyncClient(timeout=6.0) as client:
                r = await client.get(
                    "https://min-api.cryptocompare.com/data/v2/news/",
                    params={"categories": base, "lTs": 0},
                )
                if r.status_code == 200:
                    articles = r.json().get("Data", [])
                    headlines = [a["title"] for a in articles[:15]]
        except Exception as exc:
            logger.warning("News fetch failed: %s", exc)
        return headlines

    async def _fetch_social(self, symbol: str) -> list[str]:
        """Placeholder — in production connect to Twitter/Reddit API."""
        return []
