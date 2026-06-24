"""Brain #5: FinBERT NLP Sentiment Analyzer.

Analyzes crypto news headlines and social media text for sentiment.
Hybrid dual-tier: local Ollama + cloud NIM/OpenRouter fallback.
Weight in Go orchestrator: 0.10.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os

import httpx
import numpy as np

from .base_brain import BaseBrain, BrainSignal

logger = logging.getLogger(__name__)

SENTIMENT_PROMPT = """Analyze the sentiment of these crypto market headlines for {symbol}.

Headlines:
{headlines}

Return ONLY a JSON object:
{{"sentiment": "very_bullish"|"bullish"|"neutral"|"bearish"|"very_bearish", "confidence": 0.0-1.0, "key_signal": "brief explanation"}}
"""


class FinBERTBrain(BaseBrain):
    """NLP sentiment analysis brain.

    Uses dual-tier inference:
      Tier 1: Local Ollama models (2 concurrent) for raw scoring
      Tier 2: Cloud NIM/OpenRouter for consensus fallback
    Outputs a bounded scalar between -1.0 (panic) and +1.0 (euphoria).
    """

    @property
    def brain_id(self) -> str:
        return "finbert_nlp"

    def __init__(self) -> None:
        self.ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        self.ollama_models = os.getenv(
            "OLLAMA_SENTIMENT_MODELS", "llama3:8b,phi3:mini"
        ).split(",")
        self.nim_url = os.getenv("NIM_URL", "")
        self.nim_model = os.getenv("NIM_SENTIMENT_MODEL", "meta/llama3-8b-instruct")
        self.openrouter_key = os.getenv("OPENROUTER_API_KEY", "")
        self.timeout = float(os.getenv("SENTIMENT_TIMEOUT", "8.0"))
        self._http = httpx.AsyncClient(timeout=self.timeout)
        self._headlines: list[str] = []

    async def warmup(self) -> None:
        logger.info(f"[finbert_nlp] Ready (models={self.ollama_models})")

    async def compute_score(self, symbol: str) -> BrainSignal:
        if not self._headlines:
            return BrainSignal(
                brain_id=self.brain_id,
                symbol=symbol,
                score=0.0,
                confidence=0.1,
                metadata={"reason": "no_headlines"},
            )

        # Tier 1: Query 2 local Ollama models concurrently
        tasks = [
            self._query_ollama(model, symbol)
            for model in self.ollama_models[:2]
            if model.strip()
        ]
        local_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect valid local scores
        local_scores = []
        for result in local_results:
            if isinstance(result, tuple) and result[0] is not None:
                sentiment, conf = result
                local_scores.append(self._sentiment_to_score(sentiment, conf))

        # If local consensus available, use it
        if len(local_scores) >= 2:
            avg_score = float(np.mean(local_scores))
            avg_conf = min(0.9, float(np.mean([abs(s) for s in local_scores])) + 0.3)
            return BrainSignal(
                brain_id=self.brain_id,
                symbol=symbol,
                score=float(np.clip(avg_score, -1.0, 1.0)),
                confidence=avg_conf,
                metadata={"tier": "local", "scores": [round(s, 4) for s in local_scores]},
            )

        # Single local result
        if len(local_scores) == 1:
            return BrainSignal(
                brain_id=self.brain_id,
                symbol=symbol,
                score=float(np.clip(local_scores[0], -1.0, 1.0)),
                confidence=0.4,
                metadata={"tier": "local_single"},
            )

        # Tier 2: Cloud fallback
        cloud_score, cloud_conf = await self._query_cloud(symbol)
        if cloud_score is not None:
            return BrainSignal(
                brain_id=self.brain_id,
                symbol=symbol,
                score=float(np.clip(cloud_score, -1.0, 1.0)),
                confidence=cloud_conf,
                metadata={"tier": "cloud"},
            )

        # Total failure
        return BrainSignal(
            brain_id=self.brain_id,
            symbol=symbol,
            score=0.0,
            confidence=0.05,
            metadata={"tier": "failed", "reason": "all_tiers_unavailable"},
        )

    def push_headlines(self, headlines: list[str]) -> None:
        """Push new headlines for next analysis cycle."""
        self._headlines = headlines[-20:]  # keep last 20

    async def _query_ollama(self, model: str, symbol: str) -> tuple[str | None, float]:
        """Query one Ollama model for sentiment."""
        prompt = SENTIMENT_PROMPT.format(
            symbol=symbol,
            headlines="\n".join(f"- {h}" for h in self._headlines[:10]),
        )
        try:
            resp = await self._http.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": model.strip(),
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 150},
                },
            )
            if resp.status_code == 200:
                text = resp.json().get("response", "")
                return self._parse_sentiment(text)
        except (httpx.TimeoutException, httpx.ConnectError):
            logger.debug(f"[finbert_nlp] Ollama {model} timeout")
        return None, 0.0

    async def _query_cloud(self, symbol: str) -> tuple[float | None, float]:
        """Query cloud API for consensus sentiment."""
        prompt = SENTIMENT_PROMPT.format(
            symbol=symbol,
            headlines="\n".join(f"- {h}" for h in self._headlines[:10]),
        )

        # Try NIM
        if self.nim_url:
            try:
                resp = await self._http.post(
                    f"{self.nim_url}/v1/chat/completions",
                    json={
                        "model": self.nim_model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.1,
                        "max_tokens": 150,
                    },
                    headers={"Authorization": f"Bearer {os.getenv('NIM_API_KEY', '')}"},
                )
                if resp.status_code == 200:
                    text = resp.json()["choices"][0]["message"]["content"]
                    sentiment, conf = self._parse_sentiment(text)
                    if sentiment:
                        return self._sentiment_to_score(sentiment, conf), conf
            except Exception:
                pass

        # Try OpenRouter free tier
        if self.openrouter_key:
            try:
                resp = await self._http.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    json={
                        "model": "meta-llama/llama-3-8b-instruct:free",
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.1,
                        "max_tokens": 150,
                    },
                    headers={"Authorization": f"Bearer {self.openrouter_key}"},
                )
                if resp.status_code == 200:
                    text = resp.json()["choices"][0]["message"]["content"]
                    sentiment, conf = self._parse_sentiment(text)
                    if sentiment:
                        return self._sentiment_to_score(sentiment, conf), conf
            except Exception:
                pass

        return None, 0.0

    def _parse_sentiment(self, text: str) -> tuple[str | None, float]:
        """Extract sentiment JSON from LLM response."""
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(text[start:end])
                sentiment = data.get("sentiment", "neutral")
                conf = float(data.get("confidence", 0.5))
                return sentiment, conf
        except (json.JSONDecodeError, ValueError):
            pass

        # Keyword fallback
        text_lower = text.lower()
        for keyword, label in [
            ("very bullish", "very_bullish"),
            ("bullish", "bullish"),
            ("bearish", "bearish"),
            ("very bearish", "very_bearish"),
            ("neutral", "neutral"),
        ]:
            if keyword in text_lower:
                return label, 0.5
        return None, 0.0

    @staticmethod
    def _sentiment_to_score(sentiment: str, confidence: float) -> float:
        """Map sentiment label to scalar score [-1, +1]."""
        mapping = {
            "very_bullish": 0.8,
            "bullish": 0.4,
            "neutral": 0.0,
            "bearish": -0.4,
            "very_bearish": -0.8,
        }
        base = mapping.get(sentiment, 0.0)
        return base * confidence

    async def cooldown(self) -> None:
        await self._http.aclose()
