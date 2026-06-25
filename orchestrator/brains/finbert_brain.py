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
import time
from pathlib import Path

import httpx
import numpy as np

from .base_brain import BaseBrain, BrainSignal

logger = logging.getLogger(__name__)

# ── .env auto-load ─────────────────────────────────────
try:
    from dotenv import load_dotenv
    _project_root = Path(__file__).resolve().parents[2]
    _env_file = _project_root / ".env"
    if _env_file.exists():
        load_dotenv(_env_file, override=False)
except ImportError:
    pass

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

    Features:
      - Auto-fetches crypto news from CryptoCompare API
      - Auto-loads OPENROUTER_API_KEY from .env
      - Caches headlines per symbol (5-min TTL)
    """

    @property
    def brain_id(self) -> str:
        return "finbert_nlp"

    def __init__(self) -> None:
        self.ollama_url = os.getenv("OLLAMA_URL", os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
        self.ollama_models = os.getenv(
            "OLLAMA_SENTIMENT_MODELS", os.getenv("OLLAMA_SENTIMENT_MODEL", "qwen2.5:3b,phi3:mini")
        ).split(",")
        self.nim_key = os.getenv("NIM_API_KEY", os.getenv("NVIDIA_API_KEY", ""))
        self.nim_url = os.getenv("NIM_URL", "https://integrate.api.nvidia.com")
        self.nim_model = os.getenv("NIM_SENTIMENT_MODEL", "meta/llama-3.1-8b-instruct")
        # Tier 3: OpenRouter — free model fallback (gemma-4-31b-it:free, no paid models)
        self.openrouter_key = os.getenv("OPENROUTER_API_KEY", "")
        self.openrouter_model = os.getenv("OPENROUTER_SENTIMENT_MODEL", "google/gemma-4-31b-it:free")
        self.timeout = float(os.getenv("SENTIMENT_TIMEOUT", "8.0"))
        self._http = httpx.AsyncClient(timeout=self.timeout)
        self._headlines: list[str] = []
        # Cache for auto-fetched headlines
        self._headlines_cache: dict[str, list[str]] = {}
        self._headlines_cache_ts: dict[str, float] = {}

    async def warmup(self) -> None:
        # Auto-discover available Ollama models
        try:
            import httpx
            async with httpx.AsyncClient(timeout=5) as cx:
                resp = await cx.get(f"{self.ollama_url}/api/tags")
                if resp.status_code == 200:
                    available = [m["name"] for m in resp.json().get("models", [])]
                    resolved = []
                    for m in self.ollama_models:
                        if m in available:
                            resolved.append(m)
                        # try base name match (e.g. "phi3:mini" matches "phi3:mini")
                    if not resolved and available:
                        # Fallback: use first available model
                        resolved = available[:2]  # at most 2 for concurrency
                        logger.warning(f"[finbert_nlp] Configured models not found, auto-selected {resolved}")
                    self.ollama_models = resolved or self.ollama_models
        except Exception:
            pass  # Ollama offline, cloud tiers will handle
        logger.info(f"[finbert_nlp] Ready (models={self.ollama_models})")
        if self.nim_key:
            logger.info(f"[finbert_nlp] NIM configured: {self.nim_url} model={self.nim_model}")
        if self.openrouter_key:
            logger.info(f"[finbert_nlp] OpenRouter key available, model={self.openrouter_model}")
        elif not self.nim_key:
            logger.warning("[finbert_nlp] No cloud LLM configured (NIM_API_KEY / OPENROUTER_API_KEY)")

    async def compute_score(self, symbol: str) -> BrainSignal:
        # Auto-fetch headlines if empty
        self._auto_fetch_headlines(symbol)

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

        # Total failure — produce a neutral score from keyword scan
        keyword_score = self._keyword_fallback_score()
        if keyword_score != 0.0:
            return BrainSignal(
                brain_id=self.brain_id,
                symbol=symbol,
                score=float(np.clip(keyword_score, -1.0, 1.0)),
                confidence=0.2,
                metadata={"tier": "keyword_fallback"},
            )

        return BrainSignal(
            brain_id=self.brain_id,
            symbol=symbol,
            score=0.0,
            confidence=0.05,
            metadata={"tier": "failed", "reason": "all_tiers_unavailable"},
        )

    def _auto_fetch_headlines(self, symbol: str) -> None:
        """Fetch crypto news headlines from CryptoCompare API."""
        now = time.time()
        cache_ts = self._headlines_cache_ts.get(symbol, 0)
        if (now - cache_ts) < 300 and symbol in self._headlines_cache:
            self._headlines = self._headlines_cache[symbol]
            return

        try:
            import requests
            # Map symbol to CryptoCompare categories
            base = symbol.split("/")[0].upper() if "/" in symbol else symbol.upper()
            resp = requests.get(
                f"https://min-api.cryptocompare.com/data/v2/news/?categories={base}",
                timeout=5,
            )
            if resp.status_code == 200:
                articles = resp.json().get("Data", [])[:20]
                headlines = [a.get("title", "") for a in articles if a.get("title")]
                if headlines:
                    self._headlines = headlines
                    self._headlines_cache[symbol] = headlines
                    self._headlines_cache_ts[symbol] = now
                    logger.info("[finbert_nlp] Fetched %d headlines for %s from CryptoCompare",
                                len(headlines), symbol)
                    return
        except Exception as e:
            logger.debug(f"[finbert_nlp] CryptoCompare fetch failed: {e}")

        # Fallback: CoinGecko trending
        try:
            import requests
            resp = requests.get(
                "https://api.coingecko.com/api/v3/search/trending",
                timeout=5,
            )
            if resp.status_code == 200:
                coins = resp.json().get("coins", [])[:10]
                headlines = [f"{c['item'].get('name', 'Crypto')} trending #{i+1}"
                             for i, c in enumerate(coins) if c.get("item", {}).get("name")]
                if headlines:
                    self._headlines = headlines
                    self._headlines_cache[symbol] = headlines
                    self._headlines_cache_ts[symbol] = now
                    logger.info("[finbert_nlp] Fetched %d trending from CoinGecko", len(headlines))
                    return
        except Exception as e:
            logger.debug(f"[finbert_nlp] CoinGecko fetch failed: {e}")

    def _keyword_fallback_score(self) -> float:
        """Simple keyword-based sentiment when LLM is unavailable."""
        if not self._headlines:
            return 0.0
        bullish_words = {"surge", "rally", "bullish", "breakout", "moon", "pump",
                         "soar", "gain", "rise", " ATH", "all-time high", "adoption"}
        bearish_words = {"crash", "dump", "bearish", "plunge", "fall", "drop",
                         "sell-off", "ban", "hack", "rug", "collapse", "fear"}
        score = 0.0
        for h in self._headlines[:10]:
            h_lower = h.lower()
            for w in bullish_words:
                if w in h_lower:
                    score += 0.15
            for w in bearish_words:
                if w in h_lower:
                    score -= 0.15
        return float(np.clip(score, -1.0, 1.0))

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
        """Query NIM (tier 2) or OpenRouter free (tier 3) for consensus sentiment."""
        prompt = SENTIMENT_PROMPT.format(
            symbol=symbol,
            headlines="\n".join(f"- {h}" for h in self._headlines[:10]),
        )

        # Tier 2: NIM (free NVIDIA credits)
        if self.nim_key:
            try:
                resp = await self._http.post(
                    f"{self.nim_url}/v1/chat/completions",
                    json={
                        "model": self.nim_model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.1,
                        "max_tokens": 150,
                    },
                    headers={"Authorization": f"Bearer {self.nim_key}", "Content-Type": "application/json"},
                )
                if resp.status_code == 200:
                    text = resp.json()["choices"][0]["message"]["content"]
                    sentiment, conf = self._parse_sentiment(text)
                    if sentiment:
                        logger.info("[finbert_nlp] NIM tier OK")
                        return self._sentiment_to_score(sentiment, conf), conf
                else:
                    logger.debug(f"[finbert_nlp] NIM HTTP {resp.status_code}")
            except Exception as e:
                logger.debug(f"[finbert_nlp] NIM error: {e}")

        # Tier 3: OpenRouter free model
        if self.openrouter_key:
            try:
                resp = await self._http.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    json={
                        "model": self.openrouter_model,
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
                        logger.info("[finbert_nlp] OpenRouter free tier OK")
                        return self._sentiment_to_score(sentiment, conf), conf
                else:
                    logger.debug(f"[finbert_nlp] OpenRouter HTTP {resp.status_code}")
            except Exception as e:
                logger.debug(f"[finbert_nlp] OpenRouter error: {e}")

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
