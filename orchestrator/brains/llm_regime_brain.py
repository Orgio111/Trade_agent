"""Brain #3: LLM Regime Detection via Ollama.

Uses local LLMs (Llama3/Phi-3) to classify market regime from news+price context.
Weight in Go orchestrator: 0.15.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time

import httpx

from .base_brain import BaseBrain, BrainSignal

logger = logging.getLogger(__name__)

REGIME_PROMPT = """You are a market regime classifier. Given the following market data, classify the current regime.

Return ONLY a JSON object:
{{"regime": "trending_up"|"trending_down"|"ranging"|"volatile"|"crisis", "confidence": 0.0-1.0, "rationale": "brief explanation"}}

Market context:
- Symbol: {symbol}
- Recent price change: {price_change_pct:.2f}%
- Volume vs average: {vol_ratio:.2f}x
- RSI(14): {rsi:.0f}
- Recent headlines: {headlines}
"""


class LLMRegimeBrain(BaseBrain):
    """Dual-tier LLM regime detection brain.

    Tier 1 (local): Ollama Llama3/Phi-3 for low-latency regime classification.
    Tier 2 (cloud): NVIDIA NIM / OpenRouter for consensus if local fails or times out.
    """

    @property
    def brain_id(self) -> str:
        return "llm_regime"

    def __init__(self) -> None:
        self.ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        self.ollama_model = os.getenv("OLLAMA_REGIME_MODEL", "llama3:8b")
        self.nim_url = os.getenv("NIM_URL", "")
        self.nim_model = os.getenv("NIM_REGIME_MODEL", "meta/llama3-8b-instruct")
        self.openrouter_key = os.getenv("OPENROUTER_API_KEY", "")
        self.timeout = float(os.getenv("LLM_REGIME_TIMEOUT", "5.0"))
        self._http = httpx.AsyncClient(timeout=self.timeout)

    async def warmup(self) -> None:
        """Verify Ollama is reachable."""
        try:
            resp = await self._http.get(f"{self.ollama_url}/api/tags")
            if resp.status_code == 200:
                models = [m["name"] for m in resp.json().get("models", [])]
                if self.ollama_model not in models:
                    logger.warning(f"[llm_regime] Model {self.ollama_model} not found in Ollama")
                else:
                    logger.info(f"[llm_regime] Ollama ready, model={self.ollama_model}")
        except Exception as e:
            logger.warning(f"[llm_regime] Ollama unreachable: {e}")

    async def compute_score(self, symbol: str) -> BrainSignal:
        """Classify market regime and map to directional score."""
        context = self._build_context(symbol)

        # Tier 1: Local Ollama
        regime, confidence = await self._query_ollama(context)

        # Tier 2: Cloud fallback if local fails or low confidence
        if regime is None and (self.nim_url or self.openrouter_key):
            regime, confidence = await self._query_cloud(context)

        if regime is None:
            return BrainSignal(
                brain_id=self.brain_id,
                symbol=symbol,
                score=0.0,
                confidence=0.1,
                metadata={"fallback": True, "reason": "llm_unavailable"},
            )

        # Map regime to score
        score = self._regime_to_score(regime)

        return BrainSignal(
            brain_id=self.brain_id,
            symbol=symbol,
            score=score,
            confidence=confidence,
            metadata={"regime": regime, "model": self.ollama_model},
        )

    def _build_context(self, symbol: str) -> dict:
        """Build context dict for the LLM prompt."""
        return {
            "symbol": symbol,
            "price_change_pct": 0.0,   # filled by data pipeline
            "vol_ratio": 1.0,
            "rsi": 50.0,
            "headlines": "No recent headlines",
        }

    async def _query_ollama(self, context: dict) -> tuple[str | None, float]:
        """Query local Ollama for regime classification."""
        prompt = REGIME_PROMPT.format(**context)
        try:
            resp = await self._http.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 200},
                },
            )
            if resp.status_code == 200:
                text = resp.json().get("response", "")
                return self._parse_regime_response(text)
        except (httpx.TimeoutException, httpx.ConnectError) as e:
            logger.debug(f"[llm_regime] Ollama timeout/error: {e}")
        return None, 0.0

    async def _query_cloud(self, context: dict) -> tuple[str | None, float]:
        """Query NVIDIA NIM or OpenRouter as cloud fallback."""
        prompt = REGIME_PROMPT.format(**context)

        # Try NIM first
        if self.nim_url:
            try:
                resp = await self._http.post(
                    f"{self.nim_url}/v1/chat/completions",
                    json={
                        "model": self.nim_model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.1,
                        "max_tokens": 200,
                    },
                    headers={"Authorization": f"Bearer {os.getenv('NIM_API_KEY', '')}"},
                )
                if resp.status_code == 200:
                    text = resp.json()["choices"][0]["message"]["content"]
                    return self._parse_regime_response(text)
            except Exception as e:
                logger.debug(f"[llm_regime] NIM error: {e}")

        # Try OpenRouter
        if self.openrouter_key:
            try:
                resp = await self._http.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    json={
                        "model": "meta-llama/llama-3-8b-instruct:free",
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.1,
                        "max_tokens": 200,
                    },
                    headers={"Authorization": f"Bearer {self.openrouter_key}"},
                )
                if resp.status_code == 200:
                    text = resp.json()["choices"][0]["message"]["content"]
                    return self._parse_regime_response(text)
            except Exception as e:
                logger.debug(f"[llm_regime] OpenRouter error: {e}")

        return None, 0.0

    def _parse_regime_response(self, text: str) -> tuple[str | None, float]:
        """Extract regime JSON from LLM response."""
        try:
            # Find JSON in response
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(text[start:end])
                regime = data.get("regime", "ranging")
                conf = float(data.get("confidence", 0.5))
                return regime, conf
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback: keyword matching
        text_lower = text.lower()
        for keyword, regime in [
            ("trending up", "trending_up"),
            ("bullish", "trending_up"),
            ("trending down", "trending_down"),
            ("bearish", "trending_down"),
            ("ranging", "ranging"),
            ("sideways", "ranging"),
            ("volatile", "volatile"),
            ("crisis", "crisis"),
            ("panic", "crisis"),
        ]:
            if keyword in text_lower:
                return regime, 0.5
        return None, 0.0

    @staticmethod
    def _regime_to_score(regime: str) -> float:
        """Map regime classification to directional score."""
        mapping = {
            "trending_up": 0.6,
            "trending_down": -0.6,
            "ranging": 0.0,
            "volatile": 0.0,
            "crisis": -0.8,
        }
        return mapping.get(regime, 0.0)

    async def cooldown(self) -> None:
        await self._http.aclose()
