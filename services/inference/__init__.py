"""
Inference Service — vLLM token streaming for real-time trading decisions.

Provides:
- vLLM server integration with token streaming
- Fast token-level decision triggers
- Model management and routing
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, AsyncGenerator, Callable, Optional

import httpx
import websockets

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ModelConfig:
    """vLLM model configuration."""
    model_name: str
    model_path: str  # HuggingFace model ID or local path
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.90
    max_model_len: int = 4096
    dtype: str = "auto"
    quantization: str | None = None  # "awq", "gptq", "fp8"
    port: int = 8000
    host: str = "0.0.0.0"


@dataclass
class InferenceRequest:
    """Inference request with streaming support."""
    prompt: str
    model: str
    temperature: float = 0.2
    max_tokens: int = 256
    stream: bool = True
    stop: list[str] | None = None
    trace_id: str = ""


@dataclass
class InferenceResponse:
    """Inference response (streaming chunk or final)."""
    token: str = ""
    text: str = ""
    finished: bool = False
    latency_ms: float = 0.0
    tokens_generated: int = 0
    trace_id: str = ""


# ═══════════════════════════════════════════════════════════════════
# TOKEN-LEVEL DECISION ENGINE
# ═══════════════════════════════════════════════════════════════════

class TokenDecisionEngine:
    """
    Token-level decision engine.
    Parses LLM tokens in real-time to trigger trading decisions.
    """

    def __init__(
        self,
        buy_triggers: list[str] | None = None,
        sell_triggers: list[str] | None = None,
        min_confidence_tokens: int = 3,
    ):
        self.buy_triggers = buy_triggers or ["BUY", "LONG", "buy", "long"]
        self.sell_triggers = sell_triggers or ["SELL", "SHORT", "sell", "short"]
        self.min_confidence_tokens = min_confidence_tokens

        self._buy_token_count = 0
        self._sell_token_count = 0
        self._buffer = ""

    def process_token(self, token: str) -> Optional[dict]:
        """
        Process a single token and return decision if triggered.
        Returns dict with signal and confidence, or None if no decision yet.
        """
        self._buffer += token

        # Check for buy triggers
        for trigger in self.buy_triggers:
            if trigger in self._buffer.upper():
                self._buy_token_count += 1

        # Check for sell triggers
        for trigger in self.sell_triggers:
            if trigger in self._buffer.upper():
                self._sell_token_count += 1

        # Require minimum tokens for confidence
        if self._buy_token_count >= self.min_confidence_tokens:
            decision = {
                "signal": "BUY",
                "confidence": min(0.5 + self._buy_token_count * 0.1, 0.95),
                "trigger": "token_stream",
                "tokens_seen": self._buy_token_count,
            }
            self._reset()
            return decision

        if self._sell_token_count >= self.min_confidence_tokens:
            decision = {
                "signal": "SELL",
                "confidence": min(0.5 + self._sell_token_count * 0.1, 0.95),
                "trigger": "token_stream",
                "tokens_seen": self._sell_token_count,
            }
            self._reset()
            return decision

        return None

    def _reset(self):
        self._buy_token_count = 0
        self._sell_token_count = 0
        self._buffer = ""

    def reset(self):
        """Public reset method."""
        self._reset()


# ═══════════════════════════════════════════════════════════════════
# vLLM CLIENT (HTTP + WebSocket)
# ═══════════════════════════════════════════════════════════════════

class VLLMClient:
    """
    Async client for vLLM server.
    Supports both REST and WebSocket streaming.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        timeout: float = 30.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        self._client = httpx.AsyncClient(timeout=self.timeout)
        return self

    async def __aexit__(self, *args):
        if self._client:
            await self._client.aclose()

    async def generate_stream(
        self,
        request: InferenceRequest,
    ) -> AsyncGenerator[InferenceResponse, None]:
        """Generate tokens via streaming HTTP (SSE)."""
        url = f"{self.base_url}/chat/completions"

        payload = {
            "model": request.model,
            "messages": [{"role": "user", "content": request.prompt}],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stream": True,
        }

        if request.stop:
            payload["stop"] = request.stop

        start_time = time.perf_counter()
        tokens_generated = 0

        try:
            async with self._client.stream("POST", url, json=payload) as response:
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data.strip() == "[DONE]":
                            break

                        try:
                            chunk = json.loads(data)
                            choices = chunk.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                token = delta.get("content", "")
                                if token:
                                    tokens_generated += 1
                                    yield InferenceResponse(
                                        token=token,
                                        text=token,
                                        finished=False,
                                        latency_ms=(time.perf_counter() - start_time) * 1000,
                                        tokens_generated=tokens_generated,
                                        trace_id=request.trace_id,
                                    )
                        except json.JSONDecodeError:
                            continue

                # Final chunk
                yield InferenceResponse(
                    token="",
                    text="",
                    finished=True,
                    latency_ms=(time.perf_counter() - start_time) * 1000,
                    tokens_generated=tokens_generated,
                    trace_id=request.trace_id,
                )

        except Exception as e:
            logger.error(f"vLLM stream error: {e}")
            yield InferenceResponse(
                token="",
                text=f"ERROR: {e}",
                finished=True,
                latency_ms=(time.perf_counter() - start_time) * 1000,
                tokens_generated=tokens_generated,
                trace_id=request.trace_id,
            )

    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        """Non-streaming generation."""
        url = f"{self.base_url}/chat/completions"

        payload = {
            "model": request.model,
            "messages": [{"role": "user", "content": request.prompt}],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stream": False,
        }

        if request.stop:
            payload["stop"] = request.stop

        start_time = time.perf_counter()

        try:
            response = await self._client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()

            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            tokens = data.get("usage", {}).get("completion_tokens", 0)

            return InferenceResponse(
                token="",
                text=content,
                finished=True,
                latency_ms=(time.perf_counter() - start_time) * 1000,
                tokens_generated=tokens,
                trace_id=request.trace_id,
            )

        except Exception as e:
            logger.error(f"vLLM generate error: {e}")
            return InferenceResponse(
                token="",
                text=f"ERROR: {e}",
                finished=True,
                latency_ms=(time.perf_counter() - start_time) * 1000,
                tokens_generated=0,
                trace_id=request.trace_id,
            )

    async def health_check(self) -> dict:
        """Check vLLM server health."""
        try:
            response = await self._client.get(f"{self.base_url}/health")
            return {"healthy": response.status_code == 200, "details": response.json()}
        except Exception as e:
            return {"healthy": False, "error": str(e)}

    async def list_models(self) -> list[dict]:
        """List available models."""
        try:
            response = await self._client.get(f"{self.base_url}/models")
            return response.json().get("data", [])
        except Exception as e:
            logger.error(f"List models error: {e}")
            return []


# ═══════════════════════════════════════════════════════════════════
# INFERENCE SERVICE (Orchestrator)
# ═══════════════════════════════════════════════════════════════════

class InferenceService:
    """
    Complete inference service with:
    - Multiple model support
    - Token streaming
    - Token-level decision triggers
    - Request routing and load balancing
    """

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.models: dict[str, ModelConfig] = {}
        self.client: Optional[VLLMClient] = None
        self.decision_engine = TokenDecisionEngine(
            buy_triggers=self.config.get("buy_triggers"),
            sell_triggers=self.config.get("sell_triggers"),
            min_confidence_tokens=self.config.get("min_confidence_tokens", 3),
        )
        self._decision_callbacks: list[Callable] = []

    def add_model(self, name: str, config: ModelConfig):
        """Register a model."""
        self.models[name] = config

    async def start(self):
        """Start the service."""
        vllm_url = self.config.get("vllm_url", "http://localhost:8000/v1")
        self.client = VLLMClient(base_url=vllm_url, timeout=self.config.get("timeout", 30.0))
        await self.client.__aenter__()

        # Health check
        health = await self.client.health_check()
        logger.info(f"[InferenceService] vLLM health: {health}")

        models = await self.client.list_models()
        logger.info(f"[InferenceService] Available models: {[m.get('id') for m in models]}")

    async def stop(self):
        """Stop the service."""
        if self.client:
            await self.client.__aexit__(None, None, None)

    def on_decision(self, callback: Callable[[dict], None]):
        """Register callback for token-level decisions."""
        self._decision_callbacks.append(callback)

    async def infer_stream(
        self,
        prompt: str,
        model: str | None = None,
        trace_id: str = "",
        **kwargs,
    ) -> AsyncGenerator[InferenceResponse, None]:
        """
        Stream inference with token-level decision triggers.
        """
        if not self.client:
            raise RuntimeError("Service not started")

        # Use default model if not specified
        model = model or list(self.models.keys())[0] if self.models else "default"

        request = InferenceRequest(
            prompt=prompt,
            model=model,
            trace_id=trace_id or f"inf_{int(time.time() * 1000)}",
            stream=True,
            **kwargs,
        )

        async for response in self.client.generate_stream(request):
            # Process token for decision trigger
            if response.token and not response.finished:
                decision = self.decision_engine.process_token(response.token)
                if decision:
                    decision["trace_id"] = request.trace_id
                    decision["model"] = model
                    for cb in self._decision_callbacks:
                        try:
                            await cb(decision)
                        except Exception as e:
                            logger.error(f"Decision callback error: {e}")

            yield response

    async def infer(
        self,
        prompt: str,
        model: str | None = None,
        trace_id: str = "",
        **kwargs,
    ) -> InferenceResponse:
        """Non-streaming inference."""
        if not self.client:
            raise RuntimeError("Service not started")

        model = model or list(self.models.keys())[0] if self.models else "default"

        request = InferenceRequest(
            prompt=prompt,
            model=model,
            trace_id=trace_id or f"inf_{int(time.time() * 1000)}",
            stream=False,
            **kwargs,
        )

        return await self.client.generate(request)

    def reset_decision_engine(self):
        """Reset token decision engine."""
        self.decision_engine.reset()


# ═══════════════════════════════════════════════════════════════════
# PROMPT TEMPLATES FOR TRADING
# ═══════════════════════════════════════════════════════════════════

TRADING_PROMPTS = {
    "regime_classification": """You are a market regime classifier. Analyze the following market data and classify the regime.

Market Data:
{market_data}

Classify as ONE of: TRENDING_UP, TRENDING_DOWN, RANGING, HIGH_VOLATILITY, LOW_VOLATILITY, BREAKOUT_UP, BREAKOUT_DOWN

Output format:
REGIME: <regime>
CONFIDENCE: <0.0-1.0>
REASONING: <brief reasoning>""",

    "trend_analysis": """You are a trend analyst. Analyze the technical indicators and determine trend direction.

Technical Indicators:
{indicators}

Determine trend: STRONG_BULLISH, WEAK_BULLISH, NEUTRAL, WEAK_BEARISH, STRONG_BEARISH

Output format:
TREND: <trend>
CONFIDENCE: <0.0-1.0>
KEY_LEVELS: <support/resistance>
REASONING: <brief reasoning>""",

    "scalping_decision": """You are a scalping trader. Make a quick decision based on 5-candle window.

Recent Candles:
{candles}

Current Indicators:
{indicators}

Decision: BUY / SELL / HOLD

Output format:
DECISION: <BUY/SELL/HOLD>
CONFIDENCE: <0.0-1.0>
ENTRY_PRICE: <price>
STOP_LOSS: <price>
TAKE_PROFIT: <price>
REASONING: <brief reasoning>""",

    "risk_assessment": """You are a risk manager. Assess the proposed trade.

Proposed Trade:
{trade_proposal}

Portfolio State:
{portfolio_state}

Risk Limits:
{risk_limits}

Output format:
APPROVED: <true/false>
MAX_SIZE: <percentage>
LEVERAGE: <x>
REASONING: <brief reasoning>""",

    "order_flow_analysis": """You are an order flow analyst. Analyze the order book metrics.

Order Book Metrics:
{orderbook_metrics}

Determine: BUY_PRESSURE / SELL_PRESSURE / NEUTRAL

Output format:
SIGNAL: <BUY_PRESSURE/SELL_PRESSURE/NEUTRAL>
CONFIDENCE: <0.0-1.0>
KEY_METRICS: <OFI, CVD, spread, depth>
REASONING: <brief reasoning>""",
}


def get_trading_prompt(task: str, **kwargs) -> str:
    """Get formatted trading prompt."""
    template = TRADING_PROMPTS.get(task, "")
    if not template:
        return f"Task: {task}\nContext: {kwargs}"
    return template.format(**kwargs)


# ═══════════════════════════════════════════════════════════════════
# FACTORY
# ═══════════════════════════════════════════════════════════════════

def create_inference_service(config: dict | None = None) -> InferenceService:
    """Factory for InferenceService."""
    return InferenceService(config)


if __name__ == "__main__":
    import asyncio

    async def test():
        config = {
            "vllm_url": "http://localhost:8000/v1",
            "buy_triggers": ["BUY", "LONG"],
            "sell_triggers": ["SELL", "SHORT"],
            "min_confidence_tokens": 2,
        }

        service = create_inference_service(config)

        async def on_decision(d):
            print(f"TOKEN DECISION: {d['signal']} conf={d['confidence']:.2f}")

        service.on_decision(on_decision)

        await service.start()

        # Test streaming
        prompt = get_trading_prompt("scalping_decision",
            candles="Recent: 50000, 50100, 50200, 50150, 50300",
            indicators="RSI=65, MACD>0, BB_pos=0.7"
        )

        async for resp in service.infer_stream(prompt, trace_id="test_001"):
            if resp.finished:
                print(f"Done: {resp.text}")
                break

        await service.stop()

    asyncio.run(test())