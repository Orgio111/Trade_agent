"""
QUANTEX Local GPU Inference Server — vLLM + FastAPI for RTX 4050.

Ultra-low latency local inference server that bypasses cloud APIs.
Runs quantized models (Qwen2.5-7B Q4, Mistral-7B Q4) on local GPU.

Architecture:
  ┌─────────────────────────────────────────────────────────────┐
  │              Local GPU Inference Server                     │
  │                                                              │
  │  Client (LangGraph / AutoGen / Scalping)                    │
  │        ↓                                                     │
  │  FastAPI Gateway (port 8000)                                │
  │        ↓                                                     │
  │  vLLM Engine (GPU)                                          │
  │        ↓                                                     │
  │  Quantized Model (Qwen/Mistral 7B)                         │
  │        ↓                                                     │
  │  Streaming Response (SSE / WebSocket)                       │
  └─────────────────────────────────────────────────────────────┘

Usage:
    # Start server (with vLLM installed):
    python -m orchestrator.local_inference_server

    # Or with Docker:
    docker build -t quantex-inference -f Dockerfile.inference .
    docker run --gpus all -p 8000:8000 quantex-inference

    # Client call:
    from orchestrator.local_inference_server import LocalInferenceClient
    client = LocalInferenceClient("http://localhost:8000")
    result = await client.generate("Analyze BTC trend: bullish or bearish?")
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional, AsyncGenerator

import httpx

logger = logging.getLogger("quantex.inference.local")

# ── Configuration ───────────────────────────────────────────

@dataclass
class LocalInferenceConfig:
    """Configuration for local GPU inference server."""
    model: str = "Qwen/Qwen2.5-7B-Instruct"
    host: str = "0.0.0.0"
    port: int = 8000
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.85
    tensor_parallel_size: int = 1
    dtype: str = "half"
    quantization: str = "awq"  # "awq", "gptq", "None"
    max_tokens: int = 256
    temperature: float = 0.2
    batch_size: int = 1  # Trading = batch_size 1 for lowest latency

    # Model shortcuts for different use cases
    MODEL_SHORTCUTS: dict[str, str] = field(default_factory=lambda: {
        "reasoning": "Qwen/Qwen2.5-7B-Instruct",
        "analysis": "Qwen/Qwen2.5-7B-Instruct",
        "fast": "Qwen/Qwen2.5-1.5B-Instruct",
        "coding": "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
        "classification": "Qwen/Qwen2.5-1.5B-Instruct",
        "vision": "moondream/moondream2-1.6b",
    })


# ── FastAPI Server ──────────────────────────────────────────

def create_app(config: Optional[LocalInferenceConfig] = None):
    """
    Create FastAPI app with vLLM integration.

    This function is called when starting the server.
    Requires: fastapi, uvicorn, vllm (optional — fallback to httpx if not installed)
    """
    try:
        from fastapi import FastAPI
        from fastapi.responses import StreamingResponse
        from pydantic import BaseModel
    except ImportError:
        raise ImportError(
            "FastAPI not installed. Run: pip install fastapi uvicorn"
        )

    app = FastAPI(title="QUANTEX Local Inference Server")
    cfg = config or LocalInferenceConfig()

    # Try to load vLLM (optional — server can run without it for testing)
    llm = None
    try:
        from vllm import LLM, SamplingParams
        llm = LLM(
            model=cfg.model,
            dtype=cfg.dtype,
            max_model_len=cfg.max_model_len,
            gpu_memory_utilization=cfg.gpu_memory_utilization,
            tensor_parallel_size=cfg.tensor_parallel_size,
            batch_size=cfg.batch_size,
        )
        logger.info(f"vLLM loaded: {cfg.model}")
    except ImportError:
        logger.warning("vLLM not installed — running in mock mode")
    except Exception as e:
        logger.error(f"vLLM failed to load: {e}")

    class GenerateRequest(BaseModel):
        prompt: str
        max_tokens: int = cfg.max_tokens
        temperature: float = cfg.temperature
        model: Optional[str] = None
        stream: bool = False

    class GenerateResponse(BaseModel):
        response: str
        model: str
        tokens_generated: int
        latency_ms: float

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "model": cfg.model,
            "gpu_available": llm is not None,
            "server": "quantex-local-inference",
        }

    @app.post("/generate", response_model=GenerateResponse)
    async def generate(req: GenerateRequest):
        t0 = time.perf_counter()

        if llm is not None:
            # Real vLLM inference
            from vllm import SamplingParams as SP
            sampling = SP(
                temperature=req.temperature,
                max_tokens=req.max_tokens,
            )
            outputs = llm.generate([req.prompt], sampling)
            text = outputs[0].outputs[0].text
            tokens = len(outputs[0].outputs[0].token_ids)
        else:
            # Mock mode for testing
            text = '{"action": "HOLD", "confidence": 0, "size_pct": 0, "reason": "MOCK_MODE"}'
            tokens = 0

        latency_ms = (time.perf_counter() - t0) * 1000

        return GenerateResponse(
            response=text,
            model=cfg.model,
            tokens_generated=tokens,
            latency_ms=round(latency_ms, 2),
        )

    @app.post("/generate/stream")
    async def generate_stream(req: GenerateRequest):
        """Streaming generation endpoint (SSE)."""
        if llm is None:
            return {"error": "vLLM not available"}

        from vllm import SamplingParams as SP
        sampling = SP(
            temperature=req.temperature,
            max_tokens=req.max_tokens,
        )

        outputs = llm.generate([req.prompt], sampling, stream=True)

        async def event_stream():
            for out in outputs:
                token = out.outputs[0].text
                yield json.dumps({"token": token}) + "\n"
            yield json.dumps({"done": True}) + "\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
        )

    @app.get("/models")
    async def list_models():
        """List available model shortcuts."""
        return {
            "current": cfg.model,
            "shortcuts": cfg.MODEL_SHORTCUTS,
        }

    @app.post("/generate/fast")
    async def generate_fast(req: GenerateRequest):
        """
        Ultra-fast endpoint for scalping decisions.
        Uses 1.5B model if available, otherwise fastest 7B config.
        """
        # Override to fastest model
        fast_model = cfg.MODEL_SHORTCUTS.get("fast", cfg.model)
        req.model = fast_model
        req.max_tokens = min(req.max_tokens, 128)  # Scalping = short output
        req.temperature = 0.1  # Deterministic

        return await generate(req)

    return app


# ── Client ──────────────────────────────────────────────────

class LocalInferenceClient:
    """
    Client for local GPU inference server.

    Usage:
        client = LocalInferenceClient("http://localhost:8000")
        result = await client.generate("Analyze BTC trend")
        async for token in client.generate_stream("..."):
            print(token)
    """

    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout),
            )
        return self._client

    async def health(self) -> dict:
        """Check server health."""
        client = await self._get_client()
        resp = await client.get("/health")
        return resp.json()

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.2,
        model: Optional[str] = None,
    ) -> dict:
        """
        Generate response from local GPU.

        Returns:
            {"response": str, "model": str, "tokens_generated": int, "latency_ms": float}
        """
        client = await self._get_client()
        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if model:
            payload["model"] = model

        resp = await client.post("/generate", json=payload)
        return resp.json()

    async def generate_fast(self, prompt: str) -> dict:
        """
        Ultra-fast generation for scalping decisions.
        Uses 1.5B model, 128 max tokens, temperature 0.1.
        """
        client = await self._get_client()
        resp = await client.post("/generate/fast", json={
            "prompt": prompt,
            "max_tokens": 128,
            "temperature": 0.1,
        })
        return resp.json()

    async def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.2,
    ) -> AsyncGenerator[str, None]:
        """Stream tokens from local GPU."""
        client = await self._get_client()
        async with client.stream(
            "POST",
            "/generate/stream",
            json={
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
        ) as resp:
            async for line in resp.aiter_lines():
                if line:
                    data = json.loads(line)
                    if "token" in data:
                        yield data["token"]
                    if data.get("done"):
                        break

    async def close(self):
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()


# ── Server Entry Point ──────────────────────────────────────

def main():
    """Start the local inference server."""
    try:
        import uvicorn
    except ImportError:
        raise ImportError("uvicorn not installed. Run: pip install uvicorn")

    config = LocalInferenceConfig()
    app = create_app(config)

    logger.info(f"Starting QUANTEX Local Inference Server on {config.host}:{config.port}")
    logger.info(f"Model: {config.model}")
    logger.info(f"GPU memory: {config.gpu_memory_utilization * 100:.0f}%")

    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
