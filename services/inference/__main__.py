"""Inference Service — entry point with health check HTTP server."""

import asyncio
import logging
import os

from fastapi import FastAPI
import uvicorn

from services.inference import create_inference_service, get_trading_prompt

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("inference")

app = FastAPI(title="Inference Service")
_service = None


@app.get("/health")
async def health():
    return {"status": "ok", "service": "inference"}


@app.get("/api/v1/status")
async def status():
    return {"service": "inference", "vllm_url": os.getenv("VLLM_URL", "http://vllm:8000/v1")}


@app.on_event("startup")
async def startup():
    global _service
    config = {
        "vllm_url": os.getenv("VLLM_URL", "http://vllm:8000/v1"),
        "timeout": 30.0,
    }
    _service = create_inference_service(config)
    try:
        await _service.start()
        logger.info("[inference] Connected to vLLM")
    except Exception as e:
        logger.warning(f"[inference] vLLM not available yet: {e}")


@app.on_event("shutdown")
async def shutdown():
    if _service:
        await _service.stop()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8085, log_level="info")
