"""Feature Engine Service — entry point with health check HTTP server."""

import asyncio
import logging
import os

from fastapi import FastAPI
import uvicorn

from services.feature_engine import create_feature_engine_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("feature_engine")

app = FastAPI(title="Feature Engine Service")
_service = None


@app.get("/health")
async def health():
    return {"status": "ok", "service": "feature-engine"}


@app.get("/api/v1/status")
async def status():
    return {"service": "feature-engine", "symbols": list(_service.calculator.state.keys()) if _service else []}


@app.on_event("startup")
async def startup():
    global _service
    config = {
        "market_data": {
            "exchanges": ["binance"],
            "binance": {"testnet": True},
        },
    }
    _service = create_feature_engine_service(config)
    symbols = os.getenv("MARKET_SYMBOLS", "BTCUSDT,ETHUSDT").split(",")
    asyncio.create_task(_service.start(symbols))
    logger.info(f"[feature-engine] Started for {symbols}")


@app.on_event("shutdown")
async def shutdown():
    if _service:
        await _service.stop()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8082, log_level="info")
