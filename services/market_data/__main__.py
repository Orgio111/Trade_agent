"""Market Data Service — entry point with health check HTTP server."""

import asyncio
import logging
import os
import signal

from fastapi import FastAPI
import uvicorn

from services.market_data import create_market_data_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("market_data")

app = FastAPI(title="Market Data Service")
_service = None
_symbols = os.getenv("MARKET_SYMBOLS", "BTCUSDT,ETHUSDT").split(",")
_timeframes = os.getenv("MARKET_TIMEFRAMES", "1m").split(",")


@app.get("/health")
async def health():
    return {"status": "ok", "service": "market-data", "symbols": _symbols}


@app.get("/api/v1/status")
async def status():
    return {
        "service": "market-data",
        "exchanges": _service.adapters.keys() if _service else [],
        "running": _service._running if _service else False,
    }


@app.on_event("startup")
async def startup():
    global _service
    config = {
        "exchanges": ["binance"],
        "binance": {"testnet": True},
    }
    _service = create_market_data_service(config)
    asyncio.create_task(_service.start(_symbols, _timeframes))
    logger.info(f"[market-data] Started for {_symbols} {_timeframes}")


@app.on_event("shutdown")
async def shutdown():
    if _service:
        await _service.stop()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8081, log_level="info")
