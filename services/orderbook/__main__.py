"""Order Book Service — entry point with health check HTTP server."""

import asyncio
import logging
import os

from fastapi import FastAPI
import uvicorn

from services.orderbook import create_order_book_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("orderbook")

app = FastAPI(title="Order Book Service")
_service = None


@app.get("/health")
async def health():
    return {"status": "ok", "service": "orderbook"}


@app.get("/api/v1/status")
async def status():
    return {"service": "orderbook", "books": list(_service.books.keys()) if _service else []}


@app.on_event("startup")
async def startup():
    global _service
    config = {
        "market_data": {
            "exchanges": ["binance"],
            "binance": {"testnet": True},
        },
    }
    _service = create_order_book_service(config)
    symbols = os.getenv("MARKET_SYMBOLS", "BTCUSDT,ETHUSDT").split(",")
    asyncio.create_task(_service.start(symbols))
    logger.info(f"[orderbook] Started for {symbols}")


@app.on_event("shutdown")
async def shutdown():
    if _service:
        await _service.stop()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8083, log_level="info")
