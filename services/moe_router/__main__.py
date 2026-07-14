"""MoE Router Service — entry point with health check HTTP server."""

import asyncio
import logging
import os

from fastapi import FastAPI
import uvicorn

from services.moe_router import create_moe_service, MoEDecision

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("moe_router")

app = FastAPI(title="MoE Router Service")
_service = None


@app.get("/health")
async def health():
    return {"status": "ok", "service": "moe-router"}


@app.get("/api/v1/status")
async def status():
    return {
        "service": "moe-router",
        "agents": len(_service.router.agents) if _service else 0,
    }


@app.get("/api/v1/weights")
async def weights():
    if _service:
        w = _service.router.get_adaptive_weights()
        return {k.value: round(v, 4) for k, v in w.items()}
    return {}


@app.on_event("startup")
async def startup():
    global _service
    config = {
        "features": {
            "market_data": {
                "exchanges": ["binance"],
                "binance": {"testnet": True},
            },
        },
        "orderbook": {
            "market_data": {
                "exchanges": ["binance"],
                "binance": {"testnet": True},
            },
        },
        "moe": {},
    }
    _service = create_moe_service(config)
    symbols = os.getenv("MARKET_SYMBOLS", "BTCUSDT,ETHUSDT").split(",")

    async def on_decision(d: MoEDecision):
        logger.info(f"[moe-router] {d.symbol} → {d.final_signal} (conf={d.confidence:.2f})")

    _service.on_decision(on_decision)
    asyncio.create_task(_service.start(symbols))
    logger.info(f"[moe-router] Started for {symbols}")


@app.on_event("shutdown")
async def shutdown():
    if _service:
        await _service.stop()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8084, log_level="info")
