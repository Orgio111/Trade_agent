"""Execution Service — entry point with health check HTTP server."""

import asyncio
import logging
import os

from fastapi import FastAPI
import uvicorn

from services.execution import (
    ExecutionOrchestrator,
    Order, OrderSide, OrderType, ExecutionMode,
    create_execution_orchestrator,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("execution")

app = FastAPI(title="Execution Service")
_orchestrator = None


@app.get("/health")
async def health():
    return {"status": "ok", "service": "execution"}


@app.get("/api/v1/status")
async def status():
    mode = os.getenv("EXECUTION_MODE", "paper")
    return {"service": "execution", "mode": mode}


@app.get("/api/v1/account")
async def get_account():
    if _orchestrator:
        account = await _orchestrator.get_account()
        return {
            "total_balance": account.total_balance,
            "available_balance": account.available_balance,
            "unrealized_pnl": account.unrealized_pnl,
        }
    return {}


@app.get("/api/v1/positions")
async def get_positions():
    if _orchestrator:
        positions = await _orchestrator.get_positions()
        return [p.to_dict() for p in positions]
    return []


@app.on_event("startup")
async def startup():
    global _orchestrator
    mode = os.getenv("EXECUTION_MODE", "paper")
    _orchestrator = create_execution_orchestrator({
        "mode": mode,
        "binance": {
            "api_key": os.getenv("BINANCE_API_KEY", ""),
            "api_secret": os.getenv("BINANCE_API_SECRET", ""),
            "testnet": os.getenv("BINANCE_TESTNET", "true").lower() == "true",
        },
    })
    await _orchestrator.connect()
    logger.info(f"[execution] Started in {mode} mode")


@app.on_event("shutdown")
async def shutdown():
    if _orchestrator:
        await _orchestrator.disconnect()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8086, log_level="info")
