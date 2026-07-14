"""Risk Service — entry point with health check HTTP server."""

import asyncio
import logging
import os

from fastapi import FastAPI
import uvicorn

from services.risk import create_risk_service, RiskLevel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("risk")

app = FastAPI(title="Risk Service")
_service = None


@app.get("/health")
async def health():
    return {"status": "ok", "service": "risk"}


@app.get("/api/v1/status")
async def status():
    return {"service": "risk", "monitoring": _service._monitoring_task is not None if _service else False}


@app.on_event("startup")
async def startup():
    global _service
    _service = create_risk_service()
    asyncio.create_task(_service.start_monitoring(interval_seconds=60))
    logger.info("[risk] Started with background monitoring")


@app.on_event("shutdown")
async def shutdown():
    if _service:
        await _service.stop_monitoring()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8087, log_level="info")
