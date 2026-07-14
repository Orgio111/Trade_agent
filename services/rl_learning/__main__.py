"""RL Learning Service — entry point with health check HTTP server."""

import asyncio
import logging
import os

from fastapi import FastAPI
import uvicorn

from services.rl_learning import create_rl_learning_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("rl_learning")

app = FastAPI(title="RL Learning Service")
_service = None
_strategies = os.getenv("RL_STRATEGIES", "trend,scalping,mean_reversion,breakout").split(",")


@app.get("/health")
async def health():
    return {"status": "ok", "service": "rl-learning"}


@app.get("/api/v1/status")
async def status():
    if _service:
        return _service.get_status()
    return {"service": "rl-learning", "strategies": _strategies}


@app.on_event("startup")
async def startup():
    global _service
    _service = create_rl_learning_service(_strategies)
    asyncio.create_task(_service.start_training(interval_seconds=300))
    logger.info(f"[rl-learning] Started for strategies: {_strategies}")


@app.on_event("shutdown")
async def shutdown():
    if _service:
        await _service.stop_training()
        _service.ppo.save_model()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8088, log_level="info")
