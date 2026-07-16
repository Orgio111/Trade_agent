"""FastAPI application exposing no trading or broker mutation surface."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from .models import Liveness, RuntimeReadiness
from .probes import ReadinessProbe


def create_app(probe: ReadinessProbe) -> FastAPI:
    app = FastAPI(
        title="QUANTEX Local Control Plane",
        version="1.0.0",
        docs_url="/docs",
        redoc_url=None,
    )

    @app.get("/health", response_model=Liveness)
    async def health() -> Liveness:
        return Liveness()

    @app.get("/ready", response_model=RuntimeReadiness)
    async def ready() -> RuntimeReadiness | JSONResponse:
        result = await probe.check()
        if result.ready:
            return result
        return JSONResponse(status_code=503, content=result.model_dump(mode="json"))

    @app.get("/api/v1/runtime", response_model=RuntimeReadiness)
    async def runtime() -> RuntimeReadiness:
        return await probe.check()

    return app
