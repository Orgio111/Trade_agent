"""FastAPI application exposing no trading or broker mutation surface."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import JSONResponse, PlainTextResponse

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

    @app.get("/metrics", response_class=PlainTextResponse)
    async def metrics() -> str:
        result = await probe.check()
        lines = [
            "# TYPE quantex_runtime_ready gauge",
            f"quantex_runtime_ready {int(result.ready)}",
            "# TYPE quantex_execution_enabled gauge",
            f"quantex_execution_enabled {int(result.execution_enabled)}",
            "# TYPE quantex_dependency_healthy gauge",
        ]
        lines.extend(
            f'quantex_dependency_healthy{{dependency="{name}"}} {int(status.healthy)}'
            for name, status in sorted(result.dependencies.items())
        )
        return "\n".join(lines) + "\n"

    return app
