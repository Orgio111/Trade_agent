"""Start the loopback-exposed canonical control plane."""

from __future__ import annotations

import os

import uvicorn

from workers.config import WorkerSettings

from .app import create_app
from .probes import DefaultReadinessProbe


def build_app():
    settings = WorkerSettings.from_env(
        service_name="quantex-control-plane",
        durable_name="control-plane-readonly-v1",
    )
    return create_app(DefaultReadinessProbe(settings))


app = build_app()


def main() -> None:
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("CONTROL_PLANE_PORT", "8001")),
        log_level=os.getenv("LOG_LEVEL", "info").casefold(),
    )


if __name__ == "__main__":
    main()
