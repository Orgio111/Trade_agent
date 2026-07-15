"""Command-line entry point for the loopback Hermes FastAPI service."""

from __future__ import annotations

import os

import uvicorn


def _port_from_environment() -> int:
    raw = os.environ.get("HERMES_API_PORT", "8011")
    try:
        port = int(raw)
    except ValueError as exc:
        raise SystemExit("HERMES_API_PORT must be an integer") from exc
    if not 1 <= port <= 65_535:
        raise SystemExit("HERMES_API_PORT must be between 1 and 65535")
    return port


def main() -> None:
    """Start a factory-created application that cannot bind beyond loopback."""

    uvicorn.run(
        "packages.hermes.bootstrap:create_local_app_from_environment",
        factory=True,
        host="127.0.0.1",
        port=_port_from_environment(),
        access_log=True,
    )


if __name__ == "__main__":
    main()
