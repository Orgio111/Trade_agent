"""Executable entrypoint for the canonical local decision worker."""

from __future__ import annotations

import asyncio

from .runtime import run_decision_worker


def main() -> None:
    asyncio.run(run_decision_worker())


if __name__ == "__main__":
    main()
