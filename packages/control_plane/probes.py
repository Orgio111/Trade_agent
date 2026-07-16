"""Bounded readiness probes for local runtime dependencies."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
import math
from typing import Protocol

import asyncpg
import nats

from packages.local_ai import LocalOllamaHealthClient
from workers.config import WorkerSettings

from .models import DependencyStatus, RuntimeReadiness


class ReadinessProbe(Protocol):
    async def check(self) -> RuntimeReadiness: ...


class DefaultReadinessProbe:
    """Check dependencies without retaining credentials or mutable clients."""

    def __init__(
        self,
        settings: WorkerSettings,
        *,
        timeout_seconds: float = 5.0,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._settings = settings
        self._timeout_seconds = timeout_seconds

    async def check(self) -> RuntimeReadiness:
        database, nats_status, ollama = await asyncio.gather(
            self._bounded(self._check_database),
            self._bounded(self._check_nats),
            self._bounded(self._check_ollama),
        )
        execution_enabled = database[1]
        dependencies = {
            "postgres": database[0],
            "nats": nats_status[0],
            "ollama": ollama[0],
        }
        ready = all(status.healthy for status in dependencies.values())
        return RuntimeReadiness(
            ready=ready,
            execution_enabled=ready and execution_enabled,
            mode=self._settings.mode.value,
            dependencies=dependencies,
        )

    async def _bounded(
        self,
        operation: Callable[[], Awaitable[tuple[DependencyStatus, bool]]],
    ) -> tuple[DependencyStatus, bool]:
        try:
            return await asyncio.wait_for(
                operation(), timeout=self._timeout_seconds
            )
        except TimeoutError:
            return DependencyStatus(healthy=False, detail="probe timed out"), False
        except Exception as exc:
            return (
                DependencyStatus(
                    healthy=False,
                    detail=f"probe failed: {type(exc).__name__}",
                ),
                False,
            )

    async def _check_database(self) -> tuple[DependencyStatus, bool]:
        connection = await asyncpg.connect(
            self._settings.database_url.get_secret_value()
        )
        try:
            migration_count = await connection.fetchval(
                "SELECT COUNT(*) FROM schema_migrations"
            )
            kill_switch_active = await connection.fetchval(
                """
                SELECT active
                FROM kill_switch_state
                WHERE account_id = $1
                """,
                self._settings.account_id,
            )
        finally:
            await connection.close()
        if kill_switch_active is None:
            detail = f"migrations={migration_count}; kill switch is uninitialized"
            return DependencyStatus(healthy=True, detail=detail), False
        detail = f"migrations={migration_count}; kill_switch_active={bool(kill_switch_active)}"
        return DependencyStatus(healthy=True, detail=detail), not bool(
            kill_switch_active
        )

    async def _check_nats(self) -> tuple[DependencyStatus, bool]:
        client = await nats.connect(
            servers=[self._settings.nats_url],
            connect_timeout=self._timeout_seconds,
            max_reconnect_attempts=0,
        )
        try:
            await client.flush(timeout=max(1, math.ceil(self._timeout_seconds)))
        finally:
            await client.close()
        return DependencyStatus(healthy=True, detail="JetStream transport reachable"), True

    async def _check_ollama(self) -> tuple[DependencyStatus, bool]:
        async with LocalOllamaHealthClient(
            self._settings.ollama_base_url,
            timeout_seconds=self._timeout_seconds,
        ) as client:
            health = await client.health()
        detail = health.detail
        if health.missing:
            detail += "; missing=" + ",".join(health.missing)
        return DependencyStatus(healthy=health.healthy, detail=detail), health.healthy
