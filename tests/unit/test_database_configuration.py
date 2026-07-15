"""Fail-closed configuration tests for the async PostgreSQL database layer."""

from __future__ import annotations

import asyncpg
import pytest

from orchestrator.database import Database
from orchestrator.rl_memory import memory_store


@pytest.mark.asyncio
async def test_connect_requires_database_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)

    database = Database()

    with pytest.raises(RuntimeError, match="DATABASE_URL is required"):
        await database.connect()


@pytest.mark.asyncio
async def test_connect_reads_database_url_at_connection_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    database = Database()
    monkeypatch.setenv("DATABASE_URL", "postgresql://runtime-injected")
    captured: dict[str, object] = {}
    pool = object()

    async def create_pool(**kwargs: object) -> object:
        captured.update(kwargs)
        return pool

    migrations_run = False

    async def run_migrations() -> None:
        nonlocal migrations_run
        migrations_run = True

    monkeypatch.setattr(asyncpg, "create_pool", create_pool)
    monkeypatch.setattr(database, "_run_migrations", run_migrations)

    connected = await database.connect()

    assert connected is database
    assert database._pool is pool
    assert captured["dsn"] == "postgresql://runtime-injected"
    assert migrations_run is True


def test_rl_postgres_store_requires_injected_dsn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(memory_store, "POSTGRES_AVAILABLE", True)
    monkeypatch.delenv("POSTGRES_DSN", raising=False)

    with pytest.raises(RuntimeError, match="POSTGRES_DSN is required"):
        memory_store.PostgresStore()
