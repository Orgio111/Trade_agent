"""Minimal async PostgreSQL protocols shared by durable repositories."""

from __future__ import annotations

from contextlib import AbstractAsyncContextManager
import json
from typing import Any, Mapping, Protocol, Sequence


class PostgresConnection(Protocol):
    async def execute(self, query: str, *args: Any) -> str: ...

    async def fetchval(self, query: str, *args: Any) -> Any: ...

    async def fetchrow(self, query: str, *args: Any) -> Mapping[str, Any] | None: ...

    async def fetch(self, query: str, *args: Any) -> Sequence[Mapping[str, Any]]: ...

    def transaction(self) -> AbstractAsyncContextManager[Any]: ...


class PostgresPool(Protocol):
    def acquire(self) -> AbstractAsyncContextManager[PostgresConnection]: ...


def json_object(value: object, *, field_name: str) -> dict[str, Any]:
    """Normalize asyncpg JSONB output without accepting non-object payloads."""

    parsed = json.loads(value) if isinstance(value, str) else value
    if not isinstance(parsed, Mapping):
        raise ValueError(f"{field_name} must be a JSON object")
    return {str(key): item for key, item in parsed.items()}
