from __future__ import annotations

from typing import Any

import pytest

from scripts.apply_migrations import ensure_runtime_role, quote_role


class RecordingConnection:
    def __init__(self, *, exists: bool = True) -> None:
        self.exists = exists
        self.commands: list[str] = []

    async def fetchval(self, query: str, *args: Any) -> Any:
        if "pg_roles" in query:
            return self.exists
        if "quote_literal" in query:
            return "'runtime-password-at-least-16'"
        if "current_database" in query:
            return "quantex"
        raise AssertionError(f"unexpected query: {query}")

    async def execute(self, query: str, *args: Any) -> None:
        self.commands.append(" ".join(query.split()))


@pytest.mark.asyncio
async def test_runtime_role_is_demoted_and_cannot_mutate_migration_history() -> None:
    connection = RecordingConnection()

    await ensure_runtime_role(
        connection,
        role="quantex_runtime",
        password="runtime-password-at-least-16",
    )

    commands = "\n".join(connection.commands)
    assert (
        'ALTER ROLE "quantex_runtime" LOGIN NOSUPERUSER NOCREATEDB '
        "NOCREATEROLE NOINHERIT NOREPLICATION NOBYPASSRLS"
    ) in commands
    assert 'REVOKE ALL PRIVILEGES ON DATABASE "quantex" FROM "quantex_runtime"' in commands
    assert "REVOKE CREATE ON SCHEMA public FROM PUBLIC" in commands
    assert (
        "REVOKE INSERT, UPDATE, DELETE, TRUNCATE, REFERENCES, TRIGGER "
        'ON TABLE public.schema_migrations FROM "quantex_runtime"'
    ) in commands
    assert 'GRANT CONNECT ON DATABASE "quantex" TO "quantex_runtime"' in commands
    assert (
        "GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public "
        'TO "quantex_runtime"'
    ) in commands


@pytest.mark.asyncio
async def test_runtime_role_is_created_without_elevation_when_missing() -> None:
    connection = RecordingConnection(exists=False)

    await ensure_runtime_role(
        connection,
        role="quantex_runtime",
        password="runtime-password-at-least-16",
    )

    assert connection.commands[0] == (
        'CREATE ROLE "quantex_runtime" LOGIN NOSUPERUSER NOCREATEDB '
        "NOCREATEROLE NOINHERIT"
    )


@pytest.mark.parametrize(
    "role",
    ["runtime;DROP ROLE owner", "runtime-role", '"runtime"', "", "a" * 64],
)
def test_runtime_role_identifier_rejects_sql_metacharacters(role: str) -> None:
    with pytest.raises(ValueError, match="simple PostgreSQL identifier"):
        quote_role(role)
