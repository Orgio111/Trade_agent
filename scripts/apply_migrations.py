"""Apply versioned PostgreSQL migrations with checksum and advisory locking."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import os
from pathlib import Path
import re
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import asyncpg  # type: ignore[import-untyped]


ROOT = Path(__file__).resolve().parents[1]
MIGRATIONS = ROOT / "db" / "migrations"
_ROLE_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,62}$")
_MAX_SECRET_FILE_BYTES = 4096
EXPECTED_MIGRATIONS = {
    "001_init.sql": "f9394b8f8ce87cc841dd178014833161360f944730e75f8a2510ceaec5092c97",
    "002_execution_ledger.sql": "aa84d6e32918cc294c75d32de492ff7c0f699b66210e30dec9daa7c7e970008c",
    "003_runtime_durability.sql": "3c1337ce0c38420f156ab94f6d93e449f5251c7c291c2b18703c69b3aac22c57",
    "004_worker_leases.sql": "0d5d53354716b3bd0971dd72f5226d8c8f5e3e756256cf9927fe049d24249fe8",
    "005_outbox_dispatch.sql": "5de22e493e3b93f64bca5adf60e7e00714cd6f9caf3d4616c45693a4cbe234bd",
    "006_feature_checkpoints.sql": "bdf29b9c434720f207b9acb629ac7af591c264dab88cee007fe05c371ccee38b",
    "007_replay_audit.sql": "9b1eb51b54244f2024dc39a14f488277e12441bcd942fc1282d2ef9f6289fd21",
}


def quote_role(value: str) -> str:
    if not _ROLE_PATTERN.fullmatch(value):
        raise ValueError("database role must be a simple PostgreSQL identifier")
    return f'"{value}"'


async def ensure_runtime_role(connection, *, role: str, password: str) -> None:
    """Create/update one non-owner DML role after schema migration."""

    identifier = quote_role(role)
    if len(password) < 16:
        raise ValueError("DB_RUNTIME_PASSWORD must contain at least 16 characters")
    exists = await connection.fetchval("SELECT EXISTS(SELECT 1 FROM pg_roles WHERE rolname=$1)", role)
    if not exists:
        await connection.execute(f"CREATE ROLE {identifier} LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT")
    await connection.execute(
        f"ALTER ROLE {identifier} LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE "
        "NOINHERIT NOREPLICATION NOBYPASSRLS"
    )
    password_literal = await connection.fetchval("SELECT quote_literal($1)", password)
    await connection.execute(f"ALTER ROLE {identifier} PASSWORD {password_literal}")
    database = quote_role(await connection.fetchval("SELECT current_database()"))
    await connection.execute(f"REVOKE ALL PRIVILEGES ON DATABASE {database} FROM {identifier}")
    await connection.execute(f"GRANT CONNECT ON DATABASE {database} TO {identifier}")
    await connection.execute("REVOKE CREATE ON SCHEMA public FROM PUBLIC")
    await connection.execute(f"REVOKE ALL PRIVILEGES ON SCHEMA public FROM {identifier}")
    await connection.execute(f"GRANT USAGE ON SCHEMA public TO {identifier}")
    await connection.execute(f"REVOKE ALL PRIVILEGES ON ALL TABLES IN SCHEMA public FROM {identifier}")
    await connection.execute(f"REVOKE ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public FROM {identifier}")
    await connection.execute(f"GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO {identifier}")
    await connection.execute(f"GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO {identifier}")
    await connection.execute(
        f"REVOKE INSERT, UPDATE, DELETE, TRUNCATE, REFERENCES, TRIGGER "
        f"ON TABLE public.schema_migrations FROM {identifier}"
    )
    await connection.execute(f"ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO {identifier}")
    await connection.execute(f"ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT USAGE, SELECT ON SEQUENCES TO {identifier}")


def migration_files() -> tuple[Path, ...]:
    return tuple(sorted(MIGRATIONS.glob("[0-9][0-9][0-9]_*.sql")))


def checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_migration_contract(files: tuple[Path, ...] | None = None) -> None:
    selected = files or migration_files()
    names = [path.name for path in selected]
    expected_names = list(EXPECTED_MIGRATIONS)
    if names != expected_names:
        raise RuntimeError(
            f"migration set drifted: expected={expected_names}; actual={names}"
        )
    drifted = [
        path.name
        for path in selected
        if checksum(path) != EXPECTED_MIGRATIONS[path.name]
    ]
    if drifted:
        raise RuntimeError("immutable migration checksum drift: " + ",".join(drifted))


def read_secret_file(variable: str) -> str | None:
    path_value = os.getenv(variable)
    if not path_value:
        return None
    path = Path(path_value)
    if not path.is_absolute():
        raise ValueError(f"{variable} must reference an absolute path")
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{variable} must reference a regular non-symlink file")
    if path.stat().st_size > _MAX_SECRET_FILE_BYTES:
        raise ValueError(f"{variable} exceeds the secret size limit")
    value = path.read_text(encoding="utf-8").strip()
    if not value:
        raise ValueError(f"{variable} references an empty secret")
    return value


def secret_from_env_or_file(name: str) -> str | None:
    direct = os.getenv(name)
    from_file = read_secret_file(f"{name}_FILE")
    if direct and from_file:
        raise ValueError(f"{name} and {name}_FILE are mutually exclusive")
    return direct or from_file


async def apply(
    database_url: str | None,
    *,
    baseline_001: bool = False,
    connection_options: dict[str, str] | None = None,
    runtime_role: str | None = None,
    runtime_password: str | None = None,
) -> list[str]:
    connection = await asyncpg.connect(
        database_url, **(connection_options or {})
    )
    applied_now: list[str] = []
    try:
        await connection.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version TEXT PRIMARY KEY,
                checksum CHAR(64) NOT NULL,
                applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
        await connection.execute(
            "SELECT pg_advisory_lock(hashtext('quantex-schema-migrations'))"
        )
        files = migration_files()
        validate_migration_contract(files)
        if baseline_001:
            first = next((path for path in files if path.name.startswith("001_")), None)
            if first is None:
                raise RuntimeError("001 migration file is missing")
            legacy_table = await connection.fetchval("SELECT to_regclass('public.trades')")
            if legacy_table is None:
                raise RuntimeError(
                    "cannot baseline 001: expected legacy table public.trades is absent"
                )
            await connection.execute(
                """
                INSERT INTO schema_migrations(version, checksum)
                VALUES($1, $2)
                ON CONFLICT (version) DO NOTHING
                """,
                first.name,
                checksum(first),
            )

        for path in files:
            digest = checksum(path)
            recorded = await connection.fetchval(
                "SELECT checksum FROM schema_migrations WHERE version = $1",
                path.name,
            )
            if recorded is not None:
                if recorded != digest:
                    raise RuntimeError(
                        f"applied migration checksum changed: {path.name}"
                    )
                continue
            async with connection.transaction():
                await connection.execute(path.read_text(encoding="utf-8"))
                await connection.execute(
                    "INSERT INTO schema_migrations(version, checksum) VALUES($1, $2)",
                    path.name,
                    digest,
                )
            applied_now.append(path.name)
        if runtime_role or runtime_password:
            if not runtime_role or not runtime_password:
                raise ValueError("DB_RUNTIME_USER and DB_RUNTIME_PASSWORD must be set together")
            await ensure_runtime_role(connection, role=runtime_role, password=runtime_password)
        return applied_now
    finally:
        try:
            await connection.execute(
                "SELECT pg_advisory_unlock(hashtext('quantex-schema-migrations'))"
            )
        finally:
            await connection.close()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--database-url",
        default=os.getenv("DATABASE_URL"),
        help="PostgreSQL URL; defaults to DATABASE_URL (never printed)",
    )
    result.add_argument(
        "--baseline-001",
        action="store_true",
        help="Record legacy 001 as already applied after verifying public.trades",
    )
    return result


def main() -> int:
    args = parser().parse_args()
    connection_options = None
    if not args.database_url:
        required = {
            "host": os.getenv("PGHOST"),
            "port": os.getenv("PGPORT", "5432"),
            "user": os.getenv("PGUSER"),
            "password": secret_from_env_or_file("PGPASSWORD"),
            "database": os.getenv("PGDATABASE"),
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise SystemExit(
                "DATABASE_URL or complete PGHOST/PGUSER/PGPASSWORD/PGDATABASE "
                "settings are required"
            )
        connection_options = {name: str(value) for name, value in required.items()}
    applied = asyncio.run(
        apply(
            args.database_url,
            baseline_001=args.baseline_001,
            connection_options=connection_options,
            runtime_role=os.getenv("DB_RUNTIME_USER"),
            runtime_password=secret_from_env_or_file("DB_RUNTIME_PASSWORD"),
        )
    )
    print("applied=" + (",".join(applied) if applied else "none"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
