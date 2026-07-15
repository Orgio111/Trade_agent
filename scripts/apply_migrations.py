"""Apply versioned PostgreSQL migrations with checksum and advisory locking."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import os
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import asyncpg


ROOT = Path(__file__).resolve().parents[1]
MIGRATIONS = ROOT / "db" / "migrations"


def migration_files() -> tuple[Path, ...]:
    return tuple(sorted(MIGRATIONS.glob("[0-9][0-9][0-9]_*.sql")))


def checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


async def apply(
    database_url: str | None,
    *,
    baseline_001: bool = False,
    connection_options: dict[str, str] | None = None,
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
            "password": os.getenv("PGPASSWORD"),
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
        )
    )
    print("applied=" + (",".join(applied) if applied else "none"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
