"""Contracts for the explicit migration runner used on existing volumes."""

from pathlib import Path

import pytest

from scripts.apply_migrations import (
    EXPECTED_MIGRATIONS,
    checksum,
    migration_files,
    validate_migration_contract,
)


def test_migrations_are_versioned_sorted_and_checksum_stable() -> None:
    files = migration_files()
    assert [path.name for path in files] == list(EXPECTED_MIGRATIONS)
    assert {path.name: checksum(path) for path in files} == EXPECTED_MIGRATIONS
    validate_migration_contract(files)


def test_execution_migration_is_owned_by_runner_transaction() -> None:
    path = Path(__file__).parents[2] / "db" / "migrations" / "002_execution_ledger.sql"
    sql = path.read_text(encoding="utf-8").lower()
    assert "begin;" not in sql
    assert "commit;" not in sql


def test_migration_contract_rejects_tampered_bytes(tmp_path: Path) -> None:
    copies: list[Path] = []
    for source in migration_files():
        target = tmp_path / source.name
        target.write_bytes(source.read_bytes())
        copies.append(target)
    copies[-1].write_text("-- tampered\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="checksum drift"):
        validate_migration_contract(tuple(copies))
