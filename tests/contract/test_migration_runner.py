"""Contracts for the explicit migration runner used on existing volumes."""

from pathlib import Path

from scripts.apply_migrations import checksum, migration_files


def test_migrations_are_versioned_sorted_and_checksum_stable() -> None:
    files = migration_files()
    assert [path.name for path in files] == [
        "001_init.sql",
        "002_execution_ledger.sql",
    ]
    assert all(len(checksum(path)) == 64 for path in files)


def test_execution_migration_is_owned_by_runner_transaction() -> None:
    path = Path(__file__).parents[2] / "db" / "migrations" / "002_execution_ledger.sql"
    sql = path.read_text(encoding="utf-8").lower()
    assert "begin;" not in sql
    assert "commit;" not in sql
