"""Static safety contract for additive runtime-durability migration 003."""

import re
from pathlib import Path


MIGRATION = (
    Path(__file__).parents[2] / "db" / "migrations" / "003_runtime_durability.sql"
)


def test_runtime_migration_is_additive_and_contains_required_state() -> None:
    sql = MIGRATION.read_text(encoding="utf-8").lower()
    tables = set(
        re.findall(
            r"create table if not exists\s+([a-z_]+)",
            sql,
            flags=re.IGNORECASE,
        )
    )
    assert {
        "event_inbox",
        "event_outbox",
        "consumer_offsets",
        "portfolio_snapshots",
        "instrument_constraints",
        "position_projections",
        "position_lots",
    } <= tables
    assert "drop table" not in sql
    assert "truncate" not in sql
    assert "delete from" not in sql
    assert "insert into instrument_constraints" not in sql
    assert "insert into portfolio_snapshots" not in sql


def test_runtime_migration_has_idempotency_and_immutability_guards() -> None:
    sql = MIGRATION.read_text(encoding="utf-8").lower()

    assert "primary key (stream_name, durable_name, event_id)" in sql
    assert "unique (stream_name, durable_name, stream_sequence)" in sql
    assert "payload_checksum" in sql
    assert "publish_attempts between 0 and 20" in sql
    assert "trg_risk_decisions_immutable" in sql
    assert "trg_signals_immutable" in sql
    assert "trg_portfolio_snapshots_immutable" in sql
    assert "trg_instrument_constraints_immutable" in sql


def test_portfolio_and_constraint_keys_are_versioned() -> None:
    sql = MIGRATION.read_text(encoding="utf-8").lower()

    assert "primary key (account_id, state_id)" in sql
    assert "unique (account_id, source_sequence)" in sql
    assert "primary key (venue, market_type, instrument_id, version)" in sql
    assert "expires_at is null or expires_at > effective_at" in sql
