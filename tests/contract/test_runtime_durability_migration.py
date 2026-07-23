"""Static safety contract for additive runtime-durability migration 003."""

import re
from pathlib import Path


MIGRATION = (
    Path(__file__).parents[2] / "db" / "migrations" / "003_runtime_durability.sql"
)
LEASE_MIGRATION = (
    Path(__file__).parents[2] / "db" / "migrations" / "004_worker_leases.sql"
)
OUTBOX_MIGRATION = (
    Path(__file__).parents[2] / "db" / "migrations" / "005_outbox_dispatch.sql"
)
FEATURE_MIGRATION = (
    Path(__file__).parents[2] / "db" / "migrations" / "006_feature_checkpoints.sql"
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
    assert "uq_risk_decision_signal_once" in sql
    assert "candidate_event_id" in sql
    assert "uq_signal_candidate_event" in sql


def test_portfolio_and_constraint_keys_are_versioned() -> None:
    sql = MIGRATION.read_text(encoding="utf-8").lower()

    assert "primary key (account_id, state_id)" in sql
    assert "unique (account_id, source_sequence)" in sql
    assert "primary key (venue, market_type, instrument_id, version)" in sql
    assert "expires_at is null or expires_at > effective_at" in sql


def test_worker_lease_records_transport_and_consumer_health() -> None:
    sql = LEASE_MIGRATION.read_text(encoding="utf-8").lower()

    assert "create table if not exists worker_leases" in sql
    assert "lease_expires_at" in sql
    assert "consumer_lag" in sql
    assert "last_error" in sql


def test_outbox_dispatch_supports_leases_retry_and_dead_letter() -> None:
    sql = OUTBOX_MIGRATION.read_text(encoding="utf-8").lower()

    assert "publishing" in sql
    assert "lease_expires_at" in sql
    assert "next_attempt_at" in sql
    assert "create table if not exists dead_letter_events" in sql
    assert "original_subject" in sql
    assert "stack_trace_fingerprint" in sql


def test_feature_checkpoint_is_durable_and_integrity_bound() -> None:
    sql = FEATURE_MIGRATION.read_text(encoding="utf-8").lower()
    assert "create table if not exists feature_checkpoints" in sql
    assert "checkpoint_checksum" in sql
    assert "last_market_event_id" in sql
    assert "unique (symbol, last_market_event_id)" in sql
