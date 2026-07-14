"""Static safety contract for the durable execution-ledger migration."""

import re
from pathlib import Path


MIGRATION = (
    Path(__file__).parents[2] / "db" / "migrations" / "002_execution_ledger.sql"
)


def test_execution_migration_contains_authoritative_tables() -> None:
    sql = MIGRATION.read_text(encoding="utf-8")
    tables = set(
        re.findall(
            r"CREATE TABLE IF NOT EXISTS\s+([a-z_]+)",
            sql,
            flags=re.IGNORECASE,
        )
    )
    assert {
        "ingest_runs",
        "data_quality_events",
        "strategy_versions",
        "signals",
        "risk_policies",
        "risk_decisions",
        "order_intents",
        "broker_orders",
        "order_events",
        "fills",
        "cash_ledger",
        "reconciliation_runs",
        "kill_switch_state",
        "promotion_gates",
    } <= tables


def test_execution_migration_is_idempotent_and_fail_closed() -> None:
    sql = MIGRATION.read_text(encoding="utf-8").lower()
    assert "drop table" not in sql
    assert "drop schema" not in sql
    assert "client_order_id" in sql and "unique" in sql
    assert "risk_decision_id" in sql
    assert "active              boolean not null default true" in sql
    order_intent = sql.split("create table if not exists order_intents", 1)[1]
    order_intent = order_intent.split("create table if not exists broker_orders", 1)[0]
    assert "'replay', 'paper_live'" in order_intent
    assert "'live'" not in order_intent

