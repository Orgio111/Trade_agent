"""Fail-closed contracts for PostgreSQL risk-input reads."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import timedelta
from decimal import Decimal

import pytest

from packages.risk import (
    InstrumentConstraints,
    PostgresRiskInputRepository,
    RiskInputAmbiguous,
    RiskInputMissing,
    RiskInputStale,
    RiskInputUnavailable,
    RiskPolicy,
)
from tests.fakes.durable import NOW, portfolio


class FakeConnection:
    def __init__(self) -> None:
        self.rows: dict[str, list[dict]] = {}
        self.fail = False

    async def fetch(self, query, *args):
        if self.fail:
            raise ConnectionError("offline")
        sql = " ".join(query.split()).lower()
        if "from risk_policies" in sql:
            return self.rows.get("policies", [])
        if "from portfolio_snapshots" in sql:
            return self.rows.get("portfolio", [])
        if "from instrument_constraints" in sql:
            return self.rows.get("constraints", [])
        raise AssertionError(f"unexpected SQL: {sql}")


class FakePool:
    def __init__(self, connection: FakeConnection) -> None:
        self.connection = connection

    @asynccontextmanager
    async def acquire(self):
        yield self.connection


@pytest.mark.asyncio
async def test_active_policy_and_fresh_portfolio_are_exactly_loaded() -> None:
    connection = FakeConnection()
    policy = RiskPolicy(version="policy-v1")
    state = portfolio()
    connection.rows["policies"] = [
        {"version": policy.version, "policy": policy.model_dump_json()}
    ]
    connection.rows["portfolio"] = [
        {
            "state_id": state.state_id,
            "source_sequence": 11,
            "reconciled_at": state.reconciled_at,
            "payload": state.model_dump_json(),
        }
    ]
    repository = PostgresRiskInputRepository(FakePool(connection))

    assert await repository.get_active_policy() == policy
    loaded = await repository.get_portfolio_state(
        state.account_id,
        as_of=NOW + timedelta(seconds=5),
        max_age_seconds=Decimal("5"),
    )
    assert loaded == state


@pytest.mark.asyncio
async def test_missing_or_multiple_active_policy_fails_closed() -> None:
    connection = FakeConnection()
    repository = PostgresRiskInputRepository(FakePool(connection))

    with pytest.raises(RiskInputMissing, match="no active"):
        await repository.get_active_policy()

    policy = RiskPolicy(version="policy-v1")
    connection.rows["policies"] = [
        {"version": "policy-v1", "policy": policy.model_dump_json()},
        {"version": "policy-v2", "policy": policy.model_dump_json()},
    ]
    with pytest.raises(RiskInputAmbiguous, match="multiple active"):
        await repository.get_active_policy()


@pytest.mark.asyncio
async def test_stale_portfolio_is_never_returned_as_current() -> None:
    connection = FakeConnection()
    state = portfolio()
    connection.rows["portfolio"] = [
        {
            "state_id": state.state_id,
            "source_sequence": 11,
            "reconciled_at": state.reconciled_at,
            "payload": state.model_dump_json(),
        }
    ]

    with pytest.raises(RiskInputStale, match="exceeds"):
        await PostgresRiskInputRepository(FakePool(connection)).get_portfolio_state(
            state.account_id,
            as_of=NOW + timedelta(seconds=6),
            max_age_seconds=Decimal("5"),
        )


@pytest.mark.asyncio
async def test_constraint_version_must_be_effective_and_identity_bound() -> None:
    connection = FakeConnection()
    constraints = InstrumentConstraints(
        venue="binance",
        market_type="spot",
        instrument="BTCUSDT",
        tick_size=Decimal("0.01"),
        step_size=Decimal("0.001"),
        min_quantity=Decimal("0.001"),
        min_notional=Decimal("10"),
    )
    connection.rows["constraints"] = [
        {
            "version": "binance-btc-v1",
            "effective_at": NOW - timedelta(days=1),
            "expires_at": NOW + timedelta(days=1),
            "payload": constraints.model_dump_json(),
        }
    ]
    repository = PostgresRiskInputRepository(FakePool(connection))

    loaded = await repository.get_instrument_constraints(
        venue="BINANCE",
        market_type="SPOT",
        instrument="btcusdt",
        as_of=NOW,
    )
    assert loaded.version == "binance-btc-v1"
    assert loaded.constraints == constraints

    with pytest.raises(RiskInputStale, match="not effective"):
        await repository.get_instrument_constraints(
            venue="binance",
            market_type="spot",
            instrument="BTCUSDT",
            version="binance-btc-v1",
            as_of=NOW + timedelta(days=2),
        )


@pytest.mark.asyncio
async def test_database_error_is_not_converted_to_missing_defaults() -> None:
    connection = FakeConnection()
    connection.fail = True

    with pytest.raises(RiskInputUnavailable, match="failed closed"):
        await PostgresRiskInputRepository(FakePool(connection)).get_active_policy()
