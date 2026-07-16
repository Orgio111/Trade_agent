"""Semantic tests for the async PostgreSQL risk-decision authority."""

from __future__ import annotations

from contextlib import asynccontextmanager

import pytest

from packages.execution import (
    DecisionAuthorizationError,
    DecisionStoreUnavailable,
    PostgresDecisionStore,
)
from tests.fakes.durable import candidate, decision, portfolio


class FakeDecisionConnection:
    def __init__(self) -> None:
        state = portfolio()
        self.snapshots = {(state.account_id, state.state_id): state.model_dump_json()}
        self.signals: dict[str, str] = {}
        self.decisions: dict[str, str] = {}
        self.bindings: dict[tuple[str, str, str], str] = {}
        self.calls: list[str] = []
        self.fail = False

    @asynccontextmanager
    async def transaction(self):
        yield self

    async def fetchrow(self, query, *args):
        if self.fail:
            raise ConnectionError("database unavailable")
        normalized = " ".join(query.split()).lower()
        self.calls.append(normalized)
        if "from portfolio_snapshots" in normalized:
            payload = self.snapshots.get((args[0], args[1]))
            return None if payload is None else {"payload": payload}
        if "insert into signals" in normalized:
            signal_id = args[0]
            if signal_id in self.signals:
                return None
            self.signals[signal_id] = args[12]
            return {"payload": args[12]}
        if "select payload from signals" in normalized:
            payload = self.signals.get(args[0])
            return None if payload is None else {"payload": payload}
        if "insert into risk_decisions" in normalized:
            decision_id = args[0]
            binding = (args[1], args[4], args[5])
            if decision_id in self.decisions or binding in self.bindings:
                return None
            self.decisions[decision_id] = args[12]
            self.bindings[binding] = decision_id
            return {"decision_payload": args[12]}
        if "where id = $1" in normalized and "from risk_decisions" in normalized:
            payload = self.decisions.get(args[0])
            return None if payload is None else {"decision_payload": payload}
        if "where signal_id = $1" in normalized:
            identity = self.bindings.get((args[0], args[1], args[2]))
            return None if identity is None else {"id": identity}
        if "count(*)" in normalized:
            return {"count": len(self.decisions)}
        raise AssertionError(f"unexpected SQL: {normalized}")


class FakePool:
    def __init__(self, connection: FakeDecisionConnection) -> None:
        self.connection = connection

    @asynccontextmanager
    async def acquire(self):
        yield self.connection


@pytest.mark.asyncio
async def test_candidate_snapshot_and_decision_are_bound_in_one_transaction() -> None:
    connection = FakeDecisionConnection()
    store = PostgresDecisionStore(FakePool(connection))
    signal = candidate()
    verdict = decision(candidate_value=signal)

    recorded = await store.record_candidate_and_decision(
        signal,
        verdict,
        portfolio_state=portfolio(),
        market_event_id="market-event-1",
    )

    assert recorded == verdict
    assert await store.require_exact_approval(verdict) == verdict
    assert await store.count() == 1
    sql = " ".join(connection.calls)
    assert sql.index("insert into signals") < sql.index("insert into risk_decisions")


@pytest.mark.asyncio
async def test_atomic_write_rejects_signal_identity_reuse_with_changed_payload() -> (
    None
):
    connection = FakeDecisionConnection()
    store = PostgresDecisionStore(FakePool(connection))
    original = candidate()
    original_decision = decision(candidate_value=original)
    await store.record_candidate_and_decision(
        original,
        original_decision,
        portfolio_state=portfolio(),
        market_event_id="market-event-1",
    )
    changed = original.model_copy(update={"side": "sell"})
    changed_decision = decision(candidate_value=changed).model_copy(
        update={"decision_id": "risk_fedcba9876543210fedcba98"}
    )

    with pytest.raises(DecisionAuthorizationError, match="different immutable"):
        await store.record_candidate_and_decision(
            changed,
            changed_decision,
            portfolio_state=portfolio(),
            market_event_id="market-event-2",
        )

    assert len(connection.decisions) == 1


@pytest.mark.asyncio
async def test_missing_authoritative_snapshot_fails_before_signal_write() -> None:
    connection = FakeDecisionConnection()
    connection.snapshots.clear()
    store = PostgresDecisionStore(FakePool(connection))
    signal = candidate()

    with pytest.raises(DecisionAuthorizationError, match="snapshot is missing"):
        await store.record_candidate_and_decision(
            signal,
            decision(candidate_value=signal),
            portfolio_state=portfolio(),
            market_event_id="market-event-1",
        )

    assert connection.signals == {}
    assert connection.decisions == {}


@pytest.mark.asyncio
async def test_exact_approval_rejects_forged_content() -> None:
    connection = FakeDecisionConnection()
    store = PostgresDecisionStore(FakePool(connection))
    signal = candidate()
    issued = decision(candidate_value=signal)
    await store.record_candidate_and_decision(
        signal,
        issued,
        portfolio_state=portfolio(),
        market_event_id="market-event-1",
    )
    forged = issued.model_copy(
        update={"approved_quantity": issued.approved_quantity * 2}
    )

    with pytest.raises(DecisionAuthorizationError, match="content differs"):
        await store.require_exact_approval(forged)


@pytest.mark.asyncio
async def test_database_failure_never_grants_authorization() -> None:
    connection = FakeDecisionConnection()
    connection.fail = True
    store = PostgresDecisionStore(FakePool(connection))

    with pytest.raises(DecisionStoreUnavailable, match="failed closed"):
        await store.get("risk_0123456789abcdef01234567")
