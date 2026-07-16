"""Restart and authorization tests for durable paper execution orchestration."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from packages.execution import InMemoryDecisionStore, InMemoryExecutionLedger
from packages.local_ai import LocalModelRole
from tests.unit.test_execution_core import candidate, decision, market
from workers.contracts import CandidateForRiskEvent, RiskDecisionEvent
from workers.execution.async_service import AsyncPaperExecutionService


NOW = datetime(2026, 7, 16, 7, 0, tzinfo=UTC)


class AsyncDecisionAdapter:
    def __init__(self, store: InMemoryDecisionStore) -> None:
        self.store = store

    async def record(self, value):
        return self.store.record(value)

    async def require_exact_approval(self, value):
        return self.store.require_exact_approval(value)

    async def get(self, decision_id: str):
        return self.store.get(decision_id)


class AsyncLedgerAdapter:
    def __init__(self, ledger: InMemoryExecutionLedger) -> None:
        self.ledger = ledger

    def __getattr__(self, name):
        target = getattr(self.ledger, name)

        async def call(*args, **kwargs):
            return target(*args, **kwargs)

        return call


def approved_event() -> RiskDecisionEvent:
    candidate_value = candidate()
    candidate_event = CandidateForRiskEvent(
        trace_id=candidate_value.trace_id,
        market_event_id="b7a7ee48-90f6-40cc-890f-4d6bf8e2c6b0",
        model_role=LocalModelRole.FAST,
        model="phi3:3.8b",
        model_digest="a" * 64,
        candidate=candidate_value,
        market=market(at=NOW),
        created_at=NOW,
    )
    decision_value = decision(candidate_value=candidate_value).model_copy(
        update={"evaluated_at": NOW}
    )
    return RiskDecisionEvent.create(
        candidate_event,
        decision_value,
        created_at=NOW,
    )


@pytest.mark.asyncio
async def test_async_paper_service_requires_issued_exact_approval() -> None:
    store = InMemoryDecisionStore()
    ledger = InMemoryExecutionLedger()
    service = AsyncPaperExecutionService(
        AsyncDecisionAdapter(store),
        AsyncLedgerAdapter(ledger),
        account_id="paper-main",
    )

    with pytest.raises(Exception, match="not issued"):
        await service.submit(approved_event(), execution_at=NOW)

    assert ledger.list_intents() == ()


@pytest.mark.asyncio
async def test_async_paper_service_is_idempotent_across_service_restart() -> None:
    event = approved_event()
    store = InMemoryDecisionStore()
    store.record(event.decision)
    ledger = InMemoryExecutionLedger()
    async_store = AsyncDecisionAdapter(store)
    async_ledger = AsyncLedgerAdapter(ledger)

    first_service = AsyncPaperExecutionService(
        async_store, async_ledger, account_id="paper-main"
    )
    first = await first_service.submit(event, execution_at=NOW)
    restarted = AsyncPaperExecutionService(
        async_store, async_ledger, account_id="paper-main"
    )
    duplicate = await restarted.submit(event, execution_at=NOW)

    assert first.duplicate is False
    assert duplicate.duplicate is True
    assert duplicate.intent.intent_id == first.intent.intent_id
    assert len(ledger.list_orders()) == 1
    assert len(ledger.fills) == 1
