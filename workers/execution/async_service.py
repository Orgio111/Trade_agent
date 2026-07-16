"""Durable, paper-only async execution orchestration."""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal

from packages.brokers import PaperBroker
from packages.execution import (
    AmbiguousSubmissionError,
    DuplicateLedgerIdentity,
    ExecutionResult,
    OrderIntent,
    OrderStatus,
    OrderType,
)
from packages.execution.ports import AsyncDecisionStore, AsyncExecutionLedger
from workers.contracts import RiskDecisionEvent

from .main import validate_execution_context


class AsyncPaperExecutionService:
    """Persist-before-paper-fill service with restart-safe deterministic IDs.

    No live broker protocol is accepted by this class. A temporary PaperBroker
    is used only as the existing deterministic fill calculator; PostgreSQL is
    the durable state authority.
    """

    def __init__(
        self,
        decision_store: AsyncDecisionStore,
        ledger: AsyncExecutionLedger,
        *,
        account_id: str,
        max_price_deviation_bps: Decimal = Decimal("20"),
        max_decision_age: timedelta = timedelta(seconds=2),
        max_market_age: timedelta = timedelta(seconds=1),
    ) -> None:
        if not account_id.strip():
            raise ValueError("account_id cannot be blank")
        if max_price_deviation_bps < 0:
            raise ValueError("max_price_deviation_bps must be non-negative")
        if max_decision_age < timedelta(0) or max_market_age < timedelta(0):
            raise ValueError("execution freshness limits must be non-negative")
        self._decision_store = decision_store
        self._ledger = ledger
        self._account_id = account_id.strip()
        self._max_price_deviation_bps = max_price_deviation_bps
        self._max_decision_age = max_decision_age
        self._max_market_age = max_market_age

    async def submit(
        self,
        event: RiskDecisionEvent,
        *,
        execution_at: datetime,
    ) -> ExecutionResult:
        """Authorize, reserve, and deterministically settle one paper intent."""

        if not event.decision.approved:
            raise ValueError("execution consumer accepts only approved risk events")
        await self._decision_store.require_exact_approval(event.decision)
        execution_utc = validate_execution_context(
            event.decision,
            event.candidate,
            market=event.market,
            execution_at=execution_at,
            max_price_deviation_bps=self._max_price_deviation_bps,
            max_decision_age=self._max_decision_age,
            max_market_age=self._max_market_age,
        )
        intent = OrderIntent.from_approved_decision(
            event.decision,
            event.candidate,
            account_id=self._account_id,
            order_type=OrderType.MARKET,
            created_at=execution_utc,
        )
        existing = await self._ledger.get_order(intent.intent_id)
        if existing is None:
            try:
                await self._ledger.persist_intent(intent)
            except DuplicateLedgerIdentity:
                existing = await self._ledger.get_order(intent.intent_id)
                if existing is None:
                    raise
        if existing is not None:
            stored_intent = await self._ledger.get_intent(intent.intent_id)
            if stored_intent != intent:
                raise ValueError("persisted intent differs from deterministic intent")
            if existing.status.terminal:
                return await self._ledger.result_for(
                    intent.intent_id, duplicate=True
                )
            if existing.status in {
                OrderStatus.AMBIGUOUS,
                OrderStatus.PARTIALLY_FILLED,
            }:
                raise AmbiguousSubmissionError(
                    "durable paper intent requires reconciliation before retry"
                )

        # PaperBroker has no network side effect. Recreating it after a process
        # crash deterministically rebuilds the same broker/fill identifiers.
        broker = PaperBroker()
        broker_record = broker.submit_order(intent, event.market)
        await self._ledger.record_submission(broker_record)
        for fill in broker.list_fills(intent.client_order_id):
            stored_fills = await self._ledger.fills_for(intent.intent_id)
            if any(existing_fill.fill_id == fill.fill_id for existing_fill in stored_fills):
                continue
            await self._ledger.record_fill(fill)
        return await self._ledger.result_for(
            intent.intent_id,
            duplicate=existing is not None,
        )
