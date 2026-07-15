from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal, localcontext

import pytest

from packages.brokers import (
    BrokerError,
    DuplicateBrokerIntent,
    InvalidOrderError,
    PaperBroker,
    PaperBrokerConfig,
    PaperModeRequired,
)
from packages.domain.events import SourceMode
from packages.event_bus import InMemoryEventBus
from packages.execution import (
    AmbiguousSubmissionError,
    BrokerLedgerConflict,
    DiscrepancyKind,
    ExecutionService,
    InMemoryDecisionStore,
    InMemoryExecutionLedger,
    InvalidOrderTransition,
    MarketSnapshot,
    OrderIntent,
    OrderRecord,
    OrderSide,
    OrderStatus,
    OrderType,
    Reconciler,
    UnapprovedDecisionError,
    assert_transition,
)
from packages.risk import CandidateSignal, RiskDecision, RiskReason
from workers.execution import ExecutionWorker, StaleExecutionContext


NOW = datetime(2026, 7, 15, 4, 0, tzinfo=timezone.utc)


def candidate(
    *,
    signal_id: str = "sig-001",
    side: str = "buy",
    source_mode: SourceMode = SourceMode.PAPER_LIVE,
    venue: str = "binance",
    market_type: str = "spot",
) -> CandidateSignal:
    return CandidateSignal(
        trace_id="trace-001",
        signal_id=signal_id,
        strategy_id="breakout",
        strategy_version="1.0.0",
        venue=venue,
        market_type=market_type,
        instrument="BTCUSDT",
        side=side,
        reference_price=Decimal("100"),
        stop_price=Decimal("95") if side == "buy" else Decimal("105"),
        take_profit_price=Decimal("110") if side == "buy" else Decimal("90"),
        confidence=Decimal("0.8"),
        spread_bps=Decimal("2"),
        expected_slippage_bps=Decimal("1"),
        data_age_seconds=Decimal("0.1"),
        source_mode=source_mode,
        requested_risk_fraction=Decimal("0.005"),
    )


def decision(
    *,
    signal_id: str | None = None,
    approved: bool = True,
    quantity: str = "2",
    candidate_value: CandidateSignal | None = None,
    account_id: str = "paper-main",
) -> RiskDecision:
    bound_candidate = candidate_value or candidate(signal_id=signal_id or "sig-001")
    resolved_signal_id = signal_id or bound_candidate.signal_id
    approved_quantity = Decimal(quantity) if approved else Decimal("0")
    return RiskDecision(
        decision_id="risk_000000000000000000000001",
        trace_id="trace-001",
        signal_id=resolved_signal_id,
        account_id=account_id,
        candidate_hash=bound_candidate.content_hash(),
        approved=approved,
        reason_codes=(
            (RiskReason.APPROVED,) if approved else (RiskReason.KILL_SWITCH_ACTIVE,)
        ),
        approved_quantity=approved_quantity,
        approved_notional=approved_quantity * Decimal("100"),
        approved_risk_amount=approved_quantity * Decimal("5"),
        approved_risk_fraction=Decimal("0.005") if approved else Decimal("0"),
        policy_version="risk-v1",
        portfolio_state_id="portfolio-001",
        evaluated_at=NOW,
    )


def market(
    *,
    bid: str = "99",
    ask: str = "100",
    bid_quantity: str | None = None,
    ask_quantity: str | None = None,
    at: datetime = NOW,
    instrument: str = "BTCUSDT",
    venue: str = "binance",
    market_type: str = "spot",
) -> MarketSnapshot:
    return MarketSnapshot(
        venue=venue,
        market_type=market_type,
        instrument=instrument,
        bid=Decimal(bid),
        ask=Decimal(ask),
        last=Decimal(ask),
        available_bid_quantity=(
            Decimal(bid_quantity) if bid_quantity is not None else None
        ),
        available_ask_quantity=(
            Decimal(ask_quantity) if ask_quantity is not None else None
        ),
        observed_at=at,
    )


def service(
    *, config: PaperBrokerConfig | None = None
) -> tuple[ExecutionService, InMemoryExecutionLedger, PaperBroker]:
    ledger = InMemoryExecutionLedger()
    broker = PaperBroker(config)
    return ExecutionService(ledger, broker, InMemoryDecisionStore()), ledger, broker


def authorized_submit(executor: ExecutionService, decision_value, candidate_value, **kwargs):
    executor.decision_store.record(decision_value)
    return executor.submit(decision_value, candidate_value, **kwargs)


def test_execution_rejects_unapproved_decision_before_persist_or_submit() -> None:
    executor, ledger, broker = service()

    with pytest.raises(UnapprovedDecisionError):
        authorized_submit(
            executor,
            decision(approved=False),
            candidate(),
            account_id="paper-main",
            market=market(),
        )

    assert ledger.list_intents() == ()
    assert ledger.events == ()
    assert broker.list_orders() == ()


def test_execution_rejects_unissued_forged_approval() -> None:
    executor, ledger, broker = service()

    with pytest.raises(UnapprovedDecisionError, match="not issued"):
        executor.submit(
            decision(),
            candidate(),
            account_id="paper-main",
            market=market(),
        )

    assert ledger.list_intents() == ()
    assert broker.list_orders() == ()


def test_execution_service_has_a_hard_live_mode_gate() -> None:
    executor, ledger, broker = service()
    live_candidate = candidate(source_mode=SourceMode.LIVE)
    live_decision = decision(candidate_value=live_candidate)

    with pytest.raises(UnapprovedDecisionError, match="hard-gated"):
        authorized_submit(
            executor,
            live_decision,
            live_candidate,
            account_id="paper-main",
            market=market(),
        )

    assert ledger.list_intents() == ()
    assert broker.list_orders() == ()


def test_execution_worker_rejects_stale_decision_before_intent_creation() -> None:
    executor, ledger, broker = service()
    approved_candidate = candidate()
    approved_decision = decision(candidate_value=approved_candidate)
    executor.decision_store.record(approved_decision)
    worker = ExecutionWorker(
        InMemoryEventBus(), executor, account_id="paper-main"
    )

    with pytest.raises(StaleExecutionContext):
        worker.process(
            approved_decision,
            approved_candidate,
            market=market(),
            execution_at=NOW + timedelta(seconds=3),
        )

    assert ledger.list_intents() == ()
    assert broker.list_orders() == ()


@pytest.mark.parametrize("account_id", ["paper-other", "live-main"])
def test_execution_rejects_approval_replayed_to_another_account(
    account_id: str,
) -> None:
    executor, ledger, broker = service()
    approved_candidate = candidate()

    with pytest.raises(ValueError, match="account_id does not match"):
        authorized_submit(
            executor,
            decision(candidate_value=approved_candidate),
            approved_candidate,
            account_id=account_id,
            market=market(),
        )

    assert ledger.list_intents() == ()
    assert broker.list_orders() == ()


def test_execution_rejects_candidate_changed_after_risk_approval() -> None:
    executor, ledger, broker = service()
    approved_candidate = candidate()
    changed_candidate = candidate(side="sell")

    with pytest.raises(ValueError, match="candidate payload does not match"):
        authorized_submit(
            executor,
            decision(candidate_value=approved_candidate),
            changed_candidate,
            account_id="paper-main",
            market=market(),
        )

    assert ledger.list_intents() == ()
    assert broker.list_orders() == ()


def test_deterministic_duplicate_is_idempotent_across_service_restart() -> None:
    executor, ledger, broker = service()
    first = authorized_submit(
        executor,
        decision(), candidate(), account_id="paper-main", market=market(), created_at=NOW
    )
    event_count = len(ledger.events)

    restarted_service = ExecutionService(ledger, broker, executor.decision_store)
    duplicate = authorized_submit(
        restarted_service,
        decision(),
        candidate(),
        account_id="paper-main",
        market=market(at=NOW + timedelta(seconds=5)),
        created_at=NOW + timedelta(seconds=5),
    )

    assert duplicate.duplicate is True
    assert duplicate.intent.intent_id == first.intent.intent_id
    assert duplicate.intent.client_order_id == first.intent.client_order_id
    assert duplicate.order == first.order
    assert len(broker.list_orders()) == 1
    assert len(broker.list_fills()) == 1
    assert len(ledger.fills) == 1
    assert len(ledger.events) == event_count


def test_ack_lost_after_accept_is_ambiguous_then_reconciles_without_resubmit() -> None:
    ledger = InMemoryExecutionLedger()
    broker = AcceptThenTimeoutBroker()
    executor = ExecutionService(ledger, broker, InMemoryDecisionStore())

    with pytest.raises(AmbiguousSubmissionError):
        authorized_submit(
            executor,
            decision(), candidate(), account_id="paper-main", market=market(), created_at=NOW
        )

    assert ledger.list_orders()[0].status is OrderStatus.AMBIGUOUS
    assert broker.submit_calls == 1
    recovered = authorized_submit(
        executor,
        decision(), candidate(), account_id="paper-main", market=market(), created_at=NOW
    )
    assert recovered.duplicate
    assert recovered.order.status is OrderStatus.FILLED
    assert broker.submit_calls == 1


def test_unknown_submit_outcome_never_automatically_resubmits() -> None:
    ledger = InMemoryExecutionLedger()
    broker = AlwaysTimeoutBroker()
    executor = ExecutionService(ledger, broker, InMemoryDecisionStore())

    for _ in range(2):
        with pytest.raises(AmbiguousSubmissionError):
            authorized_submit(
                executor,
                decision(),
                candidate(),
                account_id="paper-main",
                market=market(),
                created_at=NOW,
            )

    assert ledger.list_orders()[0].status is OrderStatus.AMBIGUOUS
    assert broker.submit_calls == 1


def test_full_market_fill_uses_decimal_slippage_and_fee_math() -> None:
    executor, ledger, _ = service(
        config=PaperBrokerConfig(
            fee_rate=Decimal("0.001"), slippage_bps=Decimal("10")
        )
    )

    result = authorized_submit(
        executor,
        decision(quantity="2"), candidate(), account_id="paper-main", market=market()
    )

    assert result.order.status is OrderStatus.FILLED
    assert result.order.filled_quantity == Decimal("2")
    assert result.order.average_fill_price == Decimal("100.100")
    assert result.order.cumulative_fee == Decimal("0.200200")
    assert result.fills[0].notional == Decimal("200.200")
    assert result.total_fee == Decimal("0.200200")
    assert [event.event_type for event in ledger.events] == [
        "intent_persisted",
        "order_submitted",
        "fill_recorded",
    ]


@pytest.mark.parametrize("precision", [6, 10, 28, 50])
def test_intent_and_fill_are_independent_of_ambient_decimal_context(
    precision: int,
) -> None:
    approved_candidate = candidate()
    approved_decision = decision(candidate_value=approved_candidate)

    with localcontext() as context:
        context.prec = precision
        executor, _, _ = service(
            config=PaperBrokerConfig(
                fee_rate=Decimal("0.001"), slippage_bps=Decimal("10")
            )
        )
        actual = authorized_submit(
            executor,
            approved_decision,
            approved_candidate,
            account_id="paper-main",
            market=market(),
            created_at=NOW,
        )

    expected_intent = OrderIntent.from_approved_decision(
        approved_decision,
        approved_candidate,
        account_id="paper-main",
        created_at=NOW,
    )
    assert actual.intent.intent_id == expected_intent.intent_id
    assert actual.order.average_fill_price == Decimal("100.100")
    assert actual.order.cumulative_fee == Decimal("0.200200")


def test_liquidity_driven_partial_fills_accumulate_weighted_price_and_fees() -> None:
    executor, ledger, broker = service(
        config=PaperBrokerConfig(
            fee_rate=Decimal("0.001"),
            slippage_bps=Decimal("0"),
            liquidity_participation=Decimal("0.5"),
        )
    )
    first = authorized_submit(
        executor,
        decision(quantity="2"),
        candidate(),
        account_id="paper-main",
        market=market(ask_quantity="1"),
    )

    assert first.order.status is OrderStatus.PARTIALLY_FILLED
    assert first.order.filled_quantity == Decimal("0.5")
    assert first.order.average_fill_price == Decimal("100")
    assert first.order.cumulative_fee == Decimal("0.0500")

    broker.process_market(
        market(
            bid="101",
            ask="102",
            ask_quantity="3",
            at=NOW + timedelta(seconds=1),
        )
    )
    completed = executor.synchronize(first.intent.client_order_id)

    assert completed.order.status is OrderStatus.FILLED
    assert completed.order.filled_quantity == Decimal("2.0")
    assert completed.order.average_fill_price == Decimal("101.5")
    assert completed.order.cumulative_fee == Decimal("0.2030")
    assert [fill.quantity for fill in completed.fills] == [Decimal("0.5"), Decimal("1.5")]
    assert len(ledger.fills) == 2


def test_duplicate_fill_id_with_changed_money_is_quarantined() -> None:
    executor, ledger, broker = service()
    result = authorized_submit(
        executor,
        decision(),
        candidate(),
        account_id="paper-main",
        market=market(),
        created_at=NOW,
    )
    changed_broker = MutatedFillBroker(broker)
    reconciler_service = ExecutionService(
        ledger, changed_broker, executor.decision_store
    )

    with pytest.raises(BrokerLedgerConflict, match="different immutable content"):
        reconciler_service.synchronize(result.intent.client_order_id)


def test_non_marketable_limit_order_remains_open_without_a_fill() -> None:
    executor, ledger, broker = service()
    result = authorized_submit(
        executor,
        decision(),
        candidate(),
        account_id="paper-main",
        market=market(),
        order_type=OrderType.LIMIT,
        limit_price=Decimal("99"),
    )

    assert result.order.status is OrderStatus.OPEN
    assert result.order.filled_quantity == Decimal("0")
    assert result.fills == ()
    assert broker.list_fills() == ()
    assert [event.event_type for event in ledger.events] == [
        "intent_persisted",
        "order_submitted",
    ]


def test_marketable_limit_respects_limit_price_when_slippage_is_larger() -> None:
    executor, _, _ = service(
        config=PaperBrokerConfig(slippage_bps=Decimal("100"), fee_rate=Decimal("0"))
    )
    result = authorized_submit(
        executor,
        decision(quantity="1"),
        candidate(),
        account_id="paper-main",
        market=market(ask="100"),
        order_type=OrderType.LIMIT,
        limit_price=Decimal("100.5"),
    )
    assert result.order.status is OrderStatus.FILLED
    assert result.order.average_fill_price == Decimal("100.5")


def test_invalid_state_transition_is_rejected() -> None:
    assert_transition(OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED)
    assert_transition(OrderStatus.PENDING_SUBMIT, OrderStatus.AMBIGUOUS)
    assert_transition(OrderStatus.AMBIGUOUS, OrderStatus.OPEN)
    assert_transition(OrderStatus.FILLED, OrderStatus.FILLED)  # replay is a no-op

    with pytest.raises(InvalidOrderTransition):
        assert_transition(OrderStatus.FILLED, OrderStatus.OPEN)
    with pytest.raises(InvalidOrderTransition):
        assert_transition(OrderStatus.CANCELED, OrderStatus.FILLED)


def test_broker_rejects_duplicate_invalid_and_live_intents() -> None:
    broker = PaperBroker()
    intent = OrderIntent.from_approved_decision(
        decision(), candidate(), account_id="paper-main", created_at=NOW
    )
    broker.submit_order(intent, market())

    with pytest.raises(DuplicateBrokerIntent):
        broker.submit_order(intent, market())

    another = OrderIntent.from_approved_decision(
        replace_decision_id(decision(), "risk_000000000000000000000002"),
        candidate(signal_id="sig-001"),
        account_id="paper-main",
        created_at=NOW,
    )
    with pytest.raises(InvalidOrderError):
        broker.submit_order(another, market(instrument="ETHUSDT"))

    live_candidate = candidate(signal_id="sig-live", source_mode=SourceMode.LIVE)
    live_decision = decision(candidate_value=live_candidate)
    live_intent = OrderIntent.from_approved_decision(
        live_decision, live_candidate, account_id="paper-main", created_at=NOW
    )
    with pytest.raises(PaperModeRequired):
        broker.submit_order(live_intent, market())

    replay_candidate = candidate(signal_id="sig-replay", source_mode=SourceMode.REPLAY)
    replay_intent = OrderIntent.from_approved_decision(
        decision(candidate_value=replay_candidate),
        replay_candidate,
        account_id="paper-main",
        created_at=NOW,
    )
    replay_result = broker.submit_order(replay_intent, market())
    assert replay_result.status is OrderStatus.FILLED


def test_new_ledger_recovers_existing_broker_order_without_resubmitting() -> None:
    first_executor, _, broker = service()
    first = authorized_submit(
        first_executor,
        decision(), candidate(), account_id="paper-main", market=market(), created_at=NOW
    )

    recovered_ledger = InMemoryExecutionLedger()
    recovered_service = ExecutionService(
        recovered_ledger, broker, first_executor.decision_store
    )
    recovered = authorized_submit(
        recovered_service,
        decision(), candidate(), account_id="paper-main", market=market(), created_at=NOW
    )

    assert recovered.intent.intent_id == first.intent.intent_id
    assert recovered.order.status is OrderStatus.FILLED
    assert recovered.order.filled_quantity == first.order.filled_quantity
    assert len(broker.list_orders()) == 1
    assert len(broker.list_fills()) == 1
    assert len(recovered_ledger.fills) == 1


def test_reconciler_reports_missing_quantity_and_status_without_mutation() -> None:
    ledger = InMemoryExecutionLedger()
    first_intent = OrderIntent.from_approved_decision(
        decision(), candidate(), account_id="paper-main", created_at=NOW
    )
    ledger.persist_intent(first_intent)  # missing at broker

    second_intent = OrderIntent.from_approved_decision(
        replace_decision_id(
            decision(signal_id="sig-002"), "risk_000000000000000000000002"
        ),
        candidate(signal_id="sig-002"),
        account_id="paper-main",
        created_at=NOW,
    )
    ledger.persist_intent(second_intent)
    ledger_side = OrderRecord(
        intent_id=second_intent.intent_id,
        client_order_id=second_intent.client_order_id,
        account_id=second_intent.account_id,
        venue=second_intent.venue,
        market_type=second_intent.market_type,
        instrument=second_intent.instrument,
        side=second_intent.side,
        order_type=second_intent.order_type,
        requested_quantity=second_intent.quantity,
        status=OrderStatus.OPEN,
        broker_order_id="paper-shared",
        created_at=NOW,
        updated_at=NOW,
    )
    ledger.record_submission(ledger_side)

    broker_side = replace(
        ledger_side,
        requested_quantity=Decimal("3"),
        filled_quantity=Decimal("1"),
        average_fill_price=Decimal("100"),
        status=OrderStatus.PARTIALLY_FILLED,
    )
    broker_only = replace(
        ledger_side,
        intent_id="int_broker_only",
        client_order_id="qx_broker_only",
        broker_order_id="paper-broker-only",
    )
    fake_broker = StaticBroker((broker_side, broker_only))
    event_count = len(ledger.events)
    fill_count = len(ledger.fills)

    report = Reconciler(ledger, fake_broker).reconcile()

    kinds = {item.kind for item in report.discrepancies}
    assert kinds == {
        DiscrepancyKind.MISSING_AT_BROKER,
        DiscrepancyKind.MISSING_IN_LEDGER,
        DiscrepancyKind.REQUESTED_QUANTITY_MISMATCH,
        DiscrepancyKind.FILLED_QUANTITY_MISMATCH,
        DiscrepancyKind.STATUS_MISMATCH,
        DiscrepancyKind.AVERAGE_FILL_PRICE_MISMATCH,
    }
    assert report.clean is False
    assert len(ledger.events) == event_count
    assert len(ledger.fills) == fill_count


def test_decimal_boundary_rejects_binary_float_inputs() -> None:
    with pytest.raises(TypeError, match="floats are unsafe"):
        MarketSnapshot(
            venue="binance",
            market_type="spot",
            instrument="BTCUSDT",
            bid=99.0,
            ask=100.0,
        )


def replace_decision_id(value: RiskDecision, decision_id: str) -> RiskDecision:
    return value.model_copy(update={"decision_id": decision_id})


class StaticBroker:
    def __init__(self, orders: tuple[OrderRecord, ...]) -> None:
        self._orders = orders

    def submit_order(self, intent, market):  # pragma: no cover - reconciliation only
        raise AssertionError("not used")

    def get_order(self, client_order_id: str) -> OrderRecord | None:
        return next(
            (order for order in self._orders if order.client_order_id == client_order_id),
            None,
        )

    def list_orders(self) -> tuple[OrderRecord, ...]:
        return self._orders

    def list_fills(self, client_order_id: str | None = None):
        return ()


class AcceptThenTimeoutBroker(PaperBroker):
    def __init__(self) -> None:
        super().__init__()
        self.submit_calls = 0

    def submit_order(self, intent, market):
        self.submit_calls += 1
        super().submit_order(intent, market)
        raise BrokerError("ack lost after venue acceptance")


class AlwaysTimeoutBroker:
    def __init__(self) -> None:
        self.submit_calls = 0

    def submit_order(self, intent, market):
        self.submit_calls += 1
        raise BrokerError("no response")

    def get_order(self, client_order_id: str):
        return None

    def list_orders(self):
        return ()

    def list_fills(self, client_order_id: str | None = None):
        return ()


class MutatedFillBroker:
    def __init__(self, delegate: PaperBroker) -> None:
        self.delegate = delegate

    def submit_order(self, intent, market):
        return self.delegate.submit_order(intent, market)

    def get_order(self, client_order_id: str):
        return self.delegate.get_order(client_order_id)

    def list_orders(self):
        return self.delegate.list_orders()

    def list_fills(self, client_order_id: str | None = None):
        return tuple(
            replace(fill, fee=fill.fee + Decimal("0.01"))
            for fill in self.delegate.list_fills(client_order_id)
        )
