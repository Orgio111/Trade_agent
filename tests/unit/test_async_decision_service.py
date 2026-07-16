"""Deterministic, fail-closed tests for the canonical risk-decision worker."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any
from uuid import UUID

import pytest
from pydantic import SecretStr

from packages.domain import SourceMode
from packages.event_bus import CoreSubject, PublishReceipt
from packages.execution import MarketSnapshot
from packages.local_ai import LocalModelRole
from packages.risk import (
    CandidateSignal,
    InstrumentConstraints,
    PortfolioState,
    RiskDecision,
    RiskPolicy,
    RiskReason,
    VersionedInstrumentConstraints,
)
from workers.config import WorkerSettings
from workers.contracts import CandidateForRiskEvent, RiskDecisionEvent
from workers.decision.async_service import (
    AsyncRiskDecisionService,
    CandidateMetricUnderstatement,
    CandidateTimingError,
    UnsupportedCandidateMode,
)
from workers.decision.runtime import DecisionRuntime, RuntimeModeMismatch


NOW = datetime(2026, 7, 16, 8, 0, tzinfo=UTC)
MARKET_EVENT_ID = UUID("cf3fac1e-75ef-4445-9956-ece8ce1b493f")


def _candidate(**updates: object) -> CandidateSignal:
    values: dict[str, object] = {
        "trace_id": "trace-decision-runtime",
        "signal_id": "signal-decision-runtime",
        "strategy_id": "local-ollama-candidate",
        "strategy_version": "1",
        "venue": "binance",
        "market_type": "spot",
        "instrument": "BTCUSDT",
        "side": "buy",
        "reference_price": Decimal("100"),
        "stop_price": Decimal("99"),
        "take_profit_price": Decimal("102"),
        "confidence": Decimal("0.8"),
        "spread_bps": Decimal("2"),
        "expected_slippage_bps": Decimal("1"),
        "data_age_seconds": Decimal("1"),
        "source_mode": SourceMode.PAPER_LIVE,
        "requested_risk_fraction": Decimal("0.005"),
    }
    values.update(updates)
    return CandidateSignal.model_validate(values)


def _market(*, observed_at: datetime = NOW) -> MarketSnapshot:
    return MarketSnapshot(
        venue="binance",
        market_type="spot",
        instrument="BTCUSDT",
        bid=Decimal("99.99"),
        ask=Decimal("100.01"),
        last=Decimal("100"),
        observed_at=observed_at,
    )


def _candidate_event(
    *,
    candidate: CandidateSignal | None = None,
    market: MarketSnapshot | None = None,
    created_at: datetime = NOW + timedelta(seconds=1),
) -> CandidateForRiskEvent:
    return CandidateForRiskEvent(
        trace_id="trace-decision-runtime",
        market_event_id=MARKET_EVENT_ID,
        model_role=LocalModelRole.FAST,
        model="phi3:3.8b",
        model_digest="a" * 64,
        candidate=candidate or _candidate(),
        market=market or _market(),
        created_at=created_at,
    )


def _portfolio(**updates: object) -> PortfolioState:
    values: dict[str, object] = {
        "account_id": "paper-main",
        "state_id": "portfolio-state-1",
        "equity": Decimal("10000"),
        "cash": Decimal("10000"),
        "daily_pnl": Decimal("0"),
        "weekly_pnl": Decimal("0"),
        "consecutive_losses": 0,
        "open_positions": 0,
        "gross_exposure": Decimal("0"),
        "reconciled_at": NOW,
        "kill_switch_active": False,
    }
    values.update(updates)
    return PortfolioState.model_validate(values)


def _constraints() -> InstrumentConstraints:
    return InstrumentConstraints(
        venue="binance",
        market_type="spot",
        instrument="BTCUSDT",
        tick_size=Decimal("0.01"),
        step_size=Decimal("0.001"),
        min_quantity=Decimal("0.001"),
        min_notional=Decimal("10"),
    )


class FakeRiskInputs:
    def __init__(self, *, portfolio: PortfolioState | None = None) -> None:
        self.policy = RiskPolicy(version="risk-policy-v1")
        self.portfolio = portfolio or _portfolio()
        self.constraints = VersionedInstrumentConstraints(
            version="binance-btc-v1",
            constraints=_constraints(),
            effective_at=NOW - timedelta(days=1),
            expires_at=None,
        )
        self.calls: list[tuple[str, dict[str, object]]] = []

    async def get_active_policy(self) -> RiskPolicy:
        self.calls.append(("policy", {}))
        return self.policy

    async def get_portfolio_state(
        self,
        account_id: str,
        *,
        as_of: datetime,
        max_age_seconds: Decimal | int | str,
        state_id: str | None = None,
    ) -> PortfolioState:
        self.calls.append(
            (
                "portfolio",
                {
                    "account_id": account_id,
                    "as_of": as_of,
                    "max_age_seconds": max_age_seconds,
                    "state_id": state_id,
                },
            )
        )
        return self.portfolio

    async def get_instrument_constraints(
        self,
        *,
        venue: str,
        market_type: str,
        instrument: str,
        as_of: datetime,
        version: str | None = None,
    ) -> VersionedInstrumentConstraints:
        self.calls.append(
            (
                "constraints",
                {
                    "venue": venue,
                    "market_type": market_type,
                    "instrument": instrument,
                    "as_of": as_of,
                    "version": version,
                },
            )
        )
        return self.constraints


class FakeDecisionAuthority:
    def __init__(self) -> None:
        self.existing: RiskDecision | None = None
        self.lookups: list[tuple[str, str, str, str]] = []
        self.records: list[
            tuple[CandidateSignal, RiskDecision, PortfolioState, str, str]
        ] = []

    async def get_for_signal(
        self,
        signal_id: str,
        *,
        candidate_hash: str,
        market_event_id: str,
        candidate_event_id: str,
    ) -> RiskDecision | None:
        self.lookups.append(
            (signal_id, candidate_hash, market_event_id, candidate_event_id)
        )
        return self.existing

    async def record_candidate_and_decision(
        self,
        candidate: CandidateSignal,
        decision: RiskDecision,
        *,
        portfolio_state: PortfolioState,
        market_event_id: str,
        candidate_event_id: str,
        strategy_version_id: UUID | None = None,
        feature_snapshot_id: str | None = None,
    ) -> RiskDecision:
        assert strategy_version_id is None
        assert feature_snapshot_id is None
        self.records.append(
            (
                candidate,
                decision,
                portfolio_state,
                market_event_id,
                candidate_event_id,
            )
        )
        self.existing = decision
        return decision


def _service(
    *, portfolio: PortfolioState | None = None
) -> tuple[AsyncRiskDecisionService, FakeRiskInputs, FakeDecisionAuthority]:
    inputs = FakeRiskInputs(portfolio=portfolio)
    authority = FakeDecisionAuthority()
    return (
        AsyncRiskDecisionService(
            risk_inputs=inputs,
            decision_authority=authority,
            account_id="paper-main",
        ),
        inputs,
        authority,
    )


@pytest.mark.asyncio
async def test_service_evaluates_and_atomically_persists_exact_candidate() -> None:
    service, inputs, authority = _service()
    candidate_event = _candidate_event()

    result = await service.evaluate(candidate_event)

    assert result.decision.approved is True
    assert result.candidate_event_id == candidate_event.event_id
    assert result.market_event_id == MARKET_EVENT_ID
    assert result.created_at == candidate_event.created_at
    assert authority.records == [
        (
            candidate_event.candidate,
            result.decision,
            inputs.portfolio,
            str(MARKET_EVENT_ID),
            str(candidate_event.event_id),
        )
    ]
    assert [call[0] for call in inputs.calls] == [
        "policy",
        "portfolio",
        "constraints",
    ]
    assert inputs.calls[1][1]["as_of"] == candidate_event.created_at
    assert inputs.calls[2][1]["as_of"] == candidate_event.created_at


@pytest.mark.asyncio
async def test_redelivery_reuses_exact_persisted_verdict_without_new_inputs() -> None:
    service, inputs, authority = _service()
    candidate_event = _candidate_event()
    first = await service.evaluate(candidate_event)
    inputs.calls.clear()

    second = await service.evaluate(candidate_event)

    assert second == first
    assert inputs.calls == []
    assert len(authority.records) == 1
    assert authority.lookups[-1] == (
        candidate_event.candidate.signal_id,
        candidate_event.candidate.content_hash(),
        str(candidate_event.market_event_id),
        str(candidate_event.event_id),
    )


@pytest.mark.asyncio
async def test_hard_rule_rejection_is_persisted_and_never_upgraded() -> None:
    service, _, authority = _service(portfolio=_portfolio(kill_switch_active=True))

    result = await service.evaluate(_candidate_event())

    assert result.decision.approved is False
    assert result.decision.reason_codes == (RiskReason.KILL_SWITCH_ACTIVE,)
    assert authority.records[0][1] == result.decision


@pytest.mark.asyncio
async def test_declared_data_age_cannot_understate_authoritative_age() -> None:
    service, inputs, authority = _service()
    event = _candidate_event(candidate=_candidate(data_age_seconds=Decimal("0.999999")))

    with pytest.raises(CandidateMetricUnderstatement, match="data_age_seconds"):
        await service.evaluate(event)

    assert inputs.calls == []
    assert authority.records == []


@pytest.mark.asyncio
async def test_declared_spread_cannot_understate_bid_ask_spread() -> None:
    service, inputs, authority = _service()
    event = _candidate_event(candidate=_candidate(spread_bps=Decimal("1.999999")))

    with pytest.raises(CandidateMetricUnderstatement, match="spread_bps"):
        await service.evaluate(event)

    assert inputs.calls == []
    assert authority.records == []


@pytest.mark.asyncio
async def test_future_market_observation_fails_before_risk_lookup() -> None:
    service, inputs, authority = _service()
    event = _candidate_event(
        market=_market(observed_at=NOW + timedelta(seconds=2)),
    )

    with pytest.raises(CandidateTimingError, match="future"):
        await service.evaluate(event)

    assert inputs.calls == []
    assert authority.records == []


@pytest.mark.asyncio
async def test_missing_candidate_metrics_reach_existing_fail_closed_risk_rules() -> (
    None
):
    service, _, authority = _service()
    missing_age = _candidate_event(candidate=_candidate(data_age_seconds=None))

    age_result = await service.evaluate(missing_age)

    assert age_result.decision.reason_codes == (RiskReason.MISSING_MARKET_DATA,)
    assert len(authority.records) == 1


@pytest.mark.asyncio
async def test_service_itself_blocks_live_mode_before_any_authority_lookup() -> None:
    service, inputs, authority = _service()
    event = _candidate_event(candidate=_candidate(source_mode=SourceMode.LIVE))

    with pytest.raises(UnsupportedCandidateMode, match="replay or paper_live"):
        await service.evaluate(event)

    assert inputs.calls == []
    assert authority.lookups == []
    assert authority.records == []


@pytest.mark.asyncio
async def test_risk_input_failure_propagates_without_persisting_a_verdict() -> None:
    class FailingRiskInputs(FakeRiskInputs):
        async def get_active_policy(self) -> RiskPolicy:
            raise ConnectionError("database unavailable")

    inputs = FailingRiskInputs()
    authority = FakeDecisionAuthority()
    service = AsyncRiskDecisionService(
        risk_inputs=inputs,
        decision_authority=authority,
        account_id="paper-main",
    )

    with pytest.raises(ConnectionError, match="unavailable"):
        await service.evaluate(_candidate_event())

    assert authority.records == []


class FakeJetStreamBus:
    def __init__(self) -> None:
        self.published: list[tuple[CoreSubject, bytes, str, str | None]] = []
        self.subscription: tuple[CoreSubject, Any, Any] | None = None

    async def publish(
        self,
        subject: CoreSubject,
        payload: bytes,
        *,
        message_id: str,
        stream: str | None = None,
    ) -> PublishReceipt:
        self.published.append((subject, payload, message_id, stream))
        return PublishReceipt(
            stream="QUANTEX_CORE",
            sequence=1,
            duplicate=False,
        )

    async def subscribe(
        self,
        subject: CoreSubject,
        handler: Any,
        *,
        settings: Any,
        on_failure: Any = None,
    ) -> object:
        del on_failure
        self.subscription = (subject, handler, settings)
        return object()


def _settings(*, mode: SourceMode = SourceMode.PAPER_LIVE) -> WorkerSettings:
    return WorkerSettings(
        service_name="decision-worker",
        mode=mode,
        nats_url="nats://nats:4222",
        database_url=SecretStr("postgresql://quantex:local-only@postgres:5432/quantex"),
        ollama_base_url="http://host.docker.internal:11434",
        durable_name="decision-worker-v1",
    )


@pytest.mark.asyncio
async def test_runtime_publishes_exact_binding_to_verdict_subject() -> None:
    service, _, _ = _service()
    bus = FakeJetStreamBus()
    runtime = DecisionRuntime(settings=_settings(), service=service, bus=bus)
    source = _candidate_event()

    result = await runtime.handle(source.canonical_json().encode("utf-8"))

    assert result.decision.approved is True
    assert len(bus.published) == 1
    subject, payload, message_id, stream = bus.published[0]
    published = RiskDecisionEvent.model_validate_json(payload)
    assert subject is CoreSubject.RISK_APPROVED
    assert published == result
    assert message_id == str(result.event_id)
    assert stream == "QUANTEX_CORE"


@pytest.mark.asyncio
async def test_runtime_publishes_hard_rejection_only_to_rejected_subject() -> None:
    service, _, _ = _service(portfolio=_portfolio(kill_switch_active=True))
    bus = FakeJetStreamBus()
    runtime = DecisionRuntime(settings=_settings(), service=service, bus=bus)

    result = await runtime.handle(_candidate_event().canonical_json().encode("utf-8"))

    assert result.decision.approved is False
    assert len(bus.published) == 1
    assert bus.published[0][0] is CoreSubject.RISK_REJECTED


@pytest.mark.asyncio
async def test_runtime_rejects_cross_mode_candidate_without_publication() -> None:
    service, _, authority = _service()
    bus = FakeJetStreamBus()
    runtime = DecisionRuntime(
        settings=_settings(mode=SourceMode.REPLAY),
        service=service,
        bus=bus,
    )

    with pytest.raises(RuntimeModeMismatch, match="does not match"):
        await runtime.handle(_candidate_event().canonical_json().encode("utf-8"))

    assert authority.records == []
    assert bus.published == []


@pytest.mark.asyncio
async def test_runtime_subscribes_only_to_candidate_subject() -> None:
    service, _, _ = _service()
    bus = FakeJetStreamBus()
    runtime = DecisionRuntime(settings=_settings(), service=service, bus=bus)

    await runtime.start()

    assert bus.subscription is not None
    subject, handler, settings = bus.subscription
    assert subject is CoreSubject.SIGNAL_CANDIDATE
    assert handler == runtime.handle
    assert settings.durable_name == "decision-worker-v1"
