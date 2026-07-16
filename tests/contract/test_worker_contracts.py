"""Integrity and authority contracts for canonical worker events."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
import json

import pytest
from pydantic import ValidationError

from packages.domain import SourceMode
from packages.execution import MarketSnapshot
from packages.local_ai import LocalModelRole
from packages.risk import CandidateSignal, RiskDecision, RiskReason
from workers.contracts import CandidateForRiskEvent, RiskDecisionEvent


NOW = datetime(2026, 7, 16, 7, 0, tzinfo=UTC)


def candidate() -> CandidateSignal:
    return CandidateSignal(
        trace_id="trace-runtime-1",
        signal_id="sig-runtime-1",
        strategy_id="local-ollama",
        strategy_version="runtime-contract-v1",
        venue="binance",
        market_type="spot",
        instrument="BTCUSDT",
        side="buy",
        reference_price=Decimal("100"),
        stop_price=Decimal("99"),
        take_profit_price=Decimal("102"),
        confidence=Decimal("0.8"),
        spread_bps=Decimal("2"),
        expected_slippage_bps=Decimal("3"),
        data_age_seconds=Decimal("0.1"),
        source_mode=SourceMode.PAPER_LIVE,
        requested_risk_fraction=Decimal("0.005"),
    )


def market() -> MarketSnapshot:
    return MarketSnapshot(
        venue="binance",
        market_type="spot",
        instrument="BTCUSDT",
        bid=Decimal("99.99"),
        ask=Decimal("100.01"),
        last=Decimal("100"),
        observed_at=NOW,
    )


def decision(value: CandidateSignal) -> RiskDecision:
    return RiskDecision(
        decision_id="risk_000000000000000000000001",
        trace_id=value.trace_id,
        signal_id=value.signal_id,
        account_id="paper-main",
        candidate_hash=value.content_hash(),
        approved=True,
        reason_codes=(RiskReason.APPROVED,),
        approved_quantity=Decimal("1"),
        approved_notional=Decimal("100"),
        approved_risk_amount=Decimal("1"),
        approved_risk_fraction=Decimal("0.005"),
        policy_version="risk-v1",
        portfolio_state_id="portfolio-v1",
        evaluated_at=NOW,
    )


def test_candidate_event_has_deterministic_identity_and_checksum() -> None:
    first = CandidateForRiskEvent(
        trace_id="trace-runtime-1",
        market_event_id="b7a7ee48-90f6-40cc-890f-4d6bf8e2c6b0",
        model_role=LocalModelRole.FAST,
        model="phi3:3.8b",
        model_digest="a" * 64,
        candidate=candidate(),
        market=market(),
        created_at=NOW,
    )
    second = CandidateForRiskEvent.model_validate_json(first.canonical_json())

    assert second == first
    assert second.event_id == first.event_id
    assert len(first.content_sha256) == 64


def test_candidate_event_rejects_identity_and_checksum_tampering() -> None:
    event = CandidateForRiskEvent(
        trace_id="trace-runtime-1",
        market_event_id="b7a7ee48-90f6-40cc-890f-4d6bf8e2c6b0",
        model_role=LocalModelRole.FAST,
        model="phi3:3.8b",
        model_digest="a" * 64,
        candidate=candidate(),
        market=market(),
        created_at=NOW,
    )
    payload = json.loads(event.model_dump_json())
    payload["market"]["instrument"] = "ETHUSDT"

    with pytest.raises(ValidationError, match="market identity"):
        CandidateForRiskEvent.model_validate(payload)


def test_candidate_event_rejects_unapproved_model_role_mapping() -> None:
    with pytest.raises(ValidationError, match="approved local role"):
        CandidateForRiskEvent(
            trace_id="trace-runtime-1",
            market_event_id="b7a7ee48-90f6-40cc-890f-4d6bf8e2c6b0",
            model_role=LocalModelRole.FAST,
            model="mistral",
            model_digest="a" * 64,
            candidate=candidate(),
            market=market(),
            created_at=NOW,
        )


def test_risk_event_is_bound_to_exact_candidate_and_verdict() -> None:
    candidate_event = CandidateForRiskEvent(
        trace_id="trace-runtime-1",
        market_event_id="b7a7ee48-90f6-40cc-890f-4d6bf8e2c6b0",
        model_role=LocalModelRole.FAST,
        model="phi3:3.8b",
        model_digest="a" * 64,
        candidate=candidate(),
        market=market(),
        created_at=NOW,
    )
    risk_event = RiskDecisionEvent.create(
        candidate_event,
        decision(candidate_event.candidate),
        created_at=NOW,
    )

    assert RiskDecisionEvent.model_validate_json(risk_event.canonical_json()) == risk_event
    assert risk_event.decision.candidate_hash == risk_event.candidate.content_hash()


def test_risk_event_rejects_candidate_substitution() -> None:
    candidate_event = CandidateForRiskEvent(
        trace_id="trace-runtime-1",
        market_event_id="b7a7ee48-90f6-40cc-890f-4d6bf8e2c6b0",
        model_role=LocalModelRole.FAST,
        model="phi3:3.8b",
        model_digest="a" * 64,
        candidate=candidate(),
        market=market(),
        created_at=NOW,
    )
    risk_event = RiskDecisionEvent.create(
        candidate_event,
        decision(candidate_event.candidate),
        created_at=NOW,
    )
    payload = risk_event.model_dump(mode="python")
    payload["candidate"] = candidate_event.candidate.model_copy(
        update={"stop_price": Decimal("98")}
    )

    with pytest.raises(ValidationError, match="not bound"):
        RiskDecisionEvent.model_validate(payload)


def test_risk_event_cannot_relabel_embedding_model_as_trading_provenance() -> None:
    candidate_event = CandidateForRiskEvent(
        trace_id="trace-runtime-1",
        market_event_id="b7a7ee48-90f6-40cc-890f-4d6bf8e2c6b0",
        model_role=LocalModelRole.FAST,
        model="phi3:3.8b",
        model_digest="a" * 64,
        candidate=candidate(),
        market=market(),
        created_at=NOW,
    )
    risk_event = RiskDecisionEvent.create(
        candidate_event,
        decision(candidate_event.candidate),
        created_at=NOW,
    )
    payload = risk_event.model_dump(mode="python")
    payload["model_role"] = LocalModelRole.EMBEDDING
    payload["model"] = "nomic-embed-text"

    with pytest.raises(ValidationError, match="reasoning or fast"):
        RiskDecisionEvent.model_validate(payload)
