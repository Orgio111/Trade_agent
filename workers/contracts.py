"""Integrity-bound transport envelopes for the canonical worker boundary."""

from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import NAMESPACE_URL, UUID, uuid5

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from packages.domain.events import canonical_json, compute_payload_checksum
from packages.domain import SourceMode
from packages.execution import ExecutionResult, MarketSnapshot, OrderIntent
from packages.local_ai import LOCAL_MODEL_BY_ROLE, LocalModelRole
from packages.risk import CandidateSignal, RiskDecision


_ZERO_UUID = UUID(int=0)
DETERMINISTIC_BASELINE_MODEL = "baseline-feature-v1"
CandidateProvider = Literal["ollama", "deterministic_baseline"]


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(UTC)


class _StrictEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CandidateForRiskEvent(_StrictEnvelope):
    """A local-intelligence candidate awaiting deterministic authorization."""

    schema_version: Literal["1.0"] = "1.0"
    event_id: UUID = _ZERO_UUID
    trace_id: str = Field(min_length=1, max_length=128)
    market_event_id: UUID
    provider: CandidateProvider = "ollama"
    model_role: LocalModelRole | None
    model: str = Field(min_length=1, max_length=128)
    model_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate: CandidateSignal
    market: MarketSnapshot
    created_at: datetime
    content_sha256: str = Field(default="", pattern=r"^[0-9a-f]{64}$")

    @field_validator("created_at")
    @classmethod
    def normalize_created_at(cls, value: datetime) -> datetime:
        return _utc(value)

    @model_validator(mode="after")
    def bind_identity(self) -> CandidateForRiskEvent:
        if self.trace_id != self.candidate.trace_id:
            raise ValueError("event trace_id does not match candidate")
        self._validate_provenance()
        if (
            self.market.venue != self.candidate.venue
            or self.market.market_type != self.candidate.market_type
            or self.market.instrument != self.candidate.instrument
        ):
            raise ValueError("market identity does not match candidate")
        checksum = compute_payload_checksum(self.content_payload())
        if self.content_sha256 and self.content_sha256 != checksum:
            raise ValueError("content_sha256 does not match candidate event")
        if not self.content_sha256:
            object.__setattr__(self, "content_sha256", checksum)
        expected_id = uuid5(
            NAMESPACE_URL,
            f"urn:trade-agent:candidate-for-risk:v1:{checksum}",
        )
        if self.event_id not in {_ZERO_UUID, expected_id}:
            raise ValueError("event_id does not match candidate event content")
        if self.event_id == _ZERO_UUID:
            object.__setattr__(self, "event_id", expected_id)
        return self

    def _validate_provenance(self) -> None:
        if self.provider == "ollama":
            if self.model_role not in {
                LocalModelRole.REASONING,
                LocalModelRole.FAST,
            }:
                raise ValueError(
                    "Ollama candidate must come from a reasoning or fast model role"
                )
            if self.model != LOCAL_MODEL_BY_ROLE[self.model_role]:
                raise ValueError(
                    "candidate model does not match the approved local role"
                )
            return
        if self.model_role is not None or self.model != DETERMINISTIC_BASELINE_MODEL:
            raise ValueError("deterministic baseline provenance is invalid")
        if self.candidate.strategy_id != DETERMINISTIC_BASELINE_MODEL:
            raise ValueError("deterministic baseline strategy identity is invalid")

    def content_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "trace_id": self.trace_id,
            "market_event_id": self.market_event_id,
            "provider": self.provider,
            "model_role": self.model_role,
            "model": self.model,
            "model_digest": self.model_digest,
            "candidate": self.candidate,
            "market": asdict(self.market),
            "created_at": self.created_at,
        }

    def canonical_json(self) -> str:
        return canonical_json(self)


class RiskDecisionEvent(_StrictEnvelope):
    """Persisted deterministic verdict plus its exact candidate and market."""

    schema_version: Literal["1.0"] = "1.0"
    event_id: UUID = _ZERO_UUID
    trace_id: str = Field(min_length=1, max_length=128)
    candidate_event_id: UUID
    market_event_id: UUID
    provider: CandidateProvider = "ollama"
    model_role: LocalModelRole | None
    model: str = Field(min_length=1, max_length=128)
    model_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate: CandidateSignal
    decision: RiskDecision
    market: MarketSnapshot
    created_at: datetime
    content_sha256: str = Field(default="", pattern=r"^[0-9a-f]{64}$")

    @field_validator("created_at")
    @classmethod
    def normalize_created_at(cls, value: datetime) -> datetime:
        return _utc(value)

    @model_validator(mode="after")
    def bind_identity(self) -> RiskDecisionEvent:
        if (
            self.trace_id
            not in {
                self.candidate.trace_id,
                self.decision.trace_id,
            }
            or self.candidate.trace_id != self.decision.trace_id
        ):
            raise ValueError("event trace_id does not match decision and candidate")
        if self.provider == "ollama":
            if self.model_role not in {
                LocalModelRole.REASONING,
                LocalModelRole.FAST,
            }:
                raise ValueError(
                    "Ollama risk event must retain a reasoning or fast model role"
                )
            if self.model != LOCAL_MODEL_BY_ROLE[self.model_role]:
                raise ValueError(
                    "risk event model does not match the approved local role"
                )
        elif (
            self.model_role is not None
            or self.model != DETERMINISTIC_BASELINE_MODEL
            or self.candidate.strategy_id != DETERMINISTIC_BASELINE_MODEL
        ):
            raise ValueError("deterministic baseline risk provenance is invalid")
        if self.decision.signal_id != self.candidate.signal_id:
            raise ValueError("decision signal_id does not match candidate")
        if self.decision.candidate_hash != self.candidate.content_hash():
            raise ValueError("decision is not bound to the candidate content")
        if (
            self.market.venue != self.candidate.venue
            or self.market.market_type != self.candidate.market_type
            or self.market.instrument != self.candidate.instrument
        ):
            raise ValueError("market identity does not match candidate")
        checksum = compute_payload_checksum(self.content_payload())
        if self.content_sha256 and self.content_sha256 != checksum:
            raise ValueError("content_sha256 does not match risk decision event")
        if not self.content_sha256:
            object.__setattr__(self, "content_sha256", checksum)
        expected_id = uuid5(
            NAMESPACE_URL,
            f"urn:trade-agent:risk-decision:v1:{checksum}",
        )
        if self.event_id not in {_ZERO_UUID, expected_id}:
            raise ValueError("event_id does not match risk decision content")
        if self.event_id == _ZERO_UUID:
            object.__setattr__(self, "event_id", expected_id)
        return self

    @classmethod
    def create(
        cls,
        candidate_event: CandidateForRiskEvent,
        decision: RiskDecision,
        *,
        created_at: datetime,
    ) -> RiskDecisionEvent:
        return cls(
            trace_id=candidate_event.trace_id,
            candidate_event_id=candidate_event.event_id,
            market_event_id=candidate_event.market_event_id,
            provider=candidate_event.provider,
            model_role=candidate_event.model_role,
            model=candidate_event.model,
            model_digest=candidate_event.model_digest,
            candidate=candidate_event.candidate,
            decision=decision,
            market=candidate_event.market,
            created_at=created_at,
        )

    def content_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "trace_id": self.trace_id,
            "candidate_event_id": self.candidate_event_id,
            "market_event_id": self.market_event_id,
            "provider": self.provider,
            "model_role": self.model_role,
            "model": self.model,
            "model_digest": self.model_digest,
            "candidate": self.candidate,
            "decision": self.decision,
            "market": asdict(self.market),
            "created_at": self.created_at,
        }

    def canonical_json(self) -> str:
        return canonical_json(self)


class OrderIntentEvent(_StrictEnvelope):
    """A durable paper/replay intent derived from one exact risk event."""

    schema_version: Literal["1.0"] = "1.0"
    event_id: UUID = _ZERO_UUID
    trace_id: str = Field(min_length=1, max_length=128)
    risk_event_id: UUID
    risk_decision_id: str = Field(min_length=1, max_length=128)
    intent: OrderIntent
    created_at: datetime
    content_sha256: str = Field(default="", pattern=r"^[0-9a-f]{64}$")

    @field_validator("created_at")
    @classmethod
    def normalize_created_at(cls, value: datetime) -> datetime:
        return _utc(value)

    @model_validator(mode="after")
    def bind_identity(self) -> OrderIntentEvent:
        if self.intent.decision_id != self.risk_decision_id:
            raise ValueError("intent does not match risk_decision_id")
        if self.intent.source_mode not in {
            SourceMode.REPLAY,
            SourceMode.PAPER_LIVE,
        }:
            raise ValueError("order intent event admits only replay or paper_live")
        checksum = compute_payload_checksum(self.content_payload())
        if self.content_sha256 and self.content_sha256 != checksum:
            raise ValueError("content_sha256 does not match order intent event")
        if not self.content_sha256:
            object.__setattr__(self, "content_sha256", checksum)
        expected_id = uuid5(
            NAMESPACE_URL,
            f"urn:trade-agent:order-intent:v1:{checksum}",
        )
        if self.event_id not in {_ZERO_UUID, expected_id}:
            raise ValueError("event_id does not match order intent content")
        if self.event_id == _ZERO_UUID:
            object.__setattr__(self, "event_id", expected_id)
        return self

    @classmethod
    def create(
        cls,
        risk_event: RiskDecisionEvent,
        intent: OrderIntent,
    ) -> OrderIntentEvent:
        if intent.decision_id != risk_event.decision.decision_id:
            raise ValueError("intent is not derived from the supplied risk event")
        return cls(
            trace_id=risk_event.trace_id,
            risk_event_id=risk_event.event_id,
            risk_decision_id=risk_event.decision.decision_id,
            intent=intent,
            created_at=intent.created_at,
        )

    def content_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "trace_id": self.trace_id,
            "risk_event_id": self.risk_event_id,
            "risk_decision_id": self.risk_decision_id,
            "intent": asdict(self.intent),
            "created_at": self.created_at,
        }

    def canonical_json(self) -> str:
        return canonical_json(self)


class OrderUpdatedEvent(_StrictEnvelope):
    """The immutable result snapshot emitted after deterministic paper execution."""

    schema_version: Literal["1.0"] = "1.0"
    event_id: UUID = _ZERO_UUID
    trace_id: str = Field(min_length=1, max_length=128)
    risk_event_id: UUID
    intent_event_id: UUID
    result: ExecutionResult
    created_at: datetime
    content_sha256: str = Field(default="", pattern=r"^[0-9a-f]{64}$")

    @field_validator("created_at")
    @classmethod
    def normalize_created_at(cls, value: datetime) -> datetime:
        return _utc(value)

    @model_validator(mode="after")
    def bind_identity(self) -> OrderUpdatedEvent:
        if self.result.intent.source_mode not in {
            SourceMode.REPLAY,
            SourceMode.PAPER_LIVE,
        }:
            raise ValueError("order update event admits only replay or paper_live")
        if self.result.order.intent_id != self.result.intent.intent_id:
            raise ValueError("order result is not bound to its intent")
        if any(
            fill.intent_id != self.result.intent.intent_id for fill in self.result.fills
        ):
            raise ValueError("fill result is not bound to its intent")
        checksum = compute_payload_checksum(self.content_payload())
        if self.content_sha256 and self.content_sha256 != checksum:
            raise ValueError("content_sha256 does not match order update event")
        if not self.content_sha256:
            object.__setattr__(self, "content_sha256", checksum)
        expected_id = uuid5(
            NAMESPACE_URL,
            f"urn:trade-agent:order-updated:v1:{checksum}",
        )
        if self.event_id not in {_ZERO_UUID, expected_id}:
            raise ValueError("event_id does not match order update content")
        if self.event_id == _ZERO_UUID:
            object.__setattr__(self, "event_id", expected_id)
        return self

    @classmethod
    def create(
        cls,
        risk_event: RiskDecisionEvent,
        intent_event: OrderIntentEvent,
        result: ExecutionResult,
        *,
        created_at: datetime,
    ) -> OrderUpdatedEvent:
        if result.intent != intent_event.intent:
            raise ValueError("execution result does not match the intent event")
        if intent_event.risk_event_id != risk_event.event_id:
            raise ValueError("intent event is not bound to the supplied risk event")
        return cls(
            trace_id=risk_event.trace_id,
            risk_event_id=risk_event.event_id,
            intent_event_id=intent_event.event_id,
            result=result,
            created_at=created_at,
        )

    def content_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "trace_id": self.trace_id,
            "risk_event_id": self.risk_event_id,
            "intent_event_id": self.intent_event_id,
            "result": asdict(self.result),
            "created_at": self.created_at,
        }

    def canonical_json(self) -> str:
        return canonical_json(self)
