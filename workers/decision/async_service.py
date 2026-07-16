"""Deterministic, model-free authorization service for candidate signals.

The service is deliberately transport-neutral.  It verifies that the
candidate did not understate market freshness or spread, loads immutable
authoritative inputs, evaluates the pure hard-rule risk engine, and persists
the candidate plus verdict in one database transaction.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from decimal import Decimal, localcontext
from typing import Protocol
from uuid import UUID

from packages.domain import SourceMode
from packages.risk import (
    CandidateSignal,
    PortfolioState,
    RiskDecision,
    RiskEngine,
    RiskPolicy,
    VersionedInstrumentConstraints,
)
from workers.contracts import CandidateForRiskEvent, RiskDecisionEvent


_MICROSECONDS_PER_SECOND = Decimal("1000000")
_BASIS_POINTS_PER_UNIT = Decimal("10000")


class DecisionServiceError(RuntimeError):
    """Base class for failures that must never produce authorization."""


class CandidateIntegrityError(DecisionServiceError):
    """The immutable candidate envelope failed an authority-bound check."""


class CandidateMetricUnderstatement(CandidateIntegrityError):
    """A candidate claimed a safer metric than the market snapshot proves."""


class CandidateTimingError(CandidateIntegrityError):
    """Candidate and market timestamps cannot form a valid causal ordering."""


class UnsupportedCandidateMode(CandidateIntegrityError):
    """A candidate attempted to cross the paper/replay safety boundary."""


class DecisionAuthorityMismatch(DecisionServiceError):
    """The durable authority did not round-trip the exact computed verdict."""


class RiskInputRepository(Protocol):
    """Read-only inputs required by the deterministic risk engine."""

    async def get_active_policy(self) -> RiskPolicy: ...

    async def get_portfolio_state(
        self,
        account_id: str,
        *,
        as_of: datetime,
        max_age_seconds: Decimal | int | str,
        state_id: str | None = None,
    ) -> PortfolioState: ...

    async def get_instrument_constraints(
        self,
        *,
        venue: str,
        market_type: str,
        instrument: str,
        as_of: datetime,
        version: str | None = None,
    ) -> VersionedInstrumentConstraints: ...


class DecisionAuthority(Protocol):
    """Atomic candidate plus verdict persistence boundary."""

    async def get_for_signal(
        self,
        signal_id: str,
        *,
        candidate_hash: str,
        market_event_id: str,
        candidate_event_id: str,
    ) -> RiskDecision | None: ...

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
    ) -> RiskDecision: ...


class AsyncRiskDecisionService:
    """Authorize candidates with hard rules and immutable local state only."""

    def __init__(
        self,
        *,
        risk_inputs: RiskInputRepository,
        decision_authority: DecisionAuthority,
        account_id: str,
    ) -> None:
        resolved_account = account_id.strip()
        if not resolved_account:
            raise ValueError("account_id cannot be blank")
        self._risk_inputs = risk_inputs
        self._decision_authority = decision_authority
        self._account_id = resolved_account

    async def evaluate(
        self,
        candidate_event: CandidateForRiskEvent,
    ) -> RiskDecisionEvent:
        """Evaluate and persist one exact candidate deterministically.

        ``candidate_event.created_at`` is the evaluation instant.  It is part
        of the checksummed input, so a JetStream redelivery cannot create a
        different decision merely because wall-clock time advanced.
        """

        evaluated_at = candidate_event.created_at.astimezone(UTC)
        self._verify_mode(candidate_event.candidate.source_mode)
        self._verify_market_metrics(candidate_event, evaluated_at=evaluated_at)

        issued = await self._decision_authority.get_for_signal(
            candidate_event.candidate.signal_id,
            candidate_hash=candidate_event.candidate.content_hash(),
            market_event_id=str(candidate_event.market_event_id),
            candidate_event_id=str(candidate_event.event_id),
        )
        if issued is not None:
            if issued.account_id != self._account_id:
                raise DecisionAuthorityMismatch(
                    "persisted verdict belongs to a different account"
                )
            if issued.evaluated_at != evaluated_at:
                raise DecisionAuthorityMismatch(
                    "persisted verdict has a different deterministic evaluation time"
                )
            return RiskDecisionEvent.create(
                candidate_event,
                issued,
                created_at=evaluated_at,
            )

        policy = await self._risk_inputs.get_active_policy()
        portfolio, versioned_constraints = await asyncio.gather(
            self._risk_inputs.get_portfolio_state(
                self._account_id,
                as_of=evaluated_at,
                max_age_seconds=policy.max_portfolio_age_seconds,
            ),
            self._risk_inputs.get_instrument_constraints(
                venue=candidate_event.candidate.venue,
                market_type=candidate_event.candidate.market_type,
                instrument=candidate_event.candidate.instrument,
                as_of=evaluated_at,
            ),
        )
        decision = RiskEngine(policy).evaluate(
            candidate_event.candidate,
            portfolio,
            versioned_constraints.constraints,
            evaluated_at=evaluated_at,
        )
        issued = await self._decision_authority.record_candidate_and_decision(
            candidate_event.candidate,
            decision,
            portfolio_state=portfolio,
            market_event_id=str(candidate_event.market_event_id),
            candidate_event_id=str(candidate_event.event_id),
        )
        if issued != decision:
            raise DecisionAuthorityMismatch(
                "durable authority returned a different risk decision"
            )
        return RiskDecisionEvent.create(
            candidate_event,
            issued,
            created_at=evaluated_at,
        )

    @staticmethod
    def _verify_mode(mode: SourceMode) -> None:
        if mode not in {SourceMode.REPLAY, SourceMode.PAPER_LIVE}:
            raise UnsupportedCandidateMode(
                "decision service admits only replay or paper_live candidates"
            )

    @classmethod
    def _verify_market_metrics(
        cls,
        candidate_event: CandidateForRiskEvent,
        *,
        evaluated_at: datetime,
    ) -> None:
        authoritative_age = cls._elapsed_seconds(
            later=evaluated_at,
            earlier=candidate_event.market.observed_at,
        )
        if authoritative_age < 0:
            raise CandidateTimingError(
                "market observation is from the future relative to candidate creation"
            )
        declared_age = candidate_event.candidate.data_age_seconds
        if declared_age is not None and declared_age < authoritative_age:
            raise CandidateMetricUnderstatement(
                "candidate data_age_seconds understates authoritative market age"
            )

        with localcontext() as context:
            context.prec = 50
            midpoint = (candidate_event.market.bid + candidate_event.market.ask) / 2
            actual_spread = (
                (candidate_event.market.ask - candidate_event.market.bid)
                / midpoint
                * _BASIS_POINTS_PER_UNIT
            )
        declared_spread = candidate_event.candidate.spread_bps
        if declared_spread is not None and declared_spread < actual_spread:
            raise CandidateMetricUnderstatement(
                "candidate spread_bps understates authoritative bid/ask spread"
            )

    @staticmethod
    def _elapsed_seconds(*, later: datetime, earlier: datetime) -> Decimal:
        delta = later - earlier
        microseconds = (
            delta.days * 86_400_000_000 + delta.seconds * 1_000_000 + delta.microseconds
        )
        return Decimal(microseconds) / _MICROSECONDS_PER_SECOND


__all__ = [
    "AsyncRiskDecisionService",
    "CandidateIntegrityError",
    "CandidateMetricUnderstatement",
    "CandidateTimingError",
    "DecisionAuthorityMismatch",
    "DecisionServiceError",
    "UnsupportedCandidateMode",
]
