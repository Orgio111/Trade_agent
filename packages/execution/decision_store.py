"""Append-only authorization boundary for issued risk decisions."""

from __future__ import annotations

from threading import RLock

from packages.risk import RiskDecision


class DecisionAuthorizationError(RuntimeError):
    """A decision is missing, conflicting, or not approved."""


class InMemoryDecisionStore:
    """Reference store; production must implement this contract in PostgreSQL."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._decisions: dict[str, RiskDecision] = {}

    def record(self, decision: RiskDecision) -> RiskDecision:
        with self._lock:
            existing = self._decisions.get(decision.decision_id)
            if existing is not None and existing != decision:
                raise DecisionAuthorizationError(
                    "decision_id already exists with different immutable content"
                )
            if existing is None:
                self._decisions[decision.decision_id] = decision
            return decision

    def require_exact_approval(self, decision: RiskDecision) -> RiskDecision:
        with self._lock:
            issued = self._decisions.get(decision.decision_id)
            if issued is None:
                raise DecisionAuthorizationError(
                    "risk decision was not issued by the configured authority"
                )
            if issued != decision:
                raise DecisionAuthorizationError(
                    "risk decision content differs from the issued record"
                )
            if not issued.approved:
                raise DecisionAuthorizationError("issued risk decision is not approved")
            return issued

    def get(self, decision_id: str) -> RiskDecision | None:
        with self._lock:
            return self._decisions.get(decision_id)

    def __len__(self) -> int:
        with self._lock:
            return len(self._decisions)
