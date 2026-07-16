"""Canonical deterministic risk boundary.

Domain policy/engine modules have no broker, network, model, or clock
dependency.  ``postgres_inputs`` is an explicit read-only infrastructure
adapter and never participates in risk calculations itself.
"""

from .engine import RiskEngine
from .policy import InstrumentConstraints, RiskPolicy
from .postgres_inputs import (
    PostgresRiskInputRepository,
    RiskInputAmbiguous,
    RiskInputError,
    RiskInputMissing,
    RiskInputStale,
    RiskInputUnavailable,
    VersionedInstrumentConstraints,
)
from .state import CandidateSignal, PortfolioState, RiskDecision, RiskReason

__all__ = [
    "CandidateSignal",
    "InstrumentConstraints",
    "PostgresRiskInputRepository",
    "PortfolioState",
    "RiskDecision",
    "RiskEngine",
    "RiskInputAmbiguous",
    "RiskInputError",
    "RiskInputMissing",
    "RiskInputStale",
    "RiskInputUnavailable",
    "RiskPolicy",
    "RiskReason",
    "VersionedInstrumentConstraints",
]
