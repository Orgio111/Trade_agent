"""Versioned subjects used by the deterministic core."""

from enum import StrEnum


class CoreSubject(StrEnum):
    MARKET_VALIDATED = "market.validated.v1"
    MARKET_REJECTED = "market.rejected.v1"
    SIGNAL_CANDIDATE = "signals.candidate.v1"
    RISK_APPROVED = "risk.approved.v1"
    RISK_REJECTED = "risk.rejected.v1"
    ORDER_INTENT = "orders.intent.v1"
    ORDER_UPDATED = "orders.updated.v1"

