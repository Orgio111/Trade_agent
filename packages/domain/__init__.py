"""Canonical domain models shared by workers and adapters."""

from .events import (
    CandlePayload,
    MarketEvent,
    SourceMode,
    canonical_json,
    compute_payload_checksum,
)

__all__ = [
    "CandlePayload",
    "MarketEvent",
    "SourceMode",
    "canonical_json",
    "compute_payload_checksum",
]

