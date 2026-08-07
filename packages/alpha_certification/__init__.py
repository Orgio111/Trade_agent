"""Offline alpha certification and signed promotion gates."""

from .data import (
    BinanceHistoricalDownloader,
    DataGapError,
    DatasetSnapshot,
    GapPolicy,
    HistoricalCandle,
    HistoricalRequest,
    load_snapshot,
)
from .promotion import (
    AlphaGateThresholds,
    PromotionDecision,
    PromotionRegistry,
    evaluate_alpha_gates,
)
from .shadow import (
    ShadowThresholds,
    create_shadow_state,
    evaluate_shadow_gates,
    finalize_shadow_state,
    record_shadow_sample,
)
from .walk_forward import (
    PurgedWalkForwardSplitter,
    WalkForwardConfig,
    WalkForwardFold,
)

__all__ = [
    "AlphaGateThresholds",
    "BinanceHistoricalDownloader",
    "DataGapError",
    "DatasetSnapshot",
    "GapPolicy",
    "HistoricalCandle",
    "HistoricalRequest",
    "PromotionDecision",
    "PromotionRegistry",
    "PurgedWalkForwardSplitter",
    "WalkForwardConfig",
    "WalkForwardFold",
    "ShadowThresholds",
    "create_shadow_state",
    "evaluate_alpha_gates",
    "evaluate_shadow_gates",
    "finalize_shadow_state",
    "load_snapshot",
    "record_shadow_sample",
]
