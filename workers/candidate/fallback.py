"""Explicit deterministic candidate provider for supervised paper acceptance."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_EVEN, localcontext
import hashlib
from typing import Literal

from packages.domain import SourceMode
from packages.execution import MarketSnapshot
from packages.risk import CandidateSignal
from workers.contracts import (
    CandidateForRiskEvent,
    DETERMINISTIC_BASELINE_MODEL,
)
from workers.features import FeatureSnapshot


_TICK_SIZE = {
    "BTCUSDT": Decimal("0.01"),
    "ETHUSDT": Decimal("0.01"),
}


def _round_to_tick(value: Decimal, tick: Decimal, rounding: str) -> Decimal:
    return (value / tick).to_integral_value(rounding=rounding) * tick


class DeterministicBaselineCandidateProducer:
    """Create a conservative plumbing candidate from an established trend.

    This is not approved alpha. It exists to prove the real public-data paper
    path without coercing an LLM to trade. Deterministic risk remains the only
    authority that can approve the candidate.
    """

    def __init__(
        self,
        *,
        artifact_digest: str,
        minimum_candles: int = 5,
    ) -> None:
        if (
            len(artifact_digest) != 64
            or any(character not in "0123456789abcdef" for character in artifact_digest)
        ):
            raise ValueError("artifact_digest must be lowercase SHA-256")
        if minimum_candles < 5 or minimum_candles > 200:
            raise ValueError("minimum_candles must be between 5 and 200")
        self._artifact_digest = artifact_digest
        self._minimum_candles = minimum_candles

    async def generate(
        self,
        feature: FeatureSnapshot,
        market: MarketSnapshot,
        *,
        generated_at: datetime,
        source_mode: SourceMode = SourceMode.PAPER_LIVE,
    ) -> CandidateForRiskEvent | None:
        if (
            not feature.fresh
            or not feature.verify_checksum()
            or feature.candle_count < self._minimum_candles
            or feature.trend_direction == "flat"
            or market.instrument != feature.symbol
        ):
            return None
        age = Decimal(str((generated_at - market.observed_at).total_seconds()))
        if age < 0 or age > Decimal("5"):
            return None
        with localcontext() as context:
            context.prec = 50
            raw_midpoint = (market.bid + market.ask) / Decimal("2")
            tick = _TICK_SIZE.get(feature.symbol)
            if tick is None:
                return None
            midpoint = _round_to_tick(raw_midpoint, tick, ROUND_HALF_EVEN)
            spread_bps = (
                (market.ask - market.bid) / raw_midpoint * Decimal("10000")
            )
            side: Literal["buy", "sell"] = (
                "buy" if feature.trend_direction == "up" else "sell"
            )
            stop_fraction = Decimal("0.01")
            target_fraction = Decimal("0.02")
            if side == "buy":
                stop = _round_to_tick(
                    midpoint * (Decimal("1") - stop_fraction),
                    tick,
                    ROUND_FLOOR,
                )
                target = _round_to_tick(
                    midpoint * (Decimal("1") + target_fraction),
                    tick,
                    ROUND_CEILING,
                )
            else:
                stop = _round_to_tick(
                    midpoint * (Decimal("1") + stop_fraction),
                    tick,
                    ROUND_CEILING,
                )
                target = _round_to_tick(
                    midpoint * (Decimal("1") - target_fraction),
                    tick,
                    ROUND_FLOOR,
                )
        identity = hashlib.sha256(
            (
                f"{feature.input_event_id}:{feature.checksum}:"
                f"{self._artifact_digest}:{side}"
            ).encode("utf-8")
        ).hexdigest()
        candidate = CandidateSignal(
            trace_id=f"baseline:{feature.input_event_id}",
            signal_id=f"sig_{identity[:24]}",
            strategy_id=DETERMINISTIC_BASELINE_MODEL,
            strategy_version="1.0.0",
            venue=market.venue,
            market_type=market.market_type,
            instrument=feature.symbol,
            side=side,
            reference_price=midpoint,
            stop_price=stop,
            take_profit_price=target,
            confidence=Decimal("0.60"),
            spread_bps=spread_bps,
            expected_slippage_bps=spread_bps / Decimal("2"),
            data_age_seconds=age,
            source_mode=source_mode,
            requested_risk_fraction=Decimal("0.0025"),
        )
        return CandidateForRiskEvent(
            trace_id=candidate.trace_id,
            market_event_id=feature.input_event_id,
            provider="deterministic_baseline",
            model_role=None,
            model=DETERMINISTIC_BASELINE_MODEL,
            model_digest=self._artifact_digest,
            candidate=candidate,
            market=market,
            created_at=generated_at,
        )
