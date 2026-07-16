"""A deliberately simple baseline used to prove the core plumbing.

This strategy is not approved alpha. Its purpose is to create reproducible
candidate signals so data, risk, execution, ledger, and replay behavior can be
tested end to end.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, localcontext
import hashlib
from typing import Literal

from packages.domain.events import MarketEvent
from packages.risk import CandidateSignal, InstrumentConstraints


@dataclass(frozen=True, slots=True)
class BaselineStrategyConfig:
    strategy_id: str = "baseline_sma"
    version: str = "1.0.0"
    fast_window: int = 3
    slow_window: int = 5
    stop_distance_bps: Decimal = Decimal("100")
    take_profit_distance_bps: Decimal = Decimal("200")
    requested_risk_fraction: Decimal = Decimal("0.005")

    def __post_init__(self) -> None:
        if self.fast_window < 1 or self.slow_window <= self.fast_window:
            raise ValueError("strategy requires 1 <= fast_window < slow_window")
        if self.stop_distance_bps <= 0 or self.take_profit_distance_bps <= 0:
            raise ValueError("stop and take-profit distances must be positive")
        if not (Decimal("0") < self.requested_risk_fraction <= Decimal("1")):
            raise ValueError("requested_risk_fraction must be in (0, 1]")


class BaselineSmaStrategy:
    """Emit a candidate when fast and slow close averages disagree."""

    def __init__(self, config: BaselineStrategyConfig | None = None) -> None:
        self.config = config or BaselineStrategyConfig()
        self._closes: dict[str, deque[Decimal]] = defaultdict(
            lambda: deque(maxlen=self.config.slow_window)
        )
        self._last_side: dict[str, str] = {}

    def on_market_event(
        self,
        event: MarketEvent,
        constraints: InstrumentConstraints,
    ) -> CandidateSignal | None:
        with localcontext() as context:
            context.prec = 50
            return self._on_market_event(event, constraints)

    def _on_market_event(
        self,
        event: MarketEvent,
        constraints: InstrumentConstraints,
    ) -> CandidateSignal | None:
        if event.event_type != "market.candle":
            return None
        if event.instrument_id != constraints.instrument:
            raise ValueError("event and instrument constraints do not match")

        closes = self._closes[event.instrument_id]
        closes.append(event.payload.close)
        if len(closes) < self.config.slow_window:
            return None

        close_values = tuple(closes)
        fast = sum(close_values[-self.config.fast_window :], Decimal("0")) / Decimal(
            self.config.fast_window
        )
        slow = sum(close_values, Decimal("0")) / Decimal(self.config.slow_window)
        if fast == slow:
            self._last_side.pop(event.instrument_id, None)
            return None

        side: Literal["buy", "sell"] = "buy" if fast > slow else "sell"
        if self._last_side.get(event.instrument_id) == side:
            return None
        self._last_side[event.instrument_id] = side
        reference = self._round_to_tick(
            event.payload.close,
            constraints.tick_size,
            ROUND_FLOOR,
        )
        stop_fraction = self.config.stop_distance_bps / Decimal("10000")
        target_fraction = self.config.take_profit_distance_bps / Decimal("10000")
        if side == "buy":
            stop = self._round_to_tick(
                reference * (Decimal("1") - stop_fraction),
                constraints.tick_size,
                ROUND_FLOOR,
            )
            target = self._round_to_tick(
                reference * (Decimal("1") + target_fraction),
                constraints.tick_size,
                ROUND_CEILING,
            )
        else:
            stop = self._round_to_tick(
                reference * (Decimal("1") + stop_fraction),
                constraints.tick_size,
                ROUND_CEILING,
            )
            target = self._round_to_tick(
                reference * (Decimal("1") - target_fraction),
                constraints.tick_size,
                ROUND_FLOOR,
            )
        if stop <= 0 or target <= 0:
            return None

        relative_gap = abs(fast - slow) / reference
        confidence = min(
            Decimal("0.85"),
            Decimal("0.55") + relative_gap * Decimal("20"),
        )
        identity = (
            f"{self.config.strategy_id}:{self.config.version}:"
            f"{event.event_id}:{side}"
        )
        signal_id = "sig_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
        return CandidateSignal(
            trace_id=event.trace_id,
            signal_id=signal_id,
            strategy_id=self.config.strategy_id,
            strategy_version=self.config.version,
            venue=event.venue,
            market_type=event.market_type,
            instrument=event.instrument_id,
            side=side,
            reference_price=reference,
            stop_price=stop,
            take_profit_price=target,
            confidence=confidence,
            spread_bps=None,
            expected_slippage_bps=None,
            data_age_seconds=None,
            source_mode=event.source_mode,
            requested_risk_fraction=self.config.requested_risk_fraction,
        )

    @staticmethod
    def _round_to_tick(
        price: Decimal,
        tick_size: Decimal,
        rounding: str,
    ) -> Decimal:
        ticks = (price / tick_size).to_integral_value(rounding=rounding)
        return ticks * tick_size
