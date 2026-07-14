"""Validate canonical market events before they enter the decision path."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from packages.data_quality import (
    QualityVerdict,
    validate_candle,
    validate_sequence,
    validate_timestamps,
)
from packages.domain.events import MarketEvent
from packages.event_bus import CoreSubject, InMemoryEventBus


@dataclass(frozen=True, slots=True)
class MarketDataResult:
    event: MarketEvent
    verdict: QualityVerdict


class MarketDataWorker:
    """Fail-closed validation with per-stream sequence memory."""

    def __init__(
        self,
        bus: InMemoryEventBus,
        *,
        max_staleness: timedelta = timedelta(seconds=5),
        max_transport_lag: timedelta = timedelta(seconds=2),
    ) -> None:
        self.bus = bus
        self.max_staleness = max_staleness
        self.max_transport_lag = max_transport_lag
        self._last_sequence: dict[tuple[str, str, str], int] = {}

    def process(self, event: MarketEvent, *, now: datetime) -> MarketDataResult:
        verdicts = [
            validate_candle(event.payload),
            validate_timestamps(
                event.exchange_ts,
                event.received_ts,
                now=now,
                max_staleness=self.max_staleness,
                max_transport_lag=self.max_transport_lag,
            ),
        ]
        key = (event.venue, event.instrument_id, event.event_type)
        if event.sequence_start is not None or event.sequence_end is not None:
            verdicts.append(
                validate_sequence(
                    self._last_sequence.get(key),
                    event.sequence_start,
                    event.sequence_end,
                    snapshot=key not in self._last_sequence,
                )
            )
        verdict = QualityVerdict.combine(*verdicts)
        result = MarketDataResult(event=event, verdict=verdict)
        if verdict.accepted:
            if event.sequence_end is not None:
                self._last_sequence[key] = event.sequence_end
            self.bus.publish(CoreSubject.MARKET_VALIDATED, event)
        else:
            self.bus.publish(CoreSubject.MARKET_REJECTED, result)
        return result

