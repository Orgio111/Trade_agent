"""Validated Ollama output to immutable risk-candidate envelope."""

from __future__ import annotations

from collections.abc import Awaitable
from datetime import datetime
from decimal import Decimal
import hashlib
import json
from typing import Literal, Protocol
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from packages.domain import SourceMode
from packages.execution import MarketSnapshot
from packages.local_ai import LOCAL_MODEL_BY_ROLE, LocalModelRole
from packages.risk import CandidateSignal
from workers.contracts import CandidateForRiskEvent
from workers.features import FeatureSnapshot


class CandidateModel(Protocol):
    def complete(self, prompt: str) -> Awaitable[str]: ...


class _RawCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_id: UUID
    symbol: Literal["BTCUSDT", "ETHUSDT"]
    side: Literal["BUY", "SELL", "HOLD"]
    order_type: Literal["MARKET", "LIMIT"]
    entry_price: Decimal | None
    stop_loss: Decimal | None
    take_profit: Decimal | None
    confidence: Decimal = Field(ge=Decimal("0"), le=Decimal("1"), allow_inf_nan=False)
    time_horizon_seconds: int = Field(gt=0, le=3600)
    strategy_id: str = Field(min_length=1, max_length=64)
    reason_codes: tuple[str, ...] = Field(min_length=1, max_length=8)


_PROMPT_PREFIX = """Return one JSON object only. No markdown, prose, or chain-of-thought.
Allowed side: BUY, SELL, HOLD. Allowed order_type: MARKET, LIMIT.
Use HOLD when evidence is weak. Never override risk or size positions.
Required schema keys: candidate_id, symbol, side, order_type, entry_price,
stop_loss, take_profit, confidence, time_horizon_seconds, strategy_id, reason_codes.
Feature state:\n"""


class TypedCandidateProducer:
    def __init__(
        self,
        model: CandidateModel,
        *,
        model_digest: str,
        max_prompt_chars: int = 8_000,
    ) -> None:
        if len(model_digest) != 64 or any(c not in "0123456789abcdef" for c in model_digest):
            raise ValueError("model_digest must be lowercase SHA-256")
        if not 1_000 <= max_prompt_chars <= 16_000:
            raise ValueError("max_prompt_chars must be between 1000 and 16000")
        self._model = model
        self._model_digest = model_digest
        self._max_prompt_chars = max_prompt_chars

    async def generate(
        self,
        feature: FeatureSnapshot,
        market: MarketSnapshot,
        *,
        generated_at: datetime,
        source_mode: SourceMode = SourceMode.PAPER_LIVE,
    ) -> CandidateForRiskEvent | None:
        if not feature.fresh or not feature.verify_checksum():
            return None
        if market.instrument != feature.symbol:
            return None
        prompt = _PROMPT_PREFIX + feature.model_dump_json(exclude={"checksum"})
        if len(prompt) > self._max_prompt_chars:
            return None
        try:
            raw = await self._model.complete(prompt)
            parsed = _RawCandidate.model_validate_json(raw)
        except (Exception, ValidationError, json.JSONDecodeError):
            return None
        if parsed.side == "HOLD" or parsed.symbol != feature.symbol:
            return None
        if parsed.entry_price is None or parsed.stop_loss is None:
            return None
        if parsed.entry_price <= 0 or parsed.stop_loss <= 0:
            return None
        if parsed.take_profit is not None and parsed.take_profit <= 0:
            return None
        age = Decimal(str((generated_at - market.observed_at).total_seconds()))
        if age < 0 or age > Decimal(parsed.time_horizon_seconds):
            return None
        spread_bps = (
            (market.ask - market.bid)
            / ((market.ask + market.bid) / Decimal("2"))
            * Decimal("10000")
        )
        identity = hashlib.sha256(
            (
                f"{feature.input_event_id}:{feature.checksum}:"
                f"{parsed.model_dump_json()}"
            ).encode()
        ).hexdigest()
        candidate = CandidateSignal(
            trace_id=f"candidate:{feature.input_event_id}",
            signal_id=f"sig_{identity[:24]}",
            strategy_id=parsed.strategy_id,
            strategy_version="1",
            venue=market.venue,
            market_type=market.market_type,
            instrument=parsed.symbol,
            side="buy" if parsed.side == "BUY" else "sell",
            reference_price=parsed.entry_price,
            stop_price=parsed.stop_loss,
            take_profit_price=parsed.take_profit,
            confidence=parsed.confidence,
            spread_bps=spread_bps,
            # PaperBroker market orders settle exactly at top-of-book. Relative
            # to midpoint, deterministic paper slippage is half the spread.
            expected_slippage_bps=spread_bps / Decimal("2"),
            data_age_seconds=age,
            source_mode=source_mode,
            requested_risk_fraction=Decimal("0.005"),
        )
        return CandidateForRiskEvent(
            trace_id=candidate.trace_id,
            market_event_id=feature.input_event_id,
            model_role=LocalModelRole.REASONING,
            model=LOCAL_MODEL_BY_ROLE[LocalModelRole.REASONING],
            model_digest=self._model_digest,
            candidate=candidate,
            market=market,
            created_at=generated_at,
        )
