"""Strict Ollama candidate boundary tests."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from uuid import UUID

import pytest

from packages.execution import MarketSnapshot
from workers.candidate.runtime import TypedCandidateProducer
from workers.features import FeatureSnapshot


NOW = datetime(2026, 7, 15, 1, 0, tzinfo=UTC)
MARKET_ID = UUID("0cab4639-f4f7-42ca-b89f-4217051c1a85")


class FakeModel:
    def __init__(self, response: str | Exception) -> None:
        self.response = response

    async def complete(self, prompt: str) -> str:
        assert len(prompt) <= 8_000
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def feature(*, fresh: bool = True) -> FeatureSnapshot:
    return FeatureSnapshot(
        input_event_id=MARKET_ID,
        state_version=20,
        symbol="BTCUSDT",
        observed_at=NOW,
        candle_count=20,
        session_high="105",
        session_low="95",
        session_vwap="100",
        atr="2",
        atr_median="1.5",
        volatility="0.01",
        ema_9="102",
        ema_21="101",
        ema_50="99",
        rsi_14="60",
        trend_direction="up",
        trend_strength="0.03",
        support="98",
        resistance="104",
        liquidity_proxy="1000",
        market_data_age_seconds="1" if fresh else "120",
        fresh=fresh,
    )


def market() -> MarketSnapshot:
    return MarketSnapshot(
        venue="binance",
        market_type="spot",
        instrument="BTCUSDT",
        bid=Decimal("99.99"),
        ask=Decimal("100.01"),
        last=Decimal("100"),
        observed_at=NOW,
    )


VALID = """{
  "candidate_id":"2ccad15f-c12b-49bf-b7a4-fd035c837cbb",
  "symbol":"BTCUSDT",
  "side":"BUY",
  "order_type":"MARKET",
  "entry_price":"100",
  "stop_loss":"99",
  "take_profit":"102",
  "confidence":"0.8",
  "time_horizon_seconds":60,
  "strategy_id":"ollama-feature-v1",
  "reason_codes":["TREND_UP","RSI_OK"]
}"""


@pytest.mark.asyncio
async def test_valid_json_becomes_integrity_bound_candidate() -> None:
    producer = TypedCandidateProducer(FakeModel(VALID), model_digest="a" * 64)

    event = await producer.generate(
        feature(), market(), generated_at=NOW + timedelta(seconds=1)
    )

    assert event is not None
    assert event.market_event_id == MARKET_ID
    assert event.candidate.side == "buy"
    assert event.candidate.requested_risk_fraction == Decimal("0.005")
    assert event.candidate.spread_bps == Decimal("2")
    # Canonical PaperBroker fills at top-of-book, so midpoint slippage is
    # exactly half the authoritative spread rather than a model estimate.
    assert event.candidate.expected_slippage_bps == Decimal("1")
    assert event.model == "qwen3:8b"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        "not json",
        VALID[:-1] + ',"unsafe_override":true}',
        VALID.replace('"BUY"', '"HOLD"'),
        VALID.replace('"BTCUSDT"', '"BNBUSDT"'),
        TimeoutError("model timeout"),
    ],
)
async def test_invalid_hold_unsupported_or_failed_model_produces_no_candidate(
    response: str | Exception,
) -> None:
    producer = TypedCandidateProducer(FakeModel(response), model_digest="a" * 64)

    assert (
        await producer.generate(feature(), market(), generated_at=NOW + timedelta(seconds=1))
        is None
    )


@pytest.mark.asyncio
async def test_stale_feature_never_calls_model() -> None:
    class ForbiddenModel:
        async def complete(self, _prompt: str) -> str:
            raise AssertionError("stale state must not reach Ollama")

    producer = TypedCandidateProducer(ForbiddenModel(), model_digest="a" * 64)

    assert await producer.generate(feature(fresh=False), market(), generated_at=NOW) is None
