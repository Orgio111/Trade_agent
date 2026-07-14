from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal

import pytest
from pydantic import ValidationError

from packages.domain.events import (
    CandlePayload,
    MarketEvent,
    SourceMode,
    canonical_json,
    compute_payload_checksum,
)
from packages.event_contracts import FIXTURE_ROOT, fixture_path, schema_path


def candle(**overrides: object) -> CandlePayload:
    values: dict[str, object] = {
        "interval": "1m",
        "open_time": datetime(2026, 7, 15, 0, 0, tzinfo=UTC),
        "close_time": datetime(2026, 7, 15, 0, 1, tzinfo=UTC),
        "open": "60000.10",
        "high": "60100.25",
        "low": "59950.00",
        "close": "60080.50",
        "volume": "12.345",
        "quote_volume": "741234.56",
        "trade_count": 321,
    }
    values.update(overrides)
    return CandlePayload.model_validate(values)


def event(**overrides: object) -> MarketEvent:
    values: dict[str, object] = {
        "trace_id": "trace-fixture-001",
        "event_type": "market.candle.closed",
        "venue": "binance",
        "market_type": "spot",
        "instrument_id": "BTCUSDT",
        "exchange_ts": datetime(2026, 7, 15, 0, 1, tzinfo=UTC),
        "received_ts": datetime(2026, 7, 15, 0, 1, 0, 125000, tzinfo=UTC),
        "sequence_start": 123456,
        "sequence_end": 123456,
        "source_mode": SourceMode.REPLAY,
        "ingest_run_id": "fixture-run-001",
        "payload": candle(),
    }
    values.update(overrides)
    return MarketEvent.create(**values)  # type: ignore[arg-type]


def test_source_modes_are_the_complete_provenance_set() -> None:
    assert {mode.value for mode in SourceMode} == {"historical", "replay", "paper_live", "live"}


def test_candle_payload_is_decimal_safe_and_does_not_duplicate_envelope_identity() -> None:
    payload = candle()

    assert payload.open == Decimal("60000.10")
    encoded = json.loads(payload.model_dump_json())
    assert encoded["open"] == "60000.1"
    assert encoded["low"] == "59950"
    assert "venue" not in encoded
    assert "instrument_id" not in encoded


@pytest.mark.parametrize(
    ("overrides", "error_fragment"),
    [
        ({"high": "59999"}, "high must be"),
        ({"low": "60090"}, "low must be"),
        ({"volume": "-0.1"}, "volumes must be"),
        ({"close_time": datetime(2026, 7, 15, 0, 0, tzinfo=UTC)}, "close_time must be"),
        ({"open": "NaN"}, "finite"),
    ],
)
def test_candle_payload_rejects_invalid_market_data(
    overrides: dict[str, object], error_fragment: str
) -> None:
    with pytest.raises(ValidationError, match=error_fragment):
        candle(**overrides)


def test_candle_payload_rejects_identity_duplication_and_naive_time() -> None:
    data = candle().model_dump(mode="python")
    data["venue"] = "binance"
    with pytest.raises(ValidationError, match="Extra inputs"):
        CandlePayload.model_validate(data)

    with pytest.raises(ValidationError, match="timezone-aware"):
        candle(open_time=datetime(2026, 7, 15, 0, 0))


def test_factory_creates_deterministic_id_checksum_and_canonical_json() -> None:
    first = event()
    duplicate = event(received_ts=first.received_ts + timedelta(milliseconds=10))

    assert first.event_id == duplicate.event_id
    assert first.payload_checksum == compute_payload_checksum(first.payload)
    assert first.verify_checksum()
    assert canonical_json(candle(open="60000.1000")) == canonical_json(candle(open="60000.1"))
    assert first.canonical_json() == canonical_json(json.loads(first.canonical_json()))


def test_golden_fixture_round_trips_byte_for_byte() -> None:
    raw = fixture_path().read_text(encoding="utf-8")
    fixture = json.loads(raw)
    parsed = MarketEvent.model_validate(fixture)

    assert parsed == event()
    assert parsed.canonical_json() == canonical_json(fixture)
    assert str(parsed.event_id) == "5835eba0-31a0-58a0-a2c5-19a0f534eccf"
    assert parsed.payload_checksum == "b3a7dc8afe1bd4e19319391a7088323f3d18ba2664f195d902b9c48022c1cc19"


def test_tampered_fixture_fails_closed_on_checksum() -> None:
    fixture = json.loads((FIXTURE_ROOT / "tampered-candle-event-1.0.json").read_text(encoding="utf-8"))

    with pytest.raises(ValidationError, match="payload_checksum does not match"):
        MarketEvent.model_validate(fixture)


def test_market_event_normalizes_offset_times_to_utc() -> None:
    plus_eight = timezone(timedelta(hours=8))
    result = event(
        exchange_ts=datetime(2026, 7, 15, 8, 1, tzinfo=plus_eight),
        received_ts=datetime(2026, 7, 15, 8, 1, 0, 125000, tzinfo=plus_eight),
    )

    assert result.exchange_ts == datetime(2026, 7, 15, 0, 1, tzinfo=UTC)
    assert result.received_ts.utcoffset() == timedelta(0)


def test_market_event_rejects_bad_sequences_flags_and_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="sequence_end"):
        event(sequence_start=5, sequence_end=4)
    for invalid_flag in ("stale", "1_BAD"):
        with pytest.raises(ValidationError, match="quality flags"):
            event(quality_flags=(invalid_flag,))

    dumped = event().model_dump(mode="python")
    dumped["unexpected"] = True
    with pytest.raises(ValidationError, match="Extra inputs"):
        MarketEvent.model_validate(dumped)


def test_language_neutral_schemas_are_versioned_and_strict() -> None:
    market_schema = json.loads(schema_path().read_text(encoding="utf-8"))
    candle_schema = json.loads(schema_path("candle-payload-1.0.schema.json").read_text(encoding="utf-8"))

    assert market_schema["$schema"].endswith("2020-12/schema")
    assert market_schema["properties"]["schema_version"] == {"const": "1.0"}
    assert market_schema["properties"]["source_mode"]["enum"] == [
        "historical",
        "replay",
        "paper_live",
        "live",
    ]
    assert market_schema["additionalProperties"] is False
    assert candle_schema["additionalProperties"] is False
    assert "venue" not in candle_schema["properties"]
    assert "instrument_id" not in candle_schema["properties"]


def test_contract_path_helpers_do_not_allow_traversal() -> None:
    with pytest.raises(FileNotFoundError):
        schema_path("../fixtures/valid-candle-event-1.0.json")
    with pytest.raises(FileNotFoundError):
        fixture_path("../schemas/market-event-1.0.schema.json")
