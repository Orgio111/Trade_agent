"""Canonical paper risk authority bootstrap."""

from decimal import Decimal

import pytest

from workers.reconciliation.authority import constraints_from_exchange_info


def test_exchange_info_filters_create_exact_spot_constraints() -> None:
    payload = {
        "symbol": "BTCUSDT",
        "status": "TRADING",
        "isSpotTradingAllowed": True,
        "filters": [
            {"filterType": "PRICE_FILTER", "tickSize": "0.01000000"},
            {
                "filterType": "LOT_SIZE",
                "stepSize": "0.00001000",
                "minQty": "0.00001000",
            },
            {"filterType": "NOTIONAL", "minNotional": "5.00000000"},
        ],
    }

    result = constraints_from_exchange_info(payload, expected_symbol="BTCUSDT")

    assert result.tick_size == Decimal("0.01000000")
    assert result.step_size == Decimal("0.00001000")
    assert result.min_quantity == Decimal("0.00001000")
    assert result.min_notional == Decimal("5.00000000")


@pytest.mark.parametrize(
    "change",
    [
        {"symbol": "ETHUSDT"},
        {"status": "BREAK"},
        {"isSpotTradingAllowed": False},
        {"filters": []},
    ],
)
def test_invalid_or_cross_symbol_exchange_info_fails_closed(change) -> None:
    payload = {
        "symbol": "BTCUSDT",
        "status": "TRADING",
        "isSpotTradingAllowed": True,
        "filters": [
            {"filterType": "PRICE_FILTER", "tickSize": "0.01"},
            {"filterType": "LOT_SIZE", "stepSize": "0.001", "minQty": "0.001"},
            {"filterType": "MIN_NOTIONAL", "minNotional": "10"},
        ],
    }
    payload.update(change)

    with pytest.raises(ValueError):
        constraints_from_exchange_info(payload, expected_symbol="BTCUSDT")