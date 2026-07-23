"""Deterministic paper-ledger portfolio projection."""

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from workers.reconciliation.projection import (
    FillRow,
    PaperPortfolioProjector,
    project_position,
)


NOW = datetime(2026, 7, 22, 8, 0, tzinfo=UTC)


def fill(
    fill_id: str,
    *,
    side: str,
    quantity: str,
    price: str,
    fee: str = "0",
    symbol: str = "BTCUSDT",
) -> FillRow:
    return FillRow(
        fill_id=fill_id,
        instrument=symbol,
        side=side,
        quantity=Decimal(quantity),
        price=Decimal(price),
        fee=Decimal(fee),
        occurred_at=NOW,
    )


def test_empty_ledger_projects_initial_cash_with_switch_off() -> None:
    state = PaperPortfolioProjector(Decimal("10000")).project(
        account_id="paper-main", fills=(), marks={}, reconciled_at=NOW
    )

    assert state.cash == Decimal("10000")
    assert state.equity == Decimal("10000")
    assert state.open_positions == 0
    assert state.gross_exposure == 0
    assert state.kill_switch_active is False


def test_buy_then_partial_sell_projects_cash_position_and_equity() -> None:
    state = PaperPortfolioProjector(Decimal("10000")).project(
        account_id="paper-main",
        fills=(
            fill("buy-1", side="buy", quantity="2", price="100", fee="1"),
            fill("sell-1", side="sell", quantity="0.5", price="120", fee="0.5"),
        ),
        marks={"BTCUSDT": Decimal("110")},
        reconciled_at=NOW,
    )

    assert state.cash == Decimal("9858.5")
    assert state.equity == Decimal("10023.5")
    assert state.open_positions == 1
    assert state.gross_exposure == Decimal("165")
    assert state.daily_pnl == Decimal("23.5")
    assert state.weekly_pnl == Decimal("23.5")


def test_missing_mark_for_open_position_fails_closed() -> None:
    with pytest.raises(ValueError, match="mark"):
        PaperPortfolioProjector(Decimal("10000")).project(
            account_id="paper-main",
            fills=(fill("buy-1", side="buy", quantity="1", price="100"),),
            marks={},
            reconciled_at=NOW,
        )


def test_sell_larger_than_position_fails_closed() -> None:
    with pytest.raises(ValueError, match="negative"):
        PaperPortfolioProjector(Decimal("10000")).project(
            account_id="paper-main",
            fills=(fill("sell-1", side="sell", quantity="1", price="100"),),
            marks={"BTCUSDT": Decimal("100")},
            reconciled_at=NOW,
        )


def test_signed_position_flip_closes_exact_quantity_and_reopens_remainder() -> None:
    position = project_position(
        (
            fill("buy-1", side="buy", quantity="2", price="100", fee="1"),
            fill("sell-1", side="sell", quantity="3", price="120", fee="1.5"),
        )
    )

    assert position.quantity == Decimal("-1")
    assert position.average_entry_price == Decimal("120")
    assert position.realized_pnl == Decimal("40")
    assert position.fees == Decimal("2.5")


def test_signed_position_weighted_average_and_close_are_decimal_exact() -> None:
    position = project_position(
        (
            fill("buy-1", side="buy", quantity="1", price="100"),
            fill("buy-2", side="buy", quantity="2", price="130"),
            fill("sell-1", side="sell", quantity="3", price="120"),
        )
    )

    assert position.quantity == 0
    assert position.average_entry_price is None
    assert position.realized_pnl == Decimal("0")