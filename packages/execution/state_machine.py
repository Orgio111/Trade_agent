"""Explicit execution order lifecycle rules."""

from __future__ import annotations

from packages.execution.models import OrderStatus


class InvalidOrderTransition(ValueError):
    """Raised when a caller attempts an impossible order state transition."""


_ALLOWED_TRANSITIONS: dict[OrderStatus, frozenset[OrderStatus]] = {
    OrderStatus.PENDING_SUBMIT: frozenset(
        {
            OrderStatus.OPEN,
            OrderStatus.PARTIALLY_FILLED,
            OrderStatus.FILLED,
            OrderStatus.CANCELED,
            OrderStatus.REJECTED,
        }
    ),
    OrderStatus.OPEN: frozenset(
        {
            OrderStatus.PARTIALLY_FILLED,
            OrderStatus.FILLED,
            OrderStatus.CANCELED,
        }
    ),
    OrderStatus.PARTIALLY_FILLED: frozenset(
        {OrderStatus.FILLED, OrderStatus.CANCELED}
    ),
    OrderStatus.FILLED: frozenset(),
    OrderStatus.CANCELED: frozenset(),
    OrderStatus.REJECTED: frozenset(),
}


def allowed_transitions(status: OrderStatus) -> frozenset[OrderStatus]:
    return _ALLOWED_TRANSITIONS[OrderStatus(status)]


def assert_transition(current: OrderStatus, target: OrderStatus) -> None:
    """Validate a genuine state change.

    Re-applying the current state is treated as an idempotent no-op. This lets
    an at-least-once transport safely replay a status update without creating a
    second ledger event.
    """

    current = OrderStatus(current)
    target = OrderStatus(target)
    if target is current:
        return
    if target not in _ALLOWED_TRANSITIONS[current]:
        raise InvalidOrderTransition(f"cannot transition order from {current.value} to {target.value}")

