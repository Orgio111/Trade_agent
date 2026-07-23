"""Operator DLQ replay must be explicit, allowlisted, and auditable."""

from __future__ import annotations

import pytest

from scripts.replay_dlq import ReplayRequest


def test_replay_request_requires_identity_reason_and_exact_allowlist() -> None:
    request = ReplayRequest(
        event_ids=("event-1",),
        operator="operator@example.test",
        reason="validated transient NATS outage",
        dry_run=True,
    )

    assert request.event_ids == ("event-1",)
    assert request.dry_run is True


@pytest.mark.parametrize(
    ("event_ids", "operator", "reason"),
    [
        ((), "operator@example.test", "validated transient NATS outage"),
        (("event-1",), "", "validated transient NATS outage"),
        (("event-1",), "operator@example.test", "short"),
        (("event-1", "event-1"), "operator@example.test", "validated duplicate list"),
    ],
)
def test_replay_request_rejects_ambiguous_authorization(
    event_ids: tuple[str, ...], operator: str, reason: str
) -> None:
    with pytest.raises(ValueError):
        ReplayRequest(
            event_ids=event_ids,
            operator=operator,
            reason=reason,
            dry_run=True,
        )
