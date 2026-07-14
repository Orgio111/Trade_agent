"""Replay CLI accepts canonical contract artifacts without network access."""

from pathlib import Path

from scripts.replay_market import load_events


def test_load_pretty_printed_single_event_fixture() -> None:
    fixture = (
        Path(__file__).parents[2]
        / "packages"
        / "event_contracts"
        / "fixtures"
        / "valid-candle-event-1.0.json"
    )
    events = load_events(fixture)

    assert len(events) == 1
    assert events[0].verify_checksum()
