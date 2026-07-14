"""Locations and helpers for language-neutral event contract artifacts."""

from pathlib import Path

CONTRACT_ROOT = Path(__file__).resolve().parent
SCHEMA_ROOT = CONTRACT_ROOT / "schemas"
FIXTURE_ROOT = CONTRACT_ROOT / "fixtures"


def schema_path(name: str = "market-event-1.0.schema.json") -> Path:
    """Return a checked path to a bundled schema without accepting traversal."""

    path = SCHEMA_ROOT / name
    if path.parent != SCHEMA_ROOT or not path.is_file():
        raise FileNotFoundError(name)
    return path


def fixture_path(name: str = "valid-candle-event-1.0.json") -> Path:
    """Return a checked path to a bundled golden fixture."""

    path = FIXTURE_ROOT / name
    if path.parent != FIXTURE_ROOT or not path.is_file():
        raise FileNotFoundError(name)
    return path


__all__ = ["CONTRACT_ROOT", "FIXTURE_ROOT", "SCHEMA_ROOT", "fixture_path", "schema_path"]

