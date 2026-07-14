"""Replay canonical JSONL candle events through the deterministic paper core."""

from __future__ import annotations

import argparse
from decimal import Decimal
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from packages.domain import MarketEvent
from packages.replay import ReplayHarness
from packages.risk import InstrumentConstraints, RiskPolicy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("events", type=Path, help="Canonical MarketEvent JSONL file")
    parser.add_argument("--venue", default="binance")
    parser.add_argument("--market-type", default="spot")
    parser.add_argument("--instrument", default="BTCUSDT")
    parser.add_argument("--tick-size", default="0.01")
    parser.add_argument("--step-size", default="0.001")
    parser.add_argument("--min-quantity", default="0.001")
    parser.add_argument("--min-notional", default="10")
    parser.add_argument("--equity", default="10000")
    return parser


def load_events(path: Path) -> list[MarketEvent]:
    """Load one event, a JSON array, or newline-delimited JSON events."""

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        decoded: Any = json.loads(text)
    except json.JSONDecodeError:
        decoded = [json.loads(line) for line in text.splitlines() if line.strip()]
    payloads = decoded if isinstance(decoded, list) else [decoded]
    return [MarketEvent.model_validate(payload) for payload in payloads]


def main() -> int:
    args = build_parser().parse_args()
    events = load_events(args.events)
    constraints = InstrumentConstraints(
        venue=args.venue,
        market_type=args.market_type,
        instrument=args.instrument,
        tick_size=Decimal(args.tick_size),
        step_size=Decimal(args.step_size),
        min_quantity=Decimal(args.min_quantity),
        min_notional=Decimal(args.min_notional),
    )
    summary = ReplayHarness(
        constraints=constraints,
        risk_policy=RiskPolicy(version="paper-v1"),
        initial_equity=Decimal(args.equity),
    ).run(events)
    print(
        json.dumps(
            {
                "accepted_market_events": summary.accepted_market_events,
                "rejected_market_events": summary.rejected_market_events,
                "candidates": summary.candidates,
                "approved_decisions": summary.approved_decisions,
                "rejected_decisions": summary.rejected_decisions,
                "orders": summary.orders,
                "fills": summary.fills,
                "reconciliation_clean": summary.reconciliation_clean,
                "trace_digest": summary.trace_digest,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
