"""Offline latency benchmark for the Hermes canonical Markdown boundary."""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from time import perf_counter_ns


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ""}:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from packages.hermes.models import JournalEvent, JournalEventKind  # noqa: E402
from packages.hermes.obsidian import ObsidianJournalStore  # noqa: E402


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int((len(ordered) - 1) * percentile))
    return ordered[index]


async def benchmark(event_count: int) -> dict[str, float | int]:
    if not 1 <= event_count <= 10_000:
        raise ValueError("event_count must be between 1 and 10000")
    with tempfile.TemporaryDirectory(prefix="trade-agent-hermes-benchmark-") as root:
        store = ObsidianJournalStore(Path(root))
        events = [
            JournalEvent.create(
                occurred_at=datetime(2026, 7, 16, tzinfo=UTC)
                + timedelta(microseconds=index),
                kind=JournalEventKind.BENCHMARK,
                agent_id="hermes-benchmark",
                title=f"Hermes journal benchmark {index}",
                body="Deterministic local Markdown projection benchmark.",
                metadata={"sequence": index},
            )
            for index in range(event_count)
        ]

        create_ms: list[float] = []
        for event in events:
            started = perf_counter_ns()
            receipt = await store.append(event)
            create_ms.append((perf_counter_ns() - started) / 1_000_000)
            if not receipt.created:
                raise RuntimeError("benchmark event was not created")

        replay_ms: list[float] = []
        for event in events:
            started = perf_counter_ns()
            receipt = await store.append(event)
            replay_ms.append((perf_counter_ns() - started) / 1_000_000)
            if receipt.created:
                raise RuntimeError("benchmark replay was not idempotent")

    return {
        "event_count": event_count,
        "create_p50_ms": round(statistics.median(create_ms), 3),
        "create_p95_ms": round(_percentile(create_ms, 0.95), 3),
        "create_max_ms": round(max(create_ms), 3),
        "replay_p50_ms": round(statistics.median(replay_ms), 3),
        "replay_p95_ms": round(_percentile(replay_ms, 0.95), 3),
        "replay_max_ms": round(max(replay_ms), 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark append-only Hermes Markdown writes without network I/O."
    )
    parser.add_argument("--events", type=int, default=250)
    parser.add_argument(
        "--max-create-p95-ms",
        type=float,
        default=25.0,
        help="Fail when create p95 exceeds this local control-plane budget.",
    )
    arguments = parser.parse_args()
    result = asyncio.run(benchmark(arguments.events))
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["create_p95_ms"] > arguments.max_create_p95_ms:
        raise SystemExit("Hermes journal create p95 exceeded the configured budget")


if __name__ == "__main__":
    main()
