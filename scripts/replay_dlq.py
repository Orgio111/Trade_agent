"""Authorize exact dead-letter outbox events for bounded dispatcher replay."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
import json
import os
import sys
from typing import Any, Sequence

import asyncpg  # type: ignore[import-untyped]


@dataclass(frozen=True, slots=True)
class ReplayRequest:
    event_ids: tuple[str, ...]
    operator: str
    reason: str
    dry_run: bool = True

    def __post_init__(self) -> None:
        if not self.event_ids or len(self.event_ids) > 100:
            raise ValueError("event allowlist must contain between 1 and 100 IDs")
        if len(set(self.event_ids)) != len(self.event_ids):
            raise ValueError("event allowlist contains duplicates")
        if any(not event_id.strip() or len(event_id) > 128 for event_id in self.event_ids):
            raise ValueError("event IDs must be non-empty and at most 128 characters")
        if not self.operator.strip() or len(self.operator) > 128:
            raise ValueError("operator identity is required and at most 128 characters")
        if len(self.reason.strip()) < 10 or len(self.reason) > 512:
            raise ValueError("replay reason must be between 10 and 512 characters")


async def authorize_replay(database_url: str, request: ReplayRequest) -> list[dict[str, Any]]:
    connection = await asyncpg.connect(database_url)
    try:
        async with connection.transaction():
            rows = await connection.fetch(
                """
                SELECT d.original_event_id, d.original_subject, d.payload_hash,
                       d.consumer, d.retry_count, d.replayed_at,
                       o.status, o.payload_checksum
                FROM dead_letter_events d
                JOIN event_outbox o ON o.event_id = d.original_event_id
                WHERE d.original_event_id = ANY($1::text[])
                ORDER BY d.original_event_id
                FOR UPDATE OF d, o
                """,
                list(request.event_ids),
            )
            found = {str(row["original_event_id"]) for row in rows}
            missing = sorted(set(request.event_ids) - found)
            if missing:
                raise ValueError("events not found in DLQ: " + ",".join(missing))
            for row in rows:
                if row["status"] != "dead_letter":
                    raise ValueError(f"event is not dead_letter: {row['original_event_id']}")
                if row["replayed_at"] is not None:
                    raise ValueError(f"event was already replayed: {row['original_event_id']}")
                if row["payload_hash"] != row["payload_checksum"]:
                    raise ValueError(f"payload checksum mismatch: {row['original_event_id']}")

            evidence = [dict(row) for row in rows]
            if request.dry_run:
                return evidence

            for event_id in request.event_ids:
                await connection.execute(
                    """
                    UPDATE event_outbox
                    SET status = 'pending', publish_attempts = 0,
                        next_attempt_at = NOW(), lease_owner = NULL,
                        lease_expires_at = NULL, last_error_code = NULL
                    WHERE event_id = $1 AND status = 'dead_letter'
                    """,
                    event_id,
                )
                await connection.execute(
                    """
                    UPDATE dead_letter_events
                    SET replayed_at = NOW(), replayed_by = $2, replay_reason = $3
                    WHERE original_event_id = $1 AND replayed_at IS NULL
                    """,
                    event_id,
                    request.operator,
                    request.reason,
                )
                await connection.execute(
                    """
                    INSERT INTO replay_audit_log (
                        original_event_id, operator_identity, reason, dry_run,
                        payload_hash, consumer
                    )
                    SELECT original_event_id, $2, $3, FALSE, payload_hash, consumer
                    FROM dead_letter_events WHERE original_event_id = $1
                    """,
                    event_id,
                    request.operator,
                    request.reason,
                )
            return evidence
    finally:
        await connection.close()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--event-id", action="append", required=True, dest="event_ids")
    result.add_argument("--operator", required=True)
    result.add_argument("--reason", required=True)
    result.add_argument(
        "--execute",
        action="store_true",
        help="authorize replay; default is dry-run and never mutates state",
    )
    result.add_argument("--database-url", default=os.getenv("DATABASE_URL"))
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if not args.database_url:
        print("DATABASE_URL is required", file=sys.stderr)
        return 2
    try:
        request = ReplayRequest(
            event_ids=tuple(args.event_ids),
            operator=args.operator,
            reason=args.reason,
            dry_run=not args.execute,
        )
        rows = asyncio.run(authorize_replay(args.database_url, request))
    except (ValueError, asyncpg.PostgresError) as exc:
        print(f"replay denied: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"dry_run": request.dry_run, "authorized": len(rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
