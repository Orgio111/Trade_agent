"""Transactional PostgreSQL authority for incremental feature state."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from packages.domain import MarketEvent, canonical_json
from packages.event_bus import CoreSubject
from packages.execution._postgres import PostgresPool

from .runtime import FeatureSnapshot, IncrementalFeatureEngine


class PostgresFeatureRepository:
    def __init__(self, pool: PostgresPool, *, stream_name: str, durable_name: str) -> None:
        self._pool = pool
        self._stream_name = stream_name
        self._durable_name = durable_name

    async def process(self, event: MarketEvent) -> FeatureSnapshot:
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                await connection.execute(
                    "SELECT pg_advisory_xact_lock(hashtext($1))", event.instrument_id
                )
                inserted = await connection.fetchval(
                    """
                    INSERT INTO event_inbox (
                        stream_name, durable_name, event_id, subject,
                        stream_sequence, payload, payload_checksum, processed_at
                    ) VALUES ($1, $2, $3, $4, NULL, $5::jsonb, $6, NOW())
                    ON CONFLICT (stream_name, durable_name, event_id) DO NOTHING
                    RETURNING event_id
                    """,
                    self._stream_name,
                    self._durable_name,
                    str(event.event_id),
                    CoreSubject.MARKET_VALIDATED.value,
                    event.canonical_json(),
                    event.payload_checksum,
                )
                row = await connection.fetchrow(
                    """
                    SELECT checkpoint, checkpoint_checksum, feature_snapshot
                    FROM feature_checkpoints WHERE symbol = $1
                    """,
                    event.instrument_id,
                )
                if inserted is None:
                    if row is None:
                        raise RuntimeError("feature inbox exists without checkpoint")
                    return FeatureSnapshot.model_validate(row["feature_snapshot"])
                engine = self._restore(row)
                snapshot = engine.update(event)
                checkpoint = engine.checkpoint_json()
                checkpoint_checksum = hashlib.sha256(checkpoint.encode()).hexdigest()
                snapshot_json = snapshot.canonical_json()
                await connection.execute(
                    """
                    INSERT INTO feature_checkpoints (
                        symbol, state_version, last_market_event_id, checkpoint,
                        checkpoint_checksum, feature_snapshot, feature_checksum
                    ) VALUES ($1, $2, $3, $4::jsonb, $5, $6::jsonb, $7)
                    ON CONFLICT (symbol) DO UPDATE SET
                        state_version = EXCLUDED.state_version,
                        last_market_event_id = EXCLUDED.last_market_event_id,
                        checkpoint = EXCLUDED.checkpoint,
                        checkpoint_checksum = EXCLUDED.checkpoint_checksum,
                        feature_snapshot = EXCLUDED.feature_snapshot,
                        feature_checksum = EXCLUDED.feature_checksum,
                        updated_at = NOW()
                    """,
                    event.instrument_id,
                    snapshot.state_version,
                    event.event_id,
                    checkpoint,
                    checkpoint_checksum,
                    snapshot_json,
                    snapshot.checksum,
                )
                output_id = f"features-ready:{event.event_id}"
                await connection.execute(
                    """
                    INSERT INTO event_outbox (
                        event_id, aggregate_type, aggregate_id, subject, payload,
                        payload_checksum, headers, source_subject, source_event_id
                    ) VALUES ($1, 'feature', $2, $3, $4::jsonb, $5, $6::jsonb, $7, $8)
                    ON CONFLICT (event_id) DO NOTHING
                    """,
                    output_id,
                    event.instrument_id,
                    CoreSubject.FEATURES_READY.value,
                    snapshot_json,
                    hashlib.sha256(snapshot_json.encode()).hexdigest(),
                    json.dumps({"Nats-Msg-Id": output_id}, separators=(",", ":")),
                    CoreSubject.MARKET_VALIDATED.value,
                    str(event.event_id),
                )
                return snapshot

    @staticmethod
    def _restore(row: Any | None) -> IncrementalFeatureEngine:
        if row is None:
            return IncrementalFeatureEngine()
        checkpoint = row["checkpoint"]
        encoded = checkpoint if isinstance(checkpoint, str) else canonical_json(checkpoint)
        expected = hashlib.sha256(encoded.encode()).hexdigest()
        if expected != str(row["checkpoint_checksum"]):
            raise RuntimeError("feature checkpoint checksum mismatch")
        return IncrementalFeatureEngine.from_checkpoint(encoded)
