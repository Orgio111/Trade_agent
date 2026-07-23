"""PostgreSQL transactional inbox/outbox boundary for state-changing consumers."""

from __future__ import annotations

import asyncio
import hashlib
import json
from typing import Any, Protocol

from packages.domain import canonical_json
from packages.event_bus import CoreSubject, JetStreamEventBus, JetStreamTransportError
from packages.execution._postgres import PostgresPool


class EventPublisher(Protocol):
    async def publish(
        self,
        subject: CoreSubject,
        payload: bytes,
        *,
        message_id: str,
        stream: str | None = None,
    ) -> object: ...


class PostgresInboxOutbox:
    """Record input and output atomically, then publish with a leased claim."""

    def __init__(
        self,
        pool: PostgresPool,
        bus: JetStreamEventBus | EventPublisher,
        *,
        stream_name: str,
        durable_name: str,
        max_publish_attempts: int = 10,
    ) -> None:
        if max_publish_attempts < 1 or max_publish_attempts > 20:
            raise ValueError("max_publish_attempts must be between 1 and 20")
        self._pool = pool
        self._bus = bus
        self._stream_name = stream_name
        self._durable_name = durable_name
        self._max_publish_attempts = max_publish_attempts

    async def route(
        self,
        *,
        source_event_id: str,
        source_subject: CoreSubject,
        source_payload: dict[str, Any],
        payload_checksum: str,
        output_event_id: str,
        aggregate_type: str,
        aggregate_id: str,
        subject: CoreSubject,
        payload: bytes,
        message_id: str,
    ) -> bool:
        output_checksum = hashlib.sha256(payload).hexdigest()
        headers = {"Nats-Msg-Id": message_id}
        async with self._pool.acquire() as connection:
            async with connection.transaction():
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
                    source_event_id,
                    source_subject.value,
                    canonical_json(source_payload),
                    payload_checksum,
                )
                if inserted is None:
                    return False
                await connection.execute(
                    """
                    INSERT INTO event_outbox (
                        event_id, aggregate_type, aggregate_id, subject, payload,
                        payload_checksum, headers, source_subject, source_event_id
                    ) VALUES ($1, $2, $3, $4, $5::jsonb, $6, $7::jsonb, $8, $9)
                    ON CONFLICT (event_id) DO NOTHING
                    """,
                    output_event_id,
                    aggregate_type,
                    aggregate_id,
                    subject.value,
                    payload.decode("utf-8"),
                    output_checksum,
                    json.dumps(headers, sort_keys=True, separators=(",", ":")),
                    source_subject.value,
                    source_event_id,
                )
                await connection.execute(
                    """
                    UPDATE event_inbox SET processed_at = NOW()
                    WHERE stream_name = $1 AND durable_name = $2 AND event_id = $3
                    """,
                    self._stream_name,
                    self._durable_name,
                    source_event_id,
                )
        await self.dispatch(output_event_id)
        return True

    async def route_many(
        self,
        *,
        source_event_id: str,
        source_subject: CoreSubject,
        source_payload: dict[str, Any],
        payload_checksum: str,
        outputs: tuple[dict[str, Any], ...],
    ) -> bool:
        """Record one input and its bounded output set in one transaction."""

        if not 1 <= len(outputs) <= 8:
            raise ValueError("outputs must contain between 1 and 8 events")
        async with self._pool.acquire() as connection:
            async with connection.transaction():
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
                    source_event_id,
                    source_subject.value,
                    canonical_json(source_payload),
                    payload_checksum,
                )
                if inserted is None:
                    return False
                for output in outputs:
                    payload = output["payload"]
                    message_id = str(output["message_id"])
                    await connection.execute(
                        """
                        INSERT INTO event_outbox (
                            event_id, aggregate_type, aggregate_id, subject, payload,
                            payload_checksum, headers, source_subject, source_event_id
                        ) VALUES ($1, $2, $3, $4, $5::jsonb, $6, $7::jsonb, $8, $9)
                        ON CONFLICT (event_id) DO NOTHING
                        """,
                        str(output["output_event_id"]),
                        str(output["aggregate_type"]),
                        str(output["aggregate_id"]),
                        CoreSubject(output["subject"]).value,
                        payload.decode("utf-8"),
                        hashlib.sha256(payload).hexdigest(),
                        json.dumps(
                            {"Nats-Msg-Id": message_id},
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                        source_subject.value,
                        source_event_id,
                    )
        for output in outputs:
            await self.dispatch(str(output["output_event_id"]))
        return True

    async def reserve(
        self,
        *,
        source_event_id: str,
        source_subject: CoreSubject,
        source_payload: dict[str, Any],
        payload_checksum: str,
    ) -> bool:
        """Claim expensive processing before external inference begins."""

        async with self._pool.acquire() as connection:
            inserted = await connection.fetchval(
                """
                INSERT INTO event_inbox (
                    stream_name, durable_name, event_id, subject,
                    stream_sequence, payload, payload_checksum
                ) VALUES ($1, $2, $3, $4, NULL, $5::jsonb, $6)
                ON CONFLICT (stream_name, durable_name, event_id) DO UPDATE SET
                    received_at = NOW(), payload = EXCLUDED.payload,
                    payload_checksum = EXCLUDED.payload_checksum
                WHERE event_inbox.processed_at IS NULL
                  AND event_inbox.received_at <= NOW() - INTERVAL '2 minutes'
                RETURNING event_id
                """,
                self._stream_name,
                self._durable_name,
                source_event_id,
                source_subject.value,
                canonical_json(source_payload),
                payload_checksum,
            )
        return inserted is not None

    async def reservation_processed(self, source_event_id: str) -> bool:
        async with self._pool.acquire() as connection:
            return bool(
                await connection.fetchval(
                    """
                    SELECT processed_at IS NOT NULL FROM event_inbox
                    WHERE stream_name = $1 AND durable_name = $2 AND event_id = $3
                    """,
                    self._stream_name,
                    self._durable_name,
                    source_event_id,
                )
            )

    async def complete_reserved(
        self,
        *,
        source_event_id: str,
        output_event_id: str,
        aggregate_type: str,
        aggregate_id: str,
        subject: CoreSubject,
        payload: bytes,
        message_id: str,
    ) -> None:
        """Atomically persist reserved processing result and its output."""

        output_checksum = hashlib.sha256(payload).hexdigest()
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                status = await connection.execute(
                    """
                    UPDATE event_inbox SET processed_at = NOW()
                    WHERE stream_name = $1 AND durable_name = $2
                      AND event_id = $3 AND processed_at IS NULL
                    """,
                    self._stream_name,
                    self._durable_name,
                    source_event_id,
                )
                if status != "UPDATE 1":
                    return
                await connection.execute(
                    """
                    INSERT INTO event_outbox (
                        event_id, aggregate_type, aggregate_id, subject, payload,
                        payload_checksum, headers, source_subject, source_event_id
                    ) VALUES ($1, $2, $3, $4, $5::jsonb, $6, $7::jsonb, $8, $9)
                    ON CONFLICT (event_id) DO NOTHING
                    """,
                    output_event_id,
                    aggregate_type,
                    aggregate_id,
                    subject.value,
                    payload.decode("utf-8"),
                    output_checksum,
                    json.dumps({"Nats-Msg-Id": message_id}, separators=(",", ":")),
                    CoreSubject.FEATURES_READY.value,
                    source_event_id,
                )
        await self.dispatch(output_event_id)

    async def complete_reserved_without_output(self, source_event_id: str) -> None:
        async with self._pool.acquire() as connection:
            await connection.execute(
                """
                UPDATE event_inbox SET processed_at = NOW()
                WHERE stream_name = $1 AND durable_name = $2
                  AND event_id = $3 AND processed_at IS NULL
                """,
                self._stream_name,
                self._durable_name,
                source_event_id,
            )

    async def record_consumed(
        self,
        *,
        source_event_id: str,
        source_subject: CoreSubject,
        source_payload: dict[str, Any],
        payload_checksum: str,
    ) -> bool:
        """Persist valid no-output processing so redelivery does not rerun a model."""

        async with self._pool.acquire() as connection:
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
                source_event_id,
                source_subject.value,
                canonical_json(source_payload),
                payload_checksum,
            )
        return inserted is not None

    async def dispatch(self, event_id: str) -> bool:
        return await self._dispatch_claim(event_id)

    async def dispatch_next(self) -> bool:
        """Publish one pending or stale-leased event for background recovery."""

        return await self._dispatch_claim(None)

    async def run_dispatcher(self, *, idle_seconds: float = 0.25) -> None:
        """Recover pending outputs forever; database failure remains fatal."""

        while True:
            try:
                published = await self.dispatch_next()
            except JetStreamTransportError:
                published = False
            if not published:
                await asyncio.sleep(idle_seconds)

    async def _dispatch_claim(self, event_id: str | None) -> bool:
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                row = await connection.fetchrow(
                    """
                    SELECT event_id, subject, payload, headers, publish_attempts
                    FROM event_outbox
                    WHERE ($1::varchar IS NULL OR event_id = $1)
                      AND status IN ('pending', 'publishing')
                      AND next_attempt_at <= NOW()
                      AND (lease_expires_at IS NULL OR lease_expires_at <= NOW())
                    FOR UPDATE SKIP LOCKED
                    """,
                    event_id,
                )
                if row is None:
                    return False
                claimed_event_id = str(row["event_id"])
                await connection.execute(
                    """
                    UPDATE event_outbox
                    SET status = 'publishing', publish_attempts = publish_attempts + 1,
                        lease_owner = $2, lease_expires_at = NOW() + INTERVAL '10 seconds'
                    WHERE event_id = $1
                    """,
                    claimed_event_id,
                    self._durable_name,
                )
        headers = row["headers"]
        if isinstance(headers, str):
            headers = json.loads(headers)
        payload = row["payload"]
        encoded = (
            payload.encode("utf-8")
            if isinstance(payload, str)
            else canonical_json(payload).encode()
        )
        message_id = str(headers["Nats-Msg-Id"])
        try:
            await self._bus.publish(
                CoreSubject(str(row["subject"])),
                encoded,
                message_id=message_id,
                stream=self._stream_name,
            )
        except Exception as exc:
            # Lease expiry is the retry signal. Do not erase evidence of an
            # ambiguous publish; stable Nats-Msg-Id makes retry safe in-window.
            if int(row["publish_attempts"]) + 1 >= self._max_publish_attempts:
                await self._dead_letter(
                    event_id=claimed_event_id,
                    subject=str(row["subject"]),
                    payload=encoded,
                    error=exc,
                )
            raise
        async with self._pool.acquire() as connection:
            await connection.execute(
                """
                UPDATE event_outbox
                SET status = 'published', published_at = NOW(),
                    lease_owner = NULL, lease_expires_at = NULL
                WHERE event_id = $1 AND status = 'publishing'
                """,
                claimed_event_id,
            )
        return True

    async def _dead_letter(
        self,
        *,
        event_id: str,
        subject: str,
        payload: bytes,
        error: Exception,
    ) -> None:
        error_type = type(error).__name__
        fingerprint = hashlib.sha256(error_type.encode("utf-8")).hexdigest()
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                await connection.execute(
                    """
                    INSERT INTO dead_letter_events (
                        original_subject, original_event_id, payload_hash,
                        failure_class, failure_message, consumer,
                        first_failure_at, last_failure_at, retry_count,
                        stack_trace_fingerprint
                    ) VALUES ($1, $2, $3, $4, $4, $5, NOW(), NOW(), $6, $7)
                    ON CONFLICT (consumer, original_event_id) DO UPDATE SET
                        last_failure_at = NOW(), retry_count = EXCLUDED.retry_count,
                        failure_class = EXCLUDED.failure_class,
                        failure_message = EXCLUDED.failure_message,
                        stack_trace_fingerprint = EXCLUDED.stack_trace_fingerprint
                    """,
                    subject,
                    event_id,
                    hashlib.sha256(payload).hexdigest(),
                    error_type,
                    self._durable_name,
                    self._max_publish_attempts,
                    fingerprint,
                )
                await connection.execute(
                    """
                    UPDATE event_outbox
                    SET status = 'dead_letter', last_error_code = $2,
                        lease_owner = NULL, lease_expires_at = NULL
                    WHERE event_id = $1
                    """,
                    event_id,
                    error_type,
                )
