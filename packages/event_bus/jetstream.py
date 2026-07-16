"""Bounded, manual-ack NATS JetStream transport for canonical core events.

The adapter deliberately transports bytes.  Domain parsing and authorization
remain outside the event bus, so receiving a message can never grant broker or
risk authority by itself.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
import re
from typing import Any, Protocol

from nats.js.api import AckPolicy, ConsumerConfig, DeliverPolicy

from .subjects import CoreSubject


_DURABLE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_MESSAGE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")


class JetStreamTransportError(RuntimeError):
    """JetStream could not durably publish or settle a message."""


class JetStreamConfigurationError(ValueError):
    """A consumer setting would weaken durability or resource bounds."""


@dataclass(frozen=True, slots=True)
class DurableConsumerSettings:
    """Fail-closed limits for one explicit JetStream durable consumer."""

    stream: str
    durable_name: str
    ack_wait_seconds: float = 30.0
    max_deliver: int = 5
    max_ack_pending: int = 256
    retry_delay_seconds: float = 0.25
    max_payload_bytes: int = 1_048_576

    def __post_init__(self) -> None:
        if not _DURABLE_NAME.fullmatch(self.durable_name):
            raise JetStreamConfigurationError(
                "durable_name must be 1-64 safe identifier characters"
            )
        if not self.stream.strip():
            raise JetStreamConfigurationError("stream cannot be blank")
        if self.ack_wait_seconds <= 0:
            raise JetStreamConfigurationError("ack_wait_seconds must be positive")
        if self.max_deliver < 1:
            raise JetStreamConfigurationError("max_deliver must be at least one")
        if self.max_ack_pending < 1:
            raise JetStreamConfigurationError("max_ack_pending must be at least one")
        if self.retry_delay_seconds < 0:
            raise JetStreamConfigurationError("retry_delay_seconds cannot be negative")
        if self.max_payload_bytes < 1:
            raise JetStreamConfigurationError("max_payload_bytes must be positive")


@dataclass(frozen=True, slots=True)
class PublishReceipt:
    stream: str
    sequence: int
    duplicate: bool


@dataclass(frozen=True, slots=True)
class ConsumerFailure:
    """Sanitized failure metadata; message payloads are never copied here."""

    subject: str
    durable_name: str
    delivery_count: int
    terminal: bool
    error_type: str


class _MessageMetadata(Protocol):
    num_delivered: int


class JetStreamMessage(Protocol):
    subject: str
    data: bytes

    @property
    def metadata(self) -> _MessageMetadata: ...

    async def ack(self) -> None: ...

    async def nak(self, delay: float | None = None) -> None: ...

    async def term(self) -> None: ...


class JetStreamContext(Protocol):
    async def publish(
        self,
        subject: str,
        payload: bytes = b"",
        *,
        timeout: float | None = None,
        stream: str | None = None,
        headers: dict[str, Any] | None = None,
    ) -> Any: ...

    async def subscribe(self, subject: str, **kwargs: Any) -> Any: ...


MessageHandler = Callable[[bytes], Awaitable[None]]
FailureHandler = Callable[[ConsumerFailure], Awaitable[None]]


class JetStreamEventBus:
    """Durable byte transport with explicit acknowledgement ownership."""

    def __init__(
        self,
        context: JetStreamContext,
        *,
        publish_timeout_seconds: float = 2.0,
        max_publish_bytes: int = 1_048_576,
    ) -> None:
        if publish_timeout_seconds <= 0:
            raise JetStreamConfigurationError(
                "publish_timeout_seconds must be positive"
            )
        if max_publish_bytes < 1:
            raise JetStreamConfigurationError("max_publish_bytes must be positive")
        self._context = context
        self._publish_timeout_seconds = publish_timeout_seconds
        self._max_publish_bytes = max_publish_bytes

    async def publish(
        self,
        subject: CoreSubject,
        payload: bytes,
        *,
        message_id: str,
        stream: str | None = None,
    ) -> PublishReceipt:
        """Publish a canonical event with a required JetStream deduplication ID."""

        resolved = CoreSubject(subject)
        if not isinstance(payload, bytes) or not payload:
            raise JetStreamConfigurationError("payload must be non-empty bytes")
        if len(payload) > self._max_publish_bytes:
            raise JetStreamConfigurationError("payload exceeds configured byte limit")
        if not _MESSAGE_ID.fullmatch(message_id):
            raise JetStreamConfigurationError(
                "message_id must be 1-128 safe identifier characters"
            )

        try:
            ack = await self._context.publish(
                resolved.value,
                payload,
                timeout=self._publish_timeout_seconds,
                stream=stream,
                headers={"Nats-Msg-Id": message_id},
            )
            return PublishReceipt(
                stream=str(ack.stream),
                sequence=int(ack.seq),
                duplicate=bool(getattr(ack, "duplicate", False)),
            )
        except JetStreamConfigurationError:
            raise
        except Exception as exc:
            raise JetStreamTransportError("durable event publish failed") from exc

    async def subscribe(
        self,
        subject: CoreSubject,
        handler: MessageHandler,
        *,
        settings: DurableConsumerSettings,
        on_failure: FailureHandler | None = None,
    ) -> Any:
        """Create one explicit durable push consumer with bounded redelivery."""

        resolved = CoreSubject(subject)
        config = ConsumerConfig(
            durable_name=settings.durable_name,
            deliver_policy=DeliverPolicy.ALL,
            ack_policy=AckPolicy.EXPLICIT,
            ack_wait=settings.ack_wait_seconds,
            max_deliver=settings.max_deliver,
            max_ack_pending=settings.max_ack_pending,
            filter_subject=resolved.value,
        )

        async def callback(message: JetStreamMessage) -> None:
            delivery_count = self._delivery_count(message)
            if len(message.data) > settings.max_payload_bytes or not message.data:
                await self._settle_failure(
                    message,
                    settings=settings,
                    on_failure=on_failure,
                    delivery_count=delivery_count,
                    error=JetStreamConfigurationError(
                        "consumer payload is empty or exceeds configured byte limit"
                    ),
                    force_terminal=True,
                )
                return
            try:
                await handler(message.data)
            except Exception as exc:
                await self._settle_failure(
                    message,
                    settings=settings,
                    on_failure=on_failure,
                    delivery_count=delivery_count,
                    error=exc,
                )
                return
            try:
                await message.ack()
            except Exception as exc:
                await self._notify_failure(
                    on_failure,
                    ConsumerFailure(
                        subject=resolved.value,
                        durable_name=settings.durable_name,
                        delivery_count=delivery_count,
                        terminal=False,
                        error_type=type(exc).__name__,
                    ),
                )
                raise JetStreamTransportError("message acknowledgement failed") from exc

        try:
            return await self._context.subscribe(
                resolved.value,
                cb=callback,
                durable=settings.durable_name,
                stream=settings.stream,
                config=config,
                manual_ack=True,
                pending_msgs_limit=settings.max_ack_pending,
                pending_bytes_limit=(
                    settings.max_ack_pending * settings.max_payload_bytes
                ),
            )
        except Exception as exc:
            raise JetStreamTransportError("durable consumer creation failed") from exc

    async def _settle_failure(
        self,
        message: JetStreamMessage,
        *,
        settings: DurableConsumerSettings,
        on_failure: FailureHandler | None,
        delivery_count: int,
        error: Exception,
        force_terminal: bool = False,
    ) -> None:
        terminal = force_terminal or delivery_count >= settings.max_deliver
        failure = ConsumerFailure(
            subject=message.subject,
            durable_name=settings.durable_name,
            delivery_count=delivery_count,
            terminal=terminal,
            error_type=type(error).__name__,
        )
        try:
            if terminal:
                await message.term()
            else:
                exponent = min(max(delivery_count - 1, 0), 8)
                delay = settings.retry_delay_seconds * (2**exponent)
                await message.nak(delay=delay)
        except Exception as exc:
            await self._notify_failure(on_failure, failure)
            raise JetStreamTransportError("message settlement failed") from exc
        await self._notify_failure(on_failure, failure)

    @staticmethod
    def _delivery_count(message: JetStreamMessage) -> int:
        try:
            return max(1, int(message.metadata.num_delivered))
        except Exception:
            return 1

    @staticmethod
    async def _notify_failure(
        handler: FailureHandler | None, failure: ConsumerFailure
    ) -> None:
        if handler is None:
            return
        try:
            await handler(failure)
        except Exception:
            # Observability must not change the ack/nak/term outcome.
            return
