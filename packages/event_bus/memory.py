"""Synchronous deterministic event bus used by unit and replay tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .subjects import CoreSubject


@dataclass(frozen=True, slots=True)
class PublishedEvent:
    sequence: int
    subject: CoreSubject
    payload: Any


class InMemoryEventBus:
    """Record publications and deliver exact-subject handlers in order."""

    def __init__(self) -> None:
        self._events: list[PublishedEvent] = []
        self._handlers: dict[CoreSubject, list[Callable[[Any], None]]] = {}

    def subscribe(self, subject: CoreSubject, handler: Callable[[Any], None]) -> None:
        self._handlers.setdefault(CoreSubject(subject), []).append(handler)

    def publish(self, subject: CoreSubject, payload: Any) -> PublishedEvent:
        resolved = CoreSubject(subject)
        event = PublishedEvent(
            sequence=len(self._events) + 1,
            subject=resolved,
            payload=payload,
        )
        self._events.append(event)
        for handler in tuple(self._handlers.get(resolved, ())):
            handler(payload)
        return event

    @property
    def events(self) -> tuple[PublishedEvent, ...]:
        return tuple(self._events)

    def events_for(self, subject: CoreSubject) -> tuple[PublishedEvent, ...]:
        resolved = CoreSubject(subject)
        return tuple(event for event in self._events if event.subject is resolved)

