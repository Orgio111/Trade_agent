"""Typed, redacted failures for the local Hermes integration layer."""

from __future__ import annotations

from uuid import UUID


class HermesError(RuntimeError):
    """Base class for expected Hermes failures."""


class JournalError(HermesError):
    """Base class for durable Obsidian journal failures."""


class JournalConfigurationError(JournalError):
    """The configured vault root is unavailable or unsafe."""


class JournalPathError(JournalError):
    """A journal path escaped its configured vault boundary."""


class JournalConflictError(JournalError):
    """An event ID is already bound to different durable content."""


class JournalSecretError(JournalError):
    """A journal event was rejected because it resembled secret material."""


class JournalIntegrityError(JournalError):
    """Existing journal content failed an integrity check."""


class RuntimeProjectionError(HermesError):
    """Durable memory succeeded but its rebuildable vector projection failed."""

    def __init__(self, *, event_id: UUID | str, journal_path: str) -> None:
        self.event_id = str(event_id)
        self.journal_path = journal_path
        super().__init__(
            "runtime projection failed after the durable journal write for "
            f"event {self.event_id}; source path: {self.journal_path}"
        )
