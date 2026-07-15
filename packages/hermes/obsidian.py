"""Secure, create-only Markdown journal for a configured Obsidian vault."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import tempfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from uuid import UUID

from packages.knowledge.errors import KnowledgeError, SecretDetectedError
from packages.knowledge.locking import LocalProcessLock
from packages.knowledge.models import ComponentHealth
from packages.knowledge.security import scan_secrets

from .errors import (
    JournalConfigurationError,
    JournalConflictError,
    JournalIntegrityError,
    JournalPathError,
    JournalSecretError,
)
from .models import JournalEvent, JournalReceipt


_EVENT_NAMESPACE = ("trade-agent", "memory", "events")
_DIGEST_LINE = re.compile(r'^content_sha256: "([0-9a-f]{64})"$', re.MULTILINE)
_EVENT_ID_LINE = re.compile(
    r'^event_id: "([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-'
    r'[0-9a-f]{4}-[0-9a-f]{12})"$',
    re.MULTILINE,
)


def _is_link_or_junction(path: Path) -> bool:
    if path.is_symlink():
        return True
    is_junction = getattr(path, "is_junction", None)
    return bool(is_junction is not None and is_junction())


def _json_yaml(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _utc_text(value: datetime) -> str:
    return (
        value.astimezone(UTC)
        .isoformat(timespec="microseconds")
        .replace(
            "+00:00",
            "Z",
        )
    )


def _markdown_heading(value: str) -> str:
    return re.sub(r"([\\`*_[\]<>#])", r"\\\1", value)


def _event_lease_path(vault_root: Path, event_id: UUID) -> Path:
    identity = f"{vault_root}\0{event_id}".encode("utf-8")
    digest = hashlib.sha256(identity).hexdigest()[:32]
    return (
        Path(tempfile.gettempdir())
        / "trade-agent-hermes-locks"
        / f"event-{digest}.lock"
    )


def render_event_markdown(event: JournalEvent) -> str:
    """Render deterministic YAML 1.2 frontmatter plus human-readable Markdown."""

    metadata = dict(event.metadata)
    frontmatter = (
        "---\n"
        f"schema_version: {_json_yaml(event.schema_version)}\n"
        f"event_id: {_json_yaml(str(event.event_id))}\n"
        f"content_sha256: {_json_yaml(event.content_sha256)}\n"
        f"occurred_at: {_json_yaml(_utc_text(event.occurred_at))}\n"
        f"kind: {_json_yaml(event.kind.value)}\n"
        f"agent_id: {_json_yaml(event.agent_id)}\n"
        f"trace_id: {_json_yaml(event.trace_id)}\n"
        f"title: {_json_yaml(event.title)}\n"
        f"tags: {_json_yaml(list(event.tags))}\n"
        f"source_refs: {_json_yaml(list(event.source_refs))}\n"
        f"metadata: {_json_yaml(metadata)}\n"
        "---\n"
    )
    return f"{frontmatter}\n# {_markdown_heading(event.title)}\n\n{event.body}\n"


class ObsidianJournalStore:
    """Append-only event storage under one explicit Obsidian vault root."""

    def __init__(self, vault_root: str | Path) -> None:
        configured = Path(vault_root).expanduser()
        if not configured.exists() or not configured.is_dir():
            raise JournalConfigurationError(
                "configured Obsidian vault root must be an existing directory"
            )
        if _is_link_or_junction(configured):
            raise JournalConfigurationError(
                "configured Obsidian vault root cannot be a symlink or junction"
            )
        self._root = configured.resolve(strict=True)
        self._append_lock = asyncio.Lock()

    @property
    def vault_root(self) -> Path:
        return self._root

    @property
    def events_root(self) -> Path:
        return self._root.joinpath(*_EVENT_NAMESPACE)

    async def append(self, event: JournalEvent) -> JournalReceipt:
        """Create one event file or prove that an identical write already exists."""

        rendered = render_event_markdown(event)
        relative = self._event_relative_path(event)
        try:
            scan_secrets(relative.as_posix(), rendered)
        except SecretDetectedError as exc:
            raise JournalSecretError(
                "journal event rejected because it resembles secret material"
            ) from exc

        async with self._append_lock:
            return await asyncio.to_thread(
                self._append_sync,
                event,
                relative,
                rendered,
            )

    async def get_receipt(
        self,
        event_id: UUID,
        *,
        occurred_at: datetime | None = None,
    ) -> JournalReceipt | None:
        """Return durable evidence without following links outside the vault."""

        return await asyncio.to_thread(
            self._get_receipt_sync,
            event_id,
            occurred_at,
        )

    async def health(self) -> ComponentHealth:
        """Report storage health without creating or changing vault files."""

        return await asyncio.to_thread(self._health_sync)

    def _health_sync(self) -> ComponentHealth:
        try:
            if not self._root.exists() or not self._root.is_dir():
                raise JournalConfigurationError("configured vault root is unavailable")
            self._assert_safe_existing(self._root)
        except (OSError, JournalConfigurationError, JournalPathError) as exc:
            return ComponentHealth(healthy=False, detail=str(exc))
        return ComponentHealth(
            healthy=True,
            detail="Obsidian journal vault boundary is available",
        )

    def _event_relative_path(self, event: JournalEvent) -> PurePosixPath:
        occurred = event.occurred_at.astimezone(UTC)
        return PurePosixPath(
            *_EVENT_NAMESPACE,
            f"{occurred.year:04d}",
            f"{occurred.month:02d}",
            f"{occurred.day:02d}",
            f"{event.event_id}.md",
        )

    def _append_sync(
        self,
        event: JournalEvent,
        relative: PurePosixPath,
        rendered: str,
    ) -> JournalReceipt:
        try:
            with LocalProcessLock(_event_lease_path(self._root, event.event_id)):
                existing = self._find_existing_sync(event.event_id)
                target = self._root.joinpath(*relative.parts)
                if existing is not None:
                    return self._verify_existing(
                        event,
                        relative,
                        rendered,
                        existing,
                    )

                self._ensure_safe_directory(target.parent)
                self._assert_candidate_contained(target)
                created = self._create_exclusive(target, rendered)
                if not created:
                    return self._verify_existing(event, relative, rendered, target)
                self._assert_safe_existing(target)
                return self._receipt(event, relative, target, created=True)
        except KnowledgeError as exc:
            raise JournalConflictError(
                f"journal event {event.event_id} is being written; retry"
            ) from exc

    def _verify_existing(
        self,
        event: JournalEvent,
        expected_relative: PurePosixPath,
        rendered: str,
        existing: Path,
    ) -> JournalReceipt:
        self._assert_safe_existing(existing)
        actual_relative = PurePosixPath(existing.relative_to(self._root).as_posix())
        actual = existing.read_text(encoding="utf-8")
        if actual_relative != expected_relative or actual != rendered:
            raise JournalConflictError(
                f"journal event ID {event.event_id} already has different content"
            )
        return self._receipt(event, actual_relative, existing, created=False)

    def _get_receipt_sync(
        self,
        event_id: UUID,
        occurred_at: datetime | None,
    ) -> JournalReceipt | None:
        if occurred_at is not None:
            if occurred_at.tzinfo is None or occurred_at.utcoffset() is None:
                raise ValueError("occurred_at must be timezone-aware")
            timestamp = occurred_at.astimezone(UTC)
            relative = PurePosixPath(
                *_EVENT_NAMESPACE,
                f"{timestamp.year:04d}",
                f"{timestamp.month:02d}",
                f"{timestamp.day:02d}",
                f"{event_id}.md",
            )
            candidate = self._root.joinpath(*relative.parts)
            if not candidate.exists():
                return None
        else:
            found = self._find_existing_sync(event_id)
            if found is None:
                return None
            candidate = found
            relative = PurePosixPath(candidate.relative_to(self._root).as_posix())

        self._assert_safe_existing(candidate)
        text = candidate.read_text(encoding="utf-8")
        digest_match = _DIGEST_LINE.search(text)
        id_match = _EVENT_ID_LINE.search(text)
        if (
            digest_match is None
            or id_match is None
            or UUID(id_match.group(1)) != event_id
        ):
            raise JournalIntegrityError(
                f"journal event {event_id} has invalid durable frontmatter"
            )
        written_at = datetime.fromtimestamp(candidate.stat().st_mtime, tz=UTC)
        return JournalReceipt(
            event_id=event_id,
            content_sha256=digest_match.group(1),
            relative_path=relative,
            created=False,
            written_at=written_at,
        )

    def _find_existing_sync(self, event_id: UUID) -> Path | None:
        if not self.events_root.exists():
            return None
        self._assert_safe_existing(self.events_root)
        matches = list(self.events_root.glob(f"*/*/*/{event_id}.md"))
        if not matches:
            return None
        if len(matches) > 1:
            raise JournalIntegrityError(
                f"journal event {event_id} exists at multiple durable paths"
            )
        self._assert_safe_existing(matches[0])
        return matches[0]

    def _ensure_safe_directory(self, directory: Path) -> None:
        try:
            relative = directory.relative_to(self._root)
        except ValueError as exc:
            raise JournalPathError("journal directory escapes the vault") from exc

        current = self._root
        self._assert_safe_existing(current)
        for part in relative.parts:
            current = current / part
            try:
                current.mkdir()
            except FileExistsError:
                pass
            if not current.is_dir():
                raise JournalPathError("journal namespace contains a non-directory")
            self._assert_safe_existing(current)

    def _assert_candidate_contained(self, candidate: Path) -> None:
        resolved_parent = candidate.parent.resolve(strict=True)
        try:
            resolved_parent.relative_to(self._root)
        except ValueError as exc:
            raise JournalPathError("journal path escapes the configured vault") from exc
        if candidate.exists() or candidate.is_symlink():
            self._assert_safe_existing(candidate)

    def _assert_safe_existing(self, path: Path) -> None:
        if _is_link_or_junction(path):
            raise JournalPathError("symlink and junction journal paths are forbidden")
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(self._root)
        except (OSError, ValueError) as exc:
            raise JournalPathError(
                "journal path escapes the configured Obsidian vault"
            ) from exc

    def _create_exclusive(self, target: Path, rendered: str) -> bool:
        data = rendered.encode("utf-8")
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=target.parent,
                prefix=f".{target.stem}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            self._assert_safe_existing(temporary)
            try:
                os.link(temporary, target)
            except FileExistsError:
                return False
            return True
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def _receipt(
        self,
        event: JournalEvent,
        relative: PurePosixPath,
        path: Path,
        *,
        created: bool,
    ) -> JournalReceipt:
        return JournalReceipt(
            event_id=event.event_id,
            content_sha256=event.content_sha256,
            relative_path=relative,
            created=created,
            written_at=datetime.fromtimestamp(path.stat().st_mtime, tz=UTC),
        )
