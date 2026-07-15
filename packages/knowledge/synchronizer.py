"""Incremental, resumable synchronization from the lock into local Chroma."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .chunking import chunk_id, chunk_text
from .errors import LiveVerificationError, ManifestError
from .lockfile import LockData, canonical_json_bytes, lock_sha256
from .manifest import ManifestInventory
from .models import (
    ModelIdentity,
    QueryResult,
    Scalar,
    StoredRecord,
    SyncReport,
    VectorRecord,
)
from .ports import AsyncEmbedder, AsyncVectorStore


@dataclass(frozen=True, slots=True)
class DesiredRecord:
    id: str
    document: str
    metadata: dict[str, Scalar]


def _receipt_path(inventory: ManifestInventory) -> Path:
    candidate = (inventory.root / inventory.manifest.knowledge.receipt_file).resolve()
    try:
        candidate.relative_to(inventory.root.resolve())
    except ValueError as exc:
        raise ManifestError("knowledge.receipt_file escapes the repository") from exc
    return candidate


def _atomic_json_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_json_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _desired_records(
    inventory: ManifestInventory, lock: LockData
) -> dict[str, DesiredRecord]:
    """Rebuild source text and prove it matches the checked lock."""

    settings = inventory.manifest.knowledge
    locked_sources = {source["path"]: source for source in lock["sources"]}
    if set(locked_sources) != {source.path for source in inventory.sources}:
        raise LiveVerificationError("checked lock source inventory is inconsistent")
    desired: dict[str, DesiredRecord] = {}
    owners = {
        component.id: component.owner for component in inventory.manifest.components
    }
    for source in inventory.sources:
        locked = locked_sources[source.path]
        chunks = chunk_text(
            source.text,
            max_chars=settings.max_chunk_chars,
            overlap_chars=settings.overlap_chars,
        )
        locked_chunks = locked.get("chunks", [])
        if len(chunks) != len(locked_chunks):
            raise LiveVerificationError(
                f"chunk count drift after lock check: {source.path}"
            )
        for chunk, evidence in zip(chunks, locked_chunks, strict=True):
            record_id = chunk_id(
                project=inventory.manifest.project,
                model=settings.embedding_model,
                dimension=settings.embedding_dimension,
                chunker_version=settings.chunker_version,
                path=source.path,
                chunk=chunk,
            )
            if (
                evidence.get("id") != record_id
                or evidence.get("sha256") != chunk.sha256
            ):
                raise LiveVerificationError(
                    f"chunk evidence drift after lock check: {source.path}"
                )
            categories = ",".join(source.categories)
            metadata: dict[str, Scalar] = {
                "project": inventory.manifest.project,
                "component": source.component_id,
                "owner": owners[source.component_id],
                "source_path": source.path,
                "source_sha256": source.sha256,
                "chunk_sha256": chunk.sha256,
                "chunk_index": chunk.index,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "categories": categories,
            }
            document = (
                f"Source: {source.path}\n"
                f"Component: {source.component_id}\n"
                f"Categories: {categories}\n"
                f"Lines: {chunk.start_line}-{chunk.end_line}\n\n"
                f"{chunk.text}"
            )
            desired[record_id] = DesiredRecord(
                id=record_id,
                document=document,
                metadata=metadata,
            )
    return desired


def _stable_namespace_metadata(
    inventory: ManifestInventory, identity: ModelIdentity
) -> dict[str, Scalar]:
    settings = inventory.manifest.knowledge
    return {
        "schema_version": 1,
        "project": inventory.manifest.project,
        "embedding_model": identity.requested_model,
        "model_digest": identity.digest,
        "embedding_dimension": identity.dimension,
        "chunker_version": settings.chunker_version,
    }


def _metadata_matches(actual: StoredRecord, desired: DesiredRecord) -> bool:
    return dict(actual.metadata) == desired.metadata


class KnowledgeSynchronizer:
    """Coordinates checked source evidence, local Ollama, and local Chroma."""

    def __init__(
        self,
        *,
        inventory: ManifestInventory,
        lock: LockData,
        embedder: AsyncEmbedder,
        store: AsyncVectorStore,
    ) -> None:
        self.inventory = inventory
        self.lock = lock
        self.embedder = embedder
        self.store = store
        self.lock_digest = lock_sha256(lock)

    async def _identity_and_namespace(self) -> ModelIdentity:
        identity = await self.embedder.identity()
        expected = self.inventory.manifest.knowledge
        if identity.requested_model != expected.embedding_model:
            raise LiveVerificationError("embedder model does not match the manifest")
        if identity.dimension != expected.embedding_dimension:
            raise LiveVerificationError(
                "embedder dimension does not match the manifest"
            )
        await self.store.ensure_namespace(
            _stable_namespace_metadata(self.inventory, identity)
        )
        return identity

    async def sync(self) -> SyncReport:
        """Embed only missing/corrupt chunks, verify, then delete stale chunks."""

        identity = await self._identity_and_namespace()
        desired = _desired_records(self.inventory, self.lock)
        existing = await self.store.inventory()
        missing = [
            record
            for record_id, record in sorted(desired.items())
            if record_id not in existing
            or not _metadata_matches(existing[record_id], record)
        ]
        batch_size = self.inventory.manifest.knowledge.embedding_batch_size
        upsert_size = self.inventory.manifest.knowledge.upsert_batch_size
        embedded = 0
        for batch_start in range(0, len(missing), batch_size):
            batch = missing[batch_start : batch_start + batch_size]
            vectors = await self.embedder.embed([record.document for record in batch])
            vector_records = [
                VectorRecord(
                    id=record.id,
                    embedding=vector,
                    document=record.document,
                    metadata=record.metadata,
                )
                for record, vector in zip(batch, vectors, strict=True)
            ]
            for upsert_start in range(0, len(vector_records), upsert_size):
                await self.store.upsert(
                    vector_records[upsert_start : upsert_start + upsert_size]
                )
            embedded += len(batch)

        after_upsert = await self.store.inventory()
        bad = [
            record_id
            for record_id, record in desired.items()
            if record_id not in after_upsert
            or not _metadata_matches(after_upsert[record_id], record)
        ]
        if bad:
            raise LiveVerificationError(
                f"Chroma did not persist {len(bad)} desired knowledge chunks"
            )

        stale = sorted(set(after_upsert) - set(desired))
        deleted = await self.store.delete(stale)
        final_inventory = await self.store.inventory()
        if set(final_inventory) != set(desired):
            raise LiveVerificationError("Chroma inventory mismatch after stale cleanup")
        for record_id, record in desired.items():
            if not _metadata_matches(final_inventory[record_id], record):
                raise LiveVerificationError(f"Chroma metadata mismatch: {record_id}")

        active_metadata: dict[str, Scalar] = {
            "active_lock_sha256": self.lock_digest,
            "active_source_count": len(self.inventory.sources),
            "active_chunk_count": len(desired),
        }
        await self.store.activate(active_metadata)
        receipt = {
            "schema_version": 1,
            "project": self.inventory.manifest.project,
            "collection": self.inventory.manifest.knowledge.collection,
            "lock_sha256": self.lock_digest,
            "manifest_sha256": self.inventory.manifest_sha256,
            "embedding_model": identity.requested_model,
            "resolved_model": identity.resolved_model,
            "model_digest": identity.digest,
            "embedding_dimension": identity.dimension,
            "source_count": len(self.inventory.sources),
            "chunk_count": len(desired),
            "synced_at": datetime.now(UTC).isoformat(),
        }
        _atomic_json_write(_receipt_path(self.inventory), receipt)
        return SyncReport(
            source_count=len(self.inventory.sources),
            desired_chunks=len(desired),
            embedded_chunks=embedded,
            deleted_chunks=deleted,
            unchanged_chunks=len(desired) - embedded,
            lock_sha256=self.lock_digest,
            model_digest=identity.digest,
        )

    async def verify_live(self) -> dict[str, Any]:
        """Compare the real local collection and receipt to the checked lock."""

        identity = await self._identity_and_namespace()
        desired = _desired_records(self.inventory, self.lock)
        actual = await self.store.inventory()
        if set(actual) != set(desired):
            missing = len(set(desired) - set(actual))
            stale = len(set(actual) - set(desired))
            raise LiveVerificationError(
                f"live Chroma ID drift: missing={missing}, stale={stale}"
            )
        for record_id, record in desired.items():
            if not _metadata_matches(actual[record_id], record):
                raise LiveVerificationError(f"live Chroma metadata drift: {record_id}")
        namespace = await self.store.namespace_metadata()
        if namespace.get("active_lock_sha256") != self.lock_digest:
            raise LiveVerificationError(
                "Chroma active lock does not match the committed lock"
            )
        receipt_path = _receipt_path(self.inventory)
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise LiveVerificationError(
                "local sync receipt is missing or corrupt"
            ) from exc
        expected_receipt = {
            "lock_sha256": self.lock_digest,
            "manifest_sha256": self.inventory.manifest_sha256,
            "model_digest": identity.digest,
            "embedding_dimension": identity.dimension,
            "source_count": len(self.inventory.sources),
            "chunk_count": len(desired),
        }
        for key, expected in expected_receipt.items():
            if receipt.get(key) != expected:
                raise LiveVerificationError(f"local sync receipt drift: {key}")
        return {
            "verified": True,
            "collection": self.inventory.manifest.knowledge.collection,
            "lock_sha256": self.lock_digest,
            "model_digest": identity.digest,
            "source_count": len(self.inventory.sources),
            "chunk_count": len(desired),
        }

    async def query(
        self,
        query: str,
        *,
        top_k: int = 5,
        component: str | None = None,
    ) -> QueryResult:
        """Retrieve project knowledge only after live provenance verification."""

        if not query.strip():
            raise LiveVerificationError("query text cannot be empty")
        verified = await self.verify_live()
        vectors = await self.embedder.embed([query])
        filters = {"component": component} if component else None
        hits = await self.store.query(vectors[0], top_k=top_k, filters=filters)
        return QueryResult(
            query=query,
            lock_sha256=self.lock_digest,
            model_digest=str(verified["model_digest"]),
            hits=hits,
        )

    async def aclose(self) -> None:
        """Close both dependencies even if the first close fails."""

        try:
            await self.store.aclose()
        finally:
            await self.embedder.aclose()
