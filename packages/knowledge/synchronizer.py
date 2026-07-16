"""Incremental, resumable synchronization from the lock into local Chroma."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import struct
import tempfile
from collections.abc import Mapping, Sequence
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


_DOCUMENT_SHA256 = "document_sha256"
_EMBEDDING_SHA256 = "embedding_sha256"
_RECORD_INTEGRITY_VERSION = 1


@dataclass(frozen=True, slots=True)
class DesiredRecord:
    id: str
    document: str
    metadata: dict[str, Scalar]


def _document_sha256(document: str) -> str:
    return hashlib.sha256(document.encode("utf-8")).hexdigest()


def _embedding_sha256(embedding: Sequence[float], *, expected_dimension: int) -> str:
    """Fingerprint the float32 vector representation persisted by Chroma."""

    if len(embedding) != expected_dimension:
        raise ValueError("embedding dimension drift")
    digest = hashlib.sha256()
    digest.update(struct.pack("<I", expected_dimension))
    has_non_zero = False
    for raw_value in embedding:
        try:
            value = float(raw_value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("embedding contains a non-numeric value") from exc
        if not math.isfinite(value):
            raise ValueError("embedding contains NaN or infinity")
        try:
            packed = struct.pack("<f", value)
        except (OverflowError, struct.error) as exc:
            raise ValueError("embedding exceeds float32 range") from exc
        normalized = struct.unpack("<f", packed)[0]
        if not math.isfinite(normalized):
            raise ValueError("embedding exceeds float32 range")
        has_non_zero = has_non_zero or normalized != 0.0
        digest.update(packed)
    if not has_non_zero:
        raise ValueError("embedding is a zero vector")
    return digest.hexdigest()


def _record_integrity_issue(
    actual: StoredRecord,
    desired: DesiredRecord,
    *,
    expected_dimension: int,
) -> str | None:
    actual_metadata = dict(actual.metadata)
    stored_embedding_sha256 = actual_metadata.pop(_EMBEDDING_SHA256, None)
    if actual_metadata != desired.metadata:
        return "metadata drift"
    if actual.document != desired.document:
        return "document content drift"
    expected_document_sha256 = desired.metadata.get(_DOCUMENT_SHA256)
    if not isinstance(expected_document_sha256, str) or not hmac.compare_digest(
        _document_sha256(actual.document), expected_document_sha256
    ):
        return "document fingerprint drift"
    if not isinstance(stored_embedding_sha256, str):
        return "embedding fingerprint is missing"
    try:
        actual_embedding_sha256 = _embedding_sha256(
            actual.embedding,
            expected_dimension=expected_dimension,
        )
    except ValueError as exc:
        return str(exc)
    if not hmac.compare_digest(actual_embedding_sha256, stored_embedding_sha256):
        return "embedding fingerprint drift"
    return None


def _inventory_sha256(
    records: Mapping[str, StoredRecord], *, expected_dimension: int
) -> str:
    """Aggregate complete record fingerprints for receipt/namespace provenance."""

    digest = hashlib.sha256()
    for record_id, record in sorted(records.items()):
        values = (
            record_id,
            _document_sha256(record.document),
            _embedding_sha256(
                record.embedding,
                expected_dimension=expected_dimension,
            ),
        )
        for value in values:
            encoded = value.encode("utf-8")
            digest.update(struct.pack("<I", len(encoded)))
            digest.update(encoded)
    return digest.hexdigest()


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
            document = (
                f"Source: {source.path}\n"
                f"Component: {source.component_id}\n"
                f"Categories: {categories}\n"
                f"Lines: {chunk.start_line}-{chunk.end_line}\n\n"
                f"{chunk.text}"
            )
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
                _DOCUMENT_SHA256: _document_sha256(document),
            }
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

    async def _validated_identity(self) -> ModelIdentity:
        identity = await self.embedder.identity()
        expected = self.inventory.manifest.knowledge
        if identity.requested_model != expected.embedding_model:
            raise LiveVerificationError("embedder model does not match the manifest")
        if identity.dimension != expected.embedding_dimension:
            raise LiveVerificationError(
                "embedder dimension does not match the manifest"
            )
        return identity

    async def sync(self) -> SyncReport:
        """Embed only missing/corrupt chunks, verify, then delete stale chunks."""

        identity = await self._validated_identity()
        await self.store.ensure_namespace(
            _stable_namespace_metadata(self.inventory, identity)
        )
        desired = _desired_records(self.inventory, self.lock)
        existing = await self.store.inventory()
        expected_dimension = identity.dimension
        missing = [
            record
            for record_id, record in sorted(desired.items())
            if record_id not in existing
            or _record_integrity_issue(
                existing[record_id],
                record,
                expected_dimension=expected_dimension,
            )
            is not None
        ]
        batch_size = self.inventory.manifest.knowledge.embedding_batch_size
        upsert_size = self.inventory.manifest.knowledge.upsert_batch_size
        embedded = 0
        for batch_start in range(0, len(missing), batch_size):
            batch = missing[batch_start : batch_start + batch_size]
            vectors = await self.embedder.embed([record.document for record in batch])
            if len(vectors) != len(batch):
                raise LiveVerificationError(
                    "embedder result count does not match the requested batch"
                )
            vector_records: list[VectorRecord] = []
            for record, vector in zip(batch, vectors, strict=True):
                try:
                    embedding_sha256 = _embedding_sha256(
                        vector,
                        expected_dimension=expected_dimension,
                    )
                except ValueError as exc:
                    raise LiveVerificationError(
                        f"embedder returned an invalid vector for {record.id}"
                    ) from exc
                vector_records.append(
                    VectorRecord(
                        id=record.id,
                        embedding=tuple(vector),
                        document=record.document,
                        metadata={
                            **record.metadata,
                            _EMBEDDING_SHA256: embedding_sha256,
                        },
                    )
                )
            for upsert_start in range(0, len(vector_records), upsert_size):
                await self.store.upsert(
                    vector_records[upsert_start : upsert_start + upsert_size]
                )
            embedded += len(batch)

        after_upsert = await self.store.inventory()
        bad: list[tuple[str, str]] = []
        for record_id, record in desired.items():
            if record_id not in after_upsert:
                bad.append((record_id, "record is missing"))
                continue
            issue = _record_integrity_issue(
                after_upsert[record_id],
                record,
                expected_dimension=expected_dimension,
            )
            if issue is not None:
                bad.append((record_id, issue))
        if bad:
            record_id, issue = bad[0]
            raise LiveVerificationError(
                "Chroma did not persist complete knowledge records: "
                f"count={len(bad)}, first={record_id} ({issue})"
            )

        stale = sorted(set(after_upsert) - set(desired))
        deleted = await self.store.delete(stale)
        final_inventory = await self.store.inventory()
        if set(final_inventory) != set(desired):
            raise LiveVerificationError("Chroma inventory mismatch after stale cleanup")
        for record_id, record in desired.items():
            issue = _record_integrity_issue(
                final_inventory[record_id],
                record,
                expected_dimension=expected_dimension,
            )
            if issue is not None:
                raise LiveVerificationError(
                    f"Chroma record integrity mismatch: {record_id} ({issue})"
                )

        inventory_digest = _inventory_sha256(
            final_inventory,
            expected_dimension=expected_dimension,
        )

        active_metadata: dict[str, Scalar] = {
            "active_lock_sha256": self.lock_digest,
            "active_source_count": len(self.inventory.sources),
            "active_chunk_count": len(desired),
            "active_record_integrity_version": _RECORD_INTEGRITY_VERSION,
            "active_inventory_sha256": inventory_digest,
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
            "record_integrity_version": _RECORD_INTEGRITY_VERSION,
            "inventory_sha256": inventory_digest,
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

        identity = await self._validated_identity()
        await self.store.require_namespace(
            _stable_namespace_metadata(self.inventory, identity)
        )
        desired = _desired_records(self.inventory, self.lock)
        actual = await self.store.inventory()
        if set(actual) != set(desired):
            missing = len(set(desired) - set(actual))
            stale = len(set(actual) - set(desired))
            raise LiveVerificationError(
                f"live Chroma ID drift: missing={missing}, stale={stale}"
            )
        for record_id, record in desired.items():
            issue = _record_integrity_issue(
                actual[record_id],
                record,
                expected_dimension=identity.dimension,
            )
            if issue is not None:
                raise LiveVerificationError(
                    f"live Chroma record integrity drift: {record_id} ({issue})"
                )
        inventory_digest = _inventory_sha256(
            actual,
            expected_dimension=identity.dimension,
        )
        namespace = await self.store.namespace_metadata()
        if namespace.get("active_lock_sha256") != self.lock_digest:
            raise LiveVerificationError(
                "Chroma active lock does not match the committed lock"
            )
        if (
            namespace.get("active_record_integrity_version")
            != _RECORD_INTEGRITY_VERSION
            or namespace.get("active_inventory_sha256") != inventory_digest
        ):
            raise LiveVerificationError("Chroma active record integrity drift")
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
            "record_integrity_version": _RECORD_INTEGRITY_VERSION,
            "inventory_sha256": inventory_digest,
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
