"""Deterministic offline evidence for source, ownership, and chunk drift."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .chunking import chunk_id, chunk_text
from .errors import DriftError, ManifestError
from .manifest import GovernedFile, ManifestInventory, load_inventory


LockData = dict[str, Any]


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize generated evidence identically on Windows and Linux."""

    return (
        json.dumps(
            value, ensure_ascii=False, indent=2, sort_keys=True, separators=(",", ": ")
        )
        + "\n"
    ).encode("utf-8")


def lock_sha256(lock: LockData) -> str:
    """Hash the canonical lock representation."""

    return hashlib.sha256(canonical_json_bytes(lock)).hexdigest()


def _source_lock(inventory: ManifestInventory, source) -> LockData:
    settings = inventory.manifest.knowledge
    chunks = chunk_text(
        source.text,
        max_chars=settings.max_chunk_chars,
        overlap_chars=settings.overlap_chars,
    )
    if not chunks:
        raise ManifestError(f"empty embedding source is forbidden: {source.path}")
    return {
        "path": source.path,
        "component": source.component_id,
        "categories": list(source.categories),
        "sha256": source.sha256,
        "bytes": source.byte_count,
        "chunks": [
            {
                "id": chunk_id(
                    project=inventory.manifest.project,
                    model=settings.embedding_model,
                    dimension=settings.embedding_dimension,
                    chunker_version=settings.chunker_version,
                    path=source.path,
                    chunk=chunk,
                ),
                "index": chunk.index,
                "sha256": chunk.sha256,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "chars": len(chunk.text),
            }
            for chunk in chunks
        ],
    }


def _governed_file_lock(governed: GovernedFile) -> LockData:
    """Serialize ownership and canonical content evidence without file text."""

    return {
        "path": governed.path,
        "component": governed.component_id,
        "owner": governed.owner,
        "sha256": governed.sha256,
        "bytes": governed.byte_count,
    }


def build_lock(inventory: ManifestInventory) -> LockData:
    """Resolve the manifest into deterministic, vector-free CI evidence."""

    settings = inventory.manifest.knowledge
    governed_files = [
        _governed_file_lock(governed) for governed in inventory.governed_files
    ]
    sources = [_source_lock(inventory, source) for source in inventory.sources]
    component_contracts = [
        {
            "id": component.id,
            "owner": component.owner,
            "primary_responsibility": component.primary_responsibility,
            "documentation": list(component.documentation),
            "architecture": list(component.architecture),
            "tests": list(component.tests),
            "upstream_references": [
                reference.model_dump(mode="json")
                for reference in component.upstream_references
            ],
        }
        for component in sorted(inventory.manifest.components, key=lambda item: item.id)
    ]
    return {
        "schema_version": 1,
        "project": inventory.manifest.project,
        "manifest_sha256": inventory.manifest_sha256,
        "knowledge": {
            "source_of_truth": settings.source_of_truth,
            "embedding_provider": settings.embedding_provider,
            "embedding_model": settings.embedding_model,
            "embedding_dimension": settings.embedding_dimension,
            "model_digest_policy": settings.model_digest_policy,
            "vector_store": settings.vector_store,
            "chroma_client_version": settings.chroma_client_version,
            "chroma_server_image": settings.chroma_server_image,
            "collection": settings.collection,
            "chunker_version": settings.chunker_version,
            "max_chunk_chars": settings.max_chunk_chars,
            "overlap_chars": settings.overlap_chars,
        },
        "components": component_contracts,
        "governed_files": governed_files,
        "sources": sources,
        "summary": {
            "component_count": len(component_contracts),
            "governed_file_count": len(governed_files),
            "governed_bytes": sum(item["bytes"] for item in governed_files),
            "source_count": len(sources),
            "chunk_count": sum(len(source["chunks"]) for source in sources),
            "content_bytes": sum(source["bytes"] for source in sources),
        },
    }


def lock_path(inventory: ManifestInventory) -> Path:
    """Resolve the configured generated lock without allowing path escape."""

    candidate = (inventory.root / inventory.manifest.knowledge.lock_file).resolve()
    try:
        candidate.relative_to(inventory.root.resolve())
    except ValueError as exc:
        raise ManifestError("knowledge.lock_file escapes the repository") from exc
    return candidate


def write_lock(inventory: ManifestInventory, lock: LockData | None = None) -> Path:
    """Atomically write the deterministic lock; this never touches Chroma."""

    destination = lock_path(inventory)
    payload = canonical_json_bytes(lock or build_lock(inventory))
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def load_lock(path: str | Path) -> LockData:
    """Read minimally validated generated evidence."""

    lock_file = Path(path)
    try:
        data = json.loads(lock_file.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DriftError(f"cannot read generated knowledge lock: {lock_file}") from exc
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise DriftError("generated knowledge lock has an unsupported schema")
    if (
        not isinstance(data.get("governed_files"), list)
        or not isinstance(data.get("sources"), list)
        or not isinstance(data.get("summary"), dict)
    ):
        raise DriftError("generated knowledge lock is structurally invalid")
    return data


def _drift_summary(expected: LockData, actual: LockData) -> str:
    expected_governed = {
        item["path"]: item for item in expected.get("governed_files", [])
    }
    actual_governed = {item["path"]: item for item in actual.get("governed_files", [])}
    governed_added = sorted(set(expected_governed) - set(actual_governed))
    governed_removed = sorted(set(actual_governed) - set(expected_governed))
    governed_changed = sorted(
        path
        for path in set(expected_governed) & set(actual_governed)
        if expected_governed[path] != actual_governed[path]
    )
    expected_sources = {item["path"]: item for item in expected.get("sources", [])}
    actual_sources = {item["path"]: item for item in actual.get("sources", [])}
    added = sorted(set(expected_sources) - set(actual_sources))
    removed = sorted(set(actual_sources) - set(expected_sources))
    changed = sorted(
        path
        for path in set(expected_sources) & set(actual_sources)
        if expected_sources[path] != actual_sources[path]
    )
    parts: list[str] = []
    if expected.get("manifest_sha256") != actual.get("manifest_sha256"):
        parts.append("manifest changed")
    if governed_added:
        parts.append("new governed files: " + ", ".join(governed_added[:8]))
    if governed_removed:
        parts.append("removed governed files: " + ", ".join(governed_removed[:8]))
    if governed_changed:
        parts.append("changed governed files: " + ", ".join(governed_changed[:8]))
    if added:
        parts.append("new sources: " + ", ".join(added[:8]))
    if removed:
        parts.append("removed sources: " + ", ".join(removed[:8]))
    if changed:
        parts.append("changed sources: " + ", ".join(changed[:8]))
    if not parts:
        parts.append("generated structure or summary changed")
    return "; ".join(parts)


def check_lock(inventory: ManifestInventory) -> LockData:
    """Fail CI when manifest, ownership, docs, or chunk evidence drifts."""

    expected = build_lock(inventory)
    actual = load_lock(lock_path(inventory))
    if canonical_json_bytes(expected) != canonical_json_bytes(actual):
        raise DriftError(
            "project knowledge drift detected; run `python scripts/knowledge.py lock`: "
            + _drift_summary(expected, actual)
        )
    return actual


def load_checked_lock(
    manifest_path: str | Path = "project.manifest.toml",
) -> tuple[ManifestInventory, LockData]:
    """Convenience boundary used by every live operation."""

    inventory = load_inventory(manifest_path)
    return inventory, check_lock(inventory)
