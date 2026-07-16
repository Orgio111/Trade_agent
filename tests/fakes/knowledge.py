"""Offline knowledge-pipeline fakes with production-like failure semantics."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path

from packages.knowledge.errors import EmbeddingError, VectorStoreError
from packages.knowledge.models import (
    ComponentHealth,
    EmbeddingSpec,
    ModelIdentity,
    Scalar,
    SearchHit,
    StoredRecord,
    VectorRecord,
)


WIKI_DOCUMENT = """---
title: Test Component
type: concept
tags: [test]
created: 2026-07-15
updated: 2026-07-15
sources: []
status: stable
---

# Test Component

Architecture documentation for the test component.
"""


def write_test_project(
    root: Path,
    *,
    source_text: str = "def local_feature() -> str:\n    return 'offline'\n",
    embedding_dimension: int = 4,
    embedding_batch_size: int = 2,
    max_chunk_chars: int = 512,
    overlap_chars: int = 32,
    ollama_base_url: str = "http://127.0.0.1:11434",
    chroma_host: str = "127.0.0.1",
) -> Path:
    """Create the smallest valid governed repository for tests."""

    source = root / "src" / "app.py"
    document = root / "wiki" / "content" / "test-component.md"
    source.parent.mkdir(parents=True, exist_ok=True)
    document.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(source_text, encoding="utf-8", newline="\n")
    document.write_text(WIKI_DOCUMENT, encoding="utf-8", newline="\n")
    (root / "wiki" / "index.md").write_text(
        "# Index\n\n- [[test-component]] — fixture.\n",
        encoding="utf-8",
        newline="\n",
    )
    manifest = root / "project.manifest.toml"
    manifest.write_text(
        f'''schema_version = 1
project = "test-project"

[knowledge]
embedding_dimension = {embedding_dimension}
ollama_base_url = "{ollama_base_url}"
chroma_host = "{chroma_host}"
collection = "test_knowledge_v1"
chunker_version = "test-v1"
max_chunk_chars = {max_chunk_chars}
overlap_chars = {overlap_chars}
embedding_batch_size = {embedding_batch_size}
upsert_batch_size = 2

[coverage]
include = ["src/**/*.py", "wiki/index.md", "wiki/content/**/*.md"]

[[components]]
id = "test-component"
owner = "test-team"
primary_responsibility = "Provide deterministic offline test behavior"
owned_paths = ["src/**/*.py", "wiki/index.md", "wiki/content/**/*.md"]
tests = []
documentation = ["wiki/content/test-component.md"]
architecture = ["wiki/content/test-component.md"]
embedding_inputs = ["src/**/*.py", "wiki/content/**/*.md"]
forbidden_imports = ["openai"]
''',
        encoding="utf-8",
        newline="\n",
    )
    return manifest


class DeterministicFakeEmbedder:
    """Stable local embedder fake with injectable call failure."""

    def __init__(
        self,
        *,
        model: str = "nomic-embed-text",
        dimension: int = 4,
        digest: str = "sha256:test-model-digest-v1",
        fail_on_call: int | None = None,
    ) -> None:
        self._spec = EmbeddingSpec(model=model, dimension=dimension)
        self.digest = digest
        self.fail_on_call = fail_on_call
        self.embed_calls = 0
        self.embedded_texts: list[str] = []
        self.closed = False

    @property
    def spec(self) -> EmbeddingSpec:
        return self._spec

    async def identity(self) -> ModelIdentity:
        return ModelIdentity(
            requested_model=self._spec.model,
            resolved_model=f"{self._spec.model}:latest",
            digest=self.digest,
            dimension=self._spec.dimension,
        )

    async def embed(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        self.embed_calls += 1
        if self.fail_on_call == self.embed_calls:
            raise EmbeddingError("injected embedding failure")
        self.embedded_texts.extend(texts)
        vectors: list[tuple[float, ...]] = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            vectors.append(
                tuple(
                    (digest[index % len(digest)] + 1) / 256.0
                    for index in range(self._spec.dimension)
                )
            )
        return vectors

    async def health(self) -> ComponentHealth:
        return ComponentHealth(healthy=True, detail="deterministic fake")

    async def aclose(self) -> None:
        self.closed = True


class InMemoryVectorStore:
    """In-memory implementation of the vector-store port for integration tests."""

    def __init__(self) -> None:
        self.records: dict[str, VectorRecord] = {}
        self.metadata: dict[str, Scalar] | None = None
        self.activation_calls = 0
        self.deleted_ids: list[str] = []
        self.closed = False

    async def ensure_namespace(self, metadata: Mapping[str, Scalar]) -> None:
        expected = dict(metadata)
        if self.metadata is None:
            self.metadata = expected
            return
        mismatches = [
            key for key, value in expected.items() if self.metadata.get(key) != value
        ]
        if mismatches:
            raise VectorStoreError(
                "namespace metadata drift: " + ", ".join(sorted(mismatches))
            )

    async def require_namespace(self, metadata: Mapping[str, Scalar]) -> None:
        if self.metadata is None:
            raise VectorStoreError("namespace does not exist")
        await self.ensure_namespace(metadata)

    async def inventory(self) -> dict[str, StoredRecord]:
        return {
            record_id: StoredRecord(
                id=record_id,
                embedding=tuple(record.embedding),
                document=record.document,
                metadata=dict(record.metadata),
            )
            for record_id, record in self.records.items()
        }

    async def namespace_metadata(self) -> Mapping[str, Scalar]:
        if self.metadata is None:
            raise VectorStoreError("namespace is not initialized")
        return dict(self.metadata)

    async def upsert(self, records: Sequence[VectorRecord]) -> None:
        for record in records:
            self.records[record.id] = record

    async def delete(self, ids: Sequence[str]) -> int:
        deleted = 0
        for record_id in sorted(set(ids)):
            if self.records.pop(record_id, None) is not None:
                deleted += 1
                self.deleted_ids.append(record_id)
        return deleted

    async def activate(self, metadata: Mapping[str, Scalar]) -> None:
        if self.metadata is None:
            raise VectorStoreError("namespace is not initialized")
        self.metadata.update(metadata)
        self.activation_calls += 1

    async def query(
        self,
        vector: Sequence[float],
        *,
        top_k: int,
        filters: Mapping[str, Scalar] | None = None,
    ) -> list[SearchHit]:
        del vector
        matching = [
            record
            for record in self.records.values()
            if not filters
            or all(record.metadata.get(key) == value for key, value in filters.items())
        ]
        return [
            SearchHit(
                id=record.id,
                document=record.document,
                metadata=dict(record.metadata),
                distance=index / 10.0,
            )
            for index, record in enumerate(
                sorted(matching, key=lambda item: item.id)[:top_k]
            )
        ]

    async def health(self) -> ComponentHealth:
        return ComponentHealth(healthy=True, detail="in-memory fake")

    async def aclose(self) -> None:
        self.closed = True
