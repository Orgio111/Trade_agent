"""Contract tests for complete, paginated Chroma integrity inventories."""

from __future__ import annotations

from typing import Any

import pytest

from packages.knowledge.chroma import ChromaVectorStore
from packages.knowledge.errors import VectorStoreError
from packages.knowledge.models import VectorRecord


class _ArrayLike:
    """Exercise the NumPy-shaped ``tolist`` boundary used by Chroma."""

    def __init__(self, values: list[list[float]]) -> None:
        self._values = values

    def tolist(self) -> list[list[float]]:
        return self._values


class _AsyncCollection:
    def __init__(self) -> None:
        self.metadata: dict[str, str | int | float | bool] = {}
        self.rows: dict[str, dict[str, Any]] = {}
        self.get_calls: list[tuple[int, int, tuple[str, ...]]] = []

    async def count(self) -> int:
        return len(self.rows)

    async def get(
        self,
        *,
        limit: int,
        offset: int,
        include: list[str],
    ) -> dict[str, Any]:
        self.get_calls.append((limit, offset, tuple(include)))
        page = [self.rows[key] for key in sorted(self.rows)[offset : offset + limit]]
        return {
            "ids": [row["id"] for row in page],
            "metadatas": [row["metadata"] for row in page],
            "documents": [row["document"] for row in page],
            "embeddings": _ArrayLike([row["embedding"] for row in page]),
        }

    async def upsert(
        self,
        *,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, str | int | float | bool]],
    ) -> None:
        assert len(ids) == len(embeddings) == len(documents) == len(metadatas)
        for record_id, embedding, document, metadata in zip(
            ids,
            embeddings,
            documents,
            metadatas,
            strict=True,
        ):
            self.rows[record_id] = {
                "id": record_id,
                "embedding": embedding,
                "document": document,
                "metadata": metadata,
            }

    async def modify(self, *, metadata: dict[str, str | int | float | bool]) -> None:
        self.metadata = metadata


class _AsyncClient:
    def __init__(self, collection: _AsyncCollection) -> None:
        self.collection = collection

    async def get_or_create_collection(
        self,
        *,
        name: str,
        metadata: dict[str, str | int | float | bool],
        embedding_function: None,
    ) -> _AsyncCollection:
        assert name == "project_knowledge_v1"
        assert embedding_function is None
        if not self.collection.metadata:
            self.collection.metadata = dict(metadata)
        return self.collection


def _store(
    collection: _AsyncCollection,
    *,
    runtime_version: str = "1.5.9",
) -> ChromaVectorStore:
    return ChromaVectorStore(
        host="127.0.0.1",
        port=8100,
        ssl=False,
        collection="project_knowledge_v1",
        expected_dimension=4,
        expected_client_version="1.5.9",
        inventory_batch_size=2,
        client=_AsyncClient(collection),
        client_version_resolver=lambda: runtime_version,
    )


@pytest.mark.asyncio
async def test_complete_inventory_uses_bounded_async_chroma_pages() -> None:
    collection = _AsyncCollection()
    store = _store(collection)
    await store.ensure_namespace({"schema_version": 1, "project": "trade-agent"})
    records = [
        VectorRecord(
            id=f"record-{index}",
            embedding=(0.1 + index, 0.2, 0.3, 0.4),
            document=f"document-{index}",
            metadata={"component": "knowledge", "chunk_index": index},
        )
        for index in range(3)
    ]

    await store.upsert(records)
    inventory = await store.inventory()

    assert set(inventory) == {"record-0", "record-1", "record-2"}
    assert inventory["record-1"].document == "document-1"
    assert inventory["record-1"].embedding == (1.1, 0.2, 0.3, 0.4)
    assert inventory["record-1"].metadata["chunk_index"] == 1
    assert collection.get_calls == [
        (2, 0, ("metadatas", "documents", "embeddings")),
        (1, 2, ("metadatas", "documents", "embeddings")),
    ]


@pytest.mark.asyncio
async def test_inventory_rejects_corrupt_stored_embedding_dimension() -> None:
    collection = _AsyncCollection()
    store = _store(collection)
    await store.ensure_namespace({"schema_version": 1, "project": "trade-agent"})
    collection.rows["bad"] = {
        "id": "bad",
        "embedding": [0.1, 0.2, 0.3],
        "document": "document",
        "metadata": {"component": "knowledge"},
    }

    with pytest.raises(VectorStoreError, match="embedding dimension drift"):
        await store.inventory()


@pytest.mark.asyncio
async def test_exact_chroma_client_version_is_enforced() -> None:
    store = _store(_AsyncCollection(), runtime_version="1.5.8")

    with pytest.raises(
        VectorStoreError,
        match=r"expected 1\.5\.9, got 1\.5\.8",
    ):
        await store.ensure_namespace({"schema_version": 1, "project": "trade-agent"})
