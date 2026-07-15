"""Async local Chroma server adapter with externally supplied embeddings."""

from __future__ import annotations

import inspect
import math
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any

from .errors import VectorStoreError
from .locality import require_loopback_host
from .models import (
    ComponentHealth,
    Scalar,
    SearchHit,
    StoredRecord,
    VectorRecord,
    json_scalar_mapping,
)


async def _await_if_needed(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


class ChromaVectorStore:
    """Project-knowledge collection isolated from all trading-memory stores."""

    def __init__(
        self,
        *,
        host: str,
        port: int,
        ssl: bool,
        collection: str,
        expected_dimension: int,
        client: Any | None = None,
        client_factory: Callable[[], Awaitable[Any] | Any] | None = None,
    ) -> None:
        try:
            host = require_loopback_host(host, service="Chroma")
        except ValueError as exc:
            raise VectorStoreError(str(exc)) from exc
        if not 1 <= port <= 65535:
            raise VectorStoreError("Chroma port is invalid")
        if expected_dimension < 1:
            raise VectorStoreError("Chroma embedding dimension must be positive")
        self._host = host
        self._port = port
        self._ssl = ssl
        self._collection_name = collection
        self._expected_dimension = expected_dimension
        self._client = client
        self._client_factory = client_factory
        self._collection: Any | None = None
        self._closed = False

    async def _get_client(self) -> Any:
        if self._closed:
            raise VectorStoreError("Chroma vector store is closed")
        if self._client is not None:
            return self._client
        try:
            if self._client_factory is not None:
                self._client = await _await_if_needed(self._client_factory())
            else:
                import chromadb

                self._client = await chromadb.AsyncHttpClient(
                    host=self._host,
                    port=self._port,
                    ssl=self._ssl,
                )
        except Exception as exc:
            raise VectorStoreError("cannot connect to local Chroma") from exc
        return self._client

    async def _get_collection(self) -> Any:
        if self._collection is None:
            raise VectorStoreError("Chroma namespace has not been initialized")
        return self._collection

    async def ensure_namespace(self, metadata: Mapping[str, Scalar]) -> None:
        """Create once, then reject model/schema namespace mismatches."""

        expected = json_scalar_mapping(metadata)
        try:
            client = await self._get_client()
            collection = await _await_if_needed(
                client.get_or_create_collection(
                    name=self._collection_name,
                    metadata=expected,
                    embedding_function=None,
                )
            )
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError("cannot initialize Chroma collection") from exc
        actual = getattr(collection, "metadata", None)
        if not isinstance(actual, dict):
            raise VectorStoreError("Chroma collection metadata is unavailable")
        mismatches = [
            key for key, value in expected.items() if actual.get(key) != value
        ]
        if mismatches:
            raise VectorStoreError(
                "Chroma namespace metadata drift; bump the collection version: "
                + ", ".join(sorted(mismatches))
            )
        self._collection = collection

    async def inventory(self) -> dict[str, StoredRecord]:
        try:
            collection = await self._get_collection()
            result = await _await_if_needed(collection.get(include=["metadatas"]))
            ids = result.get("ids", [])
            metadatas = result.get("metadatas", [])
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError("cannot inventory the Chroma collection") from exc
        if (
            not isinstance(ids, list)
            or not isinstance(metadatas, list)
            or len(ids) != len(metadatas)
        ):
            raise VectorStoreError("Chroma inventory response is malformed")
        records: dict[str, StoredRecord] = {}
        for record_id, metadata in zip(ids, metadatas, strict=True):
            if not isinstance(record_id, str) or not isinstance(metadata, dict):
                raise VectorStoreError("Chroma inventory contains malformed records")
            records[record_id] = StoredRecord(
                id=record_id,
                metadata=json_scalar_mapping(metadata),
            )
        return records

    async def namespace_metadata(self) -> Mapping[str, Scalar]:
        collection = await self._get_collection()
        metadata = getattr(collection, "metadata", None)
        if not isinstance(metadata, dict):
            raise VectorStoreError("Chroma collection metadata is unavailable")
        return json_scalar_mapping(metadata)

    async def upsert(self, records: Sequence[VectorRecord]) -> None:
        if not records:
            return
        ids: list[str] = []
        embeddings: list[list[float]] = []
        documents: list[str] = []
        metadatas: list[dict[str, Scalar]] = []
        for record in records:
            if len(record.embedding) != self._expected_dimension:
                raise VectorStoreError("refusing an embedding with dimension drift")
            if not all(math.isfinite(value) for value in record.embedding):
                raise VectorStoreError("refusing a non-finite embedding")
            ids.append(record.id)
            embeddings.append(list(record.embedding))
            documents.append(record.document)
            metadatas.append(json_scalar_mapping(record.metadata))
        try:
            collection = await self._get_collection()
            await _await_if_needed(
                collection.upsert(
                    ids=ids,
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas,
                )
            )
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError("Chroma upsert failed") from exc

    async def delete(self, ids: Sequence[str]) -> int:
        unique = sorted(set(ids))
        if not unique:
            return 0
        try:
            collection = await self._get_collection()
            await _await_if_needed(collection.delete(ids=unique))
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError("Chroma stale-record deletion failed") from exc
        return len(unique)

    async def activate(self, metadata: Mapping[str, Scalar]) -> None:
        """Publish the active lock only after desired IDs have been verified."""

        try:
            collection = await self._get_collection()
            current = getattr(collection, "metadata", {})
            merged = json_scalar_mapping({**current, **dict(metadata)})
            await _await_if_needed(collection.modify(metadata=merged))
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError(
                "cannot activate the Chroma knowledge generation"
            ) from exc

    async def query(
        self,
        vector: Sequence[float],
        *,
        top_k: int,
        filters: Mapping[str, Scalar] | None = None,
    ) -> list[SearchHit]:
        if len(vector) != self._expected_dimension:
            raise VectorStoreError("query vector dimension drift")
        if top_k < 1:
            raise VectorStoreError("top_k must be positive")
        try:
            collection = await self._get_collection()
            result = await _await_if_needed(
                collection.query(
                    query_embeddings=[list(vector)],
                    n_results=top_k,
                    where=json_scalar_mapping(filters) if filters else None,
                    include=["documents", "metadatas", "distances"],
                )
            )
            ids = result.get("ids", [[]])[0]
            documents = result.get("documents", [[]])[0]
            metadatas = result.get("metadatas", [[]])[0]
            distances = result.get("distances", [[]])[0]
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError("Chroma query failed") from exc
        if not (len(ids) == len(documents) == len(metadatas) == len(distances)):
            raise VectorStoreError("Chroma query response is malformed")
        return [
            SearchHit(
                id=record_id,
                document=document,
                metadata=json_scalar_mapping(metadata),
                distance=float(distance),
            )
            for record_id, document, metadata, distance in zip(
                ids, documents, metadatas, distances, strict=True
            )
        ]

    async def health(self) -> ComponentHealth:
        try:
            client = await self._get_client()
            await _await_if_needed(client.heartbeat())
        except VectorStoreError as exc:
            return ComponentHealth(healthy=False, detail=str(exc))
        except Exception:
            return ComponentHealth(
                healthy=False, detail="local Chroma heartbeat failed"
            )
        return ComponentHealth(
            healthy=True,
            detail=f"{self._host}:{self._port}/{self._collection_name}",
        )

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        client = self._client
        self._client = None
        if client is None:
            return
        close = getattr(client, "close", None)
        if close is not None:
            await _await_if_needed(close())
