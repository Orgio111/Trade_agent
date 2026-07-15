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


_INVENTORY_INCLUDE = ["metadatas", "documents", "embeddings"]


async def _await_if_needed(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


def _result_list(result: Mapping[str, Any], field: str) -> list[Any]:
    """Normalize Chroma lists and NumPy arrays without importing NumPy."""

    value = result.get(field)
    to_list = getattr(value, "tolist", None)
    if callable(to_list):
        value = to_list()
    if not isinstance(value, list):
        raise VectorStoreError(f"Chroma inventory {field} field is malformed")
    return value


def _stored_embedding(value: Any, *, expected_dimension: int) -> tuple[float, ...]:
    to_list = getattr(value, "tolist", None)
    if callable(to_list):
        value = to_list()
    if not isinstance(value, list):
        raise VectorStoreError("Chroma inventory contains a malformed embedding")
    try:
        embedding = tuple(float(item) for item in value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise VectorStoreError(
            "Chroma inventory contains a malformed embedding"
        ) from exc
    if len(embedding) != expected_dimension:
        raise VectorStoreError("Chroma inventory embedding dimension drift")
    if not all(math.isfinite(item) for item in embedding):
        raise VectorStoreError("Chroma inventory contains a non-finite embedding")
    if not any(item != 0.0 for item in embedding):
        raise VectorStoreError("Chroma inventory contains a zero embedding")
    return embedding


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
        expected_client_version: str,
        inventory_batch_size: int = 256,
        client: Any | None = None,
        client_factory: Callable[[], Awaitable[Any] | Any] | None = None,
        client_version_resolver: Callable[[], str] | None = None,
    ) -> None:
        try:
            host = require_loopback_host(host, service="Chroma")
        except ValueError as exc:
            raise VectorStoreError(str(exc)) from exc
        if not 1 <= port <= 65535:
            raise VectorStoreError("Chroma port is invalid")
        if expected_dimension < 1:
            raise VectorStoreError("Chroma embedding dimension must be positive")
        if not expected_client_version.strip():
            raise VectorStoreError("expected Chroma client version is required")
        if not 1 <= inventory_batch_size <= 2048:
            raise VectorStoreError(
                "Chroma inventory batch size must be between 1 and 2048"
            )
        self._host = host
        self._port = port
        self._ssl = ssl
        self._collection_name = collection
        self._expected_dimension = expected_dimension
        self._expected_client_version = expected_client_version
        self._inventory_batch_size = inventory_batch_size
        self._client = client
        self._client_factory = client_factory
        self._client_version_resolver = client_version_resolver
        self._client_version_verified = False
        self._collection: Any | None = None
        self._closed = False

    def _verify_client_version(self) -> None:
        if self._client_version_verified:
            return
        actual: Any
        try:
            if self._client_version_resolver is not None:
                actual = self._client_version_resolver()
            else:
                import chromadb

                actual = getattr(chromadb, "__version__", None)
        except Exception as exc:
            raise VectorStoreError("local Chroma client is unavailable") from exc
        if not isinstance(actual, str) or actual != self._expected_client_version:
            rendered = actual if isinstance(actual, str) else "unknown"
            raise VectorStoreError(
                "Chroma client version drift: "
                f"expected {self._expected_client_version}, got {rendered}"
            )
        self._client_version_verified = True

    async def _get_client(self) -> Any:
        if self._closed:
            raise VectorStoreError("Chroma vector store is closed")
        self._verify_client_version()
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
        """Read complete records in bounded pages for content-integrity checks."""

        try:
            collection = await self._get_collection()
            expected_count = await _await_if_needed(collection.count())
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError("cannot inventory the Chroma collection") from exc
        if (
            not isinstance(expected_count, int)
            or isinstance(expected_count, bool)
            or expected_count < 0
        ):
            raise VectorStoreError("Chroma inventory count is malformed")

        records: dict[str, StoredRecord] = {}
        offset = 0
        while offset < expected_count:
            limit = min(self._inventory_batch_size, expected_count - offset)
            try:
                result = await _await_if_needed(
                    collection.get(
                        limit=limit,
                        offset=offset,
                        include=_INVENTORY_INCLUDE,
                    )
                )
            except Exception as exc:
                raise VectorStoreError(
                    "cannot inventory the Chroma collection"
                ) from exc
            if not isinstance(result, Mapping):
                raise VectorStoreError("Chroma inventory response is malformed")
            ids = _result_list(result, "ids")
            metadatas = _result_list(result, "metadatas")
            documents = _result_list(result, "documents")
            embeddings = _result_list(result, "embeddings")
            if not ids or not (
                len(ids) == len(metadatas) == len(documents) == len(embeddings)
            ):
                raise VectorStoreError("Chroma inventory response is malformed")
            if len(ids) > limit:
                raise VectorStoreError("Chroma inventory page exceeded its limit")
            for record_id, metadata, document, embedding in zip(
                ids, metadatas, documents, embeddings, strict=True
            ):
                if (
                    not isinstance(record_id, str)
                    or not isinstance(metadata, dict)
                    or not isinstance(document, str)
                ):
                    raise VectorStoreError(
                        "Chroma inventory contains malformed records"
                    )
                if record_id in records:
                    raise VectorStoreError(
                        "Chroma inventory contains duplicate record IDs"
                    )
                try:
                    validated_metadata = json_scalar_mapping(metadata)
                except TypeError as exc:
                    raise VectorStoreError(
                        "Chroma inventory contains malformed metadata"
                    ) from exc
                records[record_id] = StoredRecord(
                    id=record_id,
                    embedding=_stored_embedding(
                        embedding,
                        expected_dimension=self._expected_dimension,
                    ),
                    document=document,
                    metadata=validated_metadata,
                )
            offset += len(ids)

        try:
            final_count = await _await_if_needed(collection.count())
        except Exception as exc:
            raise VectorStoreError("cannot finalize Chroma inventory") from exc
        if final_count != expected_count or len(records) != expected_count:
            raise VectorStoreError(
                "Chroma inventory changed while integrity verification was running"
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
