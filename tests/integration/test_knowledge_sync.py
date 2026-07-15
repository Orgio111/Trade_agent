"""Offline end-to-end tests for manifest-to-vector synchronization."""

from __future__ import annotations

from pathlib import Path

import pytest

from packages.knowledge.errors import (
    EmbeddingError,
    LiveVerificationError,
    VectorStoreError,
)
from packages.knowledge.lockfile import build_lock
from packages.knowledge.manifest import load_inventory
from packages.knowledge.synchronizer import KnowledgeSynchronizer
from tests.fakes.knowledge import (
    DeterministicFakeEmbedder,
    InMemoryVectorStore,
    write_test_project,
)


def _service(
    manifest: Path,
    *,
    embedder: DeterministicFakeEmbedder,
    store: InMemoryVectorStore,
) -> KnowledgeSynchronizer:
    inventory = load_inventory(manifest)
    return KnowledgeSynchronizer(
        inventory=inventory,
        lock=build_lock(inventory),
        embedder=embedder,
        store=store,
    )


def _receipt(root: Path) -> Path:
    return root / ".local" / "knowledge" / "sync-receipt.json"


@pytest.mark.asyncio
async def test_initial_sync_embeds_and_activates_all_desired_chunks(
    tmp_path: Path,
) -> None:
    manifest = write_test_project(tmp_path)
    embedder = DeterministicFakeEmbedder()
    store = InMemoryVectorStore()
    service = _service(manifest, embedder=embedder, store=store)

    report = await service.sync()

    assert report.source_count == 2
    assert report.desired_chunks == len(store.records)
    assert report.embedded_chunks == report.desired_chunks
    assert report.unchanged_chunks == 0
    assert report.deleted_chunks == 0
    assert store.activation_calls == 1
    assert store.metadata is not None
    assert store.metadata["active_lock_sha256"] == report.lock_sha256
    assert _receipt(tmp_path).is_file()


@pytest.mark.asyncio
async def test_idempotent_sync_is_a_vector_no_op(tmp_path: Path) -> None:
    manifest = write_test_project(tmp_path)
    embedder = DeterministicFakeEmbedder()
    store = InMemoryVectorStore()
    service = _service(manifest, embedder=embedder, store=store)
    first = await service.sync()
    calls_after_first = embedder.embed_calls
    ids_after_first = set(store.records)

    second = await service.sync()

    assert second.embedded_chunks == 0
    assert second.deleted_chunks == 0
    assert second.unchanged_chunks == first.desired_chunks
    assert embedder.embed_calls == calls_after_first
    assert set(store.records) == ids_after_first


@pytest.mark.asyncio
async def test_changed_source_upserts_new_chunk_then_removes_stale_chunk(
    tmp_path: Path,
) -> None:
    manifest = write_test_project(tmp_path, source_text="VALUE = 'before'\n")
    embedder = DeterministicFakeEmbedder()
    store = InMemoryVectorStore()
    first_service = _service(manifest, embedder=embedder, store=store)
    await first_service.sync()
    old_source_ids = {
        record_id
        for record_id, record in store.records.items()
        if record.metadata["source_path"] == "src/app.py"
    }

    manifest = write_test_project(tmp_path, source_text="VALUE = 'after'\n")
    second_service = _service(manifest, embedder=embedder, store=store)
    report = await second_service.sync()
    new_source_ids = {
        record_id
        for record_id, record in store.records.items()
        if record.metadata["source_path"] == "src/app.py"
    }

    assert old_source_ids
    assert new_source_ids
    assert old_source_ids.isdisjoint(new_source_ids)
    assert old_source_ids <= set(store.deleted_ids)
    assert report.embedded_chunks == len(new_source_ids)
    assert report.deleted_chunks == len(old_source_ids)


@pytest.mark.asyncio
async def test_partial_embedding_failure_never_activates_or_writes_receipt(
    tmp_path: Path,
) -> None:
    long_source = (
        "\n".join(f"VALUE_{index} = '{'x' * 48}'" for index in range(30)) + "\n"
    )
    manifest = write_test_project(
        tmp_path,
        source_text=long_source,
        embedding_batch_size=1,
        max_chunk_chars=512,
        overlap_chars=0,
    )
    embedder = DeterministicFakeEmbedder(fail_on_call=2)
    store = InMemoryVectorStore()
    service = _service(manifest, embedder=embedder, store=store)

    with pytest.raises(EmbeddingError, match="injected embedding failure"):
        await service.sync()

    assert store.records, "the first batch proves the failure was genuinely partial"
    assert store.activation_calls == 0
    assert store.metadata is not None
    assert "active_lock_sha256" not in store.metadata
    assert not _receipt(tmp_path).exists()

    embedder.fail_on_call = None
    resumed = await service.sync()
    assert resumed.embedded_chunks < resumed.desired_chunks
    assert store.activation_calls == 1
    assert _receipt(tmp_path).is_file()


@pytest.mark.asyncio
async def test_live_verify_and_query_preserve_lock_and_model_provenance(
    tmp_path: Path,
) -> None:
    manifest = write_test_project(tmp_path)
    embedder = DeterministicFakeEmbedder()
    store = InMemoryVectorStore()
    service = _service(manifest, embedder=embedder, store=store)
    report = await service.sync()

    verification = await service.verify_live()
    result = await service.query(
        "Where is the local architecture documented?",
        top_k=3,
        component="test-component",
    )

    assert verification["verified"] is True
    assert verification["lock_sha256"] == report.lock_sha256
    assert result.lock_sha256 == report.lock_sha256
    assert result.model_digest == embedder.digest
    assert result.hits
    assert len(result.hits) <= 3
    assert all(hit.metadata["component"] == "test-component" for hit in result.hits)


@pytest.mark.asyncio
async def test_live_verify_detects_missing_vector(tmp_path: Path) -> None:
    manifest = write_test_project(tmp_path)
    embedder = DeterministicFakeEmbedder()
    store = InMemoryVectorStore()
    service = _service(manifest, embedder=embedder, store=store)
    await service.sync()
    store.records.pop(next(iter(store.records)))

    with pytest.raises(LiveVerificationError, match="live Chroma ID drift"):
        await service.verify_live()


@pytest.mark.asyncio
async def test_model_digest_namespace_mismatch_is_fail_closed(tmp_path: Path) -> None:
    manifest = write_test_project(tmp_path)
    store = InMemoryVectorStore()
    first_embedder = DeterministicFakeEmbedder(digest="sha256:model-digest-one")
    await _service(manifest, embedder=first_embedder, store=store).sync()
    activation_count = store.activation_calls

    replacement = DeterministicFakeEmbedder(digest="sha256:model-digest-two")
    service = _service(manifest, embedder=replacement, store=store)
    with pytest.raises(VectorStoreError, match="model_digest"):
        await service.sync()

    assert store.activation_calls == activation_count


@pytest.mark.asyncio
async def test_embedding_dimension_mismatch_is_fail_closed(tmp_path: Path) -> None:
    manifest = write_test_project(tmp_path, embedding_dimension=4)
    store = InMemoryVectorStore()
    embedder = DeterministicFakeEmbedder(dimension=5)
    service = _service(manifest, embedder=embedder, store=store)

    with pytest.raises(LiveVerificationError, match="dimension does not match"):
        await service.sync()

    assert store.activation_calls == 0
    assert not _receipt(tmp_path).exists()


@pytest.mark.asyncio
async def test_stored_dimension_metadata_drift_is_fail_closed(tmp_path: Path) -> None:
    manifest = write_test_project(tmp_path)
    embedder = DeterministicFakeEmbedder()
    store = InMemoryVectorStore()
    service = _service(manifest, embedder=embedder, store=store)
    await service.sync()
    assert store.metadata is not None
    store.metadata["embedding_dimension"] = 999

    with pytest.raises(VectorStoreError, match="embedding_dimension"):
        await service.verify_live()
