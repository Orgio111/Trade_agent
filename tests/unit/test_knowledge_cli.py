"""Tests for collection-scoped knowledge CLI coordination."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from packages.knowledge import cli as knowledge_cli
from packages.knowledge.cli import LocalProcessLock, _sync_lease_path
from packages.knowledge.errors import KnowledgeError


def _service(*, collection: str, root: Path) -> SimpleNamespace:
    settings = SimpleNamespace(
        chroma_host="127.0.0.1",
        chroma_port=8100,
        chroma_ssl=False,
        collection=collection,
    )
    manifest = SimpleNamespace(knowledge=settings)
    inventory = SimpleNamespace(root=root, manifest=manifest)
    return SimpleNamespace(inventory=inventory)


def test_sync_lease_is_shared_across_checkouts(tmp_path: Path) -> None:
    first = _sync_lease_path(_service(collection="project-v1", root=tmp_path / "first"))
    second = _sync_lease_path(
        _service(collection="project-v1", root=tmp_path / "second")
    )

    assert first == second
    assert tmp_path not in first.parents


def test_sync_lease_isolated_by_collection(tmp_path: Path) -> None:
    first = _sync_lease_path(_service(collection="project-v1", root=tmp_path))
    second = _sync_lease_path(_service(collection="project-v2", root=tmp_path))

    assert first != second


def test_second_writer_is_rejected(tmp_path: Path) -> None:
    lease = tmp_path / "knowledge.lock"

    with LocalProcessLock(lease):
        with pytest.raises(KnowledgeError, match="another knowledge operation"):
            with LocalProcessLock(lease):
                pytest.fail("the second writer must not acquire the lease")


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["verify", "query"])
async def test_live_read_operations_hold_the_collection_lease(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    command: str,
) -> None:
    events: list[str] = []

    class Service:
        def __init__(self) -> None:
            self.inventory = _service(collection="project-v1", root=tmp_path).inventory

        async def verify_live(self):
            events.append("verify")
            return {"verified": True}

        async def query(self, *_args, **_kwargs):
            events.append("query")
            return SimpleNamespace(
                query="question",
                lock_sha256="lock",
                model_digest="model",
                hits=[],
            )

        async def aclose(self) -> None:
            events.append("close")

    class Lock:
        def __init__(self, _path: Path) -> None:
            pass

        def __enter__(self):
            events.append("enter")
            return self

        def __exit__(self, *_args) -> None:
            events.append("exit")

    service = Service()
    monkeypatch.setattr(knowledge_cli, "_service", lambda _manifest: service)
    monkeypatch.setattr(knowledge_cli, "LocalProcessLock", Lock)
    args = SimpleNamespace(
        manifest="project.manifest.toml",
        command=command,
        text="question",
        top_k=1,
        component=None,
    )

    await knowledge_cli._run_live(args)

    assert events == ["enter", command, "exit", "close"]
