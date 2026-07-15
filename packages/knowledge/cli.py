"""Command-line interface for offline integrity and local live operations."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import os
from pathlib import Path
from types import TracebackType
from typing import Any

from .errors import KnowledgeError
from .lockfile import build_lock, check_lock, load_checked_lock, lock_sha256, write_lock
from .manifest import load_inventory


class LocalProcessLock:
    """Cross-platform non-blocking file lease for the single sync writer."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._handle: Any | None = None

    def __enter__(self) -> LocalProcessLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a+b")
        self._handle.seek(0, os.SEEK_END)
        if self._handle.tell() == 0:
            self._handle.write(b"0")
            self._handle.flush()
        self._handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self._handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                fcntl: Any = importlib.import_module("fcntl")
                fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self._handle.close()
            self._handle = None
            raise KnowledgeError("another knowledge sync is already running") from exc
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._handle is None:
            return
        try:
            self._handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self._handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl: Any = importlib.import_module("fcntl")
                fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        finally:
            self._handle.close()
            self._handle = None


def _json_print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _service(manifest_path: str):
    # Live-only imports keep the offline CI check free of httpx/Chroma dependencies.
    from .chroma import ChromaVectorStore
    from .ollama import OllamaEmbedder
    from .synchronizer import KnowledgeSynchronizer

    inventory, lock = load_checked_lock(manifest_path)
    settings = inventory.manifest.knowledge
    embedder = OllamaEmbedder(
        base_url=settings.ollama_base_url,
        model=settings.embedding_model,
        dimension=settings.embedding_dimension,
    )
    store = ChromaVectorStore(
        host=settings.chroma_host,
        port=settings.chroma_port,
        ssl=settings.chroma_ssl,
        collection=settings.collection,
        expected_dimension=settings.embedding_dimension,
    )
    return KnowledgeSynchronizer(
        inventory=inventory,
        lock=lock,
        embedder=embedder,
        store=store,
    )


async def _run_live(args: argparse.Namespace) -> dict[str, Any]:
    service = _service(args.manifest)
    try:
        if args.command == "sync":
            lease_path = service.inventory.root / ".local" / "knowledge" / "sync.lock"
            with LocalProcessLock(lease_path):
                report = await service.sync()
            return {
                "status": "synchronized",
                "source_count": report.source_count,
                "desired_chunks": report.desired_chunks,
                "embedded_chunks": report.embedded_chunks,
                "unchanged_chunks": report.unchanged_chunks,
                "deleted_chunks": report.deleted_chunks,
                "lock_sha256": report.lock_sha256,
                "model_digest": report.model_digest,
            }
        if args.command == "verify":
            return await service.verify_live()
        if args.command == "query":
            result = await service.query(
                args.text,
                top_k=args.top_k,
                component=args.component,
            )
            return {
                "query": result.query,
                "lock_sha256": result.lock_sha256,
                "model_digest": result.model_digest,
                "hits": [
                    {
                        "id": hit.id,
                        "distance": hit.distance,
                        "document": hit.document,
                        "metadata": dict(hit.metadata),
                    }
                    for hit in result.hits
                ],
            }
        raise KnowledgeError(f"unsupported live command: {args.command}")
    finally:
        await service.aclose()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quantex-knowledge",
        description="Manifest-driven local project-knowledge pipeline",
    )
    parser.add_argument(
        "--manifest",
        default="project.manifest.toml",
        help="path to the human-authored architecture manifest",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("lock", help="write deterministic offline evidence")
    subparsers.add_parser("check", help="fail on ownership/document/chunk drift")
    subparsers.add_parser("sync", help="incrementally embed and sync local Chroma")
    subparsers.add_parser(
        "verify", help="verify the real local Chroma index and receipt"
    )
    query = subparsers.add_parser("query", help="query verified project knowledge")
    query.add_argument("text")
    query.add_argument("--top-k", type=int, default=5)
    query.add_argument("--component")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "lock":
            inventory = load_inventory(args.manifest)
            lock = build_lock(inventory)
            destination = write_lock(inventory, lock)
            _json_print(
                {
                    "status": "lock-written",
                    "path": destination.relative_to(inventory.root).as_posix(),
                    "lock_sha256": lock_sha256(lock),
                    **lock["summary"],
                }
            )
            return 0
        if args.command == "check":
            inventory = load_inventory(args.manifest)
            lock = check_lock(inventory)
            _json_print(
                {
                    "status": "ok",
                    "manifest_sha256": inventory.manifest_sha256,
                    "lock_sha256": lock_sha256(lock),
                    **lock["summary"],
                }
            )
            return 0
        _json_print(asyncio.run(_run_live(args)))
        return 0
    except KnowledgeError as exc:
        _json_print({"status": "error", "error": str(exc)})
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
