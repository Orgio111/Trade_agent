"""Offline benchmark for the deterministic project-knowledge pipeline.

The benchmark deliberately avoids Ollama and Chroma.  It measures the real
manifest/lock planner when a manifest is available, then exercises the real
canonical hashing and chunking functions with deterministic synthetic input.
The embedding and store stages are small in-memory fakes so this command is
safe to run in CI or on an air-gapped development machine.
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import hashlib
import json
import math
import sys
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Awaitable, Callable, Sequence, TypeVar


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ""}:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from packages.knowledge.chunking import chunk_id, chunk_text, sha256_text  # noqa: E402
from packages.knowledge.lockfile import build_lock  # noqa: E402
from packages.knowledge.manifest import load_inventory  # noqa: E402


DEFAULT_SOURCE_COUNT = 64
DEFAULT_SOURCE_CHARS = 8_192
DEFAULT_CHUNK_CHARS = 4_000
DEFAULT_OVERLAP_CHARS = 400
DEFAULT_BATCH_SIZE = 32
DEFAULT_EMBEDDING_DIMENSION = 768

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class PlannedChunk:
    """Minimal immutable input passed to the fake online stages."""

    id: str
    document: str


@dataclass(frozen=True, slots=True)
class Measurement:
    """Wall-clock and allocation evidence for one benchmark phase."""

    value: Any
    elapsed_ns: int
    peak_bytes: int


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def _non_negative_int(raw: str) -> int:
    value = int(raw)
    if value < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return value


def _measure(call: Callable[[], T]) -> Measurement:
    gc.collect()
    tracemalloc.start()
    started = perf_counter_ns()
    try:
        value = call()
        elapsed_ns = max(perf_counter_ns() - started, 1)
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return Measurement(value=value, elapsed_ns=elapsed_ns, peak_bytes=peak_bytes)


async def _measure_async(call: Callable[[], Awaitable[T]]) -> Measurement:
    gc.collect()
    tracemalloc.start()
    started = perf_counter_ns()
    try:
        value = await call()
        elapsed_ns = max(perf_counter_ns() - started, 1)
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return Measurement(value=value, elapsed_ns=elapsed_ns, peak_bytes=peak_bytes)


def _elapsed_ms(elapsed_ns: int) -> float:
    return round(elapsed_ns / 1_000_000, 3)


def _throughput(count: int, elapsed_ns: int) -> float:
    return round(count * 1_000_000_000 / max(elapsed_ns, 1), 3)


def _synthetic_source(index: int, size: int) -> str:
    """Build a deterministic text source of exactly ``size`` characters."""

    line = (
        f"source={index:06d} component=knowledge-benchmark "
        "owner=local-software-engineering responsibility=drift-detection\n"
    )
    repeats = math.ceil(size / len(line))
    return (line * repeats)[:size]


def _plan_synthetic_sources(
    sources: Sequence[tuple[str, str]],
    *,
    max_chunk_chars: int,
    overlap_chars: int,
    embedding_dimension: int,
) -> tuple[list[PlannedChunk], str]:
    """Use production canonical hashing/chunking to build an immutable plan."""

    plan: list[PlannedChunk] = []
    digest = hashlib.sha256()
    for path, text in sources:
        digest.update(path.encode("utf-8"))
        digest.update(sha256_text(text).encode("ascii"))
        chunks = chunk_text(
            text,
            max_chars=max_chunk_chars,
            overlap_chars=overlap_chars,
        )
        for chunk in chunks:
            record_id = chunk_id(
                project="quantex-knowledge-benchmark",
                model="nomic-embed-text",
                dimension=embedding_dimension,
                chunker_version="benchmark-v1",
                path=path,
                chunk=chunk,
            )
            digest.update(record_id.encode("ascii"))
            plan.append(PlannedChunk(id=record_id, document=chunk.text))
    return plan, digest.hexdigest()


class _FakeEmbedder:
    """Deterministic, allocation-realistic local stand-in for Ollama."""

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension

    async def embed(self, documents: Sequence[str]) -> list[tuple[float, ...]]:
        vectors: list[tuple[float, ...]] = []
        for document in documents:
            seed = hashlib.sha256(document.encode("utf-8")).digest()
            vectors.append(
                tuple(
                    (seed[index % len(seed)] - 127.5) / 127.5
                    for index in range(self.dimension)
                )
            )
        return vectors


class _FakeStore:
    """In-memory store retaining only evidence needed to verify every upsert."""

    def __init__(self) -> None:
        self.records: dict[str, tuple[int, float]] = {}

    async def upsert(
        self,
        ids: Sequence[str],
        vectors: Sequence[tuple[float, ...]],
    ) -> None:
        if len(ids) != len(vectors):
            raise ValueError("fake store received mismatched IDs and vectors")
        for record_id, vector in zip(ids, vectors, strict=True):
            self.records[record_id] = (len(vector), vector[0] if vector else 0.0)


async def _run_fake_pipeline(
    plan: Sequence[PlannedChunk],
    *,
    batch_size: int,
    embedding_dimension: int,
) -> tuple[int, int]:
    embedder = _FakeEmbedder(embedding_dimension)
    store = _FakeStore()
    batch_count = 0
    for start in range(0, len(plan), batch_size):
        batch = plan[start : start + batch_size]
        vectors = await embedder.embed([record.document for record in batch])
        await store.upsert([record.id for record in batch], vectors)
        batch_count += 1
    if len(store.records) != len(plan):
        raise RuntimeError("fake store did not retain every planned chunk")
    return batch_count, len(store.records)


def _manifest_benchmark(manifest_path: Path) -> dict[str, Any]:
    if not manifest_path.is_file():
        return {
            "status": "not_found",
            "path": str(manifest_path),
            "source_count": 0,
            "chunk_count": 0,
            "elapsed_ms": 0.0,
            "peak_bytes": 0,
            "sources_per_second": 0.0,
            "chunks_per_second": 0.0,
            "batch_count": 0,
        }

    measurement = _measure(lambda: build_lock(load_inventory(manifest_path)))
    lock = measurement.value
    summary = lock["summary"]
    source_count = int(summary["source_count"])
    chunk_count = int(summary["chunk_count"])
    return {
        "status": "ok",
        "path": str(manifest_path),
        "source_count": source_count,
        "chunk_count": chunk_count,
        "elapsed_ms": _elapsed_ms(measurement.elapsed_ns),
        "peak_bytes": measurement.peak_bytes,
        "sources_per_second": _throughput(source_count, measurement.elapsed_ns),
        "chunks_per_second": _throughput(chunk_count, measurement.elapsed_ns),
        "batch_count": 1,
    }


async def _benchmark(args: argparse.Namespace) -> dict[str, Any]:
    if args.overlap_chars >= args.chunk_chars:
        raise ValueError("--overlap-chars must be smaller than --chunk-chars")

    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = REPOSITORY_ROOT / manifest_path
    manifest_result = _manifest_benchmark(manifest_path.resolve())

    sources = [
        (
            f"benchmark/source-{index:06d}.md",
            _synthetic_source(index, args.source_chars),
        )
        for index in range(args.sources)
    ]
    plan_measurement = _measure(
        lambda: _plan_synthetic_sources(
            sources,
            max_chunk_chars=args.chunk_chars,
            overlap_chars=args.overlap_chars,
            embedding_dimension=args.embedding_dimension,
        )
    )
    plan, plan_digest = plan_measurement.value
    plan_result = {
        "source_count": len(sources),
        "chunk_count": len(plan),
        "content_chars": sum(len(text) for _, text in sources),
        "plan_sha256": plan_digest,
        "elapsed_ms": _elapsed_ms(plan_measurement.elapsed_ns),
        "peak_bytes": plan_measurement.peak_bytes,
        "sources_per_second": _throughput(len(sources), plan_measurement.elapsed_ns),
        "chunks_per_second": _throughput(len(plan), plan_measurement.elapsed_ns),
        "batch_count": 1,
    }

    pipeline_measurement = await _measure_async(
        lambda: _run_fake_pipeline(
            plan,
            batch_size=args.batch_size,
            embedding_dimension=args.embedding_dimension,
        )
    )
    batch_count, stored_count = pipeline_measurement.value
    fake_result = {
        "source_count": len(sources),
        "chunk_count": stored_count,
        "elapsed_ms": _elapsed_ms(pipeline_measurement.elapsed_ns),
        "peak_bytes": pipeline_measurement.peak_bytes,
        "sources_per_second": _throughput(
            len(sources), pipeline_measurement.elapsed_ns
        ),
        "chunks_per_second": _throughput(stored_count, pipeline_measurement.elapsed_ns),
        "batch_count": batch_count,
    }

    return {
        "schema_version": 1,
        "benchmark": "quantex-local-knowledge-pipeline",
        "offline": True,
        "parameters": {
            "sources": args.sources,
            "source_chars": args.source_chars,
            "chunk_chars": args.chunk_chars,
            "overlap_chars": args.overlap_chars,
            "batch_size": args.batch_size,
            "embedding_dimension": args.embedding_dimension,
        },
        "manifest_plan": manifest_result,
        "synthetic_plan": plan_result,
        "fake_embedding_store": fake_result,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark the local knowledge manifest and fake sync path offline."
    )
    parser.add_argument("--manifest", default="project.manifest.toml")
    parser.add_argument("--sources", type=_positive_int, default=DEFAULT_SOURCE_COUNT)
    parser.add_argument(
        "--source-chars", type=_positive_int, default=DEFAULT_SOURCE_CHARS
    )
    parser.add_argument(
        "--chunk-chars", type=_positive_int, default=DEFAULT_CHUNK_CHARS
    )
    parser.add_argument(
        "--overlap-chars",
        type=_non_negative_int,
        default=DEFAULT_OVERLAP_CHARS,
    )
    parser.add_argument("--batch-size", type=_positive_int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--embedding-dimension",
        type=_positive_int,
        default=DEFAULT_EMBEDDING_DIMENSION,
    )
    parser.add_argument("--pretty", action="store_true", help="indent the JSON output")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = asyncio.run(_benchmark(args))
    except Exception as exc:  # JSON is the public contract, including failures.
        result = {
            "schema_version": 1,
            "benchmark": "quantex-local-knowledge-pipeline",
            "offline": True,
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 1

    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2 if args.pretty else None,
            sort_keys=True,
            separators=None if args.pretty else (",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
