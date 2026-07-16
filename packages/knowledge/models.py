"""Strict data contracts shared by the knowledge pipeline."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, model_validator

from .locality import normalize_loopback_http_url, require_loopback_host


class StrictModel(BaseModel):
    """Immutable Pydantic base that rejects unknown manifest fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class KnowledgeSettings(StrictModel):
    """Local-only embedding and vector-store policy."""

    source_of_truth: Literal["wiki"] = "wiki"
    embedding_provider: Literal["ollama"] = "ollama"
    embedding_model: Literal["nomic-embed-text"] = "nomic-embed-text"
    embedding_dimension: PositiveInt = 768
    model_digest_policy: Literal["runtime_locked"] = "runtime_locked"
    ollama_base_url: str = "http://127.0.0.1:11434"
    vector_store: Literal["chromadb"] = "chromadb"
    chroma_host: str = "127.0.0.1"
    chroma_port: PositiveInt = 8100
    chroma_ssl: bool = False
    chroma_client_version: str = "1.5.9"
    chroma_server_image: str = "chromadb/chroma:1.5.9"
    collection: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{1,61}[a-z0-9]$")
    lock_file: str = "project.manifest.lock.json"
    receipt_file: str = ".local/knowledge/sync-receipt.json"
    chunker_version: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]+$")
    max_chunk_chars: int = Field(default=4000, ge=512, le=32_000)
    overlap_chars: int = Field(default=400, ge=0, le=8_000)
    embedding_batch_size: int = Field(default=16, ge=1, le=128)
    upsert_batch_size: int = Field(default=64, ge=1, le=512)

    @model_validator(mode="after")
    def validate_chunk_window(self) -> KnowledgeSettings:
        if self.overlap_chars >= self.max_chunk_chars:
            raise ValueError("overlap_chars must be smaller than max_chunk_chars")
        normalized_ollama = normalize_loopback_http_url(
            self.ollama_base_url,
            service="Ollama",
            default_port=11434,
        )
        normalized_chroma = require_loopback_host(self.chroma_host, service="Chroma")
        object.__setattr__(self, "ollama_base_url", normalized_ollama)
        object.__setattr__(self, "chroma_host", normalized_chroma)
        return self


class CoverageSettings(StrictModel):
    """Repository paths whose ownership must be exhaustive and exclusive."""

    include: tuple[str, ...]
    tracked_include: tuple[str, ...] = ()
    tracked_exclude: tuple[str, ...] = ()
    forbidden_repository_paths: tuple[str, ...] = ()
    legacy_repository_exceptions: tuple[str, ...] = ()


class UpstreamReference(StrictModel):
    """Pinned provenance and reuse policy for one external reference system."""

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]+$")
    repository: str = Field(pattern=r"^https://github\.com/[^/\s]+/[^/\s]+$")
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    license: str = Field(min_length=2)
    reuse_policy: Literal["code-and-patterns", "patterns-only"]


class RuntimePolicy(StrictModel):
    """Repository-wide allowlists for components reachable in the default runtime."""

    local_inference_hosts: tuple[str, ...] = ()
    allowed_models: tuple[str, ...] = ()
    allowed_modes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_unique_values(self) -> RuntimePolicy:
        for label, values in (
            ("local inference hosts", self.local_inference_hosts),
            ("allowed models", self.allowed_models),
            ("allowed modes", self.allowed_modes),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"runtime policy {label} must be unique")
        return self


class Entrypoint(StrictModel):
    """One executable file exposed by an architecture component."""

    path: str = Field(min_length=1)
    kind: Literal["cli", "service", "worker"]
    default: bool = False


class AuthorityGrant(StrictModel):
    """An explicit capability owned by exactly one component when sensitive."""

    id: str = Field(pattern=r"^[a-z][a-z0-9]*(?:[.-][a-z0-9]+)*$")
    sensitive: bool = False


class EventContract(StrictModel):
    """Versioned event boundary produced, consumed, or stored by a component."""

    id: str = Field(pattern=r"^[a-z][a-z0-9]*(?:[.-][a-z0-9]+)*$")
    version: str = Field(pattern=r"^[0-9]+\.[0-9]+(?:\.[0-9]+)?$")
    role: Literal["produces", "consumes", "stores"]


class LicenseMetadata(StrictModel):
    """License provenance for code owned by a component."""

    expression: str = Field(min_length=2, max_length=128)
    source: Literal["repository", "upstream", "mixed", "none-declared"]
    notice: str | None = Field(default=None, min_length=8, max_length=512)


class Component(StrictModel):
    """One architecture component with one primary responsibility."""

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]+$")
    owner: str = Field(min_length=3)
    primary_responsibility: str = Field(min_length=8)
    runtime_status: Literal["active", "experimental", "quarantined"] = "experimental"
    depends_on: tuple[str, ...] = ()
    entrypoints: tuple[Entrypoint, ...] = ()
    authorities: tuple[AuthorityGrant, ...] = ()
    allowed_modes: tuple[str, ...] = ()
    allowed_egress: tuple[str, ...] = ()
    allowed_models: tuple[str, ...] = ()
    event_contracts: tuple[EventContract, ...] = ()
    license: LicenseMetadata | None = None
    owned_paths: tuple[str, ...]
    tests: tuple[str, ...] = ()
    documentation: tuple[str, ...]
    architecture: tuple[str, ...]
    embedding_inputs: tuple[str, ...] = ()
    forbidden_imports: tuple[str, ...] = ()
    upstream_references: tuple[UpstreamReference, ...] = ()

    @model_validator(mode="after")
    def validate_component_contract(self) -> Component:
        unique_contracts = {
            "dependency ids": self.depends_on,
            "entrypoint paths": tuple(item.path for item in self.entrypoints),
            "authority ids": tuple(item.id for item in self.authorities),
            "allowed modes": self.allowed_modes,
            "allowed egress endpoints": self.allowed_egress,
            "allowed models": self.allowed_models,
            "upstream reference ids": tuple(
                reference.id for reference in self.upstream_references
            ),
            "event contracts": tuple(
                f"{item.id}@{item.version}:{item.role}" for item in self.event_contracts
            ),
        }
        for label, values in unique_contracts.items():
            if len(values) != len(set(values)):
                raise ValueError(f"{label} must be unique per component")
        if self.runtime_status == "active" and self.entrypoints:
            if not any(item.default for item in self.entrypoints):
                raise ValueError("active component entrypoints require a default")
        return self


class Quarantine(StrictModel):
    """Frozen legacy scope that cannot grow without a manifest decision."""

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]+$")
    reason: str = Field(min_length=12)
    scope: tuple[str, ...]
    baseline_paths: tuple[str, ...]


class ProjectManifest(StrictModel):
    """Human-authored executable architecture contract."""

    schema_version: Literal[1]
    project: str = Field(min_length=2)
    knowledge: KnowledgeSettings
    runtime_policy: RuntimePolicy = RuntimePolicy()
    coverage: CoverageSettings
    components: tuple[Component, ...]
    quarantines: tuple[Quarantine, ...] = ()

    @model_validator(mode="after")
    def validate_unique_ids(self) -> ProjectManifest:
        component_ids = [component.id for component in self.components]
        if len(component_ids) != len(set(component_ids)):
            raise ValueError("component ids must be unique")
        quarantine_ids = [item.id for item in self.quarantines]
        if len(quarantine_ids) != len(set(quarantine_ids)):
            raise ValueError("quarantine ids must be unique")
        if not self.components:
            raise ValueError("at least one component is required")
        return self


@dataclass(frozen=True, slots=True)
class EmbeddingSpec:
    """Expected local model identity before its runtime digest is resolved."""

    model: str
    dimension: int


@dataclass(frozen=True, slots=True)
class ModelIdentity:
    """Ollama model identity resolved from the local daemon."""

    requested_model: str
    resolved_model: str
    digest: str
    dimension: int


@dataclass(frozen=True, slots=True)
class ComponentHealth:
    """Minimal health result that never hides an unavailable dependency."""

    healthy: bool
    detail: str


Scalar = str | int | float | bool


@dataclass(frozen=True, slots=True)
class VectorRecord:
    """One externally embedded Chroma record."""

    id: str
    embedding: tuple[float, ...]
    document: str
    metadata: Mapping[str, Scalar]


@dataclass(frozen=True, slots=True)
class StoredRecord:
    """Complete record returned by a vector-store integrity inventory."""

    id: str
    embedding: tuple[float, ...]
    document: str
    metadata: Mapping[str, Scalar]


@dataclass(frozen=True, slots=True)
class SearchHit:
    """A provenance-bearing knowledge retrieval result."""

    id: str
    document: str
    metadata: Mapping[str, Scalar]
    distance: float


@dataclass(frozen=True, slots=True)
class SyncReport:
    """Result of one idempotent local synchronization."""

    source_count: int
    desired_chunks: int
    embedded_chunks: int
    deleted_chunks: int
    unchanged_chunks: int
    lock_sha256: str
    model_digest: str


@dataclass(frozen=True, slots=True)
class QueryResult:
    """Query result plus the model/lock provenance used to produce it."""

    query: str
    lock_sha256: str
    model_digest: str
    hits: Sequence[SearchHit]


def json_scalar_mapping(value: Mapping[str, Any]) -> dict[str, Scalar]:
    """Validate metadata before it crosses the vector-store boundary."""

    result: dict[str, Scalar] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not isinstance(item, (str, int, float, bool)):
            raise TypeError("vector metadata must contain scalar JSON values")
        result[key] = item
    return result
