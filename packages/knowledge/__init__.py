"""Manifest-driven, fully local project-knowledge pipeline."""

from .errors import (
    DriftError,
    EmbeddingError,
    KnowledgeError,
    LiveVerificationError,
    ManifestError,
    SecretDetectedError,
    VectorStoreError,
)
from .lockfile import build_lock, check_lock, load_lock, write_lock
from .manifest import ManifestInventory, load_inventory
from .models import ProjectManifest

__all__ = [
    "DriftError",
    "EmbeddingError",
    "KnowledgeError",
    "LiveVerificationError",
    "ManifestError",
    "ManifestInventory",
    "ProjectManifest",
    "SecretDetectedError",
    "VectorStoreError",
    "build_lock",
    "check_lock",
    "load_inventory",
    "load_lock",
    "write_lock",
]
