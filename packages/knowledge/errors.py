"""Typed failures for the project-knowledge pipeline."""


class KnowledgeError(RuntimeError):
    """Base class for expected, user-actionable knowledge failures."""


class ManifestError(KnowledgeError):
    """The architecture manifest or repository inventory is invalid."""


class DriftError(KnowledgeError):
    """The generated lock no longer matches repository content."""


class SecretDetectedError(ManifestError):
    """An embedding input contains material that resembles a secret."""


class EmbeddingError(KnowledgeError):
    """The local embedding model failed its contract."""


class VectorStoreError(KnowledgeError):
    """The local vector store failed its contract."""


class LiveVerificationError(KnowledgeError):
    """The live local index does not match the committed lock."""
