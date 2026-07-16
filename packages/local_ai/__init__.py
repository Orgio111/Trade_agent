"""Shared local-only model registry and Ollama health boundary."""

from .health import LocalOllamaHealthClient, OllamaHealth, OllamaModel
from .locality import normalize_local_http_url
from .models import LOCAL_MODEL_BY_ROLE, LocalModelRole

__all__ = [
    "LOCAL_MODEL_BY_ROLE",
    "LocalModelRole",
    "LocalOllamaHealthClient",
    "OllamaHealth",
    "OllamaModel",
    "normalize_local_http_url",
]
