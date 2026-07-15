"""Production composition root for the local Hermes control plane."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI

from packages.knowledge.chroma import ChromaVectorStore
from packages.knowledge.lockfile import load_checked_lock
from packages.knowledge.ollama import OllamaEmbedder

from .api import create_app
from .errors import JournalConfigurationError
from .obsidian import ObsidianJournalStore
from .ollama import OllamaAgentRunner, OllamaChatGateway
from .runtime_memory import RUNTIME_MEMORY_COLLECTION, RuntimeMemoryIndex
from .service import HermesMemoryService
from .workflow import WorkflowEngine


def create_local_app_from_environment() -> FastAPI:
    """Build from the checked project manifest and one explicit vault path."""

    vault_value = os.environ.get("HERMES_VAULT_PATH", "").strip()
    if not vault_value:
        raise JournalConfigurationError("HERMES_VAULT_PATH is required")
    vault_path = Path(vault_value).expanduser()
    if not vault_path.is_absolute():
        raise JournalConfigurationError("HERMES_VAULT_PATH must be absolute")

    manifest_path = os.environ.get(
        "HERMES_PROJECT_MANIFEST",
        "project.manifest.toml",
    )
    inventory, _lock = load_checked_lock(manifest_path)
    settings = inventory.manifest.knowledge
    if settings.collection == RUNTIME_MEMORY_COLLECTION:
        raise JournalConfigurationError(
            "runtime and project knowledge collections must be different"
        )

    embedder = OllamaEmbedder(
        base_url=settings.ollama_base_url,
        model=settings.embedding_model,
        dimension=settings.embedding_dimension,
    )
    vector_store = ChromaVectorStore(
        host=settings.chroma_host,
        port=settings.chroma_port,
        ssl=settings.chroma_ssl,
        collection=RUNTIME_MEMORY_COLLECTION,
        expected_dimension=settings.embedding_dimension,
        expected_client_version=settings.chroma_client_version,
    )
    memory_index = RuntimeMemoryIndex(
        embedder=embedder,
        store=vector_store,
    )
    memory_service = HermesMemoryService(
        journal=ObsidianJournalStore(vault_path),
        memory=memory_index,
    )

    gateway = OllamaChatGateway(base_url=settings.ollama_base_url)
    workflow = WorkflowEngine(
        OllamaAgentRunner(gateway),
        max_concurrency=2,
    )
    return create_app(
        memory_service=memory_service,
        workflow_engine=workflow,
        workflow_health=gateway.health,
        workflow_close=gateway.aclose,
        loopback_only=True,
    )
