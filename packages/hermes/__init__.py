"""Local-only Hermes workflow, Obsidian journal, and retrieval integration."""

from .models import (
    LOCAL_MODEL_BY_ROLE,
    JournalEvent,
    JournalEventKind,
    JournalReceipt,
    LocalModelRole,
)
from .obsidian import ObsidianJournalStore
from .ollama import OllamaAgentRunner, OllamaChatGateway
from .runtime_memory import RUNTIME_MEMORY_COLLECTION, RuntimeMemoryIndex
from .service import HermesMemoryService
from .workflow import (
    AgentOutput,
    TaskResult,
    WorkflowEngine,
    WorkflowPlan,
    WorkflowTask,
)

__all__ = [
    "LOCAL_MODEL_BY_ROLE",
    "RUNTIME_MEMORY_COLLECTION",
    "AgentOutput",
    "HermesMemoryService",
    "JournalEvent",
    "JournalEventKind",
    "JournalReceipt",
    "LocalModelRole",
    "ObsidianJournalStore",
    "OllamaAgentRunner",
    "OllamaChatGateway",
    "RuntimeMemoryIndex",
    "TaskResult",
    "WorkflowEngine",
    "WorkflowPlan",
    "WorkflowTask",
]
