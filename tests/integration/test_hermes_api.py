"""End-to-end local Hermes API contract without external services."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from packages.hermes.api import create_app
from packages.hermes.models import JournalEvent, JournalEventKind, LocalModelRole
from packages.hermes.obsidian import ObsidianJournalStore
from packages.hermes.runtime_memory import RuntimeMemoryIndex
from packages.hermes.service import HermesMemoryService
from packages.hermes.workflow import (
    AgentOutput,
    TaskResult,
    WorkflowEngine,
    WorkflowTask,
)
from tests.fakes.knowledge import DeterministicFakeEmbedder, InMemoryVectorStore


class _LocalRunner:
    async def run(
        self,
        task: WorkflowTask,
        dependency_results: Mapping[str, TaskResult],
    ) -> AgentOutput:
        context = ",".join(sorted(dependency_results)) or "root"
        model = {
            LocalModelRole.REASONING: "qwen3:8b",
            LocalModelRole.TOOL_FORMATTING: "mistral",
        }[task.role]
        return AgentOutput(
            content=f"{task.task_id}:{context}",
            model=model,
        )


def _event() -> JournalEvent:
    return JournalEvent.create(
        occurred_at=datetime(2026, 7, 16, 10, 0, tzinfo=UTC),
        kind=JournalEventKind.RESEARCH,
        agent_id="hermes-engineering",
        title="Local integration evidence",
        body="A validated engineering artifact with no trading authority.",
        tags=("architecture", "hermes"),
    )


@pytest.mark.asyncio
async def test_loopback_api_persists_queries_and_runs_bounded_workflow(
    tmp_path: Path,
) -> None:
    embedder = DeterministicFakeEmbedder(dimension=4)
    vector_store = InMemoryVectorStore()
    memory_index = RuntimeMemoryIndex(embedder=embedder, store=vector_store)
    service = HermesMemoryService(
        journal=ObsidianJournalStore(tmp_path),
        memory=memory_index,
    )
    await service.initialize()
    app = create_app(
        memory_service=service,
        workflow_engine=WorkflowEngine(_LocalRunner()),
    )
    transport = httpx.ASGITransport(
        app=app,
        client=("127.0.0.1", 43111),
    )

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://127.0.0.1",
    ) as client:
        event = _event()
        created = await client.post(
            "/api/v1/hermes/events",
            json=event.model_dump(mode="json"),
        )
        assert created.status_code == 201
        receipt = created.json()
        assert receipt["journal"]["event_id"] == str(event.event_id)
        assert receipt["index"]["record_id"] == f"journal-event:{event.event_id}"

        replayed = await client.post(
            "/api/v1/hermes/events",
            json=event.model_dump(mode="json"),
        )
        assert replayed.status_code == 201
        assert replayed.json()["journal"]["created"] is False

        queried = await client.post(
            "/api/v1/hermes/memory/query",
            json={"query": "engineering artifact", "top_k": 3},
        )
        assert queried.status_code == 200
        assert (
            queried.json()["hits"][0]["metadata"]["source_path"]
            == receipt["journal"]["relative_path"]
        )

        workflow = await client.post(
            "/api/v1/hermes/workflows",
            json={
                "deadline_seconds": 2,
                "plan": {
                    "tasks": [
                        {
                            "task_id": "analyze",
                            "role": "reasoning",
                            "instructions": "Analyze the engineering contract.",
                        },
                        {
                            "task_id": "format",
                            "role": "tool_formatting",
                            "instructions": "Format the validated artifact.",
                            "depends_on": ["analyze"],
                        },
                    ]
                },
            },
        )
        assert workflow.status_code == 200
        assert workflow.json()["status"] == "succeeded"
        assert [item["status"] for item in workflow.json()["tasks"]] == [
            "succeeded",
            "succeeded",
        ]

        health = await client.get("/api/v1/hermes/health")
        assert health.status_code == 200
        assert health.json()["healthy"] is True

    await service.aclose()


@pytest.mark.asyncio
async def test_non_loopback_api_client_is_rejected(tmp_path: Path) -> None:
    memory_index = RuntimeMemoryIndex(
        embedder=DeterministicFakeEmbedder(dimension=4),
        store=InMemoryVectorStore(),
    )
    service = HermesMemoryService(
        journal=ObsidianJournalStore(tmp_path),
        memory=memory_index,
    )
    await service.initialize()
    app = create_app(memory_service=service)
    transport = httpx.ASGITransport(
        app=app,
        client=("203.0.113.10", 43111),
    )

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://203.0.113.10",
    ) as client:
        response = await client.get("/api/v1/hermes/health")

    assert response.status_code == 403
    await service.aclose()
