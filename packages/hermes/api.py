"""Loopback FastAPI control plane for Hermes memory and local workflows."""

from __future__ import annotations

import ipaddress
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from packages.knowledge.errors import KnowledgeError

from .errors import (
    HermesError,
    JournalConflictError,
    JournalPathError,
    JournalSecretError,
    RuntimeProjectionError,
)
from .models import JournalEvent
from .ollama import OllamaHealth
from .runtime_memory import RuntimeMemoryQuery, RuntimeMemoryQueryResult
from .service import HermesHealth, HermesIngestReceipt, HermesMemoryService
from .workflow import WorkflowEngine, WorkflowPlan, WorkflowResult


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class WorkflowRequest(_StrictModel):
    plan: WorkflowPlan
    deadline_seconds: float = Field(default=600.0, gt=0, le=3_600)


class HermesApiHealth(_StrictModel):
    healthy: bool
    memory: HermesHealth
    workflow_available: bool
    workflow: OllamaHealth | None = None


def create_app(
    *,
    memory_service: HermesMemoryService,
    workflow_engine: WorkflowEngine | None = None,
    workflow_health: Callable[[], Awaitable[OllamaHealth]] | None = None,
    workflow_close: Callable[[], Awaitable[None]] | None = None,
    loopback_only: bool = True,
) -> FastAPI:
    """Create an injected, fully local app without importing trading services."""

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        await memory_service.initialize()
        try:
            yield
        finally:
            if workflow_close is not None:
                await workflow_close()
            await memory_service.aclose()

    app = FastAPI(
        title="Trade_agent Hermes Local Integration",
        version="1.0.0",
        lifespan=lifespan,
    )

    if loopback_only:

        @app.middleware("http")
        async def require_loopback(request: Request, call_next):
            client = request.client
            try:
                allowed = (
                    client is not None and ipaddress.ip_address(client.host).is_loopback
                )
            except ValueError:
                allowed = False
            if not allowed:
                return JSONResponse(
                    status_code=status.HTTP_403_FORBIDDEN,
                    content={"detail": "Hermes API accepts loopback clients only"},
                )
            return await call_next(request)

    @app.exception_handler(JournalConflictError)
    async def journal_conflict(
        _request: Request,
        exc: JournalConflictError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content={"detail": str(exc)},
        )

    @app.exception_handler(JournalSecretError)
    @app.exception_handler(JournalPathError)
    async def journal_validation(
        _request: Request,
        exc: JournalSecretError | JournalPathError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"detail": str(exc)},
        )

    @app.exception_handler(RuntimeProjectionError)
    async def projection_failure(
        _request: Request,
        exc: RuntimeProjectionError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "detail": "runtime projection failed; retry the identical event",
                "event_id": exc.event_id,
                "journal_path": exc.journal_path,
                "journaled": True,
            },
        )

    @app.exception_handler(KnowledgeError)
    async def local_dependency_failure(
        _request: Request,
        _exc: KnowledgeError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": "required local knowledge dependency is unavailable"},
        )

    @app.exception_handler(HermesError)
    async def hermes_failure(
        _request: Request,
        _exc: HermesError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Hermes operation failed"},
        )

    @app.post(
        "/api/v1/hermes/events",
        response_model=HermesIngestReceipt,
        status_code=status.HTTP_201_CREATED,
    )
    async def append_event(event: JournalEvent) -> HermesIngestReceipt:
        return await memory_service.ingest(event)

    @app.post(
        "/api/v1/hermes/memory/query",
        response_model=RuntimeMemoryQueryResult,
    )
    async def query_memory(query: RuntimeMemoryQuery) -> RuntimeMemoryQueryResult:
        return await memory_service.query(
            query.query,
            top_k=query.top_k,
            filters=query.filters,
        )

    @app.post(
        "/api/v1/hermes/workflows",
        response_model=WorkflowResult,
    )
    async def run_workflow(request: WorkflowRequest) -> WorkflowResult:
        if workflow_engine is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="local workflow engine is not configured",
            )
        return await workflow_engine.run(
            request.plan,
            deadline_seconds=request.deadline_seconds,
        )

    @app.get(
        "/api/v1/hermes/health",
        response_model=HermesApiHealth,
    )
    async def health() -> HermesApiHealth:
        memory = await memory_service.health()
        workflow = await workflow_health() if workflow_health is not None else None
        return HermesApiHealth(
            healthy=memory.healthy and (workflow is None or workflow.healthy),
            memory=memory,
            workflow_available=workflow_engine is not None,
            workflow=workflow,
        )

    return app
