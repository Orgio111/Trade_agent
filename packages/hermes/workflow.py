"""Deterministic, bounded orchestration for non-authoritative local agents.

The workflow engine coordinates generic analysis and documentation tasks only.  It
does not contain trading, risk, sizing, broker, or execution behavior.  Agent
intelligence is injected through :class:`AgentRunner`, which keeps scheduling
policy deterministic and independently testable.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from enum import Enum
from time import monotonic
from typing import Protocol, runtime_checkable
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .models import LOCAL_MODEL_BY_ROLE, LocalModelRole


MAX_WORKFLOW_TASKS = 64
MAX_WORKFLOW_CONCURRENCY = 8
MAX_TASK_TIMEOUT_SECONDS = 600.0
MAX_WORKFLOW_DEADLINE_SECONDS = 3_600.0


class WorkflowError(RuntimeError):
    """Base error raised by the bounded workflow layer."""


class WorkflowLimitError(WorkflowError):
    """Raised when a plan exceeds an engine-specific resource limit."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class WorkflowTask(_StrictModel):
    """One local-agent task in a validated directed acyclic graph."""

    task_id: str = Field(
        min_length=1,
        max_length=64,
        pattern=r"^[a-z0-9][a-z0-9_-]*$",
    )
    role: LocalModelRole
    instructions: str = Field(min_length=1, max_length=32_000)
    depends_on: tuple[str, ...] = Field(
        default=(),
        max_length=MAX_WORKFLOW_TASKS - 1,
    )
    timeout_seconds: float = Field(
        default=120.0,
        gt=0.0,
        le=MAX_TASK_TIMEOUT_SECONDS,
    )

    @field_validator("instructions")
    @classmethod
    def normalize_instructions(cls, value: str) -> str:
        normalized = value.replace("\r\n", "\n").replace("\r", "\n").strip()
        if not normalized:
            raise ValueError("task instructions cannot be empty")
        if "\x00" in normalized:
            raise ValueError("task instructions cannot contain NUL bytes")
        return normalized

    @model_validator(mode="after")
    def validate_dependencies(self) -> WorkflowTask:
        if len(self.depends_on) != len(set(self.depends_on)):
            raise ValueError("task dependencies must be unique")
        if self.task_id in self.depends_on:
            raise ValueError("task cannot depend on itself")
        return self


class WorkflowPlan(_StrictModel):
    """A bounded DAG whose topology is fully validated before any agent runs."""

    run_id: UUID = Field(default_factory=uuid4)
    tasks: tuple[WorkflowTask, ...] = Field(
        min_length=1,
        max_length=MAX_WORKFLOW_TASKS,
    )
    max_concurrency: int = Field(
        default=2,
        ge=1,
        le=MAX_WORKFLOW_CONCURRENCY,
    )

    @model_validator(mode="after")
    def validate_dag(self) -> WorkflowPlan:
        task_ids = [task.task_id for task in self.tasks]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("workflow task ids must be unique")

        known = set(task_ids)
        for task in self.tasks:
            unknown = sorted(set(task.depends_on) - known)
            if unknown:
                raise ValueError(
                    f"task {task.task_id!r} has unknown dependencies: "
                    f"{', '.join(unknown)}"
                )

        # Kahn's algorithm both validates acyclicity and avoids recursive depth
        # depending on caller input.
        remaining = {task.task_id: set(task.depends_on) for task in self.tasks}
        resolved: set[str] = set()
        while remaining:
            ready = sorted(
                task_id
                for task_id, dependencies in remaining.items()
                if dependencies <= resolved
            )
            if not ready:
                cycle_members = ", ".join(sorted(remaining))
                raise ValueError(
                    f"workflow dependencies contain a cycle: {cycle_members}"
                )
            resolved.update(ready)
            for task_id in ready:
                del remaining[task_id]
        return self


class AgentOutput(_StrictModel):
    """Provider-neutral output returned by an injected local agent runner."""

    content: str = Field(min_length=1, max_length=1_000_000)
    model: str = Field(min_length=1, max_length=128)
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)


class TaskStatus(str, Enum):
    """Every possible final state of a task returned by the engine."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    BLOCKED = "blocked"


TERMINAL_TASK_STATUSES = frozenset(TaskStatus)


class TaskResult(_StrictModel):
    """Terminal result for exactly one task."""

    task_id: str
    status: TaskStatus
    output: AgentOutput | None = None
    error: str | None = None
    blocked_by: tuple[str, ...] = ()
    duration_ms: float = Field(default=0.0, ge=0.0)

    @model_validator(mode="after")
    def validate_status_payload(self) -> TaskResult:
        if self.status is TaskStatus.SUCCEEDED:
            if self.output is None or self.error is not None or self.blocked_by:
                raise ValueError("successful task must contain only output")
        elif self.status is TaskStatus.BLOCKED:
            if self.output is not None or not self.blocked_by:
                raise ValueError("blocked task must identify failed dependencies")
        else:
            if self.output is not None:
                raise ValueError("unsuccessful task cannot contain output")
            if not self.error:
                raise ValueError("unsuccessful task must contain an error")
        return self


class WorkflowStatus(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMED_OUT = "timed_out"


class WorkflowResult(_StrictModel):
    """Complete workflow outcome; every planned task has a terminal result."""

    run_id: UUID
    status: WorkflowStatus
    tasks: tuple[TaskResult, ...] = Field(min_length=1)
    duration_ms: float = Field(ge=0.0)

    @model_validator(mode="after")
    def validate_aggregate_status(self) -> WorkflowResult:
        task_ids = [task.task_id for task in self.tasks]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("workflow result task ids must be unique")
        statuses = {result.status for result in self.tasks}
        if statuses == {TaskStatus.SUCCEEDED}:
            expected = WorkflowStatus.SUCCEEDED
        elif TaskStatus.TIMED_OUT in statuses:
            expected = WorkflowStatus.TIMED_OUT
        else:
            expected = WorkflowStatus.FAILED
        if self.status is not expected:
            raise ValueError("workflow status does not match task results")
        return self

    @property
    def by_task_id(self) -> dict[str, TaskResult]:
        return {task.task_id: task for task in self.tasks}


@runtime_checkable
class AgentRunner(Protocol):
    """Injected boundary that performs one task using local intelligence."""

    async def run(
        self,
        task: WorkflowTask,
        dependency_results: Mapping[str, TaskResult],
    ) -> AgentOutput: ...


class WorkflowEngine:
    """Execute validated plans in deterministic, dependency-safe waves."""

    def __init__(
        self,
        runner: AgentRunner,
        *,
        max_tasks: int = MAX_WORKFLOW_TASKS,
        max_concurrency: int = MAX_WORKFLOW_CONCURRENCY,
        default_deadline_seconds: float = 600.0,
    ) -> None:
        if not 1 <= max_tasks <= MAX_WORKFLOW_TASKS:
            raise ValueError(f"max_tasks must be between 1 and {MAX_WORKFLOW_TASKS}")
        if not 1 <= max_concurrency <= MAX_WORKFLOW_CONCURRENCY:
            raise ValueError(
                f"max_concurrency must be between 1 and {MAX_WORKFLOW_CONCURRENCY}"
            )
        if not 0 < default_deadline_seconds <= MAX_WORKFLOW_DEADLINE_SECONDS:
            raise ValueError(
                "default_deadline_seconds must be positive and no greater than "
                f"{MAX_WORKFLOW_DEADLINE_SECONDS}"
            )
        self._runner = runner
        self._max_tasks = max_tasks
        self._max_concurrency = max_concurrency
        self._default_deadline_seconds = default_deadline_seconds

    async def run(
        self,
        plan: WorkflowPlan,
        *,
        deadline_seconds: float | None = None,
    ) -> WorkflowResult:
        """Run a plan and return one terminal result for every task.

        Ready tasks are grouped into topological waves and sorted by ``task_id``.
        A later wave never starts until the previous wave is fully terminal, so
        scheduling is reproducible even when individual local models complete in
        different orders.
        """

        if len(plan.tasks) > self._max_tasks:
            raise WorkflowLimitError(
                f"workflow has {len(plan.tasks)} tasks; engine limit is "
                f"{self._max_tasks}"
            )
        deadline_budget = (
            self._default_deadline_seconds
            if deadline_seconds is None
            else deadline_seconds
        )
        if not 0 < deadline_budget <= MAX_WORKFLOW_DEADLINE_SECONDS:
            raise ValueError(
                "deadline_seconds must be positive and no greater than "
                f"{MAX_WORKFLOW_DEADLINE_SECONDS}"
            )

        started_at = monotonic()
        deadline_at = started_at + deadline_budget
        concurrency = min(plan.max_concurrency, self._max_concurrency)
        results: dict[str, TaskResult] = {}

        for wave in _topological_waves(plan.tasks):
            runnable: list[WorkflowTask] = []
            for task in wave:
                blocked_by = tuple(
                    dependency
                    for dependency in task.depends_on
                    if results[dependency].status is not TaskStatus.SUCCEEDED
                )
                if blocked_by:
                    results[task.task_id] = TaskResult(
                        task_id=task.task_id,
                        status=TaskStatus.BLOCKED,
                        error="one or more dependencies did not succeed",
                        blocked_by=blocked_by,
                    )
                elif monotonic() >= deadline_at:
                    results[task.task_id] = TaskResult(
                        task_id=task.task_id,
                        status=TaskStatus.TIMED_OUT,
                        error="workflow deadline exceeded before task start",
                    )
                else:
                    runnable.append(task)

            for offset in range(0, len(runnable), concurrency):
                batch = runnable[offset : offset + concurrency]
                if monotonic() >= deadline_at:
                    for task in batch:
                        results[task.task_id] = TaskResult(
                            task_id=task.task_id,
                            status=TaskStatus.TIMED_OUT,
                            error="workflow deadline exceeded before task start",
                        )
                    continue
                batch_results = await asyncio.gather(
                    *(self._run_task(task, results, deadline_at) for task in batch)
                )
                results.update(
                    (task_result.task_id, task_result) for task_result in batch_results
                )

        # The validated DAG and wave construction make this invariant explicit.
        # Retain a fail-closed guard so future scheduler changes cannot leak a
        # non-terminal task through the API.
        missing = {task.task_id for task in plan.tasks} - results.keys()
        if missing:
            raise WorkflowError(
                "workflow ended without terminal results for: "
                + ", ".join(sorted(missing))
            )

        ordered_results = tuple(results[task.task_id] for task in plan.tasks)
        statuses = {result.status for result in ordered_results}
        if statuses == {TaskStatus.SUCCEEDED}:
            workflow_status = WorkflowStatus.SUCCEEDED
        elif TaskStatus.TIMED_OUT in statuses:
            workflow_status = WorkflowStatus.TIMED_OUT
        else:
            workflow_status = WorkflowStatus.FAILED
        return WorkflowResult(
            run_id=plan.run_id,
            status=workflow_status,
            tasks=ordered_results,
            duration_ms=max(0.0, (monotonic() - started_at) * 1_000),
        )

    async def _run_task(
        self,
        task: WorkflowTask,
        completed: Mapping[str, TaskResult],
        deadline_at: float,
    ) -> TaskResult:
        started_at = monotonic()
        remaining = deadline_at - started_at
        if remaining <= 0:
            return TaskResult(
                task_id=task.task_id,
                status=TaskStatus.TIMED_OUT,
                error="workflow deadline exceeded before task start",
            )
        timeout = min(task.timeout_seconds, remaining)
        dependency_results = {
            dependency: completed[dependency] for dependency in task.depends_on
        }
        try:
            output = await asyncio.wait_for(
                self._runner.run(task, dependency_results),
                timeout=timeout,
            )
            if not isinstance(output, AgentOutput):
                raise TypeError("agent runner must return AgentOutput")
            expected_model = LOCAL_MODEL_BY_ROLE[task.role]
            if not _model_matches(output.model, expected_model):
                raise TypeError("agent runner returned an unapproved model")
        except TimeoutError:
            deadline_limited = remaining <= task.timeout_seconds
            detail = (
                "workflow deadline exceeded during task"
                if deadline_limited
                else f"task exceeded {task.timeout_seconds:g}s timeout"
            )
            return TaskResult(
                task_id=task.task_id,
                status=TaskStatus.TIMED_OUT,
                error=detail,
                duration_ms=max(0.0, (monotonic() - started_at) * 1_000),
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return TaskResult(
                task_id=task.task_id,
                status=TaskStatus.FAILED,
                error=f"agent runner failed ({type(exc).__name__})",
                duration_ms=max(0.0, (monotonic() - started_at) * 1_000),
            )
        return TaskResult(
            task_id=task.task_id,
            status=TaskStatus.SUCCEEDED,
            output=output,
            duration_ms=max(0.0, (monotonic() - started_at) * 1_000),
        )


def _topological_waves(
    tasks: Sequence[WorkflowTask],
) -> tuple[tuple[WorkflowTask, ...], ...]:
    """Build stable topological waves from an already validated plan."""

    task_by_id = {task.task_id: task for task in tasks}
    remaining = {task.task_id: set(task.depends_on) for task in tasks}
    resolved: set[str] = set()
    waves: list[tuple[WorkflowTask, ...]] = []
    while remaining:
        ready_ids = sorted(
            task_id
            for task_id, dependencies in remaining.items()
            if dependencies <= resolved
        )
        if not ready_ids:  # Defensive: WorkflowPlan validation should catch this.
            raise WorkflowError("validated workflow unexpectedly contains a cycle")
        waves.append(tuple(task_by_id[task_id] for task_id in ready_ids))
        resolved.update(ready_ids)
        for task_id in ready_ids:
            del remaining[task_id]
    return tuple(waves)


def _model_matches(actual: str, expected: str) -> bool:
    return actual == expected or (
        ":" not in expected and actual == f"{expected}:latest"
    )
