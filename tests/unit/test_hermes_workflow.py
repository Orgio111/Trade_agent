"""Offline contract tests for deterministic Hermes workflow scheduling."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping

import pytest
from pydantic import ValidationError

from packages.hermes.models import LOCAL_MODEL_BY_ROLE, LocalModelRole
from packages.hermes.workflow import (
    AgentOutput,
    TaskResult,
    TaskStatus,
    WorkflowEngine,
    WorkflowLimitError,
    WorkflowPlan,
    WorkflowStatus,
    WorkflowTask,
)


class RecordingRunner:
    def __init__(
        self,
        *,
        delays: Mapping[str, float] | None = None,
        failures: set[str] | None = None,
    ) -> None:
        self.delays = dict(delays or {})
        self.failures = failures or set()
        self.events: list[str] = []
        self.calls: list[str] = []
        self.active = 0
        self.max_active = 0

    async def run(
        self,
        task: WorkflowTask,
        dependency_results: Mapping[str, TaskResult],
    ) -> AgentOutput:
        assert set(dependency_results) == set(task.depends_on)
        assert all(
            result.status is TaskStatus.SUCCEEDED
            for result in dependency_results.values()
        )
        self.calls.append(task.task_id)
        self.events.append(f"start:{task.task_id}")
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(self.delays.get(task.task_id, 0.0))
            if task.task_id in self.failures:
                raise RuntimeError("deliberate test failure")
            return AgentOutput(
                content=f"completed:{task.task_id}",
                model=LOCAL_MODEL_BY_ROLE[task.role],
            )
        finally:
            self.active -= 1
            self.events.append(f"end:{task.task_id}")


def task(
    task_id: str,
    *,
    depends_on: tuple[str, ...] = (),
    timeout_seconds: float = 1.0,
) -> WorkflowTask:
    return WorkflowTask(
        task_id=task_id,
        role=LocalModelRole.REASONING,
        instructions=f"Perform engineering task {task_id}",
        depends_on=depends_on,
        timeout_seconds=timeout_seconds,
    )


@pytest.mark.parametrize(
    "tasks",
    [
        (task("a", depends_on=("missing",)),),
        (task("a", depends_on=("b",)), task("b", depends_on=("a",))),
        (task("a"), task("a")),
    ],
)
def test_plan_rejects_unknown_cycle_and_duplicate_task_ids(
    tasks: tuple[WorkflowTask, ...],
) -> None:
    with pytest.raises(ValidationError):
        WorkflowPlan(tasks=tasks)


def test_task_rejects_duplicate_and_self_dependencies() -> None:
    with pytest.raises(ValidationError, match="dependencies must be unique"):
        task("a", depends_on=("b", "b"))
    with pytest.raises(ValidationError, match="cannot depend on itself"):
        task("a", depends_on=("a",))


@pytest.mark.asyncio
async def test_scheduler_uses_stable_topological_waves() -> None:
    runner = RecordingRunner(delays={"a": 0.01, "b": 0.01})
    plan = WorkflowPlan(
        tasks=(
            task("finish", depends_on=("a", "b")),
            task("b"),
            task("a"),
        ),
        max_concurrency=2,
    )

    result = await WorkflowEngine(runner).run(plan)

    assert result.status is WorkflowStatus.SUCCEEDED
    assert [item.task_id for item in result.tasks] == ["finish", "b", "a"]
    assert runner.events[:2] == ["start:a", "start:b"]
    assert runner.events.index("start:finish") > runner.events.index("end:b")
    assert runner.events.index("start:finish") > runner.events.index("end:a")
    assert runner.max_active == 2


@pytest.mark.asyncio
async def test_scheduler_never_exceeds_effective_concurrency_limit() -> None:
    runner = RecordingRunner(delays={f"t{index}": 0.01 for index in range(5)})
    plan = WorkflowPlan(
        tasks=tuple(task(f"t{index}") for index in range(5)),
        max_concurrency=4,
    )

    result = await WorkflowEngine(runner, max_concurrency=2).run(plan)

    assert result.status is WorkflowStatus.SUCCEEDED
    assert runner.max_active == 2
    assert len(result.tasks) == 5


@pytest.mark.asyncio
async def test_dependency_failure_blocks_descendants_without_running_them() -> None:
    runner = RecordingRunner(failures={"root"})
    plan = WorkflowPlan(
        tasks=(
            task("root"),
            task("child", depends_on=("root",)),
            task("grandchild", depends_on=("child",)),
            task("independent"),
        )
    )

    result = await WorkflowEngine(runner).run(plan)
    by_id = result.by_task_id

    assert result.status is WorkflowStatus.FAILED
    assert by_id["root"].status is TaskStatus.FAILED
    assert "deliberate test failure" not in (by_id["root"].error or "")
    assert by_id["child"].status is TaskStatus.BLOCKED
    assert by_id["child"].blocked_by == ("root",)
    assert by_id["grandchild"].status is TaskStatus.BLOCKED
    assert by_id["grandchild"].blocked_by == ("child",)
    assert by_id["independent"].status is TaskStatus.SUCCEEDED
    assert "child" not in runner.calls
    assert "grandchild" not in runner.calls
    assert len(by_id) == len(plan.tasks)


@pytest.mark.asyncio
async def test_per_task_timeout_is_terminal_and_blocks_dependent() -> None:
    runner = RecordingRunner(delays={"slow": 0.05})
    plan = WorkflowPlan(
        tasks=(
            task("slow", timeout_seconds=0.005),
            task("after", depends_on=("slow",)),
        )
    )

    result = await WorkflowEngine(runner).run(plan)

    assert result.status is WorkflowStatus.TIMED_OUT
    assert result.by_task_id["slow"].status is TaskStatus.TIMED_OUT
    assert "task exceeded" in (result.by_task_id["slow"].error or "")
    assert result.by_task_id["after"].status is TaskStatus.BLOCKED
    assert runner.calls == ["slow"]


@pytest.mark.asyncio
async def test_global_deadline_bounds_full_workflow_and_finishes_every_task() -> None:
    runner = RecordingRunner(delays={"second": 0.1})
    plan = WorkflowPlan(
        tasks=(
            task("first"),
            task("second", depends_on=("first",)),
            task("third", depends_on=("second",)),
        ),
        max_concurrency=1,
    )

    result = await WorkflowEngine(runner).run(plan, deadline_seconds=0.03)

    assert result.status is WorkflowStatus.TIMED_OUT
    assert result.by_task_id["first"].status is TaskStatus.SUCCEEDED
    assert result.by_task_id["second"].status is TaskStatus.TIMED_OUT
    assert "workflow deadline" in (result.by_task_id["second"].error or "")
    assert result.by_task_id["third"].status is TaskStatus.BLOCKED
    assert len(result.tasks) == len(plan.tasks)


@pytest.mark.asyncio
async def test_engine_rejects_plan_above_its_local_task_cap() -> None:
    plan = WorkflowPlan(tasks=(task("a"), task("b")))
    engine = WorkflowEngine(RecordingRunner(), max_tasks=1)

    with pytest.raises(WorkflowLimitError, match="engine limit is 1"):
        await engine.run(plan)


@pytest.mark.asyncio
async def test_nonconforming_runner_result_fails_closed() -> None:
    class InvalidRunner:
        async def run(
            self,
            _task: WorkflowTask,
            _dependency_results: Mapping[str, TaskResult],
        ) -> object:
            return "untyped output"

    result = await WorkflowEngine(InvalidRunner()).run(WorkflowPlan(tasks=(task("a"),)))

    assert result.status is WorkflowStatus.FAILED
    assert result.tasks[0].status is TaskStatus.FAILED
    assert result.tasks[0].error == "agent runner failed (TypeError)"


@pytest.mark.asyncio
async def test_runner_cannot_substitute_a_model_outside_role_registry() -> None:
    class SubstitutingRunner:
        async def run(
            self,
            _task: WorkflowTask,
            _dependency_results: Mapping[str, TaskResult],
        ) -> AgentOutput:
            return AgentOutput(content="artifact", model="remote-cloud-model")

    result = await WorkflowEngine(SubstitutingRunner()).run(
        WorkflowPlan(tasks=(task("a"),))
    )

    assert result.status is WorkflowStatus.FAILED
    assert result.tasks[0].error == "agent runner failed (TypeError)"
