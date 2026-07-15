"""Offline contract tests for the loopback-only Hermes Ollama gateway."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from packages.hermes.models import LOCAL_MODEL_BY_ROLE, LocalModelRole
from packages.hermes.ollama import (
    OllamaChatGateway,
    OllamaAgentRunner,
    OllamaGatewayError,
    OllamaMessage,
    OllamaModelError,
    OllamaProtocolError,
    OllamaUnavailableError,
)
from packages.hermes.workflow import (
    AgentOutput,
    TaskResult,
    TaskStatus,
    WorkflowTask,
)


def chat_response(
    model: str,
    *,
    content: str = "local result",
) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "model": model,
            "done": True,
            "message": {"role": "assistant", "content": content},
            "prompt_eval_count": 11,
            "eval_count": 7,
            "total_duration": 123,
            "load_duration": 45,
        },
    )


@pytest.mark.parametrize(
    "url",
    [
        "https://127.0.0.1:11434",
        "http://192.168.1.20:11434",
        "http://ollama.example.com:11434",
        "http://user:secret@127.0.0.1:11434",
    ],
)
def test_gateway_rejects_non_loopback_or_authenticated_urls(url: str) -> None:
    with pytest.raises(OllamaGatewayError):
        OllamaChatGateway(base_url=url)


@pytest.mark.asyncio
async def test_chat_uses_exact_role_registry_and_typed_payload() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return chat_response(LOCAL_MODEL_BY_ROLE[LocalModelRole.REASONING])

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    gateway = OllamaChatGateway(
        base_url="http://localhost:11434",
        client=client,
    )
    result = await gateway.chat(
        role=LocalModelRole.REASONING,
        messages=(OllamaMessage(role="user", content="Inspect architecture"),),
        temperature=0.0,
        max_tokens=256,
    )
    await gateway.aclose()

    assert result.role is LocalModelRole.REASONING
    assert result.model == "qwen3:8b"
    assert result.content == "local result"
    assert result.prompt_tokens == 11
    assert result.completion_tokens == 7
    assert len(requests) == 1
    assert requests[0].url == "http://localhost:11434/api/chat"
    assert json.loads(requests[0].content) == {
        "model": "qwen3:8b",
        "messages": [{"role": "user", "content": "Inspect architecture"}],
        "stream": False,
        "keep_alive": "5m",
        "options": {"temperature": 0.0, "num_predict": 256},
    }
    # An injected client's lifecycle belongs to the caller.
    assert not client.is_closed
    await client.aclose()


@pytest.mark.asyncio
async def test_chat_rejects_embedding_and_unapproved_roles_before_http() -> None:
    requests = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return chat_response("nomic-embed-text")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        gateway = OllamaChatGateway(client=client)
        with pytest.raises(OllamaModelError, match="embedding role"):
            await gateway.chat(
                role=LocalModelRole.EMBEDDING,
                messages=({"role": "user", "content": "not a chat"},),
            )
        with pytest.raises(OllamaModelError, match="unapproved"):
            await gateway.chat(  # type: ignore[arg-type]
                role="cloud-model",
                messages=({"role": "user", "content": "no fallback"},),
            )

    assert requests == 0


@pytest.mark.asyncio
async def test_transient_status_retries_with_bounded_backoff() -> None:
    attempts = 0
    delays: list[float] = []

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            return httpx.Response(503)
        return chat_response(LOCAL_MODEL_BY_ROLE[LocalModelRole.FAST])

    async def no_sleep(delay: float) -> None:
        delays.append(delay)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        gateway = OllamaChatGateway(
            client=client,
            max_retries=2,
            sleep=no_sleep,
        )
        result = await gateway.chat(
            role=LocalModelRole.FAST,
            messages=({"role": "user", "content": "format this"},),
        )

    assert result.model == "phi3:3.8b"
    assert attempts == 3
    assert delays == [0.1, 0.2]


@pytest.mark.asyncio
async def test_non_transient_status_never_retries() -> None:
    attempts = 0
    delays: list[float] = []

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(400)

    async def no_sleep(delay: float) -> None:
        delays.append(delay)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        gateway = OllamaChatGateway(
            client=client,
            max_retries=4,
            sleep=no_sleep,
        )
        with pytest.raises(OllamaGatewayError, match="HTTP 400"):
            await gateway.chat(
                role=LocalModelRole.REASONING,
                messages=({"role": "user", "content": "invalid"},),
            )

    assert attempts == 1
    assert delays == []


@pytest.mark.asyncio
async def test_transient_and_transport_failures_exhaust_without_fallback() -> None:
    transient_attempts = 0

    def transient(_request: httpx.Request) -> httpx.Response:
        nonlocal transient_attempts
        transient_attempts += 1
        return httpx.Response(504)

    async def no_sleep(_delay: float) -> None:
        return None

    async with httpx.AsyncClient(transport=httpx.MockTransport(transient)) as client:
        gateway = OllamaChatGateway(
            client=client,
            max_retries=1,
            sleep=no_sleep,
        )
        with pytest.raises(OllamaUnavailableError, match="transient HTTP 504"):
            await gateway.chat(
                role=LocalModelRole.RESEARCH,
                messages=({"role": "user", "content": "research"},),
            )
    assert transient_attempts == 2

    connection_attempts = 0

    def disconnected(request: httpx.Request) -> httpx.Response:
        nonlocal connection_attempts
        connection_attempts += 1
        raise httpx.ConnectError("offline", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(disconnected)) as client:
        gateway = OllamaChatGateway(
            client=client,
            max_retries=2,
            sleep=no_sleep,
        )
        with pytest.raises(OllamaUnavailableError, match="unavailable"):
            await gateway.tags()
    assert connection_attempts == 3


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, content=b"not-json"),
        httpx.Response(
            200,
            json={
                "model": "unexpected:latest",
                "done": True,
                "message": {"content": "wrong model"},
            },
        ),
        httpx.Response(
            200,
            json={
                "model": "qwen3:8b",
                "done": False,
                "message": {"content": "partial"},
            },
        ),
    ],
)
async def test_chat_rejects_malformed_or_wrong_model_responses(
    response: httpx.Response,
) -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _request: response)
    ) as client:
        gateway = OllamaChatGateway(client=client)
        with pytest.raises((OllamaProtocolError, OllamaModelError)):
            await gateway.chat(
                role=LocalModelRole.REASONING,
                messages=({"role": "user", "content": "fail closed"},),
            )


@pytest.mark.asyncio
async def test_tags_and_health_report_missing_approved_models() -> None:
    inventory = {
        "models": [
            {
                "name": "mistral:latest",
                "digest": "a" * 64,
                "size": 123,
                "modified_at": "2026-07-16T00:00:00Z",
            },
            {
                "name": "qwen3:8b",
                "digest": "b" * 64,
                "size": 456,
                "modified_at": "2026-07-16T00:00:00Z",
            },
        ]
    }
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json=inventory)
        )
    ) as client:
        gateway = OllamaChatGateway(client=client)
        tags = await gateway.tags()
        health = await gateway.health()

    assert [model.name for model in tags] == ["mistral:latest", "qwen3:8b"]
    assert not health.healthy
    assert "mistral" not in health.missing_models
    assert "qwen3:8b" not in health.missing_models
    assert "nomic-embed-text" in health.missing_models
    assert health.base_url == "http://127.0.0.1:11434"


@pytest.mark.asyncio
async def test_owned_client_close_is_idempotent_and_disables_requests() -> None:
    gateway = OllamaChatGateway()
    await gateway.aclose()
    await gateway.aclose()

    with pytest.raises(OllamaGatewayError, match="closed"):
        await gateway.tags()


def test_gateway_bounds_parallel_inference_for_constrained_gpu() -> None:
    with pytest.raises(ValueError, match="must be 1 or 2"):
        OllamaChatGateway(max_parallel_requests=0)
    with pytest.raises(ValueError, match="must be 1 or 2"):
        OllamaChatGateway(max_parallel_requests=3)


@pytest.mark.asyncio
async def test_gateway_serializes_inference_by_default() -> None:
    active = 0
    max_active = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        try:
            await asyncio.sleep(0.005)
            requested_model = json.loads(request.content)["model"]
            return chat_response(requested_model)
        finally:
            active -= 1

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        gateway = OllamaChatGateway(client=client)
        await asyncio.gather(
            gateway.chat(
                role=LocalModelRole.REASONING,
                messages=({"role": "user", "content": "first"},),
            ),
            gateway.chat(
                role=LocalModelRole.FAST,
                messages=({"role": "user", "content": "second"},),
            ),
        )

    assert max_active == 1


@pytest.mark.asyncio
async def test_agent_runner_builds_bounded_non_authoritative_context() -> None:
    requests: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return chat_response(
            LOCAL_MODEL_BY_ROLE[LocalModelRole.RESEARCH],
            content="engineering artifact",
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        runner = OllamaAgentRunner(
            OllamaChatGateway(client=client),
            max_dependency_context_chars=512,
        )
        task = WorkflowTask(
            task_id="synthesize",
            role=LocalModelRole.RESEARCH,
            instructions="Synthesize the engineering findings.",
            depends_on=("audit",),
        )
        dependencies = {
            "audit": TaskResult(
                task_id="audit",
                status=TaskStatus.SUCCEEDED,
                output=AgentOutput(
                    content="A" * 1_000,
                    model="qwen3:8b",
                ),
            )
        }

        output = await runner.run(task, dependencies)

    assert output.content == "engineering artifact"
    assert output.model == "deepseek-r1:8b"
    assert output.metadata == {
        "role": "research",
        "prompt_tokens": 11,
        "completion_tokens": 7,
        "total_duration_ns": 123,
        "load_duration_ns": 45,
    }
    assert len(requests) == 1
    messages = requests[0]["messages"]
    assert isinstance(messages, list)
    system_prompt = messages[0]["content"]
    user_prompt = messages[1]["content"]
    assert "non-authoritative software-engineering" in system_prompt
    assert "Never make market predictions" in system_prompt
    assert "## audit" in user_prompt
    assert "[dependency context truncated]" in user_prompt


@pytest.mark.asyncio
async def test_agent_runner_rejects_inconsistent_dependency_state_before_http() -> None:
    requests = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return chat_response("qwen3:8b")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        runner = OllamaAgentRunner(OllamaChatGateway(client=client))
        task = WorkflowTask(
            task_id="child",
            role=LocalModelRole.REASONING,
            instructions="Create an artifact.",
            depends_on=("parent",),
        )
        with pytest.raises(OllamaGatewayError, match="inconsistent"):
            await runner.run(task, {})

    assert requests == 0
