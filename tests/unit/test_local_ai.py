"""Tests for the shared local-only model and health boundary."""

from __future__ import annotations

import httpx
import pytest

from packages.local_ai import (
    LOCAL_MODEL_BY_ROLE,
    LocalModelRole,
    LocalOllamaHealthClient,
    normalize_local_http_url,
)


def test_registry_contains_only_the_six_approved_models() -> None:
    assert dict(LOCAL_MODEL_BY_ROLE) == {
        LocalModelRole.REASONING: "qwen3:8b",
        LocalModelRole.FAST: "phi3:3.8b",
        LocalModelRole.RESEARCH: "deepseek-r1:8b",
        LocalModelRole.VISION: "moondream",
        LocalModelRole.TOOL_FORMATTING: "mistral",
        LocalModelRole.EMBEDDING: "nomic-embed-text",
    }


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("http://localhost:11434", "http://localhost:11434"),
        ("http://127.0.0.1", "http://127.0.0.1:11434"),
        (
            "http://host.docker.internal:11434/",
            "http://host.docker.internal:11434",
        ),
        ("http://[::1]:11434", "http://[::1]:11434"),
    ],
)
def test_local_url_normalization(value: str, expected: str) -> None:
    assert (
        normalize_local_http_url(value, service="Ollama", default_port=11434)
        == expected
    )


@pytest.mark.parametrize(
    "value",
    [
        "https://localhost:11434",
        "http://example.com:11434",
        "http://user:secret@localhost:11434",
        "http://localhost:11434/api/chat",
        "http://localhost:11434?provider=cloud",
    ],
)
def test_local_url_rejects_nonlocal_or_credentialed_endpoints(value: str) -> None:
    with pytest.raises(ValueError):
        normalize_local_http_url(value, service="Ollama", default_port=11434)


@pytest.mark.asyncio
async def test_health_requires_every_approved_local_model() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(
                200,
                json={
                    "models": [
                        {"name": model, "digest": f"sha-{index}", "size": index}
                        for index, model in enumerate(
                            sorted(set(LOCAL_MODEL_BY_ROLE.values())), start=1
                        )
                    ]
                },
            )
        return httpx.Response(200, json={"models": [{"name": "phi3:3.8b"}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        health_client = LocalOllamaHealthClient(client=client)
        health = await health_client.health()

    assert health.healthy is True
    assert health.missing == ()
    assert health.running == ("phi3:3.8b",)


@pytest.mark.asyncio
async def test_health_fails_closed_for_missing_or_malformed_inventory() -> None:
    def missing_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"models": [{"name": "qwen3:8b"}]})

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(missing_handler)
    ) as client:
        health_client = LocalOllamaHealthClient(client=client)
        missing = await health_client.health()
    assert missing.healthy is False
    assert "phi3:3.8b" in missing.missing

    def malformed_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": []})

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(malformed_handler)
    ) as client:
        health_client = LocalOllamaHealthClient(client=client)
        malformed = await health_client.health()
    assert malformed.healthy is False
    assert "failed" in malformed.detail
