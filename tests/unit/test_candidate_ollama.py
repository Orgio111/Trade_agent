"""Bounded local-only Ollama candidate client."""

import json

import httpx
import pytest

from workers.candidate.ollama import OllamaCandidateClient


@pytest.mark.asyncio
async def test_complete_uses_qwen_json_mode_and_bounded_generation() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "model": "qwen3:8b",
                "done": True,
                "message": {"role": "assistant", "content": '{"side":"HOLD"}'},
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        candidate = OllamaCandidateClient(client=client)
        result = await candidate.complete("features")

    assert result == '{"side":"HOLD"}'
    assert seen["model"] == "qwen3:8b"
    assert seen["format"] == "json"
    assert seen["stream"] is False
    assert seen["options"] == {"temperature": 0, "num_predict": 512}
    assert seen["keep_alive"] == "5m"


@pytest.mark.parametrize(
    "url",
    ["http://example.com:11434", "http://user:pass@localhost:11434"],
)
def test_client_rejects_remote_or_authenticated_ollama(url: str) -> None:
    with pytest.raises(ValueError):
        OllamaCandidateClient(base_url=url)


@pytest.mark.asyncio
async def test_model_or_protocol_mismatch_fails() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "deepseek-r1:8b",
                "done": True,
                "message": {"content": "{}"},
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        candidate = OllamaCandidateClient(client=client)
        with pytest.raises(RuntimeError, match="protocol"):
            await candidate.complete("features")


@pytest.mark.asyncio
async def test_model_digest_is_loaded_from_exact_approved_model() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/tags"
        return httpx.Response(
            200,
            json={"models": [{"name": "qwen3:8b", "digest": "a" * 64}]},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        candidate = OllamaCandidateClient(client=client)
        assert await candidate.model_digest() == "a" * 64


@pytest.mark.asyncio
async def test_missing_approved_model_fails_closed() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"models": []})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        candidate = OllamaCandidateClient(client=client)
        with pytest.raises(RuntimeError, match="not installed"):
            await candidate.model_digest()
