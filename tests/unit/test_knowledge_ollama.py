"""Contract tests for the local-only async Ollama embedding adapter."""

from __future__ import annotations

import json
import math

import httpx
import pytest

from packages.knowledge.errors import EmbeddingError
from packages.knowledge.ollama import OllamaEmbedder


MODEL = "nomic-embed-text"
DIMENSION = 768


def _embedding_response(
    vectors: list[list[float]], *, model: str = MODEL
) -> httpx.Response:
    # Use a raw body so malformed local responses containing NaN reach the adapter's
    # validation instead of being rejected by httpx's strict response encoder.
    body = json.dumps(
        {"model": model, "embeddings": vectors},
        allow_nan=True,
    ).encode("utf-8")
    return httpx.Response(
        200, content=body, headers={"content-type": "application/json"}
    )


@pytest.mark.asyncio
async def test_embed_sends_one_exact_batch_and_accepts_768_dimensions() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return _embedding_response([[0.25] * DIMENSION, [0.5] * DIMENSION])

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://127.0.0.1:11434",
    ) as client:
        embedder = OllamaEmbedder(
            base_url="http://127.0.0.1:11434",
            model=MODEL,
            dimension=DIMENSION,
            client=client,
        )
        result = await embedder.embed(["first", "second"])
        await embedder.aclose()

    assert len(requests) == 1
    assert requests[0].method == "POST"
    assert requests[0].url.path == "/api/embed"
    assert json.loads(requests[0].content) == {
        "model": MODEL,
        "input": ["first", "second"],
        "truncate": False,
    }
    assert result == [tuple([0.25] * DIMENSION), tuple([0.5] * DIMENSION)]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("vectors", "message"),
    [
        ([[0.1] * DIMENSION], "count does not match"),
        ([[0.1] * (DIMENSION - 1), [0.2] * DIMENSION], "dimension drift"),
    ],
)
async def test_embed_rejects_count_and_dimension_drift(
    vectors: list[list[float]],
    message: str,
) -> None:
    transport = httpx.MockTransport(lambda _request: _embedding_response(vectors))
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://127.0.0.1:11434",
    ) as client:
        embedder = OllamaEmbedder(
            base_url="http://localhost:11434",
            model=MODEL,
            dimension=DIMENSION,
            client=client,
        )
        with pytest.raises(EmbeddingError, match=message):
            await embedder.embed(["first", "second"])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("vector", "message"),
    [
        ([0.0] * DIMENSION, "zero vector"),
        ([math.nan] + [0.1] * (DIMENSION - 1), "NaN or infinity"),
    ],
)
async def test_embed_rejects_zero_and_non_finite_vectors(
    vector: list[float],
    message: str,
) -> None:
    transport = httpx.MockTransport(lambda _request: _embedding_response([vector]))
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://127.0.0.1:11434",
    ) as client:
        embedder = OllamaEmbedder(
            base_url="http://127.0.0.1:11434",
            model=MODEL,
            dimension=DIMENSION,
            client=client,
        )
        with pytest.raises(EmbeddingError, match=message):
            await embedder.embed(["text"])


@pytest.mark.asyncio
async def test_identity_rejects_missing_model_digest() -> None:
    transport = httpx.MockTransport(
        lambda _request: httpx.Response(
            200,
            json={"models": [{"name": f"{MODEL}:latest"}]},
        )
    )
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://127.0.0.1:11434",
    ) as client:
        embedder = OllamaEmbedder(
            base_url="http://127.0.0.1:11434",
            model=MODEL,
            dimension=DIMENSION,
            client=client,
        )
        with pytest.raises(EmbeddingError, match="digest is missing or invalid"):
            await embedder.identity()


@pytest.mark.parametrize(
    "url",
    [
        "https://127.0.0.1:11434",
        "http://192.168.1.20:11434",
        "http://models.example.com:11434",
    ],
)
def test_remote_or_tls_ollama_url_is_rejected(url: str) -> None:
    with pytest.raises(EmbeddingError, match="loopback"):
        OllamaEmbedder(base_url=url, model=MODEL, dimension=DIMENSION)


@pytest.mark.asyncio
async def test_transient_responses_retry_with_bounded_backoff() -> None:
    attempts = 0
    delays: list[float] = []

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            return httpx.Response(503)
        return _embedding_response([[0.25] * DIMENSION])

    async def no_sleep(delay: float) -> None:
        delays.append(delay)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://127.0.0.1:11434",
    ) as client:
        embedder = OllamaEmbedder(
            base_url="http://127.0.0.1:11434",
            model=MODEL,
            dimension=DIMENSION,
            max_retries=2,
            client=client,
            sleep=no_sleep,
        )
        result = await embedder.embed(["retry me"])

    assert attempts == 3
    assert delays == [0.1, 0.2]
    assert len(result[0]) == DIMENSION


@pytest.mark.asyncio
async def test_transient_failure_exhausts_retries_without_fallback() -> None:
    attempts = 0
    delays: list[float] = []

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(503)

    async def no_sleep(delay: float) -> None:
        delays.append(delay)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://127.0.0.1:11434",
    ) as client:
        embedder = OllamaEmbedder(
            base_url="http://127.0.0.1:11434",
            model=MODEL,
            dimension=DIMENSION,
            max_retries=1,
            client=client,
            sleep=no_sleep,
        )
        with pytest.raises(EmbeddingError, match="transient HTTP 503"):
            await embedder.embed(["never degrade"])

    assert attempts == 2
    assert delays == [0.1]


@pytest.mark.asyncio
async def test_connection_failure_exhausts_retries() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        raise httpx.ConnectError("offline", request=request)

    async def no_sleep(_delay: float) -> None:
        return None

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://127.0.0.1:11434",
    ) as client:
        embedder = OllamaEmbedder(
            base_url="http://127.0.0.1:11434",
            model=MODEL,
            dimension=DIMENSION,
            max_retries=2,
            client=client,
            sleep=no_sleep,
        )
        with pytest.raises(EmbeddingError, match="local Ollama is unavailable"):
            await embedder.embed(["no fallback"])

    assert attempts == 3
