"""
NVIDIA NIM API integration layer with OpenRouter free tier fallback.
Provides intelligent model routing, caching, and batching.

All cloud APIs are optional — system works fully with local Ollama only.
"""
from __future__ import annotations

import json
import logging
import os
import hashlib
from typing import Optional, AsyncGenerator

logger = logging.getLogger("quantex.nim_client")

# ── Optional imports (graceful fallback) ──────────────────────
_OPENAI_AVAILABLE = False
_ASYNC_OPENAI = None
try:
    from openai import AsyncOpenAI
    _ASYNC_OPENAI = AsyncOpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    pass

_REDIS_AVAILABLE = False
try:
    import redis.asyncio as _redis_mod
    _REDIS_AVAILABLE = True
except ImportError:
    pass


class NIMOrchestrator:
    """
    Intelligent NIM API routing with caching, batching, and fallback.

    Routes between NVIDIA NIM (primary) and OpenRouter (free fallback).
    Falls back to local Ollama if no cloud keys are configured.
    """

    def __init__(self):
        # NVIDIA NIM (cloud) — optional
        nim_key = os.getenv("NVIDIA_API_KEY", "")
        self.nim_client = None
        if nim_key and _OPENAI_AVAILABLE:
            self.nim_client = _ASYNC_OPENAI(
                base_url="https://integrate.api.nvidia.com/v1",
                api_key=nim_key,
            )
            logger.info("NIM cloud client configured")
        elif nim_key and not _OPENAI_AVAILABLE:
            logger.warning("NVIDIA_API_KEY set but openai package not installed")
        else:
            logger.info("NIM not configured (no NVIDIA_API_KEY)")

        # OpenRouter (free tier fallback) — optional
        or_key = os.getenv("OPENROUTER_API_KEY", "")
        self.openrouter = None
        if or_key and _OPENAI_AVAILABLE:
            self.openrouter = _ASYNC_OPENAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=or_key,
            )
            logger.info("OpenRouter cloud client configured")
        elif or_key and not _OPENAI_AVAILABLE:
            logger.warning("OPENROUTER_API_KEY set but openai package not installed")

        # Local Ollama fallback — always available
        self.ollama_url = os.getenv("OLLAMA_URL", os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
        self._ollama_available = False

        # Redis cache (optional, graceful degradation)
        self.cache = None
        if _REDIS_AVAILABLE:
            try:
                self.cache = _redis_mod.Redis(
                    host=os.getenv("REDIS_HOST", "localhost"),
                    port=int(os.getenv("REDIS_PORT", "6379")),
                    decode_responses=True,
                )
            except Exception:
                self.cache = None

        # Track cloud availability
        self._cloud_available = bool(self.nim_client or self.openrouter)

        # Model routing table (cloud models)
        self.routing_table = {
            "reasoning": "deepseek/deepseek-r1:free",
            "planning": "deepseek/deepseek-v4-flash:free",
            "coding": "qwen/qwen3-coder-480b-a35b-instruct:free",
            "fast": "nvidia/llama-3.1-nemotron-nano-8b-v1:free",
            "analysis": "deepseek/deepseek-chat-v3-0324:free",
            "embedding": "nvidia/nv-embedqa-e5-v5",
            "multimodal": "moonshotai/kimi-vl-a3b-thinking:free",
        }

        # Local Ollama model mapping (for local-only fallback)
        self.ollama_models = {
            "reasoning": "qwen3:8b",
            "fast": "qwen2.5:3b",
            "analysis": "qwen3:8b",
            "coding": "qwen3:8b",
            "embedding": None,  # No local embedding model
        }

    async def _check_ollama(self) -> bool:
        """Check if Ollama is available."""
        if self._ollama_available:
            return True
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.ollama_url}/api/tags", timeout=aiohttp.ClientTimeout(total=3)) as resp:
                    if resp.status == 200:
                        self._ollama_available = True
                        return True
        except Exception:
            pass
        return False

    @property
    def available_tiers(self) -> list[str]:
        """Return list of available inference tiers."""
        tiers = []
        if self.nim_client:
            tiers.append("nim")
        if self.openrouter:
            tiers.append("openrouter")
        if self._ollama_available:
            tiers.append("ollama")
        if not tiers:
            tiers.append("none_configured")
        return tiers

    def _hash_messages(self, messages: list) -> str:
        """Generate cache key from messages."""
        raw = json.dumps(messages, sort_keys=True)
        return f"nim_cache:{hashlib.sha256(raw.encode()).hexdigest()}"

    async def route_inference(
        self,
        task_type: str,
        messages: list,
        stream: bool = False,
        cache_ttl: int = 300,
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ) -> str | AsyncGenerator:
        """
        Route inference to the appropriate model with caching.

        Fallback chain: NIM → OpenRouter → Local Ollama
        """
        # Check cache (non-streaming only)
        if not stream and self.cache:
            try:
                cache_key = self._hash_messages(messages)
                cached = await self.cache.get(cache_key)
                if cached:
                    return cached
            except Exception:
                pass

        # ── Tier 1: NIM cloud ──
        if self.nim_client:
            model = self.routing_table.get(task_type, "deepseek/deepseek-v4-flash:free")
            is_nvidia = model.startswith("nvidia/")
            actual_model = model if not is_nvidia else model.replace("nvidia/", "")
            client = self.nim_client if is_nvidia else (self.openrouter or self.nim_client)
            try:
                result = await self._cloud_inference(client, actual_model, messages, stream, temperature, max_tokens)
                if result is not None:
                    if not stream and self.cache:
                        try:
                            await self.cache.setex(self._hash_messages(messages), cache_ttl, result)
                        except Exception:
                            pass
                    return result
            except Exception as e:
                logger.debug(f"NIM inference failed: {e}")

        # ── Tier 2: OpenRouter free ──
        if self.openrouter:
            model = self.routing_table.get(task_type, "deepseek/deepseek-v4-flash:free")
            actual_model = model.replace(":free", "")
            try:
                result = await self._cloud_inference(self.openrouter, actual_model, messages, stream, temperature, max_tokens)
                if result is not None:
                    return result
            except Exception as e:
                logger.debug(f"OpenRouter inference failed: {e}")

        # ── Tier 3: Local Ollama ──
        ollama_result = await self._ollama_inference(task_type, messages, temperature, max_tokens)
        if ollama_result is not None:
            return ollama_result

        # All tiers exhausted
        raise RuntimeError(
            f"No inference backend available for task_type={task_type}. "
            f"Configure OLLAMA_URL for local inference, or set NVIDIA_API_KEY/OPENROUTER_API_KEY for cloud."
        )

    async def _cloud_inference(
        self, client, model: str, messages: list, stream: bool,
        temperature: float, max_tokens: int,
    ) -> str | None:
        """Try cloud inference. Returns None on failure."""
        if stream:
            return self._stream_inference(client, model, messages, temperature, max_tokens)
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.debug(f"Cloud inference error ({model}): {e}")
            return None

    async def _stream_inference(self, client, model: str, messages: list, temperature: float, max_tokens: int):
        """Stream inference for real-time agent reasoning display."""
        stream = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def _ollama_inference(
        self, task_type: str, messages: list, temperature: float, max_tokens: int,
    ) -> str | None:
        """Local Ollama inference fallback."""
        if not await self._check_ollama():
            return None

        model = self.ollama_models.get(task_type, "qwen2.5:3b")
        if not model:
            return None

        try:
            import aiohttp
            payload = {
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": temperature, "num_predict": max_tokens},
            }
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.ollama_url}/api/chat",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data.get("message", {}).get("content", "")
                    else:
                        logger.warning(f"Ollama returned HTTP {resp.status}")
        except Exception as e:
            logger.warning(f"Ollama inference failed: {e}")
        return None

    async def batch_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for memory search. Falls back to hash-based if no NIM."""
        if self.nim_client:
            try:
                response = await self.nim_client.embeddings.create(
                    model="nvidia/nv-embedqa-e5-v5",
                    input=texts,
                    encoding_format="float",
                )
                return [e.embedding for e in response.data]
            except Exception as e:
                logger.warning(f"NIM embedding failed: {e}, using hash fallback")

        # Hash-based fallback (deterministic, no API needed)
        import numpy as np
        embeddings = []
        for text in texts:
            rng = np.random.RandomState(abs(hash(text)) % (2**31))
            emb = rng.normal(0, 0.1, 1536).tolist()
            mag = np.linalg.norm(emb)
            embeddings.append([v / mag for v in emb] if mag > 0 else emb)
        return embeddings

    async def parallel_agent_inference(self, agent_tasks: list[dict]) -> list[str]:
        """Run multiple agent inferences in parallel."""
        import asyncio

        async def run_task(task: dict):
            return await self.route_inference(
                task.get("type", "fast"),
                task["messages"],
                temperature=task.get("temperature", 0.1),
            )

        tasks = [run_task(task) for task in agent_tasks]
        return await asyncio.gather(*tasks, return_exceptions=True)
