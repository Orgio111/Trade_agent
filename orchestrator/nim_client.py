"""
NVIDIA NIM API integration layer with OpenRouter free tier fallback.
Provides intelligent model routing, caching, and batching.
"""
import os
import json
import hashlib
from typing import Optional, AsyncGenerator
from openai import AsyncOpenAI
import redis.asyncio as redis


class NIMOrchestrator:
    """
    Intelligent NIM API routing with caching, batching, and fallback.
    Routes between NVIDIA NIM (primary) and OpenRouter (free fallback).
    """

    def __init__(self):
        # NVIDIA NIM (cloud)
        self.nim_client = AsyncOpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=os.getenv("NVIDIA_API_KEY", ""),
        )

        # OpenRouter (free tier fallback)
        self.openrouter = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY", ""),
        )

        # Redis cache (optional, graceful degradation)
        try:
            self.cache = redis.Redis(
                host=os.getenv("REDIS_HOST", "localhost"),
                port=int(os.getenv("REDIS_PORT", "6379")),
                decode_responses=True,
            )
        except Exception:
            self.cache = None

        # Model routing table
        self.routing_table = {
            "reasoning": "deepseek/deepseek-r1:free",
            "planning": "deepseek/deepseek-v4-flash:free",
            "coding": "qwen/qwen3-coder-480b-a35b-instruct:free",
            "fast": "nvidia/llama-3.1-nemotron-nano-8b-v1:free",
            "analysis": "deepseek/deepseek-chat-v3-0324:free",
            "embedding": "nvidia/nv-embedqa-e5-v5",
            "multimodal": "moonshotai/kimi-vl-a3b-thinking:free",
        }

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

        Args:
            task_type: Type of task ('reasoning', 'fast', 'coding', etc.)
            messages: Chat messages in OpenAI format
            stream: Whether to stream response
            cache_ttl: Cache TTL in seconds
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Response text or async generator for streaming
        """
        # Check cache (non-streaming only)
        if not stream and self.cache:
            cache_key = self._hash_messages(messages)
            cached = await self.cache.get(cache_key)
            if cached:
                return cached

        model = self.routing_table.get(task_type, "deepseek/deepseek-v4-flash:free")

        # Use NIM for NVIDIA models, OpenRouter for everything else
        is_nvidia = model.startswith("nvidia/")
        client = self.nim_client if is_nvidia else self.openrouter
        actual_model = model if not is_nvidia else model.replace("nvidia/", "")

        try:
            if stream:
                return self._stream_inference(client, actual_model, messages, temperature, max_tokens)

            response = await client.chat.completions.create(
                model=actual_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            result = response.choices[0].message.content

            # Cache result
            if self.cache:
                await self.cache.setex(cache_key, cache_ttl, result)

            return result

        except Exception as e:
            # Fallback to OpenRouter if NIM fails
            if is_nvidia:
                fallback_model = "deepseek/deepseek-v4-flash:free"
                try:
                    fallback_response = await self.openrouter.chat.completions.create(
                        model=fallback_model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    return fallback_response.choices[0].message.content
                except Exception:
                    pass
            raise e

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

    async def batch_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for memory search."""
        response = await self.nim_client.embeddings.create(
            model="nvidia/nv-embedqa-e5-v5",
            input=texts,
            encoding_format="float",
        )
        return [e.embedding for e in response.data]

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
