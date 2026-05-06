"""
Ray Serve multi-model serving configuration for Sentinel-X.
Routes LLM, embedding, and RL inference to appropriate GPU replicas
with automatic load balancing and failover.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

import ray
from ray import serve  # type: ignore[import]

log = logging.getLogger(__name__)


@serve.deployment(
    num_replicas=2,
    ray_actor_options={"num_gpus": 1, "resources": {"InferenceGPU": 1}},
    max_concurrent_queries=32,
    graceful_shutdown_timeout_s=30,
)
class LLMInferenceService:
    """
    Serves NIM LLM requests via local vLLM or remote NIM API.
    Routes to local GPU if available, falls back to NIM cloud.
    """

    def __init__(self) -> None:
        import os
        self._use_local = os.environ.get("USE_LOCAL_LLM", "false").lower() == "true"
        self._model = os.environ.get("NIM_MODEL", "meta/llama-3.1-70b-instruct")
        if self._use_local:
            self._init_local_vllm()
        else:
            self._init_nim_client()

    def _init_local_vllm(self) -> None:
        try:
            from vllm import LLM, SamplingParams  # type: ignore[import]
            self._llm = LLM(
                model=self._model,
                tensor_parallel_size=1,
                gpu_memory_utilization=0.85,
                max_model_len=8192,
                dtype="bfloat16",
            )
            log.info("Local vLLM initialized: %s", self._model)
        except ImportError:
            log.warning("vLLM not available — falling back to NIM API")
            self._use_local = False
            self._init_nim_client()

    def _init_nim_client(self) -> None:
        import openai, os
        self._client = openai.AsyncOpenAI(
            api_key=os.environ["NIM_API_KEY"],
            base_url=os.environ.get("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1"),
        )

    async def __call__(self, request: dict) -> dict:
        messages  = request.get("messages", [])
        temp      = request.get("temperature", 0.1)
        max_tokens = request.get("max_tokens", 1024)

        if self._use_local:
            from vllm import SamplingParams  # type: ignore[import]
            params = SamplingParams(temperature=temp, max_tokens=max_tokens)
            prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
            out = self._llm.generate([prompt], params)
            return {"content": out[0].outputs[0].text}
        else:
            resp = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=temp,
                max_tokens=max_tokens,
            )
            return {"content": resp.choices[0].message.content}


@serve.deployment(
    num_replicas=1,
    ray_actor_options={"num_gpus": 0.5, "resources": {"InferenceGPU": 0.5}},
    max_concurrent_queries=64,
)
class EmbeddingService:
    """Batch embedding via NIM — up to 64 concurrent requests."""

    def __init__(self) -> None:
        import openai, os
        self._client = openai.AsyncOpenAI(
            api_key=os.environ["NIM_API_KEY"],
            base_url=os.environ.get("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1"),
        )
        self._model = os.environ.get("NIM_EMBED_MODEL", "nvidia/nv-embedqa-e5-v5")

    async def __call__(self, request: dict) -> dict:
        texts = request.get("texts", [])
        resp  = await self._client.embeddings.create(model=self._model, input=texts)
        return {"embeddings": [item.embedding for item in resp.data]}


@serve.deployment(
    num_replicas=1,
    ray_actor_options={"num_gpus": 0.25},
    max_concurrent_queries=16,
)
class PPOInferenceService:
    """Serves the PPO execution agent — hot-loaded from model storage."""

    def __init__(self) -> None:
        import os
        model_path = os.environ.get("PPO_MODEL_PATH", "/models/ppo_execution/latest.zip")
        try:
            from stable_baselines3 import PPO  # type: ignore[import]
            self._model = PPO.load(model_path)
            log.info("PPO model loaded for inference")
        except Exception as e:
            log.warning("PPO model not found: %s — using random policy", e)
            self._model = None

    async def __call__(self, request: dict) -> dict:
        obs = request.get("obs", [0.0] * 40)
        import numpy as np
        obs_arr = np.array(obs, dtype=np.float32)
        if self._model is not None:
            action, _ = self._model.predict(obs_arr, deterministic=True)
            return {"action": int(action)}
        return {"action": 0}  # MARKET order fallback


# ── Deployment ────────────────────────────────────────────────────────────────
def deploy_cluster() -> None:
    if not ray.is_initialized():
        ray.init(address="auto", ignore_reinit_error=True)

    serve.start(
        detached=True,
        http_options=serve.HTTPOptions(host="0.0.0.0", port=8765),
    )

    serve.run(
        LLMInferenceService.bind(),
        name="llm",
        route_prefix="/llm",
    )
    serve.run(
        EmbeddingService.bind(),
        name="embed",
        route_prefix="/embed",
    )
    serve.run(
        PPOInferenceService.bind(),
        name="ppo",
        route_prefix="/ppo",
    )
    log.info("Ray Serve cluster deployed: LLM + Embed + PPO")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    deploy_cluster()
    import time
    while True:
        time.sleep(30)
