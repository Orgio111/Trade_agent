"""
Ray Serve multi-model serving for the Trade Agent.
Deploys GPU-accelerated inference services for LLM, embeddings, PPO execution,
and a CPU-based market data relay.

Usage:
  python infra/ray/serve_config.py

Or via Ray Serve CLI:
  serve run infra.ray.serve_config:deploy_cluster
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any

import ray
from ray import serve

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  LLM Inference Service (GPU)
# ═══════════════════════════════════════════════════════════════════════════════

@serve.deployment(
    num_replicas=2,
    ray_actor_options={"num_gpus": 1, "resources": {"InferenceGPU": 1}},
    max_concurrent_queries=32,
    graceful_shutdown_timeout_s=30,
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 8,
        "target_num_ongoing_requests_per_replica": 16,
    },
)
class LLMInference:
    """
    Serves LLM inference for the council agent and supervisor rationale.
    Routes to a local vLLM instance if available on GPU, otherwise falls
    back to the NVIDIA NIM cloud API.
    """

    def __init__(self) -> None:
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
        import openai

        self._client = openai.AsyncOpenAI(
            api_key=os.environ["NIM_API_KEY"],
            base_url=os.environ.get(
                "NIM_BASE_URL",
                "https://integrate.api.nvidia.com/v1",
            ),
        )

    async def chat(self, messages: list[dict], temperature: float = 0.1,
                   max_tokens: int = 1024) -> str:
        if self._use_local:
            from vllm import SamplingParams  # type: ignore[import]

            params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
            prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
            out = self._llm.generate([prompt], params)
            return out[0].outputs[0].text

        resp = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""

    async def chat_json(self, messages: list[dict], **kwargs: Any) -> dict:
        """Request a JSON response from the LLM."""
        text = await self.chat(messages, **kwargs)
        # Try to extract JSON from the response
        text = text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text)

    async def __call__(self, request: dict) -> dict:
        return {"content": await self.chat(
            request.get("messages", []),
            temperature=request.get("temperature", 0.1),
            max_tokens=request.get("max_tokens", 1024),
        )}


# ═══════════════════════════════════════════════════════════════════════════════
#  Embedding Service (GPU)
# ═══════════════════════════════════════════════════════════════════════════════

@serve.deployment(
    num_replicas=1,
    ray_actor_options={"num_gpus": 0.5, "resources": {"InferenceGPU": 0.5}},
    max_concurrent_queries=64,
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 4,
        "target_num_ongoing_requests_per_replica": 32,
    },
)
class EmbeddingService:
    """Batch embedding generation via NIM for the TurboVec memory engine."""

    def __init__(self) -> None:
        import openai

        self._client = openai.AsyncOpenAI(
            api_key=os.environ["NIM_API_KEY"],
            base_url=os.environ.get(
                "NIM_BASE_URL",
                "https://integrate.api.nvidia.com/v1",
            ),
        )
        self._model = os.environ.get(
            "NIM_EMBED_MODEL",
            "nvidia/nv-embedqa-e5-v5",
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts."""
        resp = await self._client.embeddings.create(
            model=self._model,
            input=texts,
        )
        return [item.embedding for item in resp.data]

    async def __call__(self, request: dict) -> dict:
        texts = request.get("texts", [])
        embeds = await self.embed(texts)
        return {"embeddings": embeds, "dimension": len(embeds[0]) if embeds else 0}


# ═══════════════════════════════════════════════════════════════════════════════
#  PPO Execution Agent (GPU)
# ═══════════════════════════════════════════════════════════════════════════════

@serve.deployment(
    num_replicas=1,
    ray_actor_options={"num_gpus": 0.25},
    max_concurrent_queries=16,
)
class PPOExecution:
    """Serves the stable-baselines3 PPO execution agent for order routing.

    Periodically polls the model registry for newer versions and hot-reloads
    the model without requiring a full deployment restart.
    """

    def __init__(self) -> None:
        import numpy as np

        self._model = None
        self._current_version: int = 0
        self._registry_path = os.environ.get(
            "MODEL_REGISTRY_PATH", "/models/mlops/registry"
        )

        # Initial load
        self._try_load_from_registry()
        if self._model is None:
            model_path = os.environ.get(
                "PPO_MODEL_PATH", "/models/ppo_execution/latest.zip",
            )
            self._try_load_path(model_path)

        # Start background hot-reload loop
        self._reload_interval = int(os.environ.get("SERVE_HOT_RELOAD_S", "300"))
        self._reload_task = asyncio.create_task(self._hot_reload_loop())

    def _try_load_from_registry(self) -> bool:
        """Load the latest model from the model registry."""
        registry_file = os.path.join(self._registry_path, "latest_version.txt")
        if not os.path.isfile(registry_file):
            return False
        try:
            with open(registry_file) as f:
                version_str = f.read().strip()
            if not version_str:
                return False
            version = int(version_str)
            model_file = os.path.join(self._registry_path, f"v{version:04d}", "ppo_model.zip")
            return self._try_load_path(model_file, version)
        except (ValueError, OSError) as exc:
            log.warning("Registry load failed: %s", exc)
            return False

    def _try_load_path(self, path: str, version: int = 0) -> bool:
        """Try to load a PPO model from a specific file path."""
        if not os.path.exists(path):
            return False
        try:
            from stable_baselines3 import PPO  # type: ignore[import]
            self._model = PPO.load(path)
            self._current_version = version
            log.info("PPO model v%d loaded from %s", version, path)
            return True
        except Exception as exc:
            log.warning("Failed to load PPO from %s: %s", path, exc)
            return False

    async def _hot_reload_loop(self) -> None:
        """Periodically check the model registry for newer versions."""
        while True:
            try:
                await asyncio.sleep(self._reload_interval)
                self._try_load_from_registry()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.warning("Hot-reload check error: %s", exc)

    async def predict(self, obs: list[float], deterministic: bool = True) -> dict:
        import numpy as np

        obs_arr = np.array(obs, dtype=np.float32)
        if self._model is not None:
            action, _ = self._model.predict(obs_arr, deterministic=deterministic)
            return {
                "action": int(action),
                "source": "ppo",
                "version": self._current_version,
            }
        return {"action": 0, "source": "fallback", "version": 0}

    async def __call__(self, request: dict) -> dict:
        return await self.predict(
            request.get("obs", [0.0] * 40),
            deterministic=request.get("deterministic", True),
        )

    async def health(self) -> dict:
        """Health check endpoint."""
        return {
            "status": "ready" if self._model is not None else "degraded",
            "version": self._current_version,
            "model_loaded": self._model is not None,
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  Health Check Service
# ═══════════════════════════════════════════════════════════════════════════════

@serve.deployment(
    num_replicas=1,
    ray_actor_options={"num_cpus": 0.1},
    max_concurrent_queries=32,
)
class HealthService:
    """Aggregate health endpoint for all Ray Serve deployments."""

    async def __call__(self, request: dict) -> dict:
        return {
            "status": "ok",
            "service": "trade-agent",
            "deployments": ["llm", "embed", "ppo", "market"],
        }

    async def health(self) -> dict:
        return {
            "status": "ok",
            "service": "trade-agent",
            "timestamp": time.time(),
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  Market Data Relay (CPU)
# ═══════════════════════════════════════════════════════════════════════════════

@serve.deployment(
    num_replicas=1,
    ray_actor_options={"num_cpus": 2},
    max_concurrent_queries=10,
)
class MarketDataRelay:
    """Relays the latest market snapshot from the MarketDataFeed to consumers."""

    def __init__(self) -> None:
        self._latest: dict[str, dict] = {}

    async def update(self, symbol: str, tick: dict) -> None:
        self._latest[symbol] = tick

    async def get_snapshot(self, symbols: list[str] | None = None) -> dict:
        if symbols:
            return {s: self._latest.get(s) for s in symbols}
        return dict(self._latest)

    async def get_prices(self) -> dict[str, float]:
        return {
            s: t.get("close", 0.0)
            for s, t in self._latest.items()
            if t
        }

    async def __call__(self, request: dict) -> dict:
        action = request.get("action", "snapshot")
        if action == "prices":
            return {"prices": await self.get_prices()}
        return {"market": await self.get_snapshot(request.get("symbols"))}


# ═══════════════════════════════════════════════════════════════════════════════
#  Deployment entrypoint
# ═══════════════════════════════════════════════════════════════════════════════

def deploy_cluster() -> None:
    """Initialize Ray and deploy all serving deployments."""
    if not ray.is_initialized():
        ray.init(address="auto", ignore_reinit_error=True)

    serve.start(
        detached=True,
        http_options=serve.HTTPOptions(host="0.0.0.0", port=8765),
    )

    serve.run(LLMInference.bind(), name="llm", route_prefix="/llm")
    serve.run(EmbeddingService.bind(), name="embed", route_prefix="/embed")
    serve.run(PPOExecution.bind(), name="ppo", route_prefix="/ppo")
    serve.run(MarketDataRelay.bind(), name="market", route_prefix="/market")
    serve.run(HealthService.bind(), name="health", route_prefix="/health")

    log.info(
        "Ray Serve cluster deployed: LLM (/llm) + Embed (/embed) "
        "+ PPO (/ppo) + Market (/market) + Health (/health)"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    deploy_cluster()
    import time
    while True:
        time.sleep(30)
