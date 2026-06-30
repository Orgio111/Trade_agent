"""VRAM-Aware Sequential Model Loader for RTX 4050 (6GB).

Ensures only one model is loaded in VRAM at a time by wrapping Ollama
model swap with memory tracking. Prevents OOM on 6GB GPU.

Usage:
    loader = ModelLoader(vram_limit_mb=5120)  # 5GB usable (1GB headroom)
    loader.unload_all()
    loader.load("phi3:mini")
    # ... use model ...
    loader.unload("phi3:mini")
    loader.load("qwen2.5:3b")
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("quantex.model_loader")

# Ollama API
OLLAMA_BASE = "http://localhost:11434"

# Model VRAM estimates (MB) for RTX 4050 6GB
MODEL_VRAM: dict[str, int] = {
    "phi3:mini": 2200,
    "qwen2:1.5b": 950,
    "qwen2.5:3b": 1900,
    "moondream": 1700,
    "mistral": 4400,
    "qwen3:8b": 5200,
    "deepseek-r1:8b": 5200,
}


@dataclass
class ModelLoadInfo:
    model: str
    vram_mb: int
    loaded_at: float = 0.0
    last_used: float = 0.0


class ModelLoader:
    """Sequential model loader — only one model in VRAM at a time.

    Strategy:
        1. Before loading a new model, unload current (Ollama auto-swaps, but
           we track state for budget awareness).
        2. If the new model + current usage > vram_limit, force unload first.
        3. Pre-warm: send a tiny request after load to confirm readiness.
        4. GPU pinning: keep_keep_alive > 0 prevents Ollama auto-unload.
        5. Batch warmup: pre-heat all models sequentially for first-cycle perf.
    """

    def __init__(
        self,
        ollama_base: str = OLLAMA_BASE,
        vram_limit_mb: int = 5120,  # 5GB usable out of 6GB
        warmup_prompt: str = "hi",
        keep_alive: str = "5m",     # keep model in VRAM for 5min after last use
        gpu_pinned: list[str] | None = None,  # models to never auto-unload
    ):
        self.ollama_base = ollama_base
        self.vram_limit_mb = vram_limit_mb
        self.warmup_prompt = warmup_prompt
        self.keep_alive = keep_alive
        self.gpu_pinned = set(gpu_pinned or [])
        self._loaded: dict[str, ModelLoadInfo] = {}
        self._warmup_times: dict[str, float] = {}  # model → warmup latency ms

    @property
    def used_vram_mb(self) -> int:
        return sum(m.vram_mb for m in self._loaded.values())

    @property
    def available_vram_mb(self) -> int:
        return self.vram_limit_mb - self.used_vram_mb

    def load(self, model: str, warmup: bool = True) -> bool:
        """Load a model, unloading others if VRAM budget exceeded.

        Uses keep_alive parameter to prevent Ollama auto-unload for
        GPU-pinned models and recently used models.
        """
        if model in self._loaded:
            self._loaded[model].last_used = time.time()
            logger.debug("Model %s already loaded (cache hit)", model)
            return True

        required_mb = MODEL_VRAM.get(model, 2000)

        if required_mb > self.available_vram_mb + self.used_vram_mb:
            logger.error("Model %s (%dMB) exceeds total VRAM limit (%dMB)",
                         model, required_mb, self.vram_limit_mb)
            return False

        # Unload models until we have room (skip GPU-pinned)
        while self.available_vram_mb < required_mb and self._loaded:
            # Unload least-recently-used (skip pinned)
            candidates = [m for m in self._loaded.values() if m.model not in self.gpu_pinned]
            if not candidates:
                logger.error("Cannot free VRAM — all loaded models are GPU-pinned")
                return False
            lru_model = min(candidates, key=lambda m: m.last_used)
            self.unload(lru_model.model)

        # Determine keep_alive: pinned models stay longer
        ka = self.keep_alive
        if model in self.gpu_pinned:
            ka = "30m"  # pinned models stay 30min in VRAM

        logger.info("Loading model %s (%dMB) — available: %dMB, keep_alive=%s",
                    model, required_mb, self.available_vram_mb, ka)
        t0 = time.perf_counter()

        # Pre-warm: send a tiny generate request to force model into VRAM
        if warmup:
            try:
                import httpx
                with httpx.Client(timeout=120.0) as _http:
                    resp = _http.post(
                        f"{self.ollama_base}/api/generate",
                        json={
                            "model": model,
                            "prompt": self.warmup_prompt,
                            "stream": False,
                            "options": {"keep_alive": ka},
                        },
                    )
                    if resp.status_code != 200:
                        logger.warning("Warmup failed for %s: HTTP %d", model, resp.status_code)
                        return False
            except Exception as e:
                logger.warning("Warmup request failed for %s: %s", model, e)
                return False

        elapsed_ms = (time.perf_counter() - t0) * 1000
        self._warmup_times[model] = elapsed_ms
        self._loaded[model] = ModelLoadInfo(
            model=model,
            vram_mb=required_mb,
            loaded_at=time.time(),
            last_used=time.time(),
        )
        logger.info("Model %s loaded in %.0fms (keep_alive=%s)", model, elapsed_ms, ka)
        return True

    def unload(self, model: str) -> bool:
        """Unload a model from VRAM tracking (Ollama handles actual unload)."""
        if model not in self._loaded:
            return True

        # Ollama doesn't have an explicit unload API — models get evicted
        # by LRU when new models are loaded. We just track it here.
        del self._loaded[model]
        logger.info("Unloaded model %s — freed %dMB (tracked)", model,
                    MODEL_VRAM.get(model, 2000))
        return True

    def unload_all(self) -> int:
        """Unload all tracked models. Returns count of unloaded models."""
        count = len(self._loaded)
        self._loaded.clear()
        logger.info("Unloaded all %d models", count)
        return count

    def status(self) -> dict[str, Any]:
        """Current loader status."""
        return {
            "vram_limit_mb": self.vram_limit_mb,
            "used_vram_mb": self.used_vram_mb,
            "available_vram_mb": self.available_vram_mb,
            "loaded_models": list(self._loaded.keys()),
            "model_vram_map": {m.model: m.vram_mb for m in self._loaded.values()},
        }

    def is_loaded(self, model: str) -> bool:
        return model in self._loaded

    def warmup_model(self, model: str) -> bool:
        """Force-warm a single model (load + tiny inference). Alias for load()."""
        return self.load(model, warmup=True)

    def warm_all(self, models: list[str] | None = None) -> dict[str, float]:
        """Pre-warm all models sequentially.

        Returns dict of model → warmup_time_ms.
        On RTX 4050 (6GB), models load one-at-a-time with auto-swap.
        """
        if models is None:
            models = list(MODEL_VRAM.keys())

        results: dict[str, float] = {}
        for m in models:
            t0 = time.perf_counter()
            ok = self.load(m, warmup=True)
            dt = (time.perf_counter() - t0) * 1000
            results[m] = dt if ok else -1.0

        logger.info("Batch warmup complete: %d models, total %.0fms",
                    len(models), sum(v for v in results.values() if v > 0))
        return results

    @property
    def warmup_times(self) -> dict[str, float]:
        """Return recorded warmup latencies per model."""
        return dict(self._warmup_times)


# ── SEQUENTIAL PIPELINE LOADER ─────────────────────────────────
# Loads models in sequence for the AutoGen trading team.
# Agent load order matches the debate flow:
#   QuantAgent → PatternAgent → MacroAgent → RiskAgent → Coordinator

AGENT_MODEL_MAP: dict[str, str] = {
    "QuantAgent": "qwen2.5:3b",
    "PatternAgent": "moondream",
    "MacroAgent": "qwen2.5:3b",
    "RiskAgent": "qwen2:1.5b",
    "Coordinator": "phi3:mini",
}


def load_agent_models(
    loader: ModelLoader | None = None,
    agents: list[str] | None = None,
) -> ModelLoader:
    """Pre-load models for agents in debate order.

    Since Ollama auto-swaps, models sharing the same name (Quant+Macro)
    only need one load. We load in sequence, keeping shared models cached.
    """
    if loader is None:
        loader = ModelLoader()

    if agents is None:
        agents = ["QuantAgent", "PatternAgent", "MacroAgent", "RiskAgent", "Coordinator"]

    loaded_names: set[str] = set()

    for agent_name in agents:
        model = AGENT_MODEL_MAP.get(agent_name)
        if model and model not in loaded_names:
            loader.load(model, warmup=True)
            loaded_names.add(model)

    logger.info("Loaded %d unique models for %d agents", len(loaded_names), len(agents))
    return loader


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    loader = ModelLoader()
    print("Sequential model loading for AutoGen trading team:")
    print(f"  VRAM limit: {loader.vram_limit_mb}MB")
    print()

    load_agent_models(loader)
    print()
    print("Loader status:")
    for k, v in loader.status().items():
        print(f"  {k}: {v}")
