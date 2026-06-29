"""
QUANTEX Inference Layer — Multi-provider intelligent router.

Architecture:
  App → InferenceRouter → Provider (try in order) → Semantic Cache → Response
                          ├── Groq (fastest, free tier)
                          ├── NVIDIA NIM (high quality, free tier)
                          ├── OpenRouter (fallback aggregator)
                          ├── Ollama (local emergency fallback)
                          └── Error → retry chain

  Each provider implements BaseProvider with:
    - infer(task) -> str
    - infer_stream(task) -> AsyncGenerator
    - is_available() -> bool
    - health() -> dict
"""

from .router import InferenceRouter, InferenceTask, RouterConfig
from .providers import GroqProvider, NvidiaNIMProvider, OpenRouterProvider, vLLMProvider, LocalOllamaProvider
from .cache import SemanticCache
from .cost_tracker import CostTracker

__all__ = [
    "InferenceRouter",
    "InferenceTask",
    "RouterConfig",
    "GroqProvider",
    "NvidiaNIMProvider",
    "OpenRouterProvider",
    "vLLMProvider",
    "LocalOllamaProvider",
    "SemanticCache",
    "CostTracker",
]
